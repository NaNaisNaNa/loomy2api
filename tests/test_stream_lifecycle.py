"""Wire-level regression fixtures for timely and truthful SSE forwarding."""
import io
import json
import threading
import unittest
from http.client import HTTPResponse, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as N
from unittest.mock import Mock

from loomy2api.httpc import open_stream
from loomy2api.server import Handler


def frame(obj):
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


TEXT = frame({"choices": [{"delta": {"content": "代码"}}]})
STOP = frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
USAGE = frame({"choices": [], "usage": {"points_consumed": 7}})


class Chunks:
    status = 200

    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read1(self, size):
        chunk = next(self.chunks, b"")
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def read(self, size=-1):
        raise AssertionError("SSE must not wait for a fixed-size read")


def handler(chunks):
    acc = N(name="fixture", session="offline", proxy="")
    h = object.__new__(Handler)
    h.gateway = N(cfg={"max_retries": 0}, log=Mock(), usage=N(record=Mock()),
                  pool=N(acquire=Mock(return_value=acc), release=Mock(),
                         report_success=Mock(), report_failure=Mock()),
                  models_client=N(stream=Mock(return_value=(N(close=Mock()), Chunks(chunks)))))
    h.wfile = io.BytesIO()
    h.send_response = Mock()
    h.send_header = Mock()
    h.end_headers = Mock()
    h._error = Mock()
    return h


def events(h):
    result = []
    for block in h.wfile.getvalue().decode().split("\n\n"):
        lines = block.splitlines()
        data = next((s[6:] for s in lines if s.startswith("data: ")), None)
        if data and data != "[DONE]":
            result.append(json.loads(data))
    return result


class StreamLifecycleTests(unittest.TestCase):
    def invoke(self, route, h):
        if route == "aggregate":
            return h._chat_stream_to_json({}, {}, "fixture-model")
        if route == "messages":
            h._messages_stream({}, "fixture-model")
        else:
            h._chat_stream({}, {}, "fixture-model")

    def test_incomplete_eof_is_failure_on_every_adapter(self):
        for route in ("aggregate", "messages", "chat"):
            with self.subTest(route=route):
                h = handler([TEXT])
                result = self.invoke(route, h)
                h.gateway.pool.report_success.assert_not_called()
                h.gateway.pool.report_failure.assert_called_once()
                h.gateway.pool.release.assert_called_once()
                if route == "aggregate":
                    self.assertEqual(result[0], 502)
                    self.assertIn("error", json.loads(result[1]))
                else:
                    wire = events(h)
                    self.assertTrue(any("error" in e for e in wire))
                    self.assertFalse(any(e.get("type") == "message_stop" for e in wire))
                    self.assertNotIn(b"data: [DONE]", h.wfile.getvalue())

    def test_error_object_cannot_become_empty_success(self):
        for route in ("aggregate", "messages", "chat"):
            with self.subTest(route=route):
                h = handler([frame({"error": {"message": "rate limited", "code": 429}})])
                result = self.invoke(route, h)
                h.gateway.pool.report_success.assert_not_called()
                self.assertEqual(h.gateway.pool.report_failure.call_args.args[1], 429)
                h.gateway.pool.release.assert_called_once()
                if route == "aggregate":
                    self.assertEqual(result[0], 429)
                else:
                    self.assertTrue(any("error" in e for e in events(h)))

    def test_timeout_and_incomplete_http_body_do_not_report_success(self):
        for failure in (TimeoutError("idle"), IncompleteRead(b"partial")):
            for route in ("aggregate", "messages", "chat"):
                with self.subTest(route=route, failure=type(failure).__name__):
                    h = handler([TEXT, failure])
                    self.invoke(route, h)
                    h.gateway.pool.report_success.assert_not_called()
                    h.gateway.pool.report_failure.assert_called_once()
                    h.gateway.pool.release.assert_called_once()

    def test_split_utf8_usage_and_terminal_markers_survive(self):
        payload = TEXT + STOP + USAGE + b"data: [DONE]\n\n"
        split = payload.index("代".encode()) + 1
        for route in ("aggregate", "messages", "chat"):
            with self.subTest(route=route):
                h = handler([payload[:split], payload[split:]])
                result = self.invoke(route, h)
                h.gateway.pool.report_failure.assert_not_called()
                self.assertEqual(h.gateway.pool.report_success.call_args.args[1]["points_consumed"], 7)
                h.gateway.pool.release.assert_called_once()
                if route == "aggregate":
                    self.assertEqual(json.loads(result[1])["choices"][0]["message"]["content"], "代码")
                elif route == "chat":
                    self.assertEqual(h.wfile.getvalue(), payload)
                else:
                    self.assertTrue(any(e.get("type") == "message_stop" for e in events(h)))

    def test_finish_reason_or_done_and_unterminated_final_line_are_accepted(self):
        for terminal in (STOP.rstrip(), b"data: [DONE]"):
            for route in ("aggregate", "messages", "chat"):
                with self.subTest(route=route, terminal=terminal):
                    h = handler([TEXT, terminal])
                    self.invoke(route, h)
                    h.gateway.pool.report_success.assert_called_once()

    def test_messages_preserves_upstream_heartbeat_as_ping(self):
        h = handler([b": heartbeat\n\n", STOP])
        self.invoke("messages", h)
        self.assertEqual([e["type"] for e in events(h)].count("ping"), 1)
        self.assertEqual(h.gateway.models_client.stream.call_args.kwargs["timeout"], 120)

    def test_no_replay_after_partial_stream_has_reached_client(self):
        h = handler([TEXT, TimeoutError("idle")])
        h.gateway.cfg["max_retries"] = 2
        self.invoke("chat", h)
        h.gateway.pool.acquire.assert_called_once()
        self.assertIn("代码", h.wfile.getvalue().decode())

    def test_initial_incomplete_chunked_response_releases_and_closes(self):
        class Socket:
            def makefile(self, *args):
                return io.BytesIO(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")

        for route in ("aggregate", "messages", "chat"):
            with self.subTest(route=route):
                h = handler([])
                conn = Mock()
                response = HTTPResponse(Socket())
                response.begin()
                h.gateway.models_client.stream.return_value = conn, response
                self.invoke(route, h)
                h.gateway.pool.release.assert_called_once()
                h.gateway.pool.report_failure.assert_called_once()
                h.gateway.pool.report_success.assert_not_called()
                conn.close.assert_called_once()

    def test_header_disconnect_releases_without_penalizing_upstream(self):
        for route in ("messages", "chat"):
            for failure in (BrokenPipeError, ConnectionResetError):
                with self.subTest(route=route, failure=failure.__name__):
                    h = handler([TEXT, STOP])
                    conn = h.gateway.models_client.stream.return_value[0]
                    h.end_headers.side_effect = failure("client closed headers")
                    self.invoke(route, h)
                    h.gateway.pool.release.assert_called_once()
                    h.gateway.pool.report_failure.assert_not_called()
                    h.gateway.pool.report_success.assert_not_called()
                    conn.close.assert_called_once()

    def test_body_disconnect_releases_without_penalizing_upstream(self):
        for route in ("messages", "chat"):
            for operation in ("write", "flush"):
                with self.subTest(route=route, operation=operation):
                    h = handler([TEXT, STOP])
                    conn = h.gateway.models_client.stream.return_value[0]
                    h.wfile = Mock()
                    getattr(h.wfile, operation).side_effect = ConnectionResetError("client closed body")
                    self.invoke(route, h)
                    h.gateway.pool.release.assert_called_once()
                    h.gateway.pool.report_failure.assert_not_called()
                    h.gateway.pool.report_success.assert_not_called()
                    conn.close.assert_called_once()

    def test_upstream_reset_after_data_is_an_explicit_failure(self):
        for route in ("aggregate", "messages", "chat"):
            with self.subTest(route=route):
                h = handler([TEXT, ConnectionResetError("upstream reset")])
                result = self.invoke(route, h)
                h.gateway.pool.report_failure.assert_called_once()
                h.gateway.pool.report_success.assert_not_called()
                h.gateway.pool.release.assert_called_once()
                if route == "aggregate":
                    self.assertEqual(result[0], 502)
                else:
                    self.assertTrue(any("error" in e for e in events(h)))

    def test_unterminated_upstream_frame_does_not_swallow_error_event(self):
        h = handler([TEXT.rstrip()])
        self.invoke("chat", h)
        wire = events(h)
        self.assertEqual(len(wire), 2)
        self.assertIn("choices", wire[0])
        self.assertIn("error", wire[1])
        self.assertNotIn(b"data: [DONE]", h.wfile.getvalue())

    def test_done_stops_reads_without_dropping_usage_after_choice_finish(self):
        payload = TEXT + STOP + USAGE + b"data: [DONE]\n\n"
        for route in ("aggregate", "messages", "chat"):
            with self.subTest(route=route):
                h = handler([payload, TimeoutError("must not read after DONE")])
                self.invoke(route, h)
                h.gateway.pool.report_failure.assert_not_called()
                h.gateway.pool.report_success.assert_called_once()
                self.assertEqual(h.gateway.pool.report_success.call_args.args[1]["points_consumed"], 7)
                h.gateway.pool.release.assert_called_once()

    def test_small_chunk_is_forwarded_before_upstream_eof(self):
        flushed, release, downstream = threading.Event(), threading.Event(), threading.Event()

        class Upstream(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                packet = TEXT + STOP
                self.wfile.write(("%x\r\n" % len(packet)).encode() + packet + b"\r\n")
                self.wfile.flush()
                flushed.set()
                release.wait(3)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                self.close_connection = True

        class Sink(io.BytesIO):
            def write(self, data):
                result = super().write(data)
                downstream.set()
                return result

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream.daemon_threads = True
        serving = threading.Thread(target=upstream.serve_forever, daemon=True)
        serving.start()
        h = handler([])
        h.wfile = Sink()
        h.gateway.models_client.stream.side_effect = lambda *a, **kw: open_stream(
            "http://127.0.0.1:%s/" % upstream.server_port, body=b"{}", timeout=3)
        worker = threading.Thread(target=lambda: self.invoke("chat", h), daemon=True)
        worker.start()
        try:
            self.assertTrue(flushed.wait(2))
            self.assertTrue(downstream.wait(1), "a flushed small frame must precede upstream EOF")
            self.assertIn(TEXT, h.wfile.getvalue())
        finally:
            release.set()
            worker.join(4)
            upstream.shutdown()
            upstream.server_close()
        self.assertFalse(worker.is_alive())
        h.gateway.pool.report_success.assert_called_once()


if __name__ == "__main__":
    unittest.main()
