"""Responses streaming timing, cancellation, failures and fragmented tools."""
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as N
from unittest.mock import Mock, patch

from loomy2api.httpc import open_stream
from loomy2api.pool import PoolError
from test_responses_stream import make_handler
from test_stream_lifecycle import Chunks, frame

TEXT = frame({"choices": [{"delta": {"content": "代码"}}]})
STOP = frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
USAGE = frame({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "points_consumed": 0}})
DONE = b"data: [DONE]\n\n"


def events(h):
    return [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
            if line.startswith("data: ")]


def use_chunks(h, source):
    connection = N(close=Mock())
    h.gateway.models_client.stream.return_value = connection, source
    return connection


class ResponsesLiveStreamTests(unittest.TestCase):
    def test_text_is_written_while_upstream_is_still_waiting(self):
        release, seen = threading.Event(), threading.Event()
        class Controlled(Chunks):
            def read1(self, size):
                self.thread = threading.current_thread()
                if getattr(self, "sent", False) is False:
                    self.sent = True
                    return TEXT
                release.wait(2)
                return super().read1(size)
        class Sink(io.BytesIO):
            def write(self, data):
                result = super().write(data)
                if b"event: response.output_text.delta" in data:
                    seen.set()
                return result
        h = make_handler(sink=Sink())
        upstream = Controlled([STOP, USAGE, DONE])
        use_chunks(h, upstream)
        errors = []
        def run():
            try: h._responses()
            except Exception as exc: errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(seen.wait(1), "first delta waited for upstream completion")
            self.assertTrue(worker.is_alive())
            self.assertNotIn(b"event: response.completed", h.wfile.getvalue())
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(upstream.thread.is_alive())
        self.assertEqual(events(h)[-1]["response"]["usage"]["input_tokens"], 7)
        h._chat_upstream.assert_not_called()
        args = h.gateway.models_client.stream.call_args
        self.assertTrue(args.args[2]["stream"])
        self.assertTrue(args.args[2]["stream_options"]["include_usage"])
        self.assertEqual(args.kwargs["proxy"], "http://account.invalid")
        h.gateway.pool.release.assert_called_once()

    def test_tool_arguments_arrive_before_completion_with_complete_metadata(self):
        release, seen = threading.Event(), threading.Event()
        metadata = frame({"choices": [{"delta": {"tool_calls": [{"index": 0,
            "id": "call_weather", "function": {"name": "weather", "arguments": ""}}]}}]})
        argument = lambda value: frame({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": value}}]}}]})
        class Controlled(Chunks):
            count = 0
            def read1(self, size):
                self.count += 1
                if self.count == 3: release.wait(2)
                return super().read1(size)
        class Sink(io.BytesIO):
            def write(self, data):
                result = super().write(data)
                if b"event: response.function_call_arguments.delta" in data: seen.set()
                return result
        h = make_handler(sink=Sink())
        use_chunks(h, Controlled([metadata, argument('{"city":'), argument('"上海"}'),
            frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}), USAGE, DONE]))
        worker = threading.Thread(target=h._responses)
        worker.start()
        try:
            self.assertTrue(seen.wait(1))
            self.assertTrue(worker.is_alive())
            self.assertFalse(any(e["type"] == "response.completed" for e in events(h)))
        finally:
            release.set()
            worker.join(2)
        final = events(h)[-1]["response"]
        self.assertEqual(json.loads(final["output"][0]["arguments"]), {"city": "上海"})

    def test_heartbeat_during_silence_and_upstream_comment_only_traffic(self):
        for comments in (False, True):
            with self.subTest(comments=comments):
                release, beat = threading.Event(), threading.Event()
                class Quiet(Chunks):
                    def read1(self, size):
                        if not release.is_set():
                            if comments:
                                release.wait(0.003)
                                return b": upstream heartbeat\n\n"
                            release.wait(2)
                        return super().read1(size)
                class Sink(io.BytesIO):
                    def write(self, data):
                        result = super().write(data)
                        if data == b": keepalive\n\n": beat.set()
                        return result
                h = make_handler(sink=Sink())
                use_chunks(h, Quiet([STOP, DONE]))
                with patch("loomy2api.server._RESPONSES_HEARTBEAT_SECONDS", 0.02):
                    worker = threading.Thread(target=h._responses)
                    worker.start()
                    try:
                        self.assertTrue(beat.wait(1), "heartbeat blocked behind upstream read")
                        self.assertEqual([e["type"] for e in events(h)],
                                         ["response.created", "response.in_progress"])
                    finally:
                        release.set()
                        worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(events(h)[-1]["type"], "response.completed")

    def test_split_utf8_and_usage_after_finish_are_preserved(self):
        h = make_handler()
        split = TEXT.index("代".encode()) + 1
        use_chunks(h, Chunks([TEXT[:split], TEXT[split:], STOP, USAGE, DONE]))
        h._responses()
        emitted = events(h)
        self.assertEqual("".join(e["delta"] for e in emitted if e["type"] == "response.output_text.delta"), "代码")
        self.assertEqual(emitted[-1]["response"]["usage"]["total_tokens"], 10)
        self.assertEqual([e["sequence_number"] for e in emitted], list(range(len(emitted))))
        self.assertEqual(h.gateway.usage.record.call_args.kwargs["usage"]["points_consumed"], 0)

    def test_upstream_failure_never_becomes_completed_or_replays(self):
        cases = [([TEXT], 502), ([TEXT, ConnectionResetError("reset")], 502),
                 ([TEXT, STOP, ConnectionResetError("reset after finish")], 502),
                 ([TEXT, TimeoutError("idle")], 504),
                 ([TEXT, b"data: invalid-json\n\n"], 502),
                 ([TEXT, frame({"error": {"message": "quota", "code": 429}})], 429)]
        for chunks, status in cases:
            with self.subTest(status=status, chunks=len(chunks)):
                h = make_handler()
                h.gateway.cfg["max_retries"] = 2
                conn = use_chunks(h, Chunks(chunks))
                h._responses()
                emitted = events(h)
                self.assertEqual(emitted[-1]["type"], "response.failed")
                self.assertEqual(emitted[-1]["response"]["output_text"], "代码")
                self.assertFalse(any(e["type"] == "response.completed" for e in emitted))
                self.assertEqual(h.gateway.usage.record.call_args.kwargs["status"], status)
                h.gateway.models_client.stream.assert_called_once()
                h.gateway.pool.release.assert_called_once()
                h.gateway.pool.report_success.assert_not_called()
                conn.close.assert_called_once()

    def test_length_and_content_filter_are_incomplete(self):
        for finish, reason in (("length", "max_output_tokens"), ("content_filter", "content_filter")):
            h = make_handler()
            use_chunks(h, Chunks([TEXT, frame({"choices": [{"delta": {}, "finish_reason": finish}]}), USAGE, DONE]))
            h._responses()
            terminal = events(h)[-1]
            self.assertEqual(terminal["type"], "response.incomplete")
            self.assertEqual(terminal["response"]["incomplete_details"], {"reason": reason})
            self.assertEqual(terminal["response"]["output"][0]["status"], "incomplete")
            h.gateway.pool.report_success.assert_called_once()

    def test_fragmented_parallel_tool_metadata_and_reasoning_keep_identity(self):
        deltas = [
            {"reasoning_content": "先想"}, {"content": "查询"},
            {"tool_calls": [{"index": 7, "id": "call_", "function": {"name": "get_", "arguments": '{"city":'}},
                            {"index": 2, "id": "call_", "function": {"name": "su", "arguments": '{"a":'}}]},
            {"tool_calls": [{"index": 7, "function": {"arguments": " "}}]},
            {"tool_calls": [{"index": 2, "id": "sum", "function": {"name": "m", "arguments": '1}'}},
                            {"index": 7, "id": "weather", "function": {"name": "weather", "arguments": '"上海"}'}}]},
        ]
        h = make_handler()
        chunks = [frame({"choices": [{"delta": d}]}) for d in deltas]
        chunks += [frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}), USAGE, DONE]
        use_chunks(h, Chunks(chunks))
        h._responses()
        emitted = events(h)
        final = emitted[-1]["response"]
        self.assertEqual(final["output"][0]["summary"][0]["text"], "先想")
        self.assertEqual(final["output_text"], "查询")
        tools = [i for i in final["output"] if i["type"] == "function_call"]
        self.assertEqual([(i["call_id"], i["name"], json.loads(i["arguments"])) for i in tools],
                         [("call_weather", "get_weather", {"city": "上海"}), ("call_sum", "sum", {"a": 1})])
        added = [e for e in emitted if e["type"] == "response.output_item.added"]
        self.assertEqual([e["output_index"] for e in added], list(range(4)))
        self.assertTrue(all(e["item"]["arguments"] == "" for e in added if e["item"]["type"] == "function_call"))
        self.assertTrue(any(e["type"] == "response.reasoning_summary_text.delta" for e in emitted))

    def test_connection_retry_excludes_failed_account_and_releases_each_lease(self):
        h = make_handler()
        first = h.gateway.pool.acquire.return_value
        second = N(name="second", session="offline2", proxy="")
        h.gateway.cfg["max_retries"] = 1
        h.gateway.pool.acquire.side_effect = [first, second]
        h.gateway.models_client.stream.side_effect = [TimeoutError("connect"),
            (N(close=Mock()), Chunks([TEXT, STOP, USAGE, DONE]))]
        h._responses()
        self.assertEqual(h.gateway.pool.acquire.call_args_list[1].kwargs["exclude"], [first.name])
        self.assertEqual([c.args[0] for c in h.gateway.pool.release.call_args_list], [first, second])
        self.assertEqual([c.kwargs["status"] for c in h.gateway.usage.record.call_args_list], [504, 200])
        self.assertEqual(events(h)[-1]["type"], "response.completed")

    def test_pre_stream_http_errors_and_no_accounts_keep_http_status(self):
        h = make_handler()
        upstream = N(status=400, read=Mock(return_value=b"bad request"))
        use_chunks(h, upstream)
        h._responses()
        h._error.assert_called_once_with(400, "bad request", "upstream_error")
        h.send_response.assert_not_called()
        h.gateway.pool.release.assert_called_once()
        empty = make_handler()
        empty.gateway.pool.acquire.side_effect = PoolError("empty")
        empty._responses()
        empty._error.assert_called_once_with(503, "empty", "upstream_error")
        empty.gateway.pool.release.assert_not_called()

    def test_disconnect_during_response_headers_releases_without_account_penalty(self):
        for method in ("send_response", "send_header", "end_headers"):
            h = make_handler()
            getattr(h, method).side_effect = BrokenPipeError("closed")
            h._responses()
            h.gateway.pool.release.assert_called_once()
            h.gateway.pool.report_failure.assert_not_called()
            self.assertEqual(h.gateway.usage.record.call_args.kwargs["status"], 499)

    def test_slow_client_bounds_upstream_prefetch_and_cancels_reader(self):
        blocked, release = threading.Event(), threading.Event()
        class Flood:
            status = 200
            count = 0
            def read1(self, size):
                self.count += 1
                return TEXT
            def read(self, size=-1):
                raise AssertionError("must use read1")
        class Slow(io.BytesIO):
            def write(self, data):
                if b"event: response.output_text.delta" in data:
                    blocked.set()
                    release.wait(2)
                    raise BrokenPipeError("cancel")
                return super().write(data)
        source, h = Flood(), make_handler(sink=Slow())
        use_chunks(h, source)
        worker = threading.Thread(target=h._responses)
        worker.start()
        try:
            self.assertTrue(blocked.wait(1))
            self.assertLessEqual(source.count, 18)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        h.gateway.pool.release.assert_called_once()
        h.gateway.pool.report_failure.assert_not_called()

    def test_cancellation_interrupts_real_connection_close_socket_read(self):
        release = threading.Event()
        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.flush()
                release.wait(2)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        serving.start()
        before = set(threading.enumerate())
        class Cancel(io.BytesIO):
            def write(self, data):
                if data.startswith(b": keepalive"): raise BrokenPipeError("cancel")
                return super().write(data)
        h = make_handler(sink=Cancel())
        def connect(*args, **kwargs):
            conn, resp = open_stream("http://127.0.0.1:%d/" % server.server_port, method="GET", timeout=2)
            self.assertIsNone(conn.sock)  # response.fp now owns the socket
            return conn, resp
        h.gateway.models_client.stream.side_effect = connect
        try:
            with patch("loomy2api.server._RESPONSES_HEARTBEAT_SECONDS", 0.02):
                started = time.monotonic()
                h._responses()
                self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(any(t.name == "responses-upstream" and t not in before for t in threading.enumerate()))
            h.gateway.pool.release.assert_called_once()
            h.gateway.pool.report_failure.assert_not_called()
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            serving.join(1)


if __name__ == "__main__":
    unittest.main()
