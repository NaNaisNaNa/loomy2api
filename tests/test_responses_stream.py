"""Offline Responses SSE contract tests against the actual request handler."""
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from loomy2api.server import Handler

TEXT = "你好，Responses stream! " * 12
TOOLS = [{"id": "call_weather", "type": "function", "function": {
    "name": "weather", "arguments": '{"city":"上海"}'}},
         {"id": "call_sum", "type": "function", "function": {
             "name": "sum", "arguments": '{"a":1,"b":2}'}}]


def make_handler(content=TEXT, tools=(), stream=True, sink=None):
    h = object.__new__(Handler)
    h.gateway = SimpleNamespace(resolve_model=lambda _: "fixture-model",
                                max_output_tokens=lambda _: 32768)
    h._read_json = mock.Mock(return_value={"input": "hello", "stream": stream})
    h._maybe_compress = mock.Mock()
    chat = {"choices": [{"message": {"content": content, "tool_calls": tools}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}}
    h._chat_upstream = mock.Mock(return_value=(200, json.dumps(chat).encode()))
    h.wfile = io.BytesIO() if sink is None else sink
    for name in ("send_response", "send_header", "end_headers", "_json"):
        setattr(h, name, mock.Mock())
    return h


class ResponsesStreamTestCase(unittest.TestCase):
    def consume(self, handler):
        frames = [block.split("\ndata: ", 1)
                  for block in handler.wfile.getvalue().decode().split("\n\n") if block]
        events = [(header.removeprefix("event: "), json.loads(data))
                  for header, data in frames]
        self.assertEqual([e["sequence_number"] for _, e in events], list(range(len(events))))
        self.assertEqual(events[0][0], "response.created")
        self.assertEqual(events[-1][0], "response.completed")
        snapshot = None
        for name, event in events:
            self.assertEqual(event["type"], name)
            kind = event["type"]  # SDK-style dispatch uses JSON, not the SSE header.
            if kind == "response.created":
                snapshot = copy.deepcopy(event["response"])
                self.assertEqual(snapshot["status"], "in_progress")
                self.assertEqual(snapshot["output"], [])
                self.assertEqual(snapshot["output_text"], "")
            elif kind == "response.completed":
                snapshot["status"] = "completed"
                self.assertEqual(snapshot, event["response"])
            elif kind == "response.output_item.added":
                item = copy.deepcopy(event["item"])
                self.assertEqual(event["output_index"], len(snapshot["output"]))
                self.assertEqual(item["status"], "in_progress")
                self.assertEqual(item.get("arguments", item.get("content")),
                                 "" if item["type"] == "function_call" else [])
                snapshot["output"].append(item)
            else:
                item = snapshot["output"][event["output_index"]]
                if "item_id" in event:
                    self.assertEqual(event["item_id"], item["id"])
                if kind == "response.function_call_arguments.delta":
                    item["arguments"] += event["delta"]
                elif kind == "response.function_call_arguments.done":
                    self.assertEqual(item["arguments"], event["arguments"])
                    self.assertEqual(item["name"], event["name"])
                elif kind == "response.content_part.added":
                    self.assertEqual(event["content_index"], len(item["content"]))
                    self.assertEqual(event["part"]["text"], "")
                    item["content"].append(copy.deepcopy(event["part"]))
                elif kind == "response.output_text.delta":
                    self.assertEqual(event["logprobs"], [])
                    item["content"][event["content_index"]]["text"] += event["delta"]
                    snapshot["output_text"] += event["delta"]
                elif kind == "response.output_text.done":
                    self.assertEqual(event["logprobs"], [])
                    self.assertEqual(item["content"][event["content_index"]]["text"], event["text"])
                elif kind == "response.content_part.done":
                    self.assertEqual(item["content"][event["content_index"]], event["part"])
                elif kind == "response.output_item.done":
                    item["status"] = "completed"
                    self.assertEqual(item, event["item"])
                else:
                    self.fail("Unexpected event type: " + kind)
        return snapshot, events

    def test_text_reconstructs_once_from_empty_snapshots(self):
        h = make_handler()
        h._responses()
        response, events = self.consume(h)
        self.assertEqual(response["output_text"], TEXT)
        self.assertEqual(response["output"][0]["content"][0]["text"], TEXT)
        deltas = [e["delta"] for _, e in events if e["type"] == "response.output_text.delta"]
        self.assertGreater(len(deltas), 1)
        self.assertEqual("".join(deltas), TEXT)
        names = [name for name, _ in events]
        self.assertEqual(names[:3], ["response.created", "response.output_item.added",
                                    "response.content_part.added"])
        self.assertEqual(names[-4:], ["response.output_text.done", "response.content_part.done",
                                     "response.output_item.done", "response.completed"])
        h.send_header.assert_any_call("Content-Type", "text/event-stream; charset=utf-8")
        h._json.assert_not_called()

    def test_parallel_tools_keep_distinct_arguments_indexes_and_item_ids(self):
        h = make_handler(tools=TOOLS)
        h._responses()
        response, events = self.consume(h)
        self.assertEqual([i["type"] for i in response["output"]],
                         ["function_call", "function_call", "message"])
        for index, tool in enumerate(TOOLS):
            item = response["output"][index]
            self.assertEqual(item["id"], tool["id"])
            self.assertEqual(item["call_id"], tool["id"])
            self.assertEqual(item["arguments"], tool["function"]["arguments"])
            self.assertEqual(json.loads(item["arguments"]), json.loads(tool["function"]["arguments"]))
        argument_events = [e for _, e in events if e["type"].startswith("response.function_call_arguments.")]
        self.assertEqual([e["output_index"] for e in argument_events], [0, 0, 1, 1])
        self.assertEqual([e["item_id"] for e in argument_events],
                         [TOOLS[0]["id"]] * 2 + [TOOLS[1]["id"]] * 2)
        self.assertTrue(all(e["output_index"] == 2 for _, e in events
                            if e["type"].startswith("response.output_text.")))

    def test_nonstream_matches_stream_final_response_without_sse(self):
        plain, streamed = make_handler(tools=TOOLS, stream=False), make_handler(tools=TOOLS)
        with mock.patch("loomy2api.server.uuid.uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                mock.patch("loomy2api.server.time.time", return_value=1700000000):
            plain._responses()
            streamed._responses()
        final, _ = self.consume(streamed)
        plain._json.assert_called_once_with(200, final)
        plain.send_response.assert_not_called()
        self.assertEqual(plain.wfile.getvalue(), b"")
        self.assertEqual(final["usage"]["total_tokens"], 18)

    def test_empty_output_still_completes_cleanly(self):
        h = make_handler(content="")
        h._responses()
        final, events = self.consume(h)
        self.assertEqual(final["output"], [])
        self.assertEqual(len(events), 2)

    def test_disconnect_stops_at_each_write_or_flush_boundary(self):
        reference = make_handler(tools=TOOLS)
        reference._responses()
        _, events = self.consume(reference)
        for operation in ("write", "flush"):
            for error in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                for fail_at in range(1, len(events) + 1):
                    with self.subTest(operation=operation, error=error.__name__, fail_at=fail_at):
                        sink = mock.Mock()
                        getattr(sink, operation).side_effect = [None] * (fail_at - 1) + [error("closed")]
                        h = make_handler(tools=TOOLS, sink=sink)
                        h._responses()
                        self.assertEqual(sink.write.call_count, fail_at)
                        self.assertEqual(sink.flush.call_count, fail_at - (operation == "write"))
                        self.assertTrue(h.close_connection)


if __name__ == "__main__":
    unittest.main()
