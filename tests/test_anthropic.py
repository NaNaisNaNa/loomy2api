"""Anthropic <-> OpenAI protocol translation tests."""

from __future__ import annotations

import json
import unittest

from loomy2api import anthropic as anth


class TestRequestTranslation(unittest.TestCase):
    def test_system_and_blocks(self):
        out = anth.anthropic_to_openai({
            "model": "m", "max_tokens": 16, "temperature": 0.2,
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"}]}],
        })
        self.assertEqual(out["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(out["messages"][1], {"role": "user", "content": "hi"})
        self.assertEqual(out["max_tokens"], 16)
        self.assertEqual(out["temperature"], 0.2)

    def test_tools_and_tool_results(self):
        out = anth.anthropic_to_openai({
            "model": "m",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "tu1", "name": "get_weather",
                     "input": {"city": "hz"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1",
                     "content": [{"type": "text", "text": "sunny"}]}]},
            ],
            "tools": [{"name": "get_weather", "description": "w",
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        })
        assistant = out["messages"][0]
        self.assertEqual(assistant["tool_calls"][0]["id"], "tu1")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
                         {"city": "hz"})
        self.assertEqual(out["messages"][1], {"role": "tool", "tool_call_id": "tu1",
                                              "content": "sunny"})
        self.assertEqual(out["tools"][0]["function"]["parameters"], {"type": "object"})
        self.assertEqual(out["tool_choice"],
                         {"type": "function", "function": {"name": "get_weather"}})

    def test_stop_sequences(self):
        out = anth.anthropic_to_openai({"model": "m", "messages": [],
                                        "stop_sequences": ["\n\n"]})
        self.assertEqual(out["stop"], ["\n\n"])


class TestResponseTranslation(unittest.TestCase):
    def test_text_reasoning_and_tool_calls(self):
        got = anth.openai_to_anthropic({
            "id": "x",
            "choices": [{"message": {
                "reasoning_content": "because",
                "content": "done",
                "tool_calls": [{"id": "c1", "function": {
                    "name": "f", "arguments": '{"a":1}'}}]},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }, "m")
        kinds = [b["type"] for b in got["content"]]
        self.assertEqual(kinds, ["thinking", "text", "tool_use"])
        self.assertEqual(got["stop_reason"], "tool_use")
        self.assertEqual(got["usage"], {"input_tokens": 7, "output_tokens": 3})

    def test_length_finish_reason(self):
        got = anth.openai_to_anthropic(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]}, "m")
        self.assertEqual(got["stop_reason"], "max_tokens")


class TestStreamTranslator(unittest.TestCase):
    def _run(self, chunks):
        translator = anth.StreamTranslator("m")
        events = list(translator.start())
        for chunk in chunks:
            events.extend(translator.feed(chunk))
        events.extend(translator.finish())
        return events

    def _snapshot(self, events):
        """Append starts and route deltas by index, as an SDK consumer does."""
        content, opened, arguments = [], set(), {}
        for event, data in events:
            self.assertEqual(event, data["type"])
            if event == "content_block_start":
                index = data["index"]
                self.assertEqual(index, len(content))
                content.append(dict(data["content_block"]))
                opened.add(index)
            elif event == "content_block_delta":
                index, delta = data["index"], data["delta"]
                self.assertIn(index, opened)
                block = content[index]
                if delta["type"] == "input_json_delta":
                    self.assertEqual(block["type"], "tool_use")
                    arguments[index] = arguments.get(index, "") + delta["partial_json"]
                else:
                    field = {"text_delta": "text", "thinking_delta": "thinking"}[
                        delta["type"]]
                    self.assertEqual(block["type"], field)
                    block[field] += delta[field]
            elif event == "content_block_stop":
                index = data["index"]
                self.assertIn(index, opened)
                if content[index]["type"] == "tool_use":
                    content[index]["input"] = json.loads(arguments.get(index) or "{}")
                opened.remove(index)
            elif event in ("message_delta", "message_stop"):
                self.assertFalse(opened)
        self.assertFalse(opened)
        return content

    def _ts_tool_callbacks(self, events):
        """Mirror indexed snapshots plus the TS SDK's last-block callback."""
        content, arguments, callbacks = [], {}, []
        for event, data in events:
            if event == "content_block_start":
                content.append(dict(data["content_block"]))
            elif event == "content_block_delta":
                index = data["index"]
                arguments[index] = arguments.get(index, "") + data["delta"]["partial_json"]
            elif event == "content_block_stop":
                index = data["index"]
                content[index]["input"] = json.loads(arguments.get(index) or "{}")
                callbacks.append(dict(content[-1]))
        return callbacks

    def test_ts_content_block_callbacks_preserve_parallel_tools(self):
        events = self._run([{"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "read_a", "arguments": '{"a":1}'}},
            {"index": 1, "id": "b", "function": {"name": "read_b", "arguments": '{"b":2}'}}]},
            "finish_reason": "tool_calls"}]}])
        callbacks = self._ts_tool_callbacks(events)
        self.assertEqual([block["id"] for block in callbacks], ["a", "b"])
        self.assertEqual([block["input"] for block in callbacks], [{"a": 1}, {"b": 2}])

    def test_fragmented_tool_metadata_and_interleaved_arguments(self):
        events = self._run([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 7, "id": "call_", "function": {
                    "name": "get_", "arguments": '{"city":'}},
                {"index": 2, "id": "call_", "function": {
                    "name": "read_", "arguments": '{"path":"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 2, "id": "b", "function": {"name": "file", "arguments": 'x"}'}},
                {"index": 7, "id": "a", "function": {
                    "name": "weather", "arguments": '"Paris"}'}}]}, "finish_reason": "tool_calls"}]},
        ])
        expected = [
            {"type": "tool_use", "id": "call_a", "name": "get_weather", "input": {"city": "Paris"}},
            {"type": "tool_use", "id": "call_b", "name": "read_file", "input": {"path": "x"}},
        ]
        self.assertEqual(self._snapshot(events), expected)
        self.assertEqual(self._ts_tool_callbacks(events), expected)

    def test_stream_with_reasoning(self):
        events = self._run([
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [], "usage": {"completion_tokens": 9, "points_consumed": 2}},
        ])
        names = [e for e, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        indexes = {d["index"] for e, d in events if e == "content_block_delta"}
        self.assertEqual(indexes, {0, 1})
        self.assertEqual(self._snapshot(events), [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "hi"},
        ])
        kinds = [d["content_block"]["type"] for e, d in events
                 if e == "content_block_start"]
        self.assertEqual(kinds, ["thinking", "text"])
        message_delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(message_delta["usage"]["output_tokens"], 9)
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")

    def test_stream_text_only(self):
        events = self._run([{"choices": [{"delta": {"content": "a"},
                                          "finish_reason": "length"}]}])
        starts = [d["content_block"]["type"] for e, d in events
                  if e == "content_block_start"]
        self.assertEqual(starts, ["text"])
        self.assertEqual([d for e, d in events if e == "message_delta"][0]
                         ["delta"]["stop_reason"], "max_tokens")

    def test_stream_tool_call(self):
        events = self._run([{"choices": [{"delta": {"tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": '{"b":2}'}}]},
            "finish_reason": "tool_calls"}]}])
        starts = [d["content_block"]["type"] for e, d in events
                  if e == "content_block_start"]
        self.assertEqual(starts, ["tool_use"])
        partial = [d for e, d in events if d.get("delta", {}).get("type")
                   == "input_json_delta"][0]
        self.assertEqual(json.loads(partial["delta"]["partial_json"]), {"b": 2})

    def test_full_parallel_tools_after_reasoning_and_text(self):
        for indexed in (True, False):
            with self.subTest(indexed=indexed):
                calls = [{"id": "c1", "function": {
                    "name": "read_a", "arguments": '{"path":"a"}'}},
                         {"id": "c2", "function": {
                    "name": "read_b", "arguments": '{"path":"b"}'}}]
                if indexed:
                    for index, call in enumerate(calls):
                        call["index"] = index
                events = self._run([{"choices": [{"delta": {
                    "reasoning_content": "plan", "content": "reading",
                    "tool_calls": calls}, "finish_reason": "tool_calls"}]}])
                self.assertEqual(self._snapshot(events), [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "c1", "name": "read_a",
                     "input": {"path": "a"}},
                    {"type": "tool_use", "id": "c2", "name": "read_b",
                     "input": {"path": "b"}},
                ])

    def test_interleaved_tool_argument_fragments(self):
        first_a, first_b = '{"path":"', '{"path":'
        middle_a, middle_b = 'a.txt","options":{"limit":', '"b.txt",'
        last_a, last_b = '2},"note":"hi 喵"}', '"flag":true}'
        batches = [
            [{"index": 7, "id": "a", "function": {
                "name": "read_a", "arguments": first_a}},
             {"index": 2, "id": "b", "function": {
                "name": "read_b", "arguments": first_b}}],
            [{"index": 2, "function": {"arguments": middle_b}},
             {"index": 7, "function": {"arguments": middle_a}}],
            [{"index": 2, "function": {"arguments": last_b}}],
            [{"index": 7, "function": {"arguments": last_a}}],
        ]
        translator = anth.StreamTranslator("m")
        events = list(translator.start())
        first_events = list(translator.feed({
            "choices": [{"delta": {"tool_calls": batches[0]}}]}))
        self.assertEqual(first_events, [])
        events.extend(first_events)
        for batch in batches[1:]:
            events.extend(translator.feed({"choices": [{"delta": {"tool_calls": batch}}]}))
        events.extend(translator.feed({
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
        events.extend(translator.finish())
        self.assertEqual(self._snapshot(events), [
            {"type": "tool_use", "id": "a", "name": "read_a", "input": {
                "path": "a.txt", "options": {"limit": 2}, "note": "hi 喵"}},
            {"type": "tool_use", "id": "b", "name": "read_b", "input": {
                "path": "b.txt", "flag": True}},
        ])
        partials = [(data["index"], data["delta"]["partial_json"])
                    for event, data in events if event == "content_block_delta"]
        self.assertEqual(partials, [(0, first_a), (0, middle_a), (0, last_a),
                                    (1, first_b), (1, middle_b), (1, last_b)])
        final_delta = next(data for event, data in events if event == "message_delta")
        self.assertEqual(final_delta["delta"]["stop_reason"], "tool_use")

    def test_tool_arguments_before_metadata(self):
        for name_first in (False, True):
            with self.subTest(name_first=name_first):
                function = {"arguments": '{"a":'}
                if name_first:
                    function["name"] = "lookup"
                events = self._run([
                    {"choices": [{"delta": {"tool_calls": [
                        {"index": 3, "function": function}]}}]},
                    {"choices": [{"delta": {"tool_calls": [
                        {"index": 3, "id": "late_id", "function": {
                            "name": None if name_first else "lookup", "arguments": '1}'}}]}}]},
                ])
                self.assertEqual(self._snapshot(events), [
                    {"type": "tool_use", "id": "late_id", "name": "lookup",
                     "input": {"a": 1}},
                ])

    def test_argument_escape_sequences_split_between_chunks(self):
        expected = {"path": r"C:\tmp\a.txt", "quote": '"hi"', "unicode": "喵"}
        arguments = json.dumps(expected, ensure_ascii=True)
        # Splits after a backslash, inside an escape and inside a Unicode escape.
        chunks = []
        for i, char in enumerate(arguments):
            call = {"index": 0, "function": {"arguments": char}}
            if i == 0:
                call["id"] = "escaped"
                call["function"]["name"] = "write"
            chunks.append({"choices": [{"delta": {"tool_calls": [call]}}]})
        events = self._run(chunks)
        self.assertEqual(self._snapshot(events), [
            {"type": "tool_use", "id": "escaped", "name": "write", "input": expected},
        ])

    def test_text_continues_around_tools_without_losing_arguments(self):
        events = self._run([
            {"choices": [{"delta": {"content": "before"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "t", "function": {
                    "name": "lookup", "arguments": '{"n":'}}]}}]},
            {"choices": [{"delta": {"content": "after"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '1}'}}]}}]},
        ])
        self.assertEqual(self._snapshot(events), [
            {"type": "text", "text": "before"},
            {"type": "text", "text": "after"},
            {"type": "tool_use", "id": "t", "name": "lookup", "input": {"n": 1}},
        ])

    def test_text_and_thinking_stay_live_while_tools_are_buffered(self):
        translator = anth.StreamTranslator("m")
        for kind, field in (("thinking", "reasoning_content"), ("text", "content")):
            events = list(translator.feed({"choices": [{"delta": {field: "live"}}]}))
            self.assertEqual(events[-1][1]["delta"], {"type": kind + "_delta", kind: "live"})
        events = list(translator.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "t", "function": {"name": "f", "arguments": "{}"}}]}}]}))
        self.assertEqual([event for event, _ in events], ["content_block_stop"])
        self.assertEqual(list(translator.feed({"choices": [{"delta": {"content": "more"}}]}))[-1]
                         [1]["delta"], {"type": "text_delta", "text": "more"})
        self.assertEqual([data["content_block"]["name"] for event, data in translator.finish()
                          if event == "content_block_start"], ["f"])

    def test_empty_and_missing_id_tool_payloads(self):
        events = self._run([{"choices": [{"delta": {"tool_calls": [
            {"function": {"name": "empty"}},
            {"function": {"name": "empty_too", "arguments": ""}}]}}]}])
        tools = self._snapshot(events)
        self.assertEqual([tool["input"] for tool in tools], [{}, {}])
        self.assertEqual([tool["name"] for tool in tools], ["empty", "empty_too"])
        self.assertEqual(len({tool["id"] for tool in tools}), 2)
        self.assertTrue(all(tool["id"].startswith("call_") for tool in tools))

    def test_repeated_content_transitions_do_not_reuse_indices(self):
        events = self._run([
            {"choices": [{"delta": {"reasoning_content": "first"}}]},
            {"choices": [{"delta": {"content": "one"}}]},
            {"choices": [{"delta": {"reasoning_content": "second"}}]},
            {"choices": [{"delta": {"content": "two"}}]},
        ])
        self.assertEqual(self._snapshot(events), [
            {"type": "thinking", "thinking": "first"},
            {"type": "text", "text": "one"},
            {"type": "thinking", "thinking": "second"},
            {"type": "text", "text": "two"},
        ])


if __name__ == "__main__":
    unittest.main()
