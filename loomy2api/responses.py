"""Incremental Chat Completions -> Responses SSE translation."""
from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, Iterable


class StreamTranslator:
    """Own response/item identities while upstream text and arguments arrive."""

    def __init__(self, model: str):
        self.sequence = 0
        self.items = []
        self.text = None
        self.reasoning = None
        self.tools = {}
        self.finish_reason = None
        self.response = {
            "id": "resp_" + uuid.uuid4().hex[:24], "object": "response",
            "created_at": int(time.time()), "status": "in_progress",
            "model": model, "output": self.items, "output_text": "",
            "error": None, "incomplete_details": None, "usage": None,
        }

    def _event(self, kind: str, **data: Any) -> Dict[str, Any]:
        event = dict(copy.deepcopy(data), type=kind, sequence_number=self.sequence)
        self.sequence += 1
        return event

    def _add(self, item: Dict[str, Any]):
        index = len(self.items)
        self.items.append(item)
        return index, self._event("response.output_item.added", output_index=index, item=item)

    def _location(self, index: int) -> Dict[str, Any]:
        return {"output_index": index, "item_id": self.items[index]["id"]}

    def start(self) -> Iterable:
        yield self._event("response.created", response=self.response)
        yield self._event("response.in_progress", response=self.response)

    def _tool_arguments(self, tool) -> Iterable:
        if tool["index"] is None:
            if not tool["id"] or not tool["name"]:
                return
            tool["index"], event = self._add({
                "type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24],
                "call_id": tool["id"], "name": tool["name"],
                "arguments": "", "status": "in_progress"})
            yield event
        item = self.items[tool["index"]]
        for arguments in tool["pending"]:
            item["arguments"] += arguments
            yield self._event("response.function_call_arguments.delta",
                              **self._location(tool["index"]), delta=arguments)
        tool["pending"].clear()

    def feed(self, obj: Dict[str, Any]) -> Iterable:
        choices = obj.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if self.reasoning is None:
                self.reasoning, event = self._add({
                    "type": "reasoning", "id": "rs_" + uuid.uuid4().hex[:24],
                    "status": "in_progress", "summary": []})
                yield event
                self.items[self.reasoning]["summary"].append({"type": "summary_text", "text": ""})
                yield self._event("response.reasoning_summary_part.added",
                                  **self._location(self.reasoning), summary_index=0,
                                  part={"type": "summary_text", "text": ""})
            self.items[self.reasoning]["summary"][0]["text"] += reasoning
            yield self._event("response.reasoning_summary_text.delta",
                              **self._location(self.reasoning), summary_index=0, delta=reasoning)
        content = delta.get("content")
        if content:
            if self.text is None:
                self.text, event = self._add({
                    "type": "message", "id": "msg_" + uuid.uuid4().hex[:24],
                    "status": "in_progress", "role": "assistant", "content": []})
                yield event
                part = {"type": "output_text", "text": "", "annotations": []}
                self.items[self.text]["content"].append(part)
                yield self._event("response.content_part.added",
                                  **self._location(self.text), content_index=0, part=part)
            self.items[self.text]["content"][0]["text"] += content
            self.response["output_text"] += content
            yield self._event("response.output_text.delta", **self._location(self.text),
                              content_index=0, delta=content, logprobs=[])
        for call in delta.get("tool_calls") or []:
            key = call.get("index")
            if key is None:
                key = call.get("id") or 0
            tool = self.tools.setdefault(key, {"id": "", "name": "", "index": None, "pending": [], "defer": False})
            fn = call.get("function") or {}
            if call.get("id") or fn.get("name"):
                if tool["index"] is not None:
                    raise ValueError("Upstream changed tool metadata after argument streaming began")
                tool["id"] += call.get("id") or ""
                tool["name"] += fn.get("name") or ""
            arguments = fn.get("arguments")
            if arguments and (call.get("id") or fn.get("name")):
                tool["defer"] = True
            if arguments:
                tool["pending"].append(arguments)
            # Metadata-bearing chunks may contain fragmented IDs/names. Publish
            # only after a subsequent argument-only chunk, or at finalization.
            if arguments and not tool["defer"] and not call.get("id") and not fn.get("name"):
                yield from self._tool_arguments(tool)
        self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def _usage(self, usage):
        details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return {"input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "input_tokens_details": {"cached_tokens": prompt_details.get("cached_tokens", 0)},
                "output_tokens_details": {"reasoning_tokens": details.get("reasoning_tokens", 0)}}

    def finish(self, usage: Dict[str, Any]) -> Iterable:
        for tool in self.tools.values():
            if not tool["id"]:
                tool["id"] = "call_" + uuid.uuid4().hex[:24]
            if not tool["name"]:
                raise ValueError("Upstream returned incomplete tool identity")
            yield from self._tool_arguments(tool)
        reason = {"length": "max_output_tokens", "content_filter": "content_filter"}.get(self.finish_reason)
        status = "incomplete" if reason else "completed"
        for index, item in enumerate(self.items):
            location = self._location(index)
            if item["type"] == "message":
                part = item["content"][0]
                yield self._event("response.output_text.done", **location,
                                  content_index=0, text=part["text"], logprobs=[])
                yield self._event("response.content_part.done", **location,
                                  content_index=0, part=part)
            elif item["type"] == "reasoning":
                part = item["summary"][0]
                yield self._event("response.reasoning_summary_text.done", **location,
                                  summary_index=0, text=part["text"])
                yield self._event("response.reasoning_summary_part.done", **location,
                                  summary_index=0, part=part)
            else:
                yield self._event("response.function_call_arguments.done", **location,
                                  arguments=item["arguments"], name=item["name"])
            item["status"] = status
            yield self._event("response.output_item.done", output_index=index, item=item)
        self.response.update(status=status, usage=self._usage(usage),
                             incomplete_details={"reason": reason} if reason else None)
        yield self._event("response." + status, response=self.response)

    def fail(self, message: str, usage: Dict[str, Any]) -> Dict[str, Any]:
        for item in self.items:
            if item["status"] == "in_progress":
                item["status"] = "incomplete"
        self.response.update(status="failed", error={"code": "server_error", "message": message},
                             usage=self._usage(usage))
        return self._event("response.failed", response=self.response)
