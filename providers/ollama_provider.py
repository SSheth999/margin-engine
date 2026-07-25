"""Ollama adapter — dev backend.

Talks to a local Ollama daemon via its OpenAI-compatible endpoint
(POST {base_url}/chat/completions). Token counts come from the real `usage` block
Ollama returns, so cost accounting is exact against real inference (only the rate
card is synthetic). Ollama does not report prompt caching over this endpoint, so
`cached_tokens` is always 0.

Canonical <-> OpenAI-wire message translation lives here.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from gateway.schema import ChatRequest, ChatResponse, ToolCall, Usage
from providers.base import Provider

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, request: ChatRequest) -> ChatResponse:
        payload = self._to_wire(request)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return self._from_wire(data)

    # --- translation ---------------------------------------------------------

    def _to_wire(self, request: ChatRequest) -> dict[str, Any]:
        messages = [self._msg_to_wire(m) for m in request.messages]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.tools:
            payload["tools"] = [self._tool_to_wire(t) for t in request.tools]
        payload.update(request.extra)
        return payload

    @staticmethod
    def _tool_to_wire(tool: dict[str, Any]) -> dict[str, Any]:
        # Canonical tool schema: {"name", "description", "parameters" (json schema)}.
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _msg_to_wire(msg: dict[str, Any]) -> dict[str, Any]:
        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in msg["tool_calls"]
                ],
            }
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"],
            }
        return {"role": role, "content": msg.get("content", "")}

    def _from_wire(self, data: dict[str, Any]) -> ChatResponse:
        choice = data["choices"][0]
        message = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        usage_raw = data.get("usage", {}) or {}
        usage = Usage(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
            cached_tokens=0,  # not reported over the OpenAI-compatible endpoint
        )

        stop_reason = _STOP_REASON_MAP.get(finish, "end_turn")
        if tool_calls and stop_reason == "end_turn":
            stop_reason = "tool_use"

        return ChatResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
            model=data.get("model", ""),
        )
