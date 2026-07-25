"""Anthropic adapter — hosted backend (final-matrix).

Uses the native `anthropic` SDK for exact token + cache accounting. Reads the API
key from ANTHROPIC_API_KEY. In this build pass it is exercised only by a unit test
with a mocked SDK response; flipping config/providers.yaml to `anthropic` activates
it for real with no code change.

Canonical <-> Anthropic Messages-API translation lives here. Key differences from
the OpenAI shape: `system` is a top-level param (not a message), tool calls are
`tool_use` content blocks on assistant turns, and tool results are `tool_result`
blocks inside a following user turn.
"""

from __future__ import annotations

from typing import Any

from gateway.schema import ChatRequest, ChatResponse, ToolCall, Usage
from providers.base import Provider

_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
}


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client: Any | None = None):
        # Lazy import so machines without the key/lib can still use Ollama.
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic()

    def complete(self, request: ChatRequest) -> ChatResponse:
        system, messages = self._to_wire(request)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = [self._tool_to_wire(t) for t in request.tools]
        kwargs.update(request.extra)

        resp = self._client.messages.create(**kwargs)
        return self._from_wire(resp)

    # --- translation ---------------------------------------------------------

    @staticmethod
    def _tool_to_wire(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        }

    def _to_wire(self, request: ChatRequest) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []

        for msg in request.messages:
            role = msg["role"]
            if role == "system":
                system_parts.append(msg.get("content", ""))
                continue

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
                # Merge consecutive tool results into one user turn.
                if messages and messages[-1]["role"] == "user" and isinstance(
                    messages[-1]["content"], list
                ):
                    messages[-1]["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
                continue

            if role == "assistant" and msg.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc.get("arguments", {}),
                        }
                    )
                messages.append({"role": "assistant", "content": content})
                continue

            messages.append({"role": role, "content": msg.get("content", "")})

        return "\n\n".join(p for p in system_parts if p), messages

    def _from_wire(self, resp: Any) -> ChatResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cached_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )

        return ChatResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=_STOP_REASON_MAP.get(resp.stop_reason, "end_turn"),
            model=getattr(resp, "model", ""),
        )
