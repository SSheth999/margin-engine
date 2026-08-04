"""OpenAI adapter — hosted backend.

OpenAI's Chat Completions wire format is the same OpenAI-compatible shape the Ollama
adapter already speaks, so this reuses OllamaProvider's request/response translation
and only adds (a) the Authorization header and (b) correct cached-token accounting.

Note on caching: OpenAI reports cached prompt tokens in
usage.prompt_tokens_details.cached_tokens, and they are a SUBSET of prompt_tokens
(unlike Anthropic, where cache reads are separate). So we split prompt_tokens into
uncached (input_tokens, full rate) and cached (cached_tokens, discounted) to match the
additive Usage contract.

Uses gpt-4.1-family models by default (standard chat completions + tools); avoids the
gpt-5 reasoning models, which use different params (max_completion_tokens, no temperature).
"""

from __future__ import annotations

from typing import Any

import httpx

from gateway.schema import ChatRequest, ChatResponse, ToolCall, Usage
from providers.ollama_provider import OllamaProvider, _STOP_REASON_MAP


def _is_reasoning_model(model: str) -> bool:
    """GPT-5 / o-series reasoning models use a different param surface."""
    return model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4")


class OpenAIProvider(OllamaProvider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 600.0):
        super().__init__(base_url=base_url, timeout=timeout)
        self._api_key = api_key

    def _to_wire(self, request: ChatRequest) -> dict[str, Any]:
        # Reasoning models (gpt-5*, o-series) reject max_tokens/temperature and need
        # reasoning_effort='none' to use function tools over chat completions.
        if not _is_reasoning_model(request.model):
            return super()._to_wire(request)
        messages = [self._msg_to_wire(m) for m in request.messages]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_completion_tokens": max(request.max_tokens, 2048),
            "reasoning_effort": "none",
            "stream": False,
        }
        if request.tools:
            payload["tools"] = [self._tool_to_wire(t) for t in request.tools]
        # Pass through extras except temperature (unsupported here).
        payload.update({k: v for k, v in request.extra.items() if k != "temperature"})
        return payload

    def complete(self, request):
        payload = self._to_wire(request)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            # GPT-5 models disagree on the reasoning_effort value that supports tools
            # (e.g. gpt-5.6-* need 'none', gpt-5-mini needs 'minimal'). Auto-fall back.
            if (
                resp.status_code == 400
                and "reasoning_effort" in resp.text
                and payload.get("reasoning_effort") == "none"
            ):
                payload["reasoning_effort"] = "minimal"
                resp = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            resp.raise_for_status()
            data = resp.json()
        return self._from_wire(data)

    def _from_wire(self, data: dict[str, Any]) -> ChatResponse:
        choice = data["choices"][0]
        message = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")

        tool_calls: list[ToolCall] = []
        import json as _json
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments", "{}")
            try:
                args = _json.loads(raw) if isinstance(raw, str) else (raw or {})
            except _json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""),
                                       arguments=args))

        u = data.get("usage", {}) or {}
        prompt = u.get("prompt_tokens", 0)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        usage = Usage(
            input_tokens=max(0, prompt - cached),  # uncached portion, full rate
            output_tokens=u.get("completion_tokens", 0),
            cached_tokens=cached,                    # discounted
        )

        stop = _STOP_REASON_MAP.get(finish, "end_turn")
        if tool_calls and stop == "end_turn":
            stop = "tool_use"
        return ChatResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop,
            model=data.get("model", ""),
        )
