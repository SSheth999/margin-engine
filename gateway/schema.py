"""Canonical chat schema — the provider-agnostic contract.

The agent harness, the gateway decision logic (detectors/interventions/cost), and
every provider adapter all speak these types. Interventions operate on `ChatRequest`
directly (downshift = swap `model`; restrict = filter `tools`; compact = trim
`messages`), so intervention code never needs to know which backend is active.

Provider adapters translate ChatRequest -> their wire format on the way out, and
their wire response -> ChatResponse on the way back, normalizing token accounting
into the common `Usage` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """Normalized token accounting, common across providers.

    Buckets are ADDITIVE, not overlapping:
      - input_tokens  : uncached input, billed at the full input rate.
      - cached_tokens : input served from a provider cache, billed at the
                        discounted rate (input_rate * cache_read_multiplier).
      - output_tokens : generated tokens, billed at the output rate.

    So total prompt tokens = input_tokens + cached_tokens. This matches Anthropic,
    where `usage.input_tokens` already excludes cache reads (reported separately as
    `cache_read_input_tokens`). Providers without caching (Ollama over the
    OpenAI-compatible endpoint) put the whole prompt in input_tokens and set
    cached_tokens = 0.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    """A model call as the agent sees it, before any provider translation.

    `messages` is a list of {"role", "content", ...} dicts in a provider-neutral
    shape (role in {system, user, assistant, tool}; content is a string, or a list
    of blocks for tool results). `tools` is a list of tool schemas the model may
    call. Interventions mutate this object in place / return a modified copy.
    """

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.0
    # Passthrough bag for provider-specific knobs we don't model explicitly.
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ChatRequest":
        import copy as _copy

        return ChatRequest(
            model=self.model,
            messages=_copy.deepcopy(self.messages),
            tools=_copy.deepcopy(self.tools),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra=_copy.deepcopy(self.extra),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatRequest":
        return cls(
            model=d["model"],
            messages=d.get("messages", []),
            tools=d.get("tools", []),
            max_tokens=d.get("max_tokens", 1024),
            temperature=d.get("temperature", 0.0),
            extra=d.get("extra", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": self.messages,
            "tools": self.tools,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra": self.extra,
        }


@dataclass
class ChatResponse:
    """A model reply, normalized across providers.

    `text` is the assistant's natural-language content (may be empty when the model
    only calls tools). `tool_calls` is the list of tool invocations requested.
    `stop_reason` is a coarse, normalized reason: "end_turn" | "tool_use" | "max_tokens".
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "usage": self.usage.to_dict(),
            "stop_reason": self.stop_reason,
            "model": self.model,
        }
