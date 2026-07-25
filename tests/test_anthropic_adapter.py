"""Anthropic adapter translation, against a mocked SDK client. No network / no key.

Proves the hosted backend path (canonical -> Anthropic wire -> canonical) works so
flipping config/providers.yaml to `anthropic` needs no code change.
"""

from types import SimpleNamespace

from gateway.schema import ChatRequest
from providers.anthropic_provider import AnthropicProvider


class _MockMessages:
    def __init__(self, captured):
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="working on it"),
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="read_file", input={"path": "x.txt"}
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=1200, output_tokens=80, cache_read_input_tokens=300
            ),
            stop_reason="tool_use",
            model="claude-sonnet-4-5",
        )


class _MockClient:
    def __init__(self, captured):
        self.messages = _MockMessages(captured)


def _make_request():
    return ChatRequest(
        model="claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "read x.txt"},
            {
                "role": "assistant",
                "content": "let me look",
                "tool_calls": [
                    {"id": "tu_0", "name": "list_dir", "arguments": {"path": "."}}
                ],
            },
            {"role": "tool", "tool_call_id": "tu_0", "content": "x.txt"},
        ],
        tools=[
            {
                "name": "read_file",
                "description": "read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )


def test_request_translation_and_response_normalization():
    captured: dict = {}
    provider = AnthropicProvider(client=_MockClient(captured))

    resp = provider.complete(_make_request())

    # System pulled out to top-level param, not a message.
    assert captured["system"] == "be helpful"
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user", "assistant", "user"]  # user / assistant(tool_use) / tool_result-as-user

    # Assistant tool call became a tool_use block.
    asst = captured["messages"][1]["content"]
    assert any(b["type"] == "tool_use" and b["name"] == "list_dir" for b in asst)
    # Tool result became a tool_result block in a following user turn.
    tr = captured["messages"][2]["content"]
    assert tr[0]["type"] == "tool_result" and tr[0]["tool_use_id"] == "tu_0"

    # Tools mapped parameters -> input_schema.
    assert captured["tools"][0]["input_schema"]["properties"]["path"]["type"] == "string"

    # Response normalized: text + tool call + additive usage buckets.
    assert resp.text == "working on it"
    assert resp.tool_calls[0].name == "read_file"
    assert resp.usage.input_tokens == 1200
    assert resp.usage.cached_tokens == 300
    assert resp.stop_reason == "tool_use"
