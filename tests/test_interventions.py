"""Intervention rewrites: downshift, restrict, compact (with pairing safety)."""

from gateway.interventions import compact, downshift, restrict_tool
from gateway.schema import ChatRequest


def _req_with_turns(n_turns: int) -> ChatRequest:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_turns):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{i}", "name": "search", "arguments": {"q": i}}],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
    return ChatRequest(model="big", messages=msgs, tools=[{"name": "search"}])


def test_downshift_swaps_model_only():
    r = ChatRequest(model="big", messages=[{"role": "user", "content": "x"}])
    out = downshift(r, "small")
    assert out.model == "small"
    assert r.model == "big"  # original untouched


def test_restrict_removes_named_tool():
    r = ChatRequest(
        model="m",
        messages=[],
        tools=[{"name": "search"}, {"name": "read_file"}],
    )
    out = restrict_tool(r, "search")
    assert [t["name"] for t in out.tools] == ["read_file"]


def test_compact_keeps_system_first_and_recent_and_notes_elision():
    req = _req_with_turns(10)
    out = compact(req, keep_recent_turns=3)

    roles = [m["role"] for m in out.messages]
    assert roles[0] == "system"
    # first non-system message is the task, carrying the elision note
    assert out.messages[1]["role"] == "user"
    assert "elided" in out.messages[1]["content"]
    # compaction actually dropped messages
    assert len(out.messages) < len(req.messages)
    # no orphan tool result at the boundary (first msg after task is assistant)
    assert out.messages[2]["role"] == "assistant"


def test_compact_noop_when_short():
    req = _req_with_turns(2)
    out = compact(req, keep_recent_turns=3)
    assert len(out.messages) == len(req.messages)


def test_compaction_gain_measures_dropped_fraction():
    """The break-even guard needs a real measure of how much a trim removes."""
    from gateway.interventions import compaction_gain

    req = _req_with_turns(10)
    gain = compaction_gain(req, keep_recent_turns=3)
    assert 0.0 < gain < 1.0
    # keeping fewer recent turns must drop strictly more
    assert compaction_gain(req, keep_recent_turns=2) > gain


def test_compaction_gain_zero_when_nothing_to_drop():
    from gateway.interventions import compaction_gain

    assert compaction_gain(_req_with_turns(2), keep_recent_turns=3) == 0.0


def test_compaction_gain_matches_what_compact_removes():
    """Guard and action must agree, or the guard is gating on fiction."""
    from gateway.interventions import compact, compaction_gain

    def chars(msgs):
        return sum(
            len(m.get("content") or "") + sum(len(str(tc.get("arguments", ""))) for tc in m.get("tool_calls") or [])
            for m in msgs
            if m.get("role") != "system"
        )

    # Big turn bodies so the fixed-size elision note stays negligible.
    req = _req_with_turns(12)
    for m in req.messages:
        if m["role"] == "tool":
            m["content"] = m["content"] * 200
    gain = compaction_gain(req, keep_recent_turns=3)
    out = compact(req, keep_recent_turns=3)
    # the elision note adds a few chars to the task turn, so allow a small tolerance
    actual = 1 - chars(out.messages) / chars(req.messages)
    assert abs(actual - gain) < 0.02
