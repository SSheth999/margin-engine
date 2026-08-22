"""The four interventions, applied by rewriting the outgoing canonical ChatRequest.

Because interventions operate on the provider-agnostic ChatRequest, they are identical
regardless of backend:
  - downshift : swap `model` to the cheap sibling.
  - compact   : trim `messages` (drop stale middle turns; keep system + task + recent).
  - restrict  : remove the looping tool from `tools`.
  - block     : not applied here — signalled by arms via Decision(block=True), because
                it means "don't call the provider at all".

Compaction works at TURN granularity so tool_use/tool_result pairing is never broken
(an orphaned tool result would be rejected by both the OpenAI and Anthropic wire formats).
A turn is either a standalone user/system message, or an assistant message plus the
tool-result messages that answer it.
"""

from __future__ import annotations

from gateway.schema import ChatRequest


def downshift(request: ChatRequest, cheap_model: str) -> ChatRequest:
    r = request.copy()
    r.model = cheap_model
    return r


def restrict_tool(request: ChatRequest, tool_name: str) -> ChatRequest:
    r = request.copy()
    r.tools = [t for t in r.tools if t.get("name") != tool_name]
    return r


def _group_turns(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """Split into (leading system messages, list of non-system turns).

    A turn groups an assistant message with the immediately following tool results,
    so compaction never splits a tool_use from its tool_result.
    """
    systems = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    turns: list[list[dict]] = []
    for m in rest:
        if m.get("role") == "tool" and turns:
            turns[-1].append(m)
        else:
            turns.append([m])
    return systems, turns


def _turn_chars(turn: list[dict]) -> int:
    total = 0
    for m in turn:
        c = m.get("content", "")
        total += len(c) if isinstance(c, str) else len(str(c))
        for tc in m.get("tool_calls") or []:
            total += len(str(tc.get("arguments", "")))
    return total


def compaction_gain(request: ChatRequest, keep_recent_turns: int) -> float:
    """Fraction of the prompt (by characters) that compacting would drop, 0..1.

    Used to gate compaction, because compaction is not free under cached pricing. It
    invalidates the provider's prompt-cache prefix, so the surviving prompt R gets
    re-billed once at the FULL input rate instead of the cached rate, while the dropped
    tokens D save only the cached rate on each later call. With a cache discount of
    `m` (e.g. 0.25):

        one-off cost  = R * (1 - m) * input_rate
        saving/call   = D * m       * input_rate
        break-even    = R * (1 - m) / (D * m) more calls   [= 3R/D when m = 0.25]

    So a small trim taken often is strictly worse than a large trim taken once: drop
    half the prompt and you need 3 more calls just to break even. In the first full
    benchmark, compaction fired 3.6x per affected run and dropped little each time —
    cached tokens went 10,577 -> 0 on the very next call, uncached input 3,440 -> 7,187,
    and the step cost went UP ($0.0174 -> $0.0197). This function makes the trim size
    checkable before paying for it.
    """
    _, turns = _group_turns(request.messages)
    if len(turns) <= keep_recent_turns + 1:
        return 0.0
    kept = {0, *range(len(turns) - keep_recent_turns, len(turns))}
    dropped_chars = sum(_turn_chars(t) for i, t in enumerate(turns) if i not in kept)
    total_chars = sum(_turn_chars(t) for t in turns)
    if total_chars <= 0:
        return 0.0
    return dropped_chars / total_chars


def compact(request: ChatRequest, keep_recent_turns: int) -> ChatRequest:
    """Keep system messages + the first (task) turn + the last N turns.

    Dropping the stale middle is where the token savings come from (cf. AgentDiet).
    If there's nothing to drop, the request is returned effectively unchanged.
    """
    r = request.copy()
    systems, turns = _group_turns(r.messages)

    if len(turns) <= keep_recent_turns + 1:
        return r  # nothing meaningful to compact

    first_turn = list(turns[0])  # shallow copy of the turn's message list
    recent = turns[-keep_recent_turns:]
    dropped = len(turns) - 1 - len(recent)

    # Fold the elision note into the first (task) message rather than inserting a
    # new turn — a separate user message would create two consecutive user turns,
    # which Anthropic rejects. Non-first turns are assistant-led in a ReAct loop, so
    # system -> user(task+note) -> assistant(recent...) alternates correctly.
    note = f"\n\n[{dropped} earlier step(s) elided to stay within budget.]"
    head = dict(first_turn[0])
    if isinstance(head.get("content"), str):
        head["content"] = head["content"] + note
    first_turn[0] = head

    new_messages: list[dict] = list(systems)
    new_messages.extend(first_turn)
    for turn in recent:
        new_messages.extend(turn)

    r.messages = new_messages
    return r
