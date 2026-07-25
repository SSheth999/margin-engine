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
