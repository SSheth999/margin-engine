"""Mock provider — a deterministic test double / failure-mode simulator.

NOT a real backend. Two jobs:
  1. Let the full rig run with zero inference/network (gateway lifecycle, headers,
     cost accounting, run-log schema, orchestrator spawn/teardown).
  2. Deliberately reproduce failure modes on the hard tasks so the detection ->
     intervention -> labeling -> C-oracle pipeline is exercised end-to-end:
       - hard_unfindable_search  -> TOOL LOOP  (repeats the same search forever)
       - hard_open_ended_report  -> CONTEXT BLOWOUT (writes an ever-longer report)
     Easy/medium tasks terminate cheaply so interventions correctly do NOT fire.

Token counts are synthesized from message length so cost accounting has real numbers.
Genuine failure modes from a real model are the point of the manual validation step;
this simulator only proves the wiring is correct.
"""

from __future__ import annotations

from gateway.schema import ChatRequest, ChatResponse, ToolCall, Usage


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _prior_tool_names(messages: list[dict]) -> list[str]:
    names = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                names.append(tc["name"])
    return names


def _first_user_prompt(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _last_tool_result(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "tool":
            return m.get("content", "")
    return ""


class MockProvider:
    name = "mock"

    def __init__(self, *_, **__):
        pass

    def complete(self, request: ChatRequest) -> ChatResponse:
        messages = request.messages
        prompt = _first_user_prompt(messages).lower()
        prior = _prior_tool_names(messages)
        tool_names = {t["name"] for t in request.tools}

        input_tokens = _approx_tokens(
            "".join(
                str(m.get("content", ""))
                + "".join(str(tc.get("arguments", "")) for tc in (m.get("tool_calls") or []))
                for m in messages
            )
        )

        action = self._choose_action(prompt, prior, tool_names, messages)
        if action is not None:
            name, args = action
            out_text = ""
            tool_calls = [ToolCall(id=f"call_{len(prior)}", name=name, arguments=args)]
            stop = "tool_use"
        else:
            out_text = self._finish_text(prompt, messages)
            tool_calls = []
            stop = "end_turn"

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=_approx_tokens(out_text) + 20,
            cached_tokens=0,
        )
        return ChatResponse(
            text=out_text, tool_calls=tool_calls, usage=usage,
            stop_reason=stop, model=request.model,
        )

    # --- action selection ----------------------------------------------------

    def _choose_action(self, prompt, prior, tools, messages):
        # ---- FAILURE-MODE SIMULATORS (hard tasks) ----
        if "zephyr-9" in prompt or "activation code" in prompt:
            # Tool loop: keep issuing the identical search while the tool is available.
            if "search" in tools:
                return "search", {"query": "zephyr-9 activation code"}
            # Once the gateway restricts `search`, give up gracefully and record.
            if "write_file" in tools and "write_file" not in prior:
                return "write_file", {"path": "answer.txt", "content": "code: unknown"}
            return None

        if "exhaustive" in prompt and "report.md" in prompt:
            # Context blowout: read then rewrite an ever-longer report each step.
            n = prior.count("write_file")
            if "write_file" in tools:
                body = "storage engine, replication, consensus, sharding. " * (8 * (n + 1))
                return "write_file", {
                    "path": "report.md",
                    "content": f"# Distributed Database Report (rev {n + 1})\n{body}",
                }
            return None

        # ---- WELL-BEHAVED TASKS (easy/medium) ----
        if "hello.txt" in prompt and "write_file" not in prior and "write_file" in tools:
            return "write_file", {"path": "hello.txt", "content": "hello world"}
        if "port" in prompt and "read_file" not in prior and "read_file" in tools:
            return "read_file", {"path": "config.txt"}
        if ("how many" in prompt or "count" in prompt) \
                and "list_dir" not in prior and "list_dir" in tools:
            return "list_dir", {"path": "."}
        if ("math_utils" in prompt or "done_marker" in prompt) and "write_file" not in prior:
            return "write_file", {"path": "DONE_MARKER.txt", "content": "module ready"}
        if "summary.txt" in prompt and "write_file" not in prior:
            return "write_file", {
                "path": "summary.txt",
                "content": "notes1.txt: apple\nnotes2.txt: banana\nnotes3.txt: cherry\n",
            }
        if "greet.py" in prompt and "write_file" not in prior:
            return "write_file", {
                "path": "greet.py",
                "content": "def greet(name):\n    return f'hello {name}'\n",
            }
        return None

    def _finish_text(self, prompt, messages):
        last = _last_tool_result(messages)
        if "how many" in prompt or "count" in prompt:
            n = len([ln for ln in last.splitlines() if ln.strip()])
            return f"DONE: there are {n} files."
        if "port" in prompt:
            return f"DONE: {last.strip()}"
        return "DONE: task complete."
