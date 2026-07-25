"""Hand-rolled ReAct agent — fixed backbone, identical across all arms.

Owns its own message list directly (no framework), points its model calls at the
gateway via base_url, and carries outcome context in headers. It must handle a
"block" stop signal from the gateway and end the run gracefully.

The harness is environment- and provider-agnostic. The caller injects:
  - tool_schemas : the tools advertised to the model,
  - tool_runner  : how a tool call is executed (local files, or a sandbox shell),
  - resolve_fn   : how resolution is graded (substring/file check, or TB2 tests),
  - system_prompt: the backbone prompt.
Because these are the same for every arm of a given task, swapping them changes the
task environment, never the per-arm behavior — so the A/B/C/C-oracle comparison holds.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx


def _call_gateway(
    client: httpx.Client,
    gateway_url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tool_schemas,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["extra"] = {"seed": seed}
    r = client.post(
        f"{gateway_url}/v1/chat", json=body, headers=headers, timeout=600.0
    )
    r.raise_for_status()
    return r.json()


def run_agent(
    *,
    gateway_url: str,
    model: str,
    outcome_id: str,
    contracted_price: float,
    target_margin: float,
    prompt: str,
    tool_schemas: list[dict[str, Any]],
    tool_runner: Callable[[str, dict[str, Any]], str],
    resolve_fn: Callable[[str, bool], bool],
    system_prompt: str,
    max_steps: int = 20,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one task to completion (or block / step-limit). Returns a result summary.

    The gateway (started per run) writes the JSON run log on /v1/finalize; this returns
    that record plus blocked/finished/final_text.
    """
    from agent.outcome_context import OutcomeContext

    ctx = OutcomeContext(
        outcome_id=outcome_id,
        contracted_price=contracted_price,
        target_margin=target_margin,
    )
    headers = ctx.headers()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    final_text = ""
    blocked = False
    finished = False

    with httpx.Client() as client:
        for _ in range(max_steps):
            result = _call_gateway(
                client, gateway_url, headers, model, messages, tool_schemas,
                max_tokens, temperature, seed,
            )
            gw = result.get("gateway", {})
            if gw.get("blocked"):
                blocked = True
                final_text = "[run blocked by gateway]"
                break

            resp = result["response"]
            text = resp.get("text", "") or ""
            tool_calls = resp.get("tool_calls", [])

            if not tool_calls:
                final_text = text
                finished = True
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {"id": tc["id"], "name": tc["name"], "arguments": tc["arguments"]}
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                out = tool_runner(tc["name"], tc["arguments"])
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": out}
                )

    resolved = bool(resolve_fn(final_text, finished))

    with httpx.Client() as client:
        fin = client.post(
            f"{gateway_url}/v1/finalize",
            json={"resolved": resolved, "contracted_price": contracted_price},
            timeout=120.0,
        )
        fin.raise_for_status()
        summary = fin.json()

    summary.update({"blocked": blocked, "finished": finished, "final_text": final_text})
    return summary
