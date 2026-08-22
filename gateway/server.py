"""Gateway HTTP server — one process per task run.

Started by the orchestrator on an ephemeral port, torn down when the run ends. The
agent points its model calls at this server via base_url. Run-level context (arm,
task_id, seed, complexity, output dir) comes from env vars set at spawn time;
per-request outcome context (price, margin) comes from headers.

Endpoints:
  GET  /health          liveness
  POST /v1/chat         one model call: pre-call decision -> provider -> accounting
  POST /v1/finalize     mark the run resolved/unresolved and write the JSON run log

Design rule: FAIL OPEN. Any error in gateway *logic* (decision, cost, accounting)
must not crash the run — we forward the request unchanged and record the error.
Provider call failures are returned to the agent so it can end gracefully.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from gateway.arms import decide
from gateway.cost import RateCard
from gateway.detectors import DetectionConfig, DetectionResult, detect
from gateway.run_log import InterventionRecord, RunLog, StepRecord
from gateway.schema import ChatRequest
from gateway.state_store import StateStore, args_hash
from providers.base import load_provider


def _estimate_context_size(request: ChatRequest) -> int:
    """Cheap proxy for context size: total characters across message contents.

    Good enough for the context-blowout detector's growth-rate signal (added later);
    exact token count isn't needed for a monotonic proxy.
    """
    total = 0
    for m in request.messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for block in c:
                total += len(str(block.get("content", block)))
        # Tool-call arguments are part of the conversation context too.
        for tc in m.get("tool_calls") or []:
            total += len(str(tc.get("arguments", "")))
    return total


def create_app() -> FastAPI:
    app = FastAPI(title="Margin Engine Gateway")

    arm = os.environ.get("MARGIN_ARM", "A")
    task_id = os.environ.get("MARGIN_TASK_ID", "unknown")
    seed = int(os.environ.get("MARGIN_SEED", "0"))
    complexity = os.environ.get("MARGIN_COMPLEXITY", "unknown")
    run_dir = Path(os.environ.get("MARGIN_RUN_DIR", "runs"))

    provider, model_pair = load_provider()
    rate_card = RateCard.load(provider.name)
    det_cfg = DetectionConfig.load()
    store = StateStore()

    # C-oracle: map of "task_id__seed" -> true_failure_mode, from post-hoc labeling of
    # Arm A runs. Empty for other arms / when the file is absent.
    labels: dict[str, str] = {}
    labels_path = os.environ.get("MARGIN_LABELS")
    if labels_path and Path(labels_path).exists():
        labels = json.loads(Path(labels_path).read_text())
    true_mode = labels.get(f"{task_id}__{seed}")

    # The single run this gateway instance serves. Built lazily on first /v1/chat
    # (so we know contracted_price / margin_ceiling from the headers).
    state = {"run_log": None, "outcome_id": None, "t_first_call": None}

    @app.get("/health")
    def health():
        return {"ok": True, "arm": arm, "provider": provider.name}

    @app.post("/v1/chat")
    async def chat(
        request: Request,
        x_outcome_id: str = Header(default="default"),
        x_contracted_price: float = Header(default=1.0),
        x_target_margin: float = Header(default=0.0),
    ):
        t_call_start = time.monotonic()
        body = await request.json()
        chat_req = ChatRequest.from_dict(body)

        st = store.get_or_create(x_outcome_id, x_contracted_price, x_target_margin)

        if state["run_log"] is None:
            state["t_first_call"] = t_call_start
            state["outcome_id"] = x_outcome_id
            state["run_log"] = RunLog(
                task_id=task_id,
                arm=arm,
                seed=seed,
                complexity_class=complexity,
                contracted_price=x_contracted_price,
                margin_ceiling=st.margin_ceiling,
                provider=provider.name,
            )
        run_log: RunLog = state["run_log"]

        # --- PRE-CALL: detect (shadow, all arms) + decide (per arm), fail open ---
        forward_req = chat_req
        action = None
        detected_mode = None
        try:
            det: DetectionResult = detect(st, det_cfg)
            detected_mode = det.mode
            if detected_mode is not None:
                run_log.detected_failure_mode = detected_mode

            decision = decide(
                arm, chat_req, st, det, model_pair.cheap, det_cfg, true_mode
            )
            forward_req = decision.request
            action = decision.action

            if decision.action is not None and not decision.block:
                run_log.interventions.append(
                    InterventionRecord(
                        step=st.step_count + 1,
                        mode=decision.mode or "generic",
                        action=decision.action,
                        cost_at_step=st.cum_cost,
                    )
                )
            if decision.block:
                run_log.interventions.append(
                    InterventionRecord(
                        step=st.step_count + 1,
                        mode=decision.mode or "unknown",
                        action="block",
                        cost_at_step=st.cum_cost,
                    )
                )
                return JSONResponse(
                    {
                        "response": None,
                        "gateway": {
                            "blocked": True,
                            "mode": decision.mode,
                            "action": "block",
                            "cum_cost": st.cum_cost,
                            "cost_ratio": st.cost_ratio,
                        },
                    }
                )
        except Exception as e:  # gateway logic error -> fail open
            run_log.error = f"decision_error: {e!r}"
            forward_req = chat_req

        # --- PROVIDER CALL ---
        try:
            resp = provider.complete(forward_req)
        except Exception as e:
            run_log.error = f"provider_error: {e!r}"
            return JSONResponse(
                status_code=502,
                content={"response": None, "gateway": {"error": str(e)}},
            )

        # --- POST-CALL: accounting (fail open) ---
        call_cost = 0.0
        try:
            if rate_card.has(forward_req.model):
                call_cost = rate_card.cost(forward_req.model, resp.usage)
            else:
                run_log.error = f"no_rate_for_model: {forward_req.model}"

            tool_name = resp.tool_calls[0].name if resp.tool_calls else None
            tool_ah = (
                args_hash(resp.tool_calls[0].arguments) if resp.tool_calls else None
            )
            ctx_size = _estimate_context_size(forward_req)

            # Token mix is recorded too: cache-aware interventions (arm D) need to know
            # how much of the prompt is currently served from cache before deciding
            # whether compacting can pay for the invalidation it causes.
            store.record_call(
                x_outcome_id,
                call_cost,
                ctx_size,
                tool_name,
                tool_ah,
                input_tokens=resp.usage.input_tokens,
                cached_tokens=resp.usage.cached_tokens,
            )
            run_log.add_step(
                StepRecord(
                    step=st.step_count,
                    model=forward_req.model,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    cached_tokens=resp.usage.cached_tokens,
                    cost=call_cost,
                    cum_cost=st.cum_cost,
                    context_size=ctx_size,
                    tool=tool_name,
                    args_hash=tool_ah,
                    detected_failure_mode=detected_mode,
                    latency_sec=round(time.monotonic() - t_call_start, 3),
                )
            )
        except Exception as e:
            run_log.error = f"accounting_error: {e!r}"

        return JSONResponse(
            {
                "response": resp.to_dict(),
                "gateway": {
                    "blocked": False,
                    "action": action,
                    "call_cost": call_cost,
                    "cum_cost": st.cum_cost,
                    "cost_ratio": st.cost_ratio,
                    "margin_ceiling": st.margin_ceiling,
                },
            }
        )

    @app.post("/v1/finalize")
    async def finalize(request: Request):
        body = await request.json()
        run_log: RunLog | None = state["run_log"]
        if run_log is None:
            # No model calls were made — still emit a record for completeness.
            run_log = RunLog(
                task_id=task_id,
                arm=arm,
                seed=seed,
                complexity_class=complexity,
                contracted_price=float(body.get("contracted_price", 0.0)),
                margin_ceiling=0.0,
                provider=provider.name,
            )
        run_log.resolved = bool(body.get("resolved", False))
        if state["t_first_call"] is not None:
            run_log.agent_loop_sec = round(time.monotonic() - state["t_first_call"], 3)
        # Known at runtime only for C-oracle; step-5 labeling backfills it for other arms.
        if true_mode is not None:
            run_log.true_failure_mode = true_mode
        path = run_log.write(run_dir)
        return {"written": str(path), "final_cost": run_log.final_cost,
                "margin_held": run_log.margin_held, "resolved": run_log.resolved}

    return app


app = create_app()
