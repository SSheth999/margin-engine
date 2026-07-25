"""The arm decision switch — the ONLY place gateway behavior forks by arm.

Keeping every per-arm difference here is what makes the A/B/C/C-oracle comparison
valid: agent, environment, tasks, provider, and state store are identical across
arms; only `decide()` changes.

  A        — control: unconditional passthrough.
  B        — generic: one fixed intervention (downshift) once cost_ratio crosses the
             trigger. No failure-mode diagnosis. Isolates matching vs intervention-alone.
  C        — matched: diagnose the mode (detector), apply the matching intervention.
  C-oracle — matched, but using the TRUE mode from post-hoc labeling instead of the
             detector's guess. Isolates the matching idea from detector quality.

Detection is computed once by the caller (server) and passed in, so it can also be
logged in shadow on every arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from gateway import interventions
from gateway.detectors import (
    CONTEXT_BLOWOUT,
    DetectionConfig,
    DetectionResult,
    INHERENT_DIFFICULTY,
    TOOL_LOOP,
    WRONG_MODEL,
    _max_tool_repeat,
    mode_triggered,
)
from gateway.schema import ChatRequest
from gateway.state_store import OutcomeState


@dataclass
class Decision:
    request: ChatRequest
    block: bool = False
    mode: str | None = None       # the mode that drove the decision
    action: str | None = None     # "downshift"|"compact"|"restrict"|"block"|None


# Which intervention matches which failure mode (the tested policy).
_MATCH = {
    TOOL_LOOP: "restrict",
    CONTEXT_BLOWOUT: "compact",
    WRONG_MODEL: "downshift",
    INHERENT_DIFFICULTY: "block",
}


def _apply_matched(
    mode: str,
    request: ChatRequest,
    detection: DetectionResult,
    cheap_model: str,
    cfg: DetectionConfig,
    state: OutcomeState,
) -> Decision:
    action = _MATCH.get(mode)
    if action == "restrict":
        # Prefer the detector's tool; else fall back to the most-repeated tool in
        # state (needed when C-oracle acts on a true tool_loop the live detector missed).
        tool_name = detection.tool_name or _max_tool_repeat(state)[0]
        if tool_name:
            return Decision(
                request=interventions.restrict_tool(request, tool_name),
                mode=mode, action="restrict",
            )
        return Decision(request=request, mode=mode, action=None)
    if action == "compact":
        return Decision(
            request=interventions.compact(request, cfg.compact_keep_recent_turns),
            mode=mode, action="compact",
        )
    if action == "downshift":
        return Decision(
            request=interventions.downshift(request, cheap_model),
            mode=mode, action="downshift",
        )
    if action == "block":
        return Decision(request=request, block=True, mode=mode, action="block")
    return Decision(request=request)  # unknown/None mode -> passthrough


def decide(
    arm: str,
    request: ChatRequest,
    state: OutcomeState,
    detection: DetectionResult,
    cheap_model: str,
    cfg: DetectionConfig,
    true_mode: str | None = None,
) -> Decision:
    if arm == "A":
        return Decision(request=request)

    triggered = state.cost_ratio >= cfg.trigger_cost_ratio

    if arm == "B":
        # Generic graduated intervention: downshift once over budget-trend, no diagnosis.
        if triggered:
            return Decision(
                request=interventions.downshift(request, cheap_model),
                mode=None, action="downshift",
            )
        return Decision(request=request)

    if arm == "C":
        # A tool loop can be acted on pre-trigger (detector returns it regardless of
        # ratio); other modes are gated on the trigger inside detect().
        if detection.mode is not None:
            return _apply_matched(
                detection.mode, request, detection, cheap_model, cfg, state
            )
        return Decision(request=request)

    if arm == "C-oracle":
        # Identical to C except the mode comes from post-hoc labeling, not the live
        # detector. Act at the same point C would for that mode (mode_triggered),
        # so the ONLY difference from C is diagnosis correctness, not timing.
        if true_mode is not None and mode_triggered(true_mode, state, cfg):
            return _apply_matched(true_mode, request, detection, cheap_model, cfg, state)
        return Decision(request=request)

    raise ValueError(f"unknown arm: {arm!r}")
