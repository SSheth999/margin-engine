"""The arm decision switch — the ONLY place gateway behavior forks by arm.

Keeping every per-arm difference here is what makes the A/B/C/C-oracle comparison
valid: agent, environment, tasks, provider, and state store are identical across
arms; only `decide()` changes.

  A        — control: unconditional passthrough.
  A-prime  — PLACEBO: byte-for-byte the same policy as A. Not a redundant arm — it
             measures the noise floor. Agent runs are nondeterministic, so two runs of
             the identical policy differ; without that measurement there is no way to
             know whether an arm difference is an effect. In the first full benchmark
             the C-oracle arm passed through unconditionally on the 37 tasks labeled
             with no failure mode, making it an accidental placebo: paired against A it
             showed a median 21% / mean 71% cost divergence, with one task going
             $0.053 (7 steps) -> $0.524 (30 steps). That single same-policy run was the
             third-largest contributor to C-oracle's headline P99. Run this arm
             deliberately so the floor is a measured quantity, not a discovered one.
  B        — generic: one fixed intervention (downshift) once cost_ratio crosses the
             trigger. No failure-mode diagnosis. Isolates matching vs intervention-alone.
  C        — matched: diagnose the mode (detector), apply the matching intervention.
  C-oracle — matched, but using the TRUE mode from post-hoc labeling instead of the
             detector's guess. Isolates the matching idea from detector quality.
  D        — laddered: escalate on the cost ratio alone (downshift -> cache-aware
             compaction -> hard stop). Not a matched arm; it is the policy the first
             benchmark's evidence actually points to. See `_decide_d`.

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


def _decide_d(
    request: ChatRequest,
    state: OutcomeState,
    cheap_model: str,
    cfg: DetectionConfig,
) -> Decision:
    """Arm D: a graduated ladder on the cost ratio, ignoring failure-mode diagnosis.

    Each rung is justified by a measurement from the first full benchmark rather than
    by the taxonomy:

      1. hard stop at the ceiling. Blowing the ceiling almost never bought the outcome
         (Arm A resolved 3/36 blown runs vs 23/51 held), so stopping there caps P99 at
         the ceiling by construction and forfeits little quality. Arm D is the first
         arm where `block` can actually fire: in 349 runs the matched arms never once
         reached it, because `inherent_difficulty` was never diagnosed.
      2. compaction, but at most `d_compact_max_per_run` times, only when the trim is
         big enough to out-earn the prompt-cache invalidation it causes, and never when
         the prompt is already mostly cache hits. Repeated small trims were measurably
         cost-INCREASING.
      3. downshift as the base rung, because it was the only intervention that moved
         the tail (margin-blown 41.4% -> 2.3%).

    The rungs compose: past the compaction threshold a call is both compacted and
    downshifted, since they attack different halves of the bill.
    """
    ratio = state.cost_ratio
    ceiling = state.margin_ceiling

    # Stop on the PROJECTED cost, not the spent cost. Checking only what's already
    # spent waves a run through at ratio 0.99 and lands it past the ceiling — the
    # overshoot is one full call, which on the observed step costs is 10-20% of ceiling.
    stop_at = cfg.d_stop_ratio * ceiling
    if ceiling > 0 and (
        state.cum_cost >= stop_at
        or (
            cfg.d_stop_lookahead
            and state.projected_cost_after_next_call() >= stop_at
        )
    ):
        return Decision(request=request, block=True, mode="over_ceiling", action="block")

    if ratio < cfg.d_downshift_ratio:
        return Decision(request=request)

    req = interventions.downshift(request, cheap_model)
    action = "downshift"

    if (
        ratio >= cfg.d_compact_ratio
        and state.compaction_count < cfg.d_compact_max_per_run
        and state.cache_hit_share <= cfg.d_compact_max_cache_share
        and interventions.compaction_gain(req, cfg.d_compact_keep_recent_turns)
        >= cfg.d_compact_min_drop_fraction
    ):
        req = interventions.compact(req, cfg.d_compact_keep_recent_turns)
        state.note_compaction()
        action = "compact+downshift"

    return Decision(request=req, mode="over_budget_trend", action=action)


def decide(
    arm: str,
    request: ChatRequest,
    state: OutcomeState,
    detection: DetectionResult,
    cheap_model: str,
    cfg: DetectionConfig,
    true_mode: str | None = None,
) -> Decision:
    # A-prime is the placebo: it MUST stay identical to A, including here.
    if arm in ("A", "A-prime"):
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

    if arm == "D":
        return _decide_d(request, state, cheap_model, cfg)

    raise ValueError(f"unknown arm: {arm!r}")
