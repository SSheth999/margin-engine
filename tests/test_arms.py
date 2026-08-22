"""Arm switch: passthrough vs matched/generic intervention, and no-fire on easy runs."""

from gateway.arms import decide
from gateway.detectors import DetectionConfig, DetectionResult, TOOL_LOOP, detect
from gateway.schema import ChatRequest
from gateway.state_store import OutcomeState

CFG = DetectionConfig()
CHEAP = "small"


def _state(cum_cost, step_count=1, context_sizes=None, tool_history=None):
    st = OutcomeState(outcome_id="t", contracted_price=1.0, target_margin=0.0)
    st.cum_cost = cum_cost
    st.step_count = step_count
    st.context_sizes = context_sizes or [10]
    st.tool_history = tool_history or []
    return st


def _req():
    return ChatRequest(
        model="big",
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "search"}],
    )


def test_arm_a_always_passthrough():
    st = _state(0.99)
    d = decide("A", _req(), st, detect(st, CFG), CHEAP, CFG)
    assert not d.block and d.action is None and d.request.model == "big"


def test_arm_b_downshifts_when_triggered():
    st = _state(0.8)  # ratio 0.8 >= trigger 0.6
    d = decide("B", _req(), st, DetectionResult(None), CHEAP, CFG)
    assert d.action == "downshift" and d.request.model == CHEAP


def test_arm_b_passthrough_when_cheap():
    st = _state(0.1)
    d = decide("B", _req(), st, DetectionResult(None), CHEAP, CFG)
    assert d.action is None


def test_arm_c_restricts_on_tool_loop():
    st = _state(0.2, tool_history=[("search", "h")] * 3)
    det = detect(st, CFG)
    d = decide("C", _req(), st, det, CHEAP, CFG)
    assert d.action == "restrict"
    assert "search" not in [t["name"] for t in d.request.tools]


def test_arm_c_oracle_uses_true_mode_and_matches_timing():
    # No live detection (empty history, low-ish cost), but true mode is tool_loop
    # and a loop IS present -> oracle restricts at the loop trigger.
    st = _state(0.2, tool_history=[("search", "h")] * 3)
    det = DetectionResult(None)  # pretend detector missed it
    d = decide("C-oracle", _req(), st, det, CHEAP, CFG, true_mode=TOOL_LOOP)
    assert d.action == "restrict"


def test_easy_run_no_intervention_any_arm():
    st = _state(0.15)  # well under trigger; no waste signals
    det = detect(st, CFG)
    for arm in ("A", "B", "C", "C-oracle"):
        d = decide(arm, _req(), st, det, CHEAP, CFG, true_mode=None)
        assert d.action is None and not d.block


def _req_many_turns(n_turns: int, big: int = 400) -> ChatRequest:
    """A prompt with enough droppable middle mass for compaction to be worthwhile."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n_turns):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{i}", "name": "run_command", "arguments": {"q": i}}],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * big})
    return ChatRequest(model="big", messages=msgs, tools=[{"name": "run_command"}])


def test_a_prime_is_byte_identical_to_a():
    """The placebo must never diverge from the control, or it measures nothing."""
    for cum in (0.1, 0.65, 0.95, 1.5):
        st_a = _state(cum, step_count=6, context_sizes=[100, 400, 900],
                      tool_history=[("run_command", "h")] * 4)
        st_p = _state(cum, step_count=6, context_sizes=[100, 400, 900],
                      tool_history=[("run_command", "h")] * 4)
        da = decide("A", _req(), st_a, detect(st_a, CFG), CHEAP, CFG, true_mode=TOOL_LOOP)
        dp = decide("A-prime", _req(), st_p, detect(st_p, CFG), CHEAP, CFG, true_mode=TOOL_LOOP)
        assert (da.action, da.block, da.request.model) == (dp.action, dp.block, dp.request.model)
        assert dp.action is None and not dp.block


def test_arm_d_passthrough_under_trigger():
    st = _state(0.3)
    assert decide("D", _req(), st, DetectionResult(None), CHEAP, CFG).action is None


def test_arm_d_downshifts_at_first_rung():
    st = _state(0.7)  # >= 0.60, < 0.85
    d = decide("D", _req(), st, DetectionResult(None), CHEAP, CFG)
    assert d.action == "downshift" and d.request.model == CHEAP


def test_arm_d_compacts_and_downshifts_at_second_rung():
    st = _state(0.9, step_count=10)
    d = decide("D", _req_many_turns(12), st, DetectionResult(None), CHEAP, CFG)
    assert d.action == "compact+downshift"
    assert d.request.model == CHEAP
    assert st.compaction_count == 1


def test_arm_d_compacts_at_most_once_per_run():
    st = _state(0.9, step_count=10)
    decide("D", _req_many_turns(12), st, DetectionResult(None), CHEAP, CFG)
    again = decide("D", _req_many_turns(12), st, DetectionResult(None), CHEAP, CFG)
    assert again.action == "downshift"  # ladder falls back, no second compaction
    assert st.compaction_count == 1


def test_arm_d_skips_compaction_when_prompt_is_mostly_cached():
    """Compaction invalidates the cached prefix; when the prefix is cheap, don't."""
    st = _state(0.9, step_count=10)
    st.last_cached_tokens, st.last_input_tokens = 9500, 500  # 95% cache hits
    d = decide("D", _req_many_turns(12), st, DetectionResult(None), CHEAP, CFG)
    assert d.action == "downshift" and st.compaction_count == 0


def test_arm_d_skips_compaction_when_trim_too_small():
    st = _state(0.9, step_count=10)
    d = decide("D", _req_many_turns(4), st, DetectionResult(None), CHEAP, CFG)
    assert d.action == "downshift" and st.compaction_count == 0


def test_arm_d_hard_stops_at_ceiling():
    st = _state(1.05)
    d = decide("D", _req(), st, DetectionResult(None), CHEAP, CFG)
    assert d.block and d.action == "block"


def test_unknown_arm_still_raises():
    import pytest

    st = _state(0.5)
    with pytest.raises(ValueError):
        decide("Z", _req(), st, DetectionResult(None), CHEAP, CFG)


def test_arm_d_stop_is_predictive_not_retrospective():
    """A spent-only budget check always overshoots by one call. At ratio 0.95 with a
    per-call cost of ~0.08 the NEXT call breaches the ceiling, so the stop must fire
    now rather than after the fact."""
    st = _state(0.95, step_count=9)
    st.step_costs = [0.08, 0.08, 0.08]
    d = decide("D", _req(), st, DetectionResult(None), CHEAP, CFG)
    assert d.block and d.action == "block"


def test_arm_d_does_not_stop_when_next_call_still_fits():
    st = _state(0.70, step_count=9)
    st.step_costs = [0.01, 0.01, 0.01]  # projected 0.71 < ceiling 1.0
    d = decide("D", _req(), st, DetectionResult(None), CHEAP, CFG)
    assert not d.block and d.action == "downshift"


def test_arm_d_lookahead_needs_no_history():
    """First call of a run has no cost history to extrapolate from; must not stop."""
    st = _state(0.0, step_count=0)
    assert decide("D", _req(), st, DetectionResult(None), CHEAP, CFG).action is None
