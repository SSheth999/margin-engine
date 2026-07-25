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
