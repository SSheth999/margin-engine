"""Detector rules across all four failure modes + the easy-task no-fire guarantee."""

from gateway.detectors import (
    CONTEXT_BLOWOUT,
    DetectionConfig,
    INHERENT_DIFFICULTY,
    TOOL_LOOP,
    WRONG_MODEL,
    detect,
    mode_triggered,
)
from gateway.state_store import OutcomeState

CFG = DetectionConfig()  # defaults


def _state(cum_cost, step_count, context_sizes, tool_history):
    # price 1.0, margin 0.0 -> ceiling 1.0, so cost_ratio == cum_cost.
    st = OutcomeState(outcome_id="t", contracted_price=1.0, target_margin=0.0)
    st.cum_cost = cum_cost
    st.step_count = step_count
    st.context_sizes = context_sizes
    st.tool_history = tool_history
    return st


def test_tool_loop_fires_regardless_of_cost():
    st = _state(0.1, 3, [10, 10, 10], [("search", "h1")] * 3)
    r = detect(st, CFG)
    assert r.mode == TOOL_LOOP and r.tool_name == "search"


def test_context_blowout():
    st = _state(0.7, 4, [100, 200, 400, 800], [("x", str(i)) for i in range(4)])
    assert detect(st, CFG).mode == CONTEXT_BLOWOUT


def test_inherent_difficulty_early_and_expensive():
    st = _state(0.95, 2, [100, 100], [("x", "1"), ("y", "2")])
    assert detect(st, CFG).mode == INHERENT_DIFFICULTY


def test_wrong_model_high_cost_flat_context_late():
    st = _state(0.7, 8, [100, 100, 100, 100], [("x", str(i)) for i in range(8)])
    assert detect(st, CFG).mode == WRONG_MODEL


def test_easy_task_does_not_fire():
    st = _state(0.2, 2, [50, 55], [("x", "1"), ("y", "2")])
    assert detect(st, CFG).mode is None


def test_mode_triggered_matches_detector_for_loop():
    st = _state(0.1, 3, [10, 10, 10], [("search", "h1")] * 3)
    assert mode_triggered(TOOL_LOOP, st, CFG)
    assert not mode_triggered(CONTEXT_BLOWOUT, st, CFG)  # low cost ratio


def test_mode_triggered_none_is_false():
    st = _state(0.2, 1, [10], [])
    assert not mode_triggered(None, st, CFG)
