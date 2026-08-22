"""Post-hoc labeling v2: every mode must be reachable, and the label must key on
where the money actually went rather than on raw context growth.

v1 separated context_blowout from wrong_model with max(context)/context[0] >= 1.5,
which 99% of real runs pass (context accumulates monotonically as tool output piles
up). That made wrong_model and inherent_difficulty unreachable and silently reduced
the C-oracle arm to a compaction-only arm.
"""

from analysis.labeling import input_cost_share, label_run
from gateway.detectors import (
    CONTEXT_BLOWOUT,
    DetectionConfig,
    INHERENT_DIFFICULTY,
    TOOL_LOOP,
    WRONG_MODEL,
)

CFG = DetectionConfig()
CEILING = 0.10


def _log(steps, ceiling=CEILING, provider="openai"):
    return {
        "provider": provider,
        "margin_ceiling": ceiling,
        "final_cost": steps[-1]["cum_cost"] if steps else 0.0,
        "per_step": steps,
    }


def _step(i, cum, inp=0, out=0, cached=0, tool="run_command", ah=None, model="gpt-5.6-terra"):
    return {
        "step": i, "model": model, "input_tokens": inp, "output_tokens": out,
        "cached_tokens": cached, "cost": 0.0, "cum_cost": cum, "context_size": 100 * i,
        "tool": tool, "args_hash": ah if ah is not None else f"h{i}",
    }


def test_cheap_run_with_no_waste_is_unlabeled():
    steps = [_step(i, 0.005 * i, inp=100, out=50) for i in range(1, 5)]
    assert label_run(_log(steps), CFG) is None


def test_tool_loop_wins_regardless_of_cost():
    steps = [_step(i, 0.002 * i, inp=100, out=50, ah="same") for i in range(1, 6)]
    assert label_run(_log(steps), CFG) == TOOL_LOOP


# Cost ramps gradually so the run ends over ceiling without tripping the
# inherent-difficulty rule (which only fires when it is expensive within 4 steps).
def _blowout_steps():
    return [_step(i, 0.008 * i, inp=40_000, out=200) for i in range(1, 11)]


def _wrong_model_steps():
    return [_step(i, 0.008 * i, inp=500, out=20_000) for i in range(1, 11)]


def test_context_blowout_when_spend_is_input_heavy():
    assert label_run(_log(_blowout_steps()), CFG) == CONTEXT_BLOWOUT


def test_wrong_model_when_spend_is_output_heavy():
    """v1 could never produce this label; the oracle could therefore never downshift."""
    assert label_run(_log(_wrong_model_steps()), CFG) == WRONG_MODEL


def test_inherent_difficulty_when_expensive_immediately():
    steps = [_step(1, 0.095, inp=1000, out=1000), _step(2, 0.099, inp=1000, out=1000)]
    assert label_run(_log(steps), CFG) == INHERENT_DIFFICULTY


def test_cached_tokens_are_discounted_in_the_split():
    """Cached input is ~4x cheaper, so a cache-heavy prompt is not automatically a
    context blowout — weighting by token count alone would misclassify it."""
    # 100k cached input @ 0.25*2.50 = $0.0625, vs 10k output @ $10 = $0.10 -> output-heavy
    log = _log([_step(1, 0.16, inp=0, out=10_000, cached=100_000)])
    share = input_cost_share(log)
    assert share is not None and share < 0.5


def test_unpriced_model_yields_no_share():
    log = _log([_step(1, 0.1, inp=100, out=100, model="not-a-real-model")])
    assert input_cost_share(log) is None


def test_all_four_modes_are_reachable():
    """The v1 regression test: a labeler that can only emit a subset of modes turns the
    C-oracle arm into a test of one intervention."""
    cases = [
        [_step(i, 0.002 * i, inp=100, out=50, ah="same") for i in range(1, 6)],
        _blowout_steps(),
        _wrong_model_steps(),
        [_step(1, 0.095, inp=1000, out=1000), _step(2, 0.099, inp=1000, out=1000)],
    ]
    got = {label_run(_log(s), CFG) for s in cases}
    assert got == {TOOL_LOOP, CONTEXT_BLOWOUT, WRONG_MODEL, INHERENT_DIFFICULTY}
