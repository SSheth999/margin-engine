"""Significance tests over the run logs.

The point of these is that the experiment's success criteria used to be `>` on two
point estimates. With a measured same-policy noise floor of median 21% / mean 71% cost
divergence, a point-estimate comparison cannot distinguish an effect from noise, so
every criterion is now decided by a test.
"""

import math

from analysis.metrics import (
    _binom_two_sided_p,
    compare_arms,
    non_inferiority_resolution,
    paired_cost_test,
    paired_resolution_test,
    seeds_needed,
    two_proportion_test,
)


def _run(arm, task, cost, ceiling=0.10, resolved=False, seed=1):
    return {
        "arm": arm, "task_id": task, "seed": seed, "final_cost": cost,
        "margin_ceiling": ceiling, "margin_held": cost <= ceiling,
        "resolved": resolved, "step_count": 5, "per_step": [],
    }


# --- two-proportion test ---------------------------------------------------------

def test_two_proportion_detects_large_difference():
    r = two_proportion_test(36, 87, 2, 87)  # the observed A-vs-B margin-blown split
    assert r["significant"] and r["p_value"] < 1e-6
    assert r["diff_pp"] > 0  # first arm blows budget more often


def test_two_proportion_ns_on_small_difference():
    r = two_proportion_test(36, 87, 28, 87)  # A vs C: 41.4% vs 32.2%
    assert not r["significant"] and r["p_value"] > 0.05


def test_two_proportion_handles_identical_rates():
    r = two_proportion_test(10, 50, 10, 50)
    assert r["z"] == 0.0 and r["p_value"] == 1.0 and not r["significant"]


def test_two_proportion_needs_both_arms():
    assert not two_proportion_test(1, 0, 1, 10)["applicable"]


# --- exact binomial / sign test --------------------------------------------------

def test_binom_symmetric_and_bounded():
    assert _binom_two_sided_p(5, 10) == 1.0
    assert _binom_two_sided_p(9, 10) == _binom_two_sided_p(1, 10)
    assert _binom_two_sided_p(0, 0) == 1.0
    assert 0.0 < _binom_two_sided_p(10, 10) < 0.01


# --- paired tests ----------------------------------------------------------------

def _paired_runs(deltas):
    """Arm X costs `base`, arm Y costs `base - delta` on each task."""
    runs = []
    for i, (base, delta) in enumerate(deltas):
        runs.append(_run("X", f"t{i}", base))
        runs.append(_run("Y", f"t{i}", base - delta))
    return runs


def test_paired_cost_test_finds_consistent_saving():
    """A small consistent per-task saving is detectable even when absolute costs vary
    over an order of magnitude — which is exactly why the test is paired."""
    runs = _paired_runs([(0.02 * (i + 1), 0.01) for i in range(40)])
    r = paired_cost_test(runs, "X", "Y", iters=2000)
    assert r["mean_cost_diff"] > 0  # X costs more
    assert r["cost_diff_significant"]
    lo, hi = r["cost_diff_ci"]
    assert lo > 0 and hi > lo
    assert r["a_costlier_on"] == 40 and r["sign_test_p"] < 0.001


def test_paired_cost_test_null_when_no_difference():
    runs = _paired_runs([(0.02 * (i + 1), 0.0) for i in range(40)])
    r = paired_cost_test(runs, "X", "Y", iters=2000)
    assert not r["cost_diff_significant"]
    assert r["sign_test_p"] == 1.0  # no discordant pairs


def test_paired_cost_test_reports_p99_difference():
    runs = _paired_runs([(0.02 * (i + 1), 0.01) for i in range(40)])
    r = paired_cost_test(runs, "X", "Y", iters=2000)
    assert r["p99_diff"] > 0 and r["p99_diff_significant"]


def test_paired_cost_test_needs_pairs():
    runs = [_run("X", "t0", 0.1), _run("Y", "t9", 0.1)]  # no shared task
    assert not paired_cost_test(runs, "X", "Y")["applicable"]


def test_paired_resolution_uses_only_discordant_pairs():
    runs = []
    # 20 tasks both arms solve (concordant — carry no information)
    for i in range(20):
        runs += [_run("X", f"c{i}", 0.01, resolved=True),
                 _run("Y", f"c{i}", 0.01, resolved=True)]
    # 8 tasks only X solves
    for i in range(8):
        runs += [_run("X", f"d{i}", 0.01, resolved=True),
                 _run("Y", f"d{i}", 0.01, resolved=False)]
    r = paired_resolution_test(runs, "X", "Y")
    assert r["n_paired"] == 28 and r["n_discordant"] == 8
    assert r["a_only"] == 8 and r["b_only"] == 0
    assert r["significant"]  # 8-0 split is unlikely under p=0.5


def test_paired_resolution_ns_on_even_split():
    runs = []
    for i in range(4):
        runs += [_run("X", f"a{i}", 0.01, resolved=True),
                 _run("Y", f"a{i}", 0.01, resolved=False)]
        runs += [_run("X", f"b{i}", 0.01, resolved=False),
                 _run("Y", f"b{i}", 0.01, resolved=True)]
    r = paired_resolution_test(runs, "X", "Y")
    assert r["a_only"] == 4 and r["b_only"] == 4 and not r["significant"]


def test_compare_arms_bundles_all_three():
    runs = _paired_runs([(0.2, 0.15) for _ in range(20)])
    c = compare_arms(runs, "X", "Y")
    assert set(c) == {"arm_a", "arm_b", "margin_blown", "cost", "resolution"}
    assert c["margin_blown"]["significant"]  # X always blows, Y never does


# --- non-inferiority: the feasibility guard --------------------------------------

def _resolution_runs(n, p_control, p_treatment):
    runs = []
    for i in range(n):
        runs.append(_run("A", f"t{i}", 0.01, resolved=i < round(p_control * n)))
        runs.append(_run("C", f"t{i}", 0.01, resolved=i < round(p_treatment * n)))
    return runs


def test_non_inferiority_flags_underpowered_design():
    """The real defect: at n=87 with a 3pp margin the test cannot pass even if the true
    difference is exactly zero. That is a property of the design, not of the data, and
    must not be reported as evidence of inferiority."""
    runs = _resolution_runs(87, 0.30, 0.30)
    r = non_inferiority_resolution(runs, "A", "C", margin_pp=0.03)
    assert r["diff_pp"] == 0.0            # identical rates
    assert not r["non_inferior"]          # still cannot conclude
    assert r["feasible"] is False
    assert r["best_case_z"] < 1.645
    assert "INCONCLUSIVE" in r["verdict"]
    assert r["min_n_per_arm"] > 1000
    assert r["min_feasible_margin_pp"] > 0.10


def test_non_inferiority_feasible_with_wide_margin():
    runs = _resolution_runs(87, 0.30, 0.30)
    r = non_inferiority_resolution(runs, "A", "C", margin_pp=0.20)
    assert r["feasible"] is True and r["non_inferior"] and r["verdict"] == "NON-INFERIOR"


def test_non_inferiority_never_says_inferior():
    """Failing to establish non-inferiority is absence of evidence. The verdict string
    must not collapse that into 'INFERIOR'."""
    for margin in (0.03, 0.10, 0.20):
        for pt in (0.10, 0.30, 0.50):
            v = non_inferiority_resolution(_resolution_runs(87, 0.30, pt), "A", "C",
                                           margin_pp=margin)["verdict"]
            assert v.startswith(("NON-INFERIOR", "INCONCLUSIVE", "NOT SHOWN"))
            assert v != "INFERIOR"


def test_non_inferiority_scales_with_sample_size():
    small = non_inferiority_resolution(_resolution_runs(87, 0.3, 0.3), "A", "C", 0.03)
    large = non_inferiority_resolution(_resolution_runs(1400, 0.3, 0.3), "A", "C", 0.03)
    assert not small["feasible"] and large["feasible"]
    assert large["se"] < small["se"]


def test_seeds_needed_converts_n_to_seeds():
    runs = _resolution_runs(89, 0.3, 0.3)
    r = seeds_needed(runs, 1217)
    assert r["n_tasks"] == 89 and r["seeds_needed"] == math.ceil(1217 / 89) == 14


def test_seeds_needed_inapplicable_without_target():
    assert not seeds_needed(_resolution_runs(10, 0.3, 0.3), None)["applicable"]


# --- paired non-inferiority: the fix that turns 14 seeds into 3 ----------------------

from analysis.metrics import (  # noqa: E402
    decidable_margin_by_seeds,
    infrastructure_cost,
    non_inferiority_resolution_paired,
)


def _disc_runs(n, control_only, treatment_only, both=0):
    """n paired tasks: some solved only by control, some only by treatment, some by both."""
    runs = []
    for i in range(n):
        if i < control_only:
            c, t = True, False
        elif i < control_only + treatment_only:
            c, t = False, True
        elif i < control_only + treatment_only + both:
            c, t = True, True
        else:
            c, t = False, False
        runs.append(_run("A", f"t{i}", 0.01, resolved=c))
        runs.append(_run("C", f"t{i}", 0.01, resolved=t))
    return runs


def test_paired_is_tighter_than_unpaired():
    """The whole point: arms share tasks, so pairing removes between-task variance the
    unpaired test pays for. On the observed A-vs-C shape this is a 2.3x SE reduction."""
    runs = _disc_runs(87, control_only=5, treatment_only=2, both=21)
    paired = non_inferiority_resolution_paired(runs, "A", "C", margin_pp=0.03)
    unpaired = non_inferiority_resolution(runs, "A", "C", margin_pp=0.03)
    assert paired["se"] < unpaired["se"] / 2
    assert paired["min_n_per_arm"] < unpaired["min_n_per_arm"] / 4


def test_paired_counts_only_discordant_pairs():
    runs = _disc_runs(87, control_only=5, treatment_only=2, both=21)
    r = non_inferiority_resolution_paired(runs, "A", "C")
    assert r["n_paired"] == 87
    assert r["control_only"] == 5 and r["treatment_only"] == 2
    assert r["n_discordant"] == 7
    assert r["diff_pp"] == (2 - 5) / 87


def test_paired_margin_feasibility_at_one_seed():
    """87 tasks x 1 seed cannot decide 3pp, but can decide ~5pp. That is the finding
    that makes a cheap experiment viable: argue about the margin, not the sample size."""
    runs = _disc_runs(87, control_only=5, treatment_only=2, both=21)
    tight = non_inferiority_resolution_paired(runs, "A", "C", margin_pp=0.03)
    loose = non_inferiority_resolution_paired(runs, "A", "C", margin_pp=0.06)
    assert tight["feasible"] is False
    assert loose["feasible"] is True
    assert 0.04 < tight["min_feasible_margin_pp"] < 0.06


def test_paired_perfect_agreement_is_non_inferior():
    """No discordant pairs at all: the arms resolve identically, so non-inferiority
    holds by construction and must not divide by zero."""
    runs = _disc_runs(40, control_only=0, treatment_only=0, both=20)
    r = non_inferiority_resolution_paired(runs, "A", "C", margin_pp=0.03)
    assert r["se"] == 0.0 and r["diff_pp"] == 0.0
    assert r["non_inferior"] and r["verdict"] == "NON-INFERIOR"


def test_paired_never_says_inferior():
    for margin in (0.01, 0.03, 0.10):
        for co, to in ((10, 0), (0, 10), (5, 5)):
            v = non_inferiority_resolution_paired(
                _disc_runs(87, co, to, 20), "A", "C", margin_pp=margin
            )["verdict"]
            assert v.startswith(("NON-INFERIOR", "INCONCLUSIVE", "NOT SHOWN"))


def test_decidable_margin_shrinks_with_seeds():
    runs = _disc_runs(87, control_only=5, treatment_only=2, both=21)
    m = decidable_margin_by_seeds(runs, "A", "C")["margins_pp"]
    assert m[1] > m[2] > m[3] > m[5]
    # SE ~ 1/sqrt(seeds), so 4x the seeds should roughly halve the margin
    assert abs(m[1] / m[5] - math.sqrt(5)) < 0.01


# --- infrastructure cost ------------------------------------------------------------

def _timed(arm, task, cost, secs):
    r = _run(arm, task, cost)
    r["sandbox_sec"] = secs
    return r


def test_infrastructure_reports_unmeasurable_without_timing():
    """The state the first full benchmark left us in: sandbox cost unrecoverable."""
    r = infrastructure_cost([_run("A", "t0", 0.1)], 0.10)
    assert not r["applicable"] and r["n_timed"] == 0 and "re-run" in r["note"]


def test_infrastructure_totals_hours_per_arm():
    runs = [_timed("A", "t0", 0.10, 3600), _timed("A", "t1", 0.10, 1800),
            _timed("B", "t0", 0.05, 900)]
    r = infrastructure_cost(runs, None)
    assert r["applicable"] and r["n_timed"] == 3
    assert r["by_arm"]["A"]["sandbox_hours"] == 1.5
    assert r["by_arm"]["B"]["sandbox_hours"] == 0.25
    assert r["total_sandbox_hours"] == 1.75
    assert "infra_usd" not in r["by_arm"]["A"]  # no rate configured


def test_infrastructure_applies_rate_and_shows_share():
    runs = [_timed("A", "t0", 0.10, 3600)]  # 1 h, $0.10 model
    r = infrastructure_cost(runs, 0.30)
    a = r["by_arm"]["A"]
    assert a["infra_usd"] == 0.30
    assert abs(a["total_usd"] - 0.40) < 1e-9
    assert abs(a["infra_share"] - 0.75) < 1e-9  # infra dominates at this rate
    assert abs(r["total_usd"] - 0.40) < 1e-9


def test_infrastructure_ignores_untimed_runs():
    runs = [_timed("A", "t0", 0.10, 3600), _run("A", "t1", 0.10)]
    r = infrastructure_cost(runs, None)
    assert r["n_timed"] == 1 and r["n_runs"] == 2
    assert r["by_arm"]["A"]["n"] == 1
