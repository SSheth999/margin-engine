"""Metrics over the run logs (build step 7).

Primary metrics are tail/variance (the hypothesis): P99, P90, margin-blown rate, spread.
Secondary is mean (reported, not led with). The quality constraint is resolution rate
with a non-inferiority test. Everything can be broken down per failure mode.

No scipy dependency: bootstrap CIs and the two-proportion non-inferiority z-test are
implemented directly on numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]


@dataclass
class ArmStats:
    arm: str
    n: int
    mean: float
    p90: float
    p99: float
    std: float
    margin_blown_rate: float
    resolution_rate: float
    p99_ci: tuple[float, float] = (0.0, 0.0)
    margin_blown_ci: tuple[float, float] = (0.0, 0.0)


def _percentile(costs: np.ndarray, q: float) -> float:
    if costs.size == 0:
        return float("nan")
    return float(np.percentile(costs, q))


def _bootstrap_ci(
    values: np.ndarray, stat_fn, iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary statistic (tail stats are noisy)."""
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(iters)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        stats[i] = stat_fn(sample)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def arm_stats(runs: list[dict], arm: str) -> ArmStats:
    arm_runs = [r for r in runs if r["arm"] == arm]
    costs = np.array([r["final_cost"] for r in arm_runs], dtype=float)
    blown = np.array([1.0 if not r["margin_held"] else 0.0 for r in arm_runs])
    resolved = np.array([1.0 if r["resolved"] else 0.0 for r in arm_runs])

    return ArmStats(
        arm=arm,
        n=len(arm_runs),
        mean=float(costs.mean()) if costs.size else float("nan"),
        p90=_percentile(costs, 90),
        p99=_percentile(costs, 99),
        std=float(costs.std(ddof=1)) if costs.size > 1 else 0.0,
        margin_blown_rate=float(blown.mean()) if blown.size else float("nan"),
        resolution_rate=float(resolved.mean()) if resolved.size else float("nan"),
        p99_ci=_bootstrap_ci(costs, lambda s: np.percentile(s, 99)),
        margin_blown_ci=_bootstrap_ci(blown, np.mean),
    )


Z_CRIT_ONE_SIDED = 1.645  # alpha = 0.05
Z_CRIT_TWO_SIDED = 1.96


def _norm_sf(z: float) -> float:
    """Upper-tail probability of the standard normal (no scipy)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _two_sided_p(z: float) -> float:
    return float(min(1.0, 2.0 * _norm_sf(abs(z))))


def _binom_two_sided_p(successes: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5. Returns 1.0 when n == 0."""
    if n <= 0:
        return 1.0
    k = max(successes, n - successes)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return float(min(1.0, 2.0 * tail / 2**n))


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-sided two-proportion z-test with pooled SE.

    Used on margin-blown and resolution rates. Unpaired: it ignores the fact that arms
    share tasks, so it is the conservative counterpart to the paired tests below.
    """
    if n1 == 0 or n2 == 0:
        return {"applicable": False}
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        z = 0.0 if p1 == p2 else math.inf
    else:
        z = (p1 - p2) / se
    p = 0.0 if math.isinf(z) else _two_sided_p(z)
    return {
        "applicable": True,
        "rate_a": p1,
        "rate_b": p2,
        "diff_pp": p1 - p2,
        "z": z,
        "p_value": p,
        "significant": bool(p < 0.05),
    }


def _pair_by_task(runs: list[dict], arm_a: str, arm_b: str) -> list[tuple[dict, dict]]:
    """Match runs of two arms on (task_id, seed). Arms share the task set by design."""
    a = {(r["task_id"], r["seed"]): r for r in runs if r["arm"] == arm_a}
    b = {(r["task_id"], r["seed"]): r for r in runs if r["arm"] == arm_b}
    return [(a[k], b[k]) for k in sorted(set(a) & set(b))]


def paired_cost_test(
    runs: list[dict], arm_a: str, arm_b: str, iters: int = 10000, seed: int = 0
) -> dict:
    """Paired test on cost: bootstrap CI on the mean per-task difference, plus a sign test.

    Pairing matters here. The arms run the SAME tasks, and per-task cost varies over
    orders of magnitude, so an unpaired comparison throws away most of the signal:
    between-task variance swamps the between-arm effect. Bootstrapping the paired
    differences also gives a CI on the *difference* itself, which is what the hypothesis
    is about — per-arm CIs that happen to overlap do not settle it either way.

    Positive values mean arm_a costs more.
    """
    pairs = _pair_by_task(runs, arm_a, arm_b)
    if len(pairs) < 2:
        return {"applicable": False, "arm_a": arm_a, "arm_b": arm_b}

    diffs = np.array([a["final_cost"] - b["final_cost"] for a, b in pairs])
    rng = np.random.default_rng(seed)
    boot = np.array(
        [diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(iters)]
    )
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    nonzero = int((diffs != 0).sum())
    a_costlier = int((diffs > 0).sum())

    # Paired bootstrap on the P99 DIFFERENCE (the primary metric), resampling tasks so
    # both arms are recomputed on the same resampled task set.
    costs_a = np.array([a["final_cost"] for a, _ in pairs])
    costs_b = np.array([b["final_cost"] for _, b in pairs])
    p99_boot = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, costs_a.size, costs_a.size)
        p99_boot[i] = np.percentile(costs_a[idx], 99) - np.percentile(costs_b[idx], 99)
    p99_lo, p99_hi = (float(x) for x in np.percentile(p99_boot, [2.5, 97.5]))

    return {
        "applicable": True,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_paired": len(pairs),
        "mean_cost_diff": float(diffs.mean()),
        "cost_diff_ci": (lo, hi),
        "cost_diff_significant": bool(lo > 0 or hi < 0),
        "p99_diff": float(np.percentile(costs_a, 99) - np.percentile(costs_b, 99)),
        "p99_diff_ci": (p99_lo, p99_hi),
        "p99_diff_significant": bool(p99_lo > 0 or p99_hi < 0),
        "a_costlier_on": a_costlier,
        "n_nonzero": nonzero,
        "sign_test_p": _binom_two_sided_p(a_costlier, nonzero),
    }


def paired_resolution_test(runs: list[dict], arm_a: str, arm_b: str) -> dict:
    """McNemar-style exact test on paired resolution outcomes.

    Only discordant pairs carry information: tasks both arms solved (or both failed)
    say nothing about which arm is better. With few discordant pairs the test has almost
    no power, which is itself worth reporting rather than hiding behind a point estimate.
    """
    pairs = _pair_by_task(runs, arm_a, arm_b)
    if not pairs:
        return {"applicable": False, "arm_a": arm_a, "arm_b": arm_b}
    a_only = sum(1 for a, b in pairs if a["resolved"] and not b["resolved"])
    b_only = sum(1 for a, b in pairs if b["resolved"] and not a["resolved"])
    n_disc = a_only + b_only
    return {
        "applicable": True,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_paired": len(pairs),
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_disc,
        "p_value": _binom_two_sided_p(a_only, n_disc),
        "significant": bool(_binom_two_sided_p(a_only, n_disc) < 0.05),
    }


def compare_arms(runs: list[dict], arm_a: str, arm_b: str) -> dict:
    """Full head-to-head: blown rate, paired cost/P99, and paired resolution."""
    def blown(arm):
        rs = [r for r in runs if r["arm"] == arm]
        return sum(1 for r in rs if not r["margin_held"]), len(rs)

    x1, n1 = blown(arm_a)
    x2, n2 = blown(arm_b)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "margin_blown": two_proportion_test(x1, n1, x2, n2),
        "cost": paired_cost_test(runs, arm_a, arm_b),
        "resolution": paired_resolution_test(runs, arm_a, arm_b),
    }


def non_inferiority_resolution(
    runs: list[dict], control: str, treatment: str, margin_pp: float = 0.03
) -> dict:
    """One-sided non-inferiority test on resolution rate (two-proportion z-test).

    H0: p_treatment <= p_control - margin  (treatment IS inferior by > margin)
    H1: p_treatment >  p_control - margin  (treatment is non-inferior)
    Non-inferiority is concluded when we reject H0 (z > 1.645 at alpha=0.05, one-sided).

    Two things this reports that the naive version did not, both of which matter for
    reading the result honestly:

    * FAILING to reject H0 is not evidence of inferiority. It is absence of evidence.
      The `verdict` field says which of the three states we are in, rather than
      collapsing "not proven non-inferior" into "inferior".
    * The test can be UNFALSIFIABLE at a given sample size. Its best case is a true
      difference of exactly zero, which yields z = margin / SE. If that is already below
      the critical value, the test cannot pass no matter what the data look like, and
      `feasible` is False. At n=87/arm with p~0.30 the SE on the difference is ~6.9pp, so
      the configured 3pp margin gives a best-case z of 0.43 against a 1.645 threshold —
      the criterion was decided by the design, not by the runs. `min_n_per_arm` and
      `min_feasible_margin_pp` say what it would take.
    """
    def _res(arm):
        return np.array([1.0 if r["resolved"] else 0.0 for r in runs if r["arm"] == arm])

    a, b = _res(control), _res(treatment)
    if a.size == 0 or b.size == 0:
        return {"applicable": False}

    p_c, p_t = float(a.mean()), float(b.mean())
    n_c, n_t = a.size, b.size
    diff = p_t - p_c
    p_pool = (a.sum() + b.sum()) / (n_c + n_t)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))) or 1e-9
    z = (diff + margin_pp) / se
    non_inferior = bool(z > Z_CRIT_ONE_SIDED)

    # Best case for this design: true difference exactly zero.
    best_case_z = margin_pp / se
    feasible = bool(best_case_z > Z_CRIT_ONE_SIDED)
    min_feasible_margin = Z_CRIT_ONE_SIDED * se
    # n per arm needed for the CONFIGURED margin to be conclusive at a true diff of 0.
    var_term = 2.0 * p_pool * (1 - p_pool)
    target_se = margin_pp / Z_CRIT_ONE_SIDED
    min_n = int(math.ceil(var_term / target_se**2)) if target_se > 0 else None

    if non_inferior:
        verdict = "NON-INFERIOR"
    elif not feasible:
        verdict = "INCONCLUSIVE (test cannot pass at this n — underpowered by design)"
    else:
        verdict = "NOT SHOWN NON-INFERIOR (absence of evidence, not evidence of inferiority)"

    return {
        "applicable": True,
        "control": control,
        "treatment": treatment,
        "p_control": p_c,
        "p_treatment": p_t,
        "diff_pp": float(diff),
        "margin_pp": margin_pp,
        "se": se,
        "z": float(z),
        "non_inferior": non_inferior,
        "feasible": feasible,
        "best_case_z": float(best_case_z),
        "min_feasible_margin_pp": float(min_feasible_margin),
        "min_n_per_arm": min_n,
        "n_control": int(n_c),
        "n_treatment": int(n_t),
        "verdict": verdict,
    }


def non_inferiority_resolution_paired(
    runs: list[dict], control: str, treatment: str, margin_pp: float = 0.03
) -> dict:
    """PAIRED non-inferiority test on resolution rate (McNemar-based).

    Same hypothesis as the unpaired version, but it uses the fact that the arms run the
    same tasks. Only discordant pairs carry information, so the standard error is

        SE = sqrt(b + c - (b - c)^2 / n) / n

    where b and c are the counts of tasks solved by only one arm. This is dramatically
    tighter than the unpaired SE, which pays for between-task variance the design has
    already controlled: on the observed A-vs-C data the arms agree on 80 of 87 tasks,
    and the SE drops 6.82pp -> 3.02pp. That is the difference between needing 1,217 runs
    per arm and needing 239.

    The practical consequence: a single seed over 87 tasks supports a ~6.4pp margin,
    which is well inside the measured noise floor of the apparatus (re-running the
    identical policy flips 10.8% of resolution outcomes). Asking for a 3pp guarantee is
    asking for a tolerance ~3.6x tighter than the measurement noise, which no sample
    size makes meaningful — so the margin, not the sample size, is the thing to argue
    about.
    """
    pairs = _pair_by_task(runs, control, treatment)
    if not pairs:
        return {"applicable": False}

    n = len(pairs)
    b = sum(1 for c_run, t_run in pairs if c_run["resolved"] and not t_run["resolved"])
    c = sum(1 for c_run, t_run in pairs if t_run["resolved"] and not c_run["resolved"])
    p_control = sum(1 for c_run, _ in pairs if c_run["resolved"]) / n
    p_treatment = sum(1 for _, t_run in pairs if t_run["resolved"]) / n
    diff = (c - b) / n

    var = b + c - (b - c) ** 2 / n
    se = math.sqrt(var) / n if var > 0 else 0.0

    if se == 0:
        # The arms agree on every task (or discordances exactly cancel with zero
        # variance): non-inferiority holds iff the point difference clears the margin.
        non_inferior = bool(diff >= -margin_pp)
        z = math.inf if non_inferior else -math.inf
        best_case_z = math.inf
        feasible = True
        min_feasible_margin = 0.0
        min_n = n
    else:
        z = (diff + margin_pp) / se
        non_inferior = bool(z > Z_CRIT_ONE_SIDED)
        best_case_z = margin_pp / se
        feasible = bool(best_case_z > Z_CRIT_ONE_SIDED)
        min_feasible_margin = Z_CRIT_ONE_SIDED * se
        # SE scales as 1/sqrt(n) at a fixed discordance rate.
        min_n = int(math.ceil(n * (se / (margin_pp / Z_CRIT_ONE_SIDED)) ** 2))

    if non_inferior:
        verdict = "NON-INFERIOR"
    elif not feasible:
        verdict = "INCONCLUSIVE (test cannot pass at this n — underpowered by design)"
    else:
        verdict = "NOT SHOWN NON-INFERIOR (absence of evidence, not evidence of inferiority)"

    return {
        "applicable": True,
        "paired": True,
        "control": control,
        "treatment": treatment,
        "n_paired": n,
        "control_only": b,
        "treatment_only": c,
        "n_discordant": b + c,
        "p_control": p_control,
        "p_treatment": p_treatment,
        "diff_pp": diff,
        "margin_pp": margin_pp,
        "se": se,
        "z": float(z),
        "non_inferior": non_inferior,
        "feasible": feasible,
        "best_case_z": float(best_case_z),
        "min_feasible_margin_pp": float(min_feasible_margin),
        "min_n_per_arm": min_n,
        "verdict": verdict,
    }


def decidable_margin_by_seeds(
    runs: list[dict], control: str, treatment: str, seed_counts=(1, 2, 3, 5)
) -> dict:
    """Smallest non-inferiority margin decidable at each seed count (paired test).

    Answers the question the budget actually turns on: not "how many seeds do I need"
    in the abstract, but "what quality claim does each seed count buy me". SE scales as
    1/sqrt(seeds) at a fixed discordance rate.
    """
    base = non_inferiority_resolution_paired(runs, control, treatment)
    if not base.get("applicable") or base["se"] == 0:
        return {"applicable": False}
    n_seeds_now = max(1, len({r["seed"] for r in runs if r["arm"] == control}))
    return {
        "applicable": True,
        "control": control,
        "treatment": treatment,
        "seeds_now": n_seeds_now,
        "margins_pp": {
            int(s): Z_CRIT_ONE_SIDED * base["se"] * math.sqrt(n_seeds_now / s)
            for s in seed_counts
        },
    }


def seeds_needed(runs: list[dict], min_n_per_arm: int | None) -> dict:
    """How many seeds a conclusive re-run needs, given the task set actually available."""
    tasks = len({r["task_id"] for r in runs})
    if not min_n_per_arm or tasks == 0:
        return {"applicable": False}
    return {
        "applicable": True,
        "n_tasks": tasks,
        "min_n_per_arm": min_n_per_arm,
        "seeds_needed": int(math.ceil(min_n_per_arm / tasks)),
    }


@dataclass
class PerInterventionRow:
    applied: str
    arm: str
    n: int
    mean: float
    p99: float
    margin_blown_rate: float
    resolution_rate: float


def _applied_key(run: dict) -> str:
    """The set of intervention actions a run actually received, as a stable label.

    The per-mode table below is keyed on the run's diagnosed MODE, which is not the
    same thing as what was done to it — and conflating the two produced a false claim
    in the first write-up ("for tool_loop, tool restriction -> 87.5% blown", when
    `restrict` fired exactly once in arm C across 349 runs; those runs were mostly
    compacted and downshifted instead). Keying on the applied action makes each row a
    statement about an intervention rather than about a label.
    """
    actions = sorted({i["action"] for i in (run.get("interventions") or [])})
    return "+".join(actions) if actions else "none"


def per_intervention_breakdown(runs: list[dict], arms: list[str]) -> list[PerInterventionRow]:
    """Break metrics down by (interventions actually applied) x arm."""
    keys = sorted({_applied_key(r) for r in runs})
    rows: list[PerInterventionRow] = []
    for key in keys:
        for arm in arms:
            sel = [r for r in runs if r["arm"] == arm and _applied_key(r) == key]
            if not sel:
                continue
            costs = np.array([r["final_cost"] for r in sel], dtype=float)
            rows.append(
                PerInterventionRow(
                    applied=key,
                    arm=arm,
                    n=len(sel),
                    mean=float(costs.mean()),
                    p99=_percentile(costs, 99),
                    margin_blown_rate=float(
                        np.mean([0.0 if r["margin_held"] else 1.0 for r in sel])
                    ),
                    resolution_rate=float(
                        np.mean([1.0 if r["resolved"] else 0.0 for r in sel])
                    ),
                )
            )
    return rows


def placebo_noise(runs: list[dict], arm_a: str, arm_b: str, only_no_mode: bool = False) -> dict:
    """Run-to-run noise floor between two arms that ran the SAME policy.

    Agent trajectories are nondeterministic, so identical policies produce different
    costs. Any arm-vs-arm difference smaller than this floor is not an effect. Pair by
    (task_id, seed) and report the divergence in cost, steps and resolution.

    `only_no_mode=True` restricts to runs with no true failure mode, which is how a
    same-policy channel can be extracted from A vs C-oracle: C-oracle passes through
    unconditionally when it has no label, so on those tasks it IS arm A.
    """
    def index(arm):
        out = {}
        for r in runs:
            if r["arm"] != arm:
                continue
            if only_no_mode and r.get("true_failure_mode") is not None:
                continue
            out[(r["task_id"], r["seed"])] = r
        return out

    a, b = index(arm_a), index(arm_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"applicable": False, "arm_a": arm_a, "arm_b": arm_b}

    rel = np.array(
        [
            abs(a[k]["final_cost"] - b[k]["final_cost"]) / max(a[k]["final_cost"], 1e-9)
            for k in keys
        ]
    )
    step_diff = np.array([abs(a[k]["step_count"] - b[k]["step_count"]) for k in keys])
    flips = sum(1 for k in keys if a[k]["resolved"] != b[k]["resolved"])
    worst_i = int(np.argmax(rel))
    worst_key = keys[worst_i]

    return {
        "applicable": True,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "only_no_mode": only_no_mode,
        "n_paired": len(keys),
        "median_rel_cost_diff": float(np.median(rel)),
        "mean_rel_cost_diff": float(rel.mean()),
        "p90_rel_cost_diff": float(np.percentile(rel, 90)),
        "median_step_diff": float(np.median(step_diff)),
        "resolution_flips": flips,
        "resolution_flip_rate": flips / len(keys),
        "worst": {
            "task_id": worst_key[0],
            "seed": worst_key[1],
            "rel_cost_diff": float(rel[worst_i]),
            f"cost_{arm_a}": a[worst_key]["final_cost"],
            f"cost_{arm_b}": b[worst_key]["final_cost"],
            f"steps_{arm_a}": a[worst_key]["step_count"],
            f"steps_{arm_b}": b[worst_key]["step_count"],
        },
    }


def infrastructure_cost(runs: list[dict], usd_per_sandbox_hour: float | None) -> dict:
    """Sandbox-hours per arm, and a dollar estimate when a rate is configured.

    Model spend is only part of the bill. Sandboxes are billed by the second, so a run
    that grinds to the step cap costs infrastructure money even after its token cost is
    capped by an intervention — which means an intervention that lowers token spend but
    RAISES step count can be a net loss. That interaction is invisible without timing,
    and the first full benchmark recorded none, so its infrastructure cost is
    unrecoverable. Runs logged before timing existed report `n_timed = 0`.
    """
    timed = [r for r in runs if r.get("sandbox_sec") is not None]
    if not timed:
        return {
            "applicable": False,
            "n_timed": 0,
            "n_runs": len(runs),
            "note": "no run logs carry sandbox_sec — re-run to measure infrastructure cost",
        }

    by_arm: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in timed}):
        sel = [r for r in timed if r["arm"] == arm]
        secs = np.array([r["sandbox_sec"] for r in sel], dtype=float)
        model_usd = float(sum(r["final_cost"] for r in sel))
        entry = {
            "n": len(sel),
            "sandbox_hours": float(secs.sum() / 3600.0),
            "mean_sandbox_sec": float(secs.mean()),
            "p90_sandbox_sec": float(np.percentile(secs, 90)),
            "model_usd": model_usd,
        }
        if usd_per_sandbox_hour:
            infra = entry["sandbox_hours"] * usd_per_sandbox_hour
            entry["infra_usd"] = infra
            entry["total_usd"] = model_usd + infra
            entry["infra_share"] = infra / (model_usd + infra) if model_usd + infra else 0.0
        by_arm[arm] = entry

    total_hours = sum(v["sandbox_hours"] for v in by_arm.values())
    out = {
        "applicable": True,
        "n_timed": len(timed),
        "n_runs": len(runs),
        "usd_per_sandbox_hour": usd_per_sandbox_hour,
        "total_sandbox_hours": total_hours,
        "by_arm": by_arm,
    }
    if usd_per_sandbox_hour:
        out["total_infra_usd"] = total_hours * usd_per_sandbox_hour
        out["total_model_usd"] = sum(v["model_usd"] for v in by_arm.values())
        out["total_usd"] = out["total_infra_usd"] + out["total_model_usd"]
    return out


def label_diagnostics(runs: list[dict]) -> dict:
    """Compare the label each run EXECUTED against with the current labeler's verdict.

    `true_failure_mode` is what the C-oracle arm acted on at runtime; the v2 labeler
    writes `relabeled_failure_mode` alongside it rather than overwriting. Divergence
    here means the completed oracle runs cannot be read as "the oracle with a correct
    diagnosis" — they need a re-run under the new labels.
    """
    a_runs = [r for r in runs if r["arm"] == "A"]
    def dist(field, rs):
        c: dict[str, int] = {}
        for r in rs:
            k = r.get(field) or "none"
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items()))

    changed = [
        {
            "task_id": r["task_id"],
            "executed": r.get("true_failure_mode") or "none",
            "relabeled": r.get("relabeled_failure_mode") or "none",
        }
        for r in a_runs
        if (r.get("true_failure_mode") or "none") != (r.get("relabeled_failure_mode") or "none")
    ]
    return {
        "executed_label_distribution": dist("true_failure_mode", a_runs),
        "relabeled_distribution": dist("relabeled_failure_mode", a_runs),
        "n_changed": len(changed),
        "n_arm_a_runs": len(a_runs),
        "changed": changed,
    }


@dataclass
class PerModeRow:
    mode: str
    arm: str
    n: int
    mean: float
    p99: float
    margin_blown_rate: float
    resolution_rate: float


def per_mode_breakdown(runs: list[dict], arms: list[str]) -> list[PerModeRow]:
    """Break metrics down by true_failure_mode x arm."""
    modes = sorted({r.get("true_failure_mode") or "none" for r in runs})
    rows: list[PerModeRow] = []
    for mode in modes:
        for arm in arms:
            sel = [
                r
                for r in runs
                if r["arm"] == arm and (r.get("true_failure_mode") or "none") == mode
            ]
            if not sel:
                continue
            costs = np.array([r["final_cost"] for r in sel], dtype=float)
            blown = np.mean([0.0 if r["margin_held"] else 1.0 for r in sel])
            res = np.mean([1.0 if r["resolved"] else 0.0 for r in sel])
            rows.append(
                PerModeRow(
                    mode=mode,
                    arm=arm,
                    n=len(sel),
                    mean=float(costs.mean()),
                    p99=_percentile(costs, 99),
                    margin_blown_rate=float(blown),
                    resolution_rate=float(res),
                )
            )
    return rows
