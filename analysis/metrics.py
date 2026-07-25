"""Metrics over the run logs (build step 7).

Primary metrics are tail/variance (the hypothesis): P99, P90, margin-blown rate, spread.
Secondary is mean (reported, not led with). The quality constraint is resolution rate
with a non-inferiority test. Everything can be broken down per failure mode.

No scipy dependency: bootstrap CIs and the two-proportion non-inferiority z-test are
implemented directly on numpy.
"""

from __future__ import annotations

import json
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


def non_inferiority_resolution(
    runs: list[dict], control: str, treatment: str, margin_pp: float = 0.03
) -> dict:
    """One-sided non-inferiority test on resolution rate (two-proportion z-test).

    H0: p_treatment <= p_control - margin  (treatment IS inferior by > margin)
    H1: p_treatment >  p_control - margin  (treatment is non-inferior)
    Non-inferiority is concluded when we reject H0 (z > 1.645 at alpha=0.05, one-sided).
    """
    def _res(arm):
        rr = np.array(
            [1.0 if r["resolved"] else 0.0 for r in runs if r["arm"] == arm]
        )
        return rr

    a = _res(control)
    b = _res(treatment)
    if a.size == 0 or b.size == 0:
        return {"applicable": False}

    p_c, p_t = a.mean(), b.mean()
    n_c, n_t = a.size, b.size
    diff = p_t - p_c
    # Pooled SE for the difference of proportions.
    p_pool = (a.sum() + b.sum()) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)) or 1e-9
    z = (diff + margin_pp) / se
    non_inferior = bool(z > 1.645)
    return {
        "applicable": True,
        "control": control,
        "treatment": treatment,
        "p_control": float(p_c),
        "p_treatment": float(p_t),
        "diff_pp": float(diff),
        "margin_pp": margin_pp,
        "z": float(z),
        "non_inferior": non_inferior,
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
