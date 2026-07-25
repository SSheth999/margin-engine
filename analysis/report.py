"""Verdict report (build step 7) — reads run logs, prints the verdict, writes summary.

Evaluates the three AGENTS.MD success criteria and prints the honest sub-cases:
  1. C shows lower P99 AND lower margin-blown than A.
  2. C beats B on P99 and margin-blown (matching > generic).
  3. C resolution rate non-inferior to A (quality preserved).
SUPPORTED iff all three. Otherwise reports which honest branch we're in
(C beats A but ties B; C ties B but C-oracle beats B; etc.).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from analysis.metrics import (
    RUNS_DIR,
    arm_stats,
    load_runs,
    non_inferiority_resolution,
    per_mode_breakdown,
)
from analysis.plots import cost_distributions

ARMS = ["A", "B", "C", "C-oracle"]


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def build_report(runs_dir: Path = RUNS_DIR, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (runs_dir.parent / "analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    present = [a for a in ARMS if any(r["arm"] == a for r in runs)]

    stats = {a: arm_stats(runs, a) for a in present}
    have = lambda a: a in stats and stats[a].n > 0  # noqa: E731

    # --- criteria ---
    crit = {}
    if have("A") and have("C"):
        crit["c_beats_a"] = (
            stats["C"].p99 < stats["A"].p99
            and stats["C"].margin_blown_rate < stats["A"].margin_blown_rate
        )
    if have("B") and have("C"):
        crit["c_beats_b"] = (
            stats["C"].p99 < stats["B"].p99
            and stats["C"].margin_blown_rate <= stats["B"].margin_blown_rate
        )
    noninf = (
        non_inferiority_resolution(runs, "A", "C")
        if have("A") and have("C")
        else {"applicable": False}
    )
    crit["c_quality_ok"] = bool(noninf.get("non_inferior", False))

    oracle_beats_b = None
    if have("B") and have("C-oracle"):
        oracle_beats_b = (
            stats["C-oracle"].p99 < stats["B"].p99
            and stats["C-oracle"].margin_blown_rate <= stats["B"].margin_blown_rate
        )

    supported = all(crit.get(k, False) for k in ("c_beats_a", "c_beats_b", "c_quality_ok"))

    # --- honest narrative ---
    if supported:
        verdict = "HYPOTHESIS SUPPORTED: C beats A and B on tail, quality preserved."
    elif crit.get("c_beats_a") and not crit.get("c_beats_b"):
        if oracle_beats_b:
            verdict = ("PARTIAL: C beats A but ties B, while C-oracle beats B -> the "
                       "matching idea is right, the DETECTOR needs work.")
        else:
            verdict = ("PARTIAL: C beats A but ties B -> intervention helps, but "
                       "MATCHING is not yet proven over generic.")
    elif not crit.get("c_quality_ok"):
        verdict = "NOT SUPPORTED: resolution-rate non-inferiority failed (quality regressed)."
    else:
        verdict = "NOT SUPPORTED: C did not beat control on the tail metrics."

    per_mode = per_mode_breakdown(runs, present)
    plot_path = cost_distributions(runs, out_dir / "cost_distributions.png", present)

    report = {
        "n_runs": len(runs),
        "arms_present": present,
        "arm_stats": {a: asdict(stats[a]) for a in present},
        "criteria": crit,
        "non_inferiority": noninf,
        "oracle_beats_b": oracle_beats_b,
        "supported": supported,
        "verdict": verdict,
        "per_mode": [asdict(r) for r in per_mode],
        "plot": str(plot_path) if plot_path else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _print(report: dict) -> None:
    print(f"\n=== MARGIN ENGINE VERDICT ({report['n_runs']} runs) ===\n")
    hdr = f"{'arm':<9} {'n':>3} {'mean':>9} {'P90':>9} {'P99':>9} {'std':>9} {'blown%':>8} {'resolved%':>10}"
    print(hdr)
    print("-" * len(hdr))
    for a in report["arms_present"]:
        s = report["arm_stats"][a]
        print(
            f"{a:<9} {s['n']:>3} {_fmt(s['mean']):>9} {_fmt(s['p90']):>9} "
            f"{_fmt(s['p99']):>9} {_fmt(s['std']):>9} "
            f"{100*s['margin_blown_rate']:>7.1f}% {100*s['resolution_rate']:>9.1f}%"
        )

    ni = report["non_inferiority"]
    if ni.get("applicable"):
        print(
            f"\nnon-inferiority (C vs A resolution): diff={100*ni['diff_pp']:+.1f}pp "
            f"margin={100*ni['margin_pp']:.0f}pp z={ni['z']:.2f} "
            f"-> {'NON-INFERIOR' if ni['non_inferior'] else 'INFERIOR'}"
        )

    print("\ncriteria:")
    for k, v in report["criteria"].items():
        print(f"  {k}: {v}")
    if report["oracle_beats_b"] is not None:
        print(f"  oracle_beats_b: {report['oracle_beats_b']}")

    if report["per_mode"]:
        print("\nper true_failure_mode:")
        print(f"  {'mode':<20} {'arm':<9} {'n':>3} {'mean':>9} {'P99':>9} {'blown%':>8} {'res%':>7}")
        for r in report["per_mode"]:
            print(
                f"  {r['mode']:<20} {r['arm']:<9} {r['n']:>3} {_fmt(r['mean']):>9} "
                f"{_fmt(r['p99']):>9} {100*r['margin_blown_rate']:>7.1f}% "
                f"{100*r['resolution_rate']:>6.1f}%"
            )

    print(f"\n>>> {report['verdict']}\n")
    if report["plot"]:
        print(f"distribution plot: {report['plot']}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = ap.parse_args()
    report = build_report(Path(args.runs_dir))
    _print(report)


if __name__ == "__main__":
    _main()
