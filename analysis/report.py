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
    compare_arms,
    decidable_margin_by_seeds,
    infrastructure_cost,
    label_diagnostics,
    load_runs,
    non_inferiority_resolution,
    non_inferiority_resolution_paired,
    per_intervention_breakdown,
    per_mode_breakdown,
    placebo_noise,
    seeds_needed,
)
from analysis.plots import cost_distributions

# A-prime is the placebo (same policy as A, measures the noise floor); D is the
# laddered arm. Order is the presentation order.
ARMS = ["A", "A-prime", "B", "C", "C-oracle", "D"]


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def build_report(runs_dir: Path = RUNS_DIR, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (runs_dir.parent / "analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    present = [a for a in ARMS if any(r["arm"] == a for r in runs)]

    stats = {a: arm_stats(runs, a) for a in present}
    have = lambda a: a in stats and stats[a].n > 0  # noqa: E731

    # --- head-to-head tests (paired where the arms share tasks) ---
    # Every criterion below is decided by a TEST, not by `>` on two point estimates.
    # Point estimates alone cannot settle this: the measured same-policy noise floor
    # (see `placebo_noise`) is larger than most of the arm gaps.
    pairs_of_interest = [
        (a, b)
        for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("B", "C-oracle"),
                     ("A", "C-oracle"), ("A", "D"), ("B", "D"), ("C", "D"))
        if have(a) and have(b)
    ]
    head_to_head = {f"{a}_vs_{b}": compare_arms(runs, a, b) for a, b in pairs_of_interest}

    def _oriented(a: str, b: str) -> dict | None:
        """The a-minus-b comparison, flipping a cached b-minus-a entry if that's what we
        stored. Comparisons are computed once per unordered pair; asking for the other
        orientation must not silently return nothing."""
        cmp = head_to_head.get(f"{a}_vs_{b}")
        if cmp is not None:
            return cmp
        rev = head_to_head.get(f"{b}_vs_{a}")
        if rev is None:
            return None
        cost, blown, res = rev["cost"], rev["margin_blown"], rev["resolution"]
        flip_ci = lambda ci: (-ci[1], -ci[0])  # noqa: E731
        return {
            "arm_a": a,
            "arm_b": b,
            "margin_blown": {
                **blown,
                "rate_a": blown.get("rate_b"),
                "rate_b": blown.get("rate_a"),
                "diff_pp": -blown["diff_pp"] if blown.get("applicable") else None,
                "z": -blown["z"] if blown.get("applicable") else None,
            },
            "cost": (
                {
                    **cost,
                    "arm_a": a,
                    "arm_b": b,
                    "mean_cost_diff": -cost["mean_cost_diff"],
                    "cost_diff_ci": flip_ci(cost["cost_diff_ci"]),
                    "p99_diff": -cost["p99_diff"],
                    "p99_diff_ci": flip_ci(cost["p99_diff_ci"]),
                    "a_costlier_on": cost["n_nonzero"] - cost["a_costlier_on"],
                }
                if cost.get("applicable")
                else cost
            ),
            "resolution": (
                {**res, "arm_a": a, "arm_b": b,
                 "a_only": res["b_only"], "b_only": res["a_only"]}
                if res.get("applicable")
                else res
            ),
        }

    def _beats(a: str, b: str) -> dict:
        """Does arm `b` significantly beat arm `a` on the tail? (b cheaper than a.)"""
        cmp = _oriented(a, b)
        if cmp is None:
            return {"applicable": False}
        cost, blown = cmp["cost"], cmp["margin_blown"]
        # b wins on cost if the paired difference (a - b) is significantly positive.
        cost_win = bool(cost.get("cost_diff_significant") and cost.get("mean_cost_diff", 0) > 0)
        p99_win = bool(cost.get("p99_diff_significant") and cost.get("p99_diff", 0) > 0)
        blown_win = bool(blown.get("significant") and blown.get("diff_pp", 0) > 0)
        return {
            "applicable": True,
            "point_estimate_favors": bool(
                stats[b].p99 < stats[a].p99
                and stats[b].margin_blown_rate <= stats[a].margin_blown_rate
            ),
            "cost_significant": cost_win,
            "p99_significant": p99_win,
            "margin_blown_significant": blown_win,
            # The tail is the hypothesis, so require a significant tail win: either the
            # P99 difference or the margin-blown rate, not merely the mean.
            "significant": bool(p99_win or blown_win),
        }

    crit = {}
    if have("A") and have("C"):
        crit["c_beats_a"] = _beats("A", "C")
    if have("B") and have("C"):
        crit["c_beats_b"] = _beats("B", "C")

    # Quality: the PAIRED test is primary. The arms share the task set, so the unpaired
    # test pays for between-task variance the design already controls — on this data it
    # inflates the required sample from 239 runs/arm to 1,217, which is the difference
    # between a 1-seed and a 14-seed experiment. The unpaired result is kept as a
    # conservative cross-check, not as the criterion.
    noninf = (
        non_inferiority_resolution_paired(runs, "A", "C")
        if have("A") and have("C")
        else {"applicable": False}
    )
    noninf_unpaired = (
        non_inferiority_resolution(runs, "A", "C")
        if have("A") and have("C")
        else {"applicable": False}
    )
    crit["c_quality_ok"] = {
        "applicable": noninf.get("applicable", False),
        "significant": bool(noninf.get("non_inferior", False)),
        "feasible": noninf.get("feasible"),
        "verdict": noninf.get("verdict"),
    }
    power = seeds_needed(runs, noninf.get("min_n_per_arm"))
    margin_by_seeds = (
        decidable_margin_by_seeds(runs, "A", "C")
        if have("A") and have("C")
        else {"applicable": False}
    )

    # Infrastructure cost, if the sweep recorded timings and a rate is configured.
    infra_rate = None
    try:
        import yaml

        with open(runs_dir.parent / "config" / "experiment.yaml") as f:
            infra_rate = (yaml.safe_load(f) or {}).get("e2b_usd_per_sandbox_hour")
    except Exception:
        pass
    infra = infrastructure_cost(runs, infra_rate)

    oracle_beats_b = _beats("B", "C-oracle") if have("B") and have("C-oracle") else None

    def _sig(key) -> bool:
        c = crit.get(key)
        return bool(c and c.get("significant"))

    supported = all(_sig(k) for k in ("c_beats_a", "c_beats_b", "c_quality_ok"))

    # --- honest narrative ---
    # A reversal (the other arm significantly beating C) is a different finding from a
    # tie, and a quality test that cannot pass at this n is different again from a
    # quality regression. Keep the three apart.
    b_beats_c = _beats("C", "B") if have("B") and have("C") else {"applicable": False}
    quality_untestable = crit["c_quality_ok"].get("feasible") is False

    if supported:
        verdict = "HYPOTHESIS SUPPORTED: C beats A and B on tail, quality preserved."
    elif b_beats_c.get("significant"):
        verdict = (
            "REJECTED IN REVERSE: generic (B) significantly beats matched (C) on the "
            "tail -> matching as implemented is worse than the generic policy."
        )
    elif _sig("c_beats_a") and not _sig("c_beats_b"):
        if oracle_beats_b and oracle_beats_b.get("significant"):
            verdict = ("PARTIAL: C beats A but ties B, while C-oracle beats B -> the "
                       "matching idea is right, the DETECTOR needs work.")
        else:
            verdict = ("PARTIAL: C beats A but ties B -> intervention helps, but "
                       "MATCHING is not yet proven over generic.")
    elif quality_untestable:
        verdict = (
            "INCONCLUSIVE: C did not significantly beat the control on the tail, and the "
            "quality criterion cannot be decided at this sample size."
        )
    elif not _sig("c_quality_ok"):
        verdict = "NOT SUPPORTED: resolution non-inferiority not established."
    else:
        verdict = "NOT SUPPORTED: C did not beat control on the tail metrics."

    # --- arm D (laddered): does it beat the best of A/B/C on tail AND quality? ---
    d_vs = None
    if have("D"):
        d_vs = {}
        for other in ("A", "B", "C", "C-oracle"):
            if have(other):
                d_vs[other] = {
                    "p99_lower": stats["D"].p99 < stats[other].p99,
                    "blown_lower_or_equal": (
                        stats["D"].margin_blown_rate <= stats[other].margin_blown_rate
                    ),
                    "resolution_delta_pp": (
                        stats["D"].resolution_rate - stats[other].resolution_rate
                    ),
                }
        d_noninf = non_inferiority_resolution(runs, "A", "D") if have("A") else {"applicable": False}
        d_vs["non_inferiority_vs_A"] = d_noninf

    # --- noise floor: any two arms running the SAME policy ---
    # Prefer the deliberate placebo; fall back to the accidental one (C-oracle passes
    # through unconditionally on runs it has no label for, so there it IS arm A).
    if have("A") and have("A-prime"):
        noise = placebo_noise(runs, "A", "A-prime")
        noise["source"] = "deliberate placebo arm (A vs A-prime)"
    elif have("A") and have("C-oracle"):
        noise = placebo_noise(runs, "A", "C-oracle", only_no_mode=True)
        noise["source"] = (
            "accidental placebo: A vs C-oracle on runs with no true failure mode, "
            "where C-oracle passes through unconditionally. Run arm A-prime for a "
            "clean measurement."
        )
    else:
        noise = {"applicable": False}

    per_mode = per_mode_breakdown(runs, present)
    per_intervention = per_intervention_breakdown(runs, present)
    labels = label_diagnostics(runs)
    plot_path = cost_distributions(runs, out_dir / "cost_distributions.png", present)

    report = {
        "n_runs": len(runs),
        "arms_present": present,
        "arm_stats": {a: asdict(stats[a]) for a in present},
        "criteria": crit,
        "head_to_head": head_to_head,
        "non_inferiority": noninf,
        "non_inferiority_unpaired": noninf_unpaired,
        "power": power,
        "decidable_margin_by_seeds": margin_by_seeds,
        "infrastructure": infra,
        "oracle_beats_b": oracle_beats_b,
        "supported": supported,
        "verdict": verdict,
        "noise_floor": noise,
        "arm_d": d_vs,
        "per_mode": [asdict(r) for r in per_mode],
        "per_intervention": [asdict(r) for r in per_intervention],
        "label_diagnostics": labels,
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

    h2h = report.get("head_to_head") or {}
    if h2h:
        print("\nHEAD-TO-HEAD TESTS (paired on task; positive = first arm costs more):")
        hdr2 = (f"  {'comparison':<16} {'blown diff':>11} {'p':>9} "
                f"{'paired cost diff [95% CI]':>30} {'P99 diff [95% CI]':>28} {'res p':>7}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for key, c in h2h.items():
            b, cost, res = c["margin_blown"], c["cost"], c["resolution"]
            if not cost.get("applicable"):
                continue
            star = (
                "***" if b.get("p_value", 1) < 0.001
                else "**" if b.get("p_value", 1) < 0.01
                else "*" if b.get("p_value", 1) < 0.05
                else "ns"
            )
            ci = cost["cost_diff_ci"]
            pci = cost["p99_diff_ci"]
            print(
                f"  {key:<16} {100*b.get('diff_pp', float('nan')):+10.1f}pp "
                f"{b.get('p_value', float('nan')):8.1e}{star:<3} "
                f"{cost['mean_cost_diff']:+9.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
                f"{'!' if cost['cost_diff_significant'] else ' '} "
                f"{cost['p99_diff']:+8.4f} [{pci[0]:+.4f},{pci[1]:+.4f}]"
                f"{'!' if cost['p99_diff_significant'] else ' '} "
                f"{res.get('p_value', float('nan')):6.3f}"
            )
        print("  ! = CI excludes zero;  *p<.05 **p<.01 ***p<.001;  res p = paired "
              "exact test on resolution")

    ni = report["non_inferiority"]
    if ni.get("applicable"):
        print(
            f"\nnon-inferiority PAIRED ({ni['treatment']} vs {ni['control']} resolution): "
            f"diff={100*ni['diff_pp']:+.1f}pp margin={100*ni['margin_pp']:.0f}pp "
            f"z={ni['z']:.2f} (need >{1.645:.3f})"
        )
        print(
            f"  discordant pairs {ni['n_discordant']}/{ni['n_paired']} "
            f"({ni['control_only']} control-only, {ni['treatment_only']} treatment-only)"
            f"  SE={100*ni['se']:.2f}pp"
        )
        print(f"  -> {ni['verdict']}")
        nu = report.get("non_inferiority_unpaired") or {}
        if nu.get("applicable"):
            print(
                f"  (unpaired cross-check: SE={100*nu['se']:.2f}pp, "
                f"needs n>={nu['min_n_per_arm']}/arm — pairing is {nu['se']/ni['se']:.1f}x "
                f"tighter because the arms share tasks)"
                if ni["se"] > 0 else "  (unpaired cross-check unavailable)"
            )
        if not ni.get("feasible"):
            # Absence of evidence is not evidence of inferiority. Say so explicitly,
            # and say what it would take, so the criterion is not read as a finding.
            print(
                f"  UNDERPOWERED BY DESIGN: best possible z at this n (true diff = 0) is "
                f"{ni['best_case_z']:.2f} < 1.645, so this test CANNOT pass regardless of "
                f"the data."
            )
            print(
                f"  needs n>={ni['min_n_per_arm']}/arm for a "
                f"{100*ni['margin_pp']:.0f}pp margin, or a margin "
                f">={100*ni['min_feasible_margin_pp']:.1f}pp at n={ni['n_paired']}."
            )
            pw = report.get("power") or {}
            if pw.get("applicable"):
                print(
                    f"  => a conclusive re-run needs ~{pw['seeds_needed']} seeds "
                    f"({pw['min_n_per_arm']} runs/arm over {pw['n_tasks']} tasks)."
                )

        mbs = report.get("decidable_margin_by_seeds") or {}
        if mbs.get("applicable"):
            # The budget turns on this: not "how many seeds" but "what claim each buys".
            cells = "  ".join(
                f"{s} seed{'s' if s != 1 else ' '}: {100*m:.1f}pp"
                for s, m in sorted(mbs["margins_pp"].items())
            )
            print(f"  margin decidable at each seed count -> {cells}")

    infra = report.get("infrastructure") or {}
    if infra.get("applicable"):
        rate = infra.get("usd_per_sandbox_hour")
        print(
            f"\nINFRASTRUCTURE ({infra['n_timed']}/{infra['n_runs']} runs timed, "
            f"{infra['total_sandbox_hours']:.1f} sandbox-hours total):"
        )
        hdr3 = f"  {'arm':<9} {'n':>4} {'sandbox h':>10} {'mean s':>8} {'p90 s':>8} {'model $':>9}"
        if rate:
            hdr3 += f" {'infra $':>9} {'total $':>9} {'infra%':>7}"
        print(hdr3)
        for arm, v in infra["by_arm"].items():
            line = (f"  {arm:<9} {v['n']:>4} {v['sandbox_hours']:>10.2f} "
                    f"{v['mean_sandbox_sec']:>8.0f} {v['p90_sandbox_sec']:>8.0f} "
                    f"{v['model_usd']:>9.2f}")
            if rate:
                line += (f" {v['infra_usd']:>9.2f} {v['total_usd']:>9.2f} "
                         f"{100*v['infra_share']:>6.0f}%")
            print(line)
        if rate:
            print(
                f"  TOTAL model ${infra['total_model_usd']:.2f} + infra "
                f"${infra['total_infra_usd']:.2f} = ${infra['total_usd']:.2f} "
                f"@ ${rate}/sandbox-hour"
            )
        else:
            print("  set e2b_usd_per_sandbox_hour in config/experiment.yaml for a $ estimate")
    elif infra.get("n_timed") == 0:
        print(f"\nINFRASTRUCTURE: not measurable — {infra['note']}")

    print("\ncriteria (significance-based, not point estimates):")
    for k, v in report["criteria"].items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
            continue
        mark = "PASS" if v.get("significant") else "fail"
        extra = []
        if v.get("point_estimate_favors") and not v.get("significant"):
            extra.append("point estimate favors it but not significantly")
        if v.get("feasible") is False:
            extra.append("untestable at this n")
        note = f"  ({'; '.join(extra)})" if extra else ""
        print(f"  {k}: {mark}{note}")
    if report["oracle_beats_b"] is not None:
        ob = report["oracle_beats_b"]
        print(f"  oracle_beats_b: {'PASS' if ob.get('significant') else 'fail'}")

    nf = report.get("noise_floor") or {}
    if nf.get("applicable"):
        w = nf["worst"]
        print(f"\nNOISE FLOOR — same policy, {nf['n_paired']} paired runs ({nf['source']}):")
        print(
            f"  |cost diff|: median {100*nf['median_rel_cost_diff']:.0f}%  "
            f"mean {100*nf['mean_rel_cost_diff']:.0f}%  p90 {100*nf['p90_rel_cost_diff']:.0f}%"
            f"   resolution flips {nf['resolution_flips']}/{nf['n_paired']}"
        )
        print(
            f"  worst: {w['task_id']} seed{w['seed']} "
            f"{100*w['rel_cost_diff']:+.0f}%  "
            f"({nf['arm_a']} ${w['cost_'+nf['arm_a']]:.4f}/{w['steps_'+nf['arm_a']]}st vs "
            f"{nf['arm_b']} ${w['cost_'+nf['arm_b']]:.4f}/{w['steps_'+nf['arm_b']]}st)"
        )
        print("  -> arm differences smaller than this floor are not effects.")

    d = report.get("arm_d")
    if d:
        print("\narm D (laddered) vs each arm:")
        for other, v in d.items():
            if other == "non_inferiority_vs_A":
                continue
            print(
                f"  vs {other:<9} P99 lower: {str(v['p99_lower']):<5} "
                f"blown <=: {str(v['blown_lower_or_equal']):<5} "
                f"resolution {100*v['resolution_delta_pp']:+.1f}pp"
            )
        dni = d.get("non_inferiority_vs_A") or {}
        if dni.get("applicable"):
            print(
                f"  non-inferiority (D vs A resolution): diff={100*dni['diff_pp']:+.1f}pp "
                f"z={dni['z']:.2f} -> {'NON-INFERIOR' if dni['non_inferior'] else 'INFERIOR'}"
            )

    if report["per_mode"]:
        print("\nper true_failure_mode (the label the run EXECUTED against):")
        print(f"  {'mode':<20} {'arm':<9} {'n':>3} {'mean':>9} {'P99':>9} {'blown%':>8} {'res%':>7}")
        for r in report["per_mode"]:
            print(
                f"  {r['mode']:<20} {r['arm']:<9} {r['n']:>3} {_fmt(r['mean']):>9} "
                f"{_fmt(r['p99']):>9} {100*r['margin_blown_rate']:>7.1f}% "
                f"{100*r['resolution_rate']:>6.1f}%"
            )

    if report.get("per_intervention"):
        print("\nper intervention ACTUALLY APPLIED (not per diagnosed mode):")
        print(f"  {'applied':<20} {'arm':<9} {'n':>3} {'mean':>9} {'P99':>9} {'blown%':>8} {'res%':>7}")
        for r in report["per_intervention"]:
            print(
                f"  {r['applied']:<20} {r['arm']:<9} {r['n']:>3} {_fmt(r['mean']):>9} "
                f"{_fmt(r['p99']):>9} {100*r['margin_blown_rate']:>7.1f}% "
                f"{100*r['resolution_rate']:>6.1f}%"
            )

    ld = report.get("label_diagnostics") or {}
    if ld.get("n_arm_a_runs"):
        print(f"\nlabels (Arm A, n={ld['n_arm_a_runs']}):")
        print(f"  executed  : {ld['executed_label_distribution']}")
        print(f"  relabeled : {ld['relabeled_distribution']}")
        print(f"  changed   : {ld['n_changed']}/{ld['n_arm_a_runs']} runs")
        if ld["n_changed"]:
            print("  -> completed C-oracle runs acted on the OLD labels; they do not")
            print("     measure 'the oracle with a correct diagnosis'. Re-run needed.")

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
