"""Post-hoc failure-mode labeling (build step 5).

Runs OFFLINE against the Arm A run logs. Because it sees the entire trajectory (not
just the state before one call), it can assign `true_failure_mode` more reliably than
the live detector. That label:
  - is written to a labels file ("task_id__seed" -> mode) consumed by the C-oracle arm,
  - is backfilled into every run log (all arms) for the same (task_id, seed), populating
    the `true_failure_mode` field for analysis.

Labeling uses the SAME thresholds as live detection where possible, but applied over
the whole run. A run whose cost stayed comfortably under ceiling with no waste signal
is labeled None (no failure mode) — e.g. easy tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gateway.detectors import (
    CONTEXT_BLOWOUT,
    DetectionConfig,
    INHERENT_DIFFICULTY,
    TOOL_LOOP,
    WRONG_MODEL,
)

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"


def label_run(log: dict, cfg: DetectionConfig) -> str | None:
    """Assign a true failure mode from a full run log, or None."""
    steps = log.get("per_step", [])
    if not steps:
        return None

    ceiling = log.get("margin_ceiling", 0.0) or 0.0
    final_cost = log.get("final_cost", 0.0)
    blown = ceiling > 0 and final_cost > ceiling
    near = ceiling > 0 and final_cost >= cfg.trigger_cost_ratio * ceiling

    # Tool-loop: any (tool, args_hash) repeated >= threshold across the whole run.
    pairs = [
        (s["tool"], s["args_hash"])
        for s in steps
        if s.get("tool") is not None
    ]
    if pairs:
        _, top = Counter(pairs).most_common(1)[0]
        if top >= cfg.tool_loop_repeats:
            return TOOL_LOOP

    # Context blowout: peak context size grew past the ratio vs the run's start.
    sizes = [s.get("context_size", 0) for s in steps]
    if sizes and min(sizes) > 0 and max(sizes) / sizes[0] >= cfg.context_growth_ratio:
        if near or blown:
            return CONTEXT_BLOWOUT

    # If the run never approached the ceiling and shows no waste, it's not a failure.
    if not near and not blown:
        return None

    # Expensive from very early -> inherent difficulty.
    early = steps[: cfg.early_steps]
    if early:
        early_ratio = early[-1]["cum_cost"] / ceiling if ceiling > 0 else 0.0
        if early_ratio >= cfg.inherent_difficulty_ratio:
            return INHERENT_DIFFICULTY

    # High cost, no loop/blowout, gradual -> wrong (too-expensive) model.
    return WRONG_MODEL


def build_labels(runs_dir: Path = RUNS_DIR) -> dict[str, str]:
    """Label from Arm A logs; return {"task_id__seed": mode} (only non-None modes)."""
    cfg = DetectionConfig.load()
    labels: dict[str, str] = {}
    for path in sorted(runs_dir.glob("*__A__seed*.json")):
        log = json.loads(path.read_text())
        mode = label_run(log, cfg)
        if mode is not None:
            labels[f"{log['task_id']}__{log['seed']}"] = mode
    return labels


def backfill_true_mode(labels: dict[str, str], runs_dir: Path = RUNS_DIR) -> int:
    """Write true_failure_mode into every run log sharing a labeled (task_id, seed)."""
    n = 0
    for path in sorted(runs_dir.glob("*.json")):
        log = json.loads(path.read_text())
        key = f"{log['task_id']}__{log['seed']}"
        if key in labels and log.get("true_failure_mode") != labels[key]:
            log["true_failure_mode"] = labels[key]
            path.write_text(json.dumps(log, indent=2))
            n += 1
    return n


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out", default=str(REPO / "config" / "labels.json"))
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    labels = build_labels(runs_dir)
    Path(args.out).write_text(json.dumps(labels, indent=2))
    print(f"wrote {len(labels)} labels -> {args.out}")
    for k, v in labels.items():
        print(f"  {k}: {v}")

    if args.backfill:
        n = backfill_true_mode(labels, runs_dir)
        print(f"backfilled true_failure_mode into {n} run logs")


if __name__ == "__main__":
    _main()
