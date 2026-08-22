"""Post-hoc failure-mode labeling (build step 5).

Runs OFFLINE against the Arm A run logs. Because it sees the entire trajectory (not
just the state before one call), it can assign `true_failure_mode` more reliably than
the live detector. That label:
  - is written to a labels file ("task_id__seed" -> mode) consumed by the C-oracle arm,
  - is backfilled into every run log (all arms) for the same (task_id, seed).

A run whose cost stayed comfortably under ceiling with no waste signal is labeled None
(no failure mode) — e.g. easy tasks.

--------------------------------------------------------------------------------------
LABEL VERSIONS

v1 separated context_blowout from wrong_model with `max(context)/context[0] >= 1.5`.
That rule is degenerate on real agent trajectories: context grows monotonically as
tool output accumulates, so 99% of runs pass it (median total growth 17.8x), and every
one of the 50 near-or-over-ceiling Arm A runs landed on context_blowout. `wrong_model`
and `inherent_difficulty` were therefore unreachable, which meant the C-oracle arm
could only ever compact or restrict — it was never allowed to downshift, the one lever
that demonstrably controls cost. The negative result from that sweep is a statement
about this labeler, not about the matching hypothesis.

v2 discriminates on WHERE THE MONEY WENT instead of on raw context growth, which also
ties each label to the fix that can actually address it:

  input_cost_share = (uncached-input $ + cached-input $) / total $

  - high share -> the spend is carrying context forward   -> context_blowout -> compact
  - low share  -> the spend is generating tokens per call -> wrong_model     -> downshift

Windowed growth rate was evaluated as an alternative discriminator and rejected: at
every threshold from 1.2x to 2.0x it fired on 50/50 near-ceiling runs, i.e. it carries
no information here. input_cost_share has median 0.81 and p10 0.52 over the same runs,
so it actually separates them.

Because relabeling changes the meaning of an already-executed C-oracle run, v2 labels
are backfilled into a SEPARATE field (`relabeled_failure_mode`) and `true_failure_mode`
is left as the label the run was actually executed under. Analysis can then compare the
two without rewriting history.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gateway.cost import RateCard
from gateway.detectors import (
    CONTEXT_BLOWOUT,
    DetectionConfig,
    INHERENT_DIFFICULTY,
    TOOL_LOOP,
    WRONG_MODEL,
)

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "runs"

LABEL_VERSION = 2

# Fraction of a run's spend that must be input/context cost for the run to count as a
# context blowout rather than a wrong-model (per-call generation) problem. Sits between
# the p10 (0.52) and p90 (0.91) of the observed distribution on near-ceiling runs.
INPUT_COST_SHARE_THRESHOLD = 0.75

_RATE_CARDS: dict[str, RateCard] = {}


def _rate_card(provider: str) -> RateCard:
    if provider not in _RATE_CARDS:
        _RATE_CARDS[provider] = RateCard.load(provider)
    return _RATE_CARDS[provider]


def input_cost_share(log: dict) -> float | None:
    """Fraction of the run's spend that went on input (context) rather than output.

    Uses the same rate card the gateway billed with, so this is the real split, not a
    token-count proxy: cached input is genuinely ~4x cheaper and must be weighted so.
    Returns None when the models in the log aren't priced (then the caller can't rely
    on this signal).
    """
    card = _rate_card(log.get("provider", ""))
    input_cost = 0.0
    output_cost = 0.0
    for s in log.get("per_step", []):
        model = s.get("model")
        if not model or not card.has(model):
            return None
        rate = card.rate(model)
        input_cost += s.get("input_tokens", 0) * rate.input_per_mtok / 1e6
        input_cost += (
            s.get("cached_tokens", 0)
            * rate.input_per_mtok
            * rate.cache_read_multiplier
            / 1e6
        )
        output_cost += s.get("output_tokens", 0) * rate.output_per_mtok / 1e6
    total = input_cost + output_cost
    if total <= 0:
        return None
    return input_cost / total


def label_run(log: dict, cfg: DetectionConfig) -> str | None:
    """Assign a true failure mode from a full run log, or None.

    Order matters: a tool loop is the clearest waste signal and is checked first, then
    the run has to have actually approached its ceiling to count as a failure at all,
    then early-and-expensive is separated out, and finally the spend split decides
    between carrying context and generating tokens.
    """
    steps = log.get("per_step", [])
    if not steps:
        return None

    ceiling = log.get("margin_ceiling", 0.0) or 0.0
    final_cost = log.get("final_cost", 0.0)
    blown = ceiling > 0 and final_cost > ceiling
    near = ceiling > 0 and final_cost >= cfg.trigger_cost_ratio * ceiling

    # Tool-loop: any (tool, args_hash) repeated >= threshold across the whole run.
    pairs = [(s["tool"], s["args_hash"]) for s in steps if s.get("tool") is not None]
    if pairs:
        _, top = Counter(pairs).most_common(1)[0]
        if top >= cfg.tool_loop_repeats:
            return TOOL_LOOP

    # If the run never approached the ceiling and shows no loop, it's not a failure.
    if not near and not blown:
        return None

    # Expensive from very early -> the task is just costly, no waste to reclaim.
    early = steps[: cfg.early_steps]
    if early and ceiling > 0:
        if early[-1]["cum_cost"] / ceiling >= cfg.inherent_difficulty_ratio:
            return INHERENT_DIFFICULTY

    # Where did the money go? Context (compactable) or generation (downshiftable)?
    share = input_cost_share(log)
    if share is not None and share >= INPUT_COST_SHARE_THRESHOLD:
        return CONTEXT_BLOWOUT
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


def backfill_true_mode(
    labels: dict[str, str],
    runs_dir: Path = RUNS_DIR,
    field: str = "relabeled_failure_mode",
) -> int:
    """Write the label into `field` on every run log sharing a labeled (task_id, seed).

    Defaults to the non-destructive field. `true_failure_mode` is the label a run was
    EXECUTED under (it is what the C-oracle arm acted on at runtime), so overwriting it
    after the fact would silently change the meaning of completed runs. Pass
    field="true_failure_mode" only when labeling ahead of a sweep.
    """
    n = 0
    for path in sorted(runs_dir.glob("*.json")):
        log = json.loads(path.read_text())
        key = f"{log['task_id']}__{log['seed']}"
        if key in labels and log.get(field) != labels[key]:
            log[field] = labels[key]
            path.write_text(json.dumps(log, indent=2))
            n += 1
    return n


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out", default=str(REPO / "config" / "labels.json"))
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument(
        "--field",
        default="relabeled_failure_mode",
        help="run-log field to backfill into. Use true_failure_mode only when "
             "labeling BEFORE a sweep (it is the label runs execute against).",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    labels = build_labels(runs_dir)
    Path(args.out).write_text(json.dumps(labels, indent=2))
    print(f"wrote {len(labels)} labels (v{LABEL_VERSION}) -> {args.out}")
    print("  distribution:", dict(Counter(labels.values())))

    if args.backfill:
        n = backfill_true_mode(labels, runs_dir, field=args.field)
        print(f"backfilled {args.field} into {n} run logs")


if __name__ == "__main__":
    _main()
