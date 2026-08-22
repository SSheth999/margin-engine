"""Heuristic failure-mode detection.

Four deterministic rules over the per-outcome trajectory (see AGENTS.MD table). The
detector is intentionally simple — the tested novelty is matching an intervention to
the detected mode, not the detector's sophistication.

`detect()` returns a DetectionResult with the diagnosed mode (or None) and, for a tool
loop, the offending tool name so the matched intervention can restrict exactly it.

Detection runs on the state as it stands BEFORE the upcoming call. It is also run in
"shadow" on every arm (logged, not necessarily acted on) so the run logs support
post-hoc detector-quality analysis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from gateway.state_store import OutcomeState

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Canonical failure-mode names, shared across detectors, interventions, and analysis.
TOOL_LOOP = "tool_loop"
CONTEXT_BLOWOUT = "context_blowout"
WRONG_MODEL = "wrong_model"
INHERENT_DIFFICULTY = "inherent_difficulty"


@dataclass
class DetectionConfig:
    trigger_cost_ratio: float = 0.60
    tool_loop_repeats: int = 3
    context_growth_ratio: float = 1.50
    context_growth_lookback: int = 3
    context_flat_ratio: float = 1.15
    inherent_difficulty_ratio: float = 0.90
    early_steps: int = 4
    compact_keep_recent_turns: int = 3

    # Arm D ladder (cost-ratio thresholds + cache-aware compaction guards).
    d_downshift_ratio: float = 0.60
    d_compact_ratio: float = 0.85
    d_stop_ratio: float = 1.00
    d_stop_lookahead: bool = True
    d_compact_keep_recent_turns: int = 2
    d_compact_max_per_run: int = 1
    d_compact_min_drop_fraction: float = 0.60
    d_compact_max_cache_share: float = 0.85

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "DetectionConfig":
        cfg_dir = config_dir or CONFIG_DIR
        with open(cfg_dir / "detection.yaml") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: data[k] for k in data if k in cls.__annotations__})


@dataclass
class DetectionResult:
    mode: str | None
    tool_name: str | None = None  # set for tool_loop -> which tool to restrict


def _max_tool_repeat(state: OutcomeState) -> tuple[str | None, int]:
    if not state.tool_history:
        return None, 0
    counts = Counter(state.tool_history)  # keys: (name, args_hash)
    (name, _), n = counts.most_common(1)[0]
    return name, n


def _context_growth(state: OutcomeState, lookback: int) -> float:
    """Ratio of latest context size to the size `lookback` steps earlier (>=1.0)."""
    sizes = state.context_sizes
    if len(sizes) < 2:
        return 1.0
    latest = sizes[-1]
    ref_idx = max(0, len(sizes) - 1 - lookback)
    ref = sizes[ref_idx]
    if ref <= 0:
        return 1.0
    return latest / ref


def mode_triggered(mode: str, state: OutcomeState, cfg: DetectionConfig) -> bool:
    """Would the trigger condition for a SPECIFIC mode fire on the current state?

    Used by the C-oracle arm: it knows the true mode and must act at the same point
    the detector would for that mode — so C and C-oracle differ ONLY in which mode
    is used, never in timing.
    """
    if mode is None:
        return False
    _, loop_n = _max_tool_repeat(state)
    growth = _context_growth(state, cfg.context_growth_lookback)
    ratio = state.cost_ratio

    if mode == TOOL_LOOP:
        return loop_n >= cfg.tool_loop_repeats
    if ratio < cfg.trigger_cost_ratio:
        return False
    if mode == CONTEXT_BLOWOUT:
        return growth >= cfg.context_growth_ratio
    if mode == INHERENT_DIFFICULTY:
        return state.step_count <= cfg.early_steps and ratio >= cfg.inherent_difficulty_ratio
    if mode == WRONG_MODEL:
        return growth <= cfg.context_flat_ratio
    return False


def detect(state: OutcomeState, cfg: DetectionConfig) -> DetectionResult:
    """Diagnose the failure mode from the trajectory. None if no signal (yet)."""
    loop_tool, loop_n = _max_tool_repeat(state)
    growth = _context_growth(state, cfg.context_growth_lookback)
    ratio = state.cost_ratio

    # 1) Tool loop — clearest waste signal; check first regardless of cost ratio
    #    once repeats pile up (a tight loop can blow budget fast).
    if loop_n >= cfg.tool_loop_repeats:
        return DetectionResult(TOOL_LOOP, tool_name=loop_tool)

    # Everything below is gated on the run actually trending over budget.
    if ratio < cfg.trigger_cost_ratio:
        return DetectionResult(None)

    # 2) Context blowout — context growing fast while cost climbs.
    if growth >= cfg.context_growth_ratio:
        return DetectionResult(CONTEXT_BLOWOUT)

    # 3) Inherent difficulty — expensive from very early, no waste signal.
    if state.step_count <= cfg.early_steps and ratio >= cfg.inherent_difficulty_ratio:
        return DetectionResult(INHERENT_DIFFICULTY)

    # 4) Wrong model — high cost, context flat, no loop. Cheaper model likely suffices.
    if growth <= cfg.context_flat_ratio:
        return DetectionResult(WRONG_MODEL)

    return DetectionResult(None)
