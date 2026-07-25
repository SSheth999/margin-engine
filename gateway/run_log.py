"""Run-log builder + writer.

One structured JSON record per run, matching the schema in AGENTS.MD. This is the
single source of truth for offline analysis. Per-step trajectory data is appended on
EVERY call (even with no intervention) so post-hoc labeling and the C-oracle arm are
possible later.

In this pass the gateway is Arm A only, so `detected_failure_mode`, `true_failure_mode`,
and `interventions` are left null/empty — the fields exist so the schema is stable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    step: int
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost: float
    cum_cost: float
    context_size: int
    tool: str | None = None
    args_hash: str | None = None
    detected_failure_mode: str | None = None  # populated in shadow-mode pass


@dataclass
class InterventionRecord:
    step: int
    mode: str
    action: str
    cost_at_step: float


@dataclass
class RunLog:
    task_id: str
    arm: str
    seed: int
    complexity_class: str
    contracted_price: float
    margin_ceiling: float
    provider: str
    final_cost: float = 0.0
    margin_held: bool = True
    resolved: bool = False
    step_count: int = 0
    detected_failure_mode: str | None = None
    true_failure_mode: str | None = None
    error: str | None = None
    interventions: list[InterventionRecord] = field(default_factory=list)
    per_step: list[StepRecord] = field(default_factory=list)

    def add_step(self, step: StepRecord) -> None:
        self.per_step.append(step)
        self.step_count = len(self.per_step)
        self.final_cost = step.cum_cost
        self.margin_held = self.final_cost <= self.margin_ceiling

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.task_id}__{self.arm}__seed{self.seed}.json"
        path = out_dir / fname
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
