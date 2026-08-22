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
    # Wall-clock for the provider call plus the tool exec that followed it. Model spend
    # is only half the bill: sandboxes are billed by the second, so a run's DURATION is
    # what prices the infrastructure. Without it, sandbox cost is unknowable after the
    # fact — which is exactly the position the first full benchmark left us in.
    latency_sec: float | None = None


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

    # --- timing (all wall-clock seconds; None when not measured) ---
    # agent_loop_sec is measured by the gateway (first model call -> finalize).
    # The sandbox fields are patched in by the orchestrator, which owns the sandbox
    # lifecycle and therefore knows the billable window. See `attach_timing`.
    agent_loop_sec: float | None = None
    sandbox_sec: float | None = None       # create -> close: the billable window
    sandbox_create_sec: float | None = None
    verify_sec: float | None = None

    interventions: list[InterventionRecord] = field(default_factory=list)
    per_step: list[StepRecord] = field(default_factory=list)

    def add_step(self, step: StepRecord) -> None:
        self.per_step.append(step)
        self.step_count = len(self.per_step)
        self.final_cost = step.cum_cost
        self.margin_held = self.final_cost <= self.margin_ceiling

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def attach_timing(path: Path, **fields: float | None) -> None:
        """Merge timing fields into an already-written run log.

        The gateway writes the log on /v1/finalize, but only the orchestrator knows how
        long the sandbox was alive. Rather than thread that back through the agent's
        finalize call, the orchestrator patches the file afterwards. Best-effort: a
        failure here must never sink a run whose real work already succeeded.
        """
        try:
            log = json.loads(path.read_text())
            log.update({k: v for k, v in fields.items() if v is not None})
            path.write_text(json.dumps(log, indent=2))
        except Exception:
            pass

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.task_id}__{self.arm}__seed{self.seed}.json"
        path = out_dir / fname
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
