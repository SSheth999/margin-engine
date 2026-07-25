"""Per-outcome state store.

The gateway runs one process per task run (see plan: gateway is nested inside each
sandbox), so a single instance only ever tracks one outcome. An in-memory object is
therefore correct and sufficient — no Redis needed. We still key by outcome_id so the
shape generalizes if a gateway is ever shared.

Tracks everything the detectors (added in a later pass) will need: cumulative cost,
step count, per-step context size history, and (tool_name, args_hash) history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class OutcomeState:
    outcome_id: str
    contracted_price: float
    target_margin: float
    cum_cost: float = 0.0
    step_count: int = 0
    context_sizes: list[int] = field(default_factory=list)
    tool_history: list[tuple[str, str]] = field(default_factory=list)  # (name, args_hash)

    @property
    def margin_ceiling(self) -> float:
        """price * (1 - target_margin). The cost budget for this outcome."""
        return self.contracted_price * (1.0 - self.target_margin)

    @property
    def cost_ratio(self) -> float:
        ceiling = self.margin_ceiling
        if ceiling <= 0:
            return float("inf")
        return self.cum_cost / ceiling


def args_hash(arguments: dict) -> str:
    """Stable short hash of tool arguments, for loop detection."""
    blob = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class StateStore:
    def __init__(self):
        self._states: dict[str, OutcomeState] = {}

    def get_or_create(
        self, outcome_id: str, contracted_price: float, target_margin: float
    ) -> OutcomeState:
        st = self._states.get(outcome_id)
        if st is None:
            st = OutcomeState(
                outcome_id=outcome_id,
                contracted_price=contracted_price,
                target_margin=target_margin,
            )
            self._states[outcome_id] = st
        return st

    def get(self, outcome_id: str) -> OutcomeState | None:
        return self._states.get(outcome_id)

    def record_call(
        self,
        outcome_id: str,
        call_cost: float,
        context_size: int,
        tool_name: str | None,
        tool_args_hash: str | None,
    ) -> OutcomeState:
        """Post-call accounting: advance step, add cost, append trajectory data."""
        st = self._states[outcome_id]
        st.step_count += 1
        st.cum_cost += call_cost
        st.context_sizes.append(context_size)
        if tool_name is not None:
            st.tool_history.append((tool_name, tool_args_hash or ""))
        return st
