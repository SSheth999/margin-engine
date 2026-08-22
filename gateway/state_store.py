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

    # Token mix, needed by cache-aware interventions: compaction invalidates the
    # provider's prompt cache, so how much of the prompt is currently being served
    # from cache determines whether compacting can pay for itself at all.
    input_tokens_total: int = 0
    cached_tokens_total: int = 0
    last_cached_tokens: int = 0
    last_input_tokens: int = 0

    # Per-call costs, so a laddered arm can stop BEFORE the call that would breach the
    # ceiling rather than one call after.
    step_costs: list[float] = field(default_factory=list)

    # Intervention bookkeeping for laddered arms (hysteresis: compact at most N times).
    compaction_count: int = 0
    last_compaction_step: int | None = None

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

    @property
    def cache_hit_share(self) -> float:
        """Fraction of the LAST call's prompt that was served from cache (0..1).

        High values mean the prompt prefix is cached and cheap; compacting would throw
        that away and re-bill the surviving suffix at the full input rate.
        """
        prompt = self.last_input_tokens + self.last_cached_tokens
        if prompt <= 0:
            return 0.0
        return self.last_cached_tokens / prompt

    def projected_cost_after_next_call(self, lookback: int = 3) -> float:
        """cum_cost plus an estimate of what the next call will cost.

        Estimated from the trailing mean of recent per-call costs. A budget check made
        only on money ALREADY spent always overshoots by one call: the run is waved
        through at ratio 0.99 and lands well past the ceiling. Projecting forward is
        what lets a hard stop actually hold the margin. Returns cum_cost unchanged when
        there is no history to extrapolate from.
        """
        if not self.step_costs:
            return self.cum_cost
        recent = self.step_costs[-lookback:]
        return self.cum_cost + sum(recent) / len(recent)

    def note_compaction(self) -> None:
        self.compaction_count += 1
        self.last_compaction_step = self.step_count


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
        input_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> OutcomeState:
        """Post-call accounting: advance step, add cost, append trajectory data."""
        st = self._states[outcome_id]
        st.step_count += 1
        st.cum_cost += call_cost
        st.step_costs.append(call_cost)
        st.context_sizes.append(context_size)
        if tool_name is not None:
            st.tool_history.append((tool_name, tool_args_hash or ""))
        st.input_tokens_total += input_tokens
        st.cached_tokens_total += cached_tokens
        st.last_input_tokens = input_tokens
        st.last_cached_tokens = cached_tokens
        return st
