"""Outcome context carried on every model call.

The agent attaches the contracted outcome economics to each request as headers, so
the gateway can compute the margin ceiling and cost ratio. The agent itself does not
act on these — it just forwards them, identically across all arms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutcomeContext:
    outcome_id: str
    contracted_price: float
    target_margin: float

    def headers(self) -> dict[str, str]:
        return {
            "X-Outcome-Id": self.outcome_id,
            "X-Contracted-Price": str(self.contracted_price),
            "X-Target-Margin": str(self.target_margin),
        }
