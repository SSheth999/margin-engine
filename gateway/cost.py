"""Rate-card cost computation.

Turns a normalized `Usage` into a dollar cost using the per-model rate card
(config/rate_card.yaml). Buckets are additive (see schema.Usage):

    cost = input_tokens  * input_rate
         + cached_tokens * input_rate * cache_read_multiplier
         + output_tokens * output_rate

Rates in the card are USD per 1,000,000 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from gateway.schema import Usage

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_MTOK = 1_000_000


@dataclass
class ModelRate:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float


class RateCard:
    """Loaded rate card, keyed by model name within the active provider block."""

    def __init__(self, rates: dict[str, ModelRate]):
        self._rates = rates

    @classmethod
    def load(cls, provider: str, config_dir: Path | None = None) -> "RateCard":
        cfg_dir = config_dir or CONFIG_DIR
        with open(cfg_dir / "rate_card.yaml") as f:
            data = yaml.safe_load(f)
        block = data.get(provider, {})
        rates = {
            model: ModelRate(
                input_per_mtok=float(r["input_per_mtok"]),
                output_per_mtok=float(r["output_per_mtok"]),
                cache_read_multiplier=float(r.get("cache_read_multiplier", 0.0)),
            )
            for model, r in block.items()
        }
        return cls(rates)

    def has(self, model: str) -> bool:
        return model in self._rates

    def cost(self, model: str, usage: Usage) -> float:
        """USD cost for one call. Raises KeyError if the model isn't in the card."""
        r = self._rates[model]
        input_cost = usage.input_tokens * r.input_per_mtok / _MTOK
        cached_cost = (
            usage.cached_tokens * r.input_per_mtok * r.cache_read_multiplier / _MTOK
        )
        output_cost = usage.output_tokens * r.output_per_mtok / _MTOK
        return input_cost + cached_cost + output_cost
