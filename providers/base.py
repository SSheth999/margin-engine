"""Provider interface + loader.

A provider adapter is the only code that knows a backend's wire format. It converts
a canonical `ChatRequest` into that backend's request, calls it, and converts the
reply back into a canonical `ChatResponse` (with normalized `Usage`).

`load_provider()` reads config/providers.yaml, picks the active backend, and returns
an instantiated adapter plus its resolved model pair. This is the single switch that
selects Ollama (dev) vs Anthropic (hosted).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gateway.schema import ChatRequest, ChatResponse

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class ModelPair:
    """The primary model an arm runs on, and the cheap sibling used for downshift."""

    primary: str
    cheap: str


class Provider(abc.ABC):
    """Backend adapter. Stateless w.r.t. a single call."""

    name: str

    @abc.abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute one model call and return a normalized response.

        Implementations must populate `response.usage` with real token counts from
        the backend (never estimates), and normalize `stop_reason`.
        """
        raise NotImplementedError


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def load_provider(
    config_dir: Path | None = None,
) -> tuple[Provider, ModelPair]:
    """Instantiate the active provider from config/providers.yaml.

    Returns (provider, model_pair). Import of concrete adapters is deferred so that,
    e.g., a machine without the `anthropic` key set can still run the Ollama path.
    """
    cfg_dir = config_dir or CONFIG_DIR
    with open(cfg_dir / "providers.yaml") as f:
        cfg = yaml.safe_load(f)

    active = cfg["active_provider"]
    pconf = cfg["providers"][active]
    pair = ModelPair(
        primary=pconf["model_pair"]["primary"],
        cheap=pconf["model_pair"]["cheap"],
    )

    if active == "ollama":
        from providers.ollama_provider import OllamaProvider

        provider: Provider = OllamaProvider(base_url=pconf["base_url"])
    elif active == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
    elif active == "mock":
        from providers.mock_provider import MockProvider

        provider = MockProvider()
    else:
        raise ValueError(f"unknown active_provider: {active!r}")

    return provider, pair
