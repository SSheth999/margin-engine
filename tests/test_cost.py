"""Rate-card cost math, for both provider usage shapes. No network."""

import math

from gateway.cost import RateCard
from gateway.schema import Usage


def test_ollama_no_cache():
    rc = RateCard.load("ollama")
    # qwen2.5:32b -> 3.00 input, 15.00 output per Mtok.
    cost = rc.cost("qwen2.5:32b", Usage(input_tokens=2000, output_tokens=500))
    expected = 2000 * 3.0 / 1e6 + 500 * 15.0 / 1e6  # 0.006 + 0.0075
    assert math.isclose(cost, expected, rel_tol=1e-9)


def test_ollama_cheap_sibling_is_cheaper():
    rc = RateCard.load("ollama")
    u = Usage(input_tokens=5000, output_tokens=1000)
    assert rc.cost("qwen2.5:7b", u) < rc.cost("qwen2.5:32b", u)


def test_anthropic_additive_cache_bucket():
    rc = RateCard.load("anthropic")
    # sonnet: 3.00 input, 15.00 output, cache_read_multiplier 0.10.
    u = Usage(input_tokens=1000, output_tokens=200, cached_tokens=500)
    expected = (
        1000 * 3.0 / 1e6          # uncached input
        + 500 * 3.0 * 0.10 / 1e6  # cached input at discount
        + 200 * 15.0 / 1e6        # output
    )
    assert math.isclose(rc.cost("claude-sonnet-5", u), expected, rel_tol=1e-9)


def test_cached_tokens_cheaper_than_uncached():
    rc = RateCard.load("anthropic")
    uncached = Usage(input_tokens=1000, output_tokens=0)
    cached = Usage(input_tokens=0, output_tokens=0, cached_tokens=1000)
    assert rc.cost("claude-sonnet-5", cached) < rc.cost("claude-sonnet-5", uncached)


def test_has_and_missing_model():
    rc = RateCard.load("ollama")
    assert rc.has("qwen2.5:32b")
    assert not rc.has("nonexistent-model")
