from __future__ import annotations

import os

import pytest

from actionlab_ablation_runner.research.clients import OpenRouterClient, TavilyClient


@pytest.mark.integration
def test_live_tavily_and_openrouter() -> None:
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("Live API keys are not configured")

    tavily = TavilyClient()
    sources = tavily.search_topic("peer reviewed ablation methodology machine learning")
    assert len(sources) >= 1

    openrouter = OpenRouterClient()
    raw = openrouter.generate_section(
        "Return JSON with keys abstract, methods, interpretation, hypothesis_framing, "
        "statistical_testing, threats_to_validity, ethical_considerations, citations"
    )
    assert "choices" in raw
