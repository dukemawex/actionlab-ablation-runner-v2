from __future__ import annotations

import os

import pytest

from actionlab_ablation_runner.research.clients import GeminiClient, TavilyClient


@pytest.mark.integration
def test_live_tavily_and_gemini() -> None:
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Live API keys are not configured")

    tavily = TavilyClient()
    sources = tavily.search_topic("peer reviewed ablation methodology machine learning")
    assert len(sources) >= 1

    gemini = GeminiClient()
    raw = gemini.generate_section(
        "Return JSON with keys abstract, methods, interpretation, hypothesis_framing, "
        "statistical_testing, threats_to_validity, ethical_considerations, citations"
    )
    assert "candidates" in raw
