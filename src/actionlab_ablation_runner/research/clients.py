from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from actionlab_ablation_runner.schemas import GeminiSection, SourceItem
from actionlab_ablation_runner.utils.retry import with_backoff


class TavilyClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")

    @with_backoff
    def search_topic(self, topic: str) -> list[SourceItem]:
        if not self.api_key:
            raise RuntimeError("Missing TAVILY_API_KEY")
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": topic, "max_results": 5, "search_depth": "advanced"},
            )
            response.raise_for_status()
            data = response.json()
        items: list[SourceItem] = []
        for result in data.get("results", []):
            items.append(
                SourceItem(
                    title=result.get("title", "untitled"),
                    url=result["url"],
                    snippet=result.get("content", ""),
                    topic=topic,
                )
            )
        return items


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model

    @with_backoff
    def generate_section(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(endpoint, json=body)
            response.raise_for_status()
            return response.json()


def synthesize_research(
    sources: list[SourceItem],
    ablation_table_markdown: str,
    out_dir: Path,
    client: GeminiClient,
) -> GeminiSection:
    citation_lines = "\n".join(f"- {s.title}: {s.url}" for s in sources)
    prompt = (
        "Produce JSON with keys abstract, methods, interpretation, hypothesis_framing, statistical_testing, "
        "threats_to_validity, ethical_considerations, citations. Interpret this ablation table:\n"
        f"{ablation_table_markdown}\nUse these verified sources and cite only listed URLs:\n{citation_lines}"
    )
    raw = client.generate_section(prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gemini_raw.json").write_text(json.dumps(raw, indent=2))
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    section = GeminiSection.model_validate_json(text)
    return section
