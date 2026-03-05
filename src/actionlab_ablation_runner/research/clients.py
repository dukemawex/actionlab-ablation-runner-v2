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
                json={
                    "api_key": self.api_key,
                    "query": topic,
                    "max_results": 5,
                    "search_depth": "advanced",
                },
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


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openrouter/openrouter:free",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model

    @with_backoff
    def generate_section(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result


def synthesize_research(
    sources: list[SourceItem],
    ablation_table_markdown: str,
    out_dir: Path,
    client: OpenRouterClient,
) -> GeminiSection:
    citation_lines = "\n".join(f"- {s.title}: {s.url}" for s in sources)
    prompt = (
        "Produce JSON with keys abstract, methods, interpretation, hypothesis_framing, "
        "statistical_testing, threats_to_validity, ethical_considerations, citations. "
        "Interpret this ablation table:\n"
        f"{ablation_table_markdown}\n"
        f"Use these verified sources and cite only listed URLs:\n{citation_lines}"
    )
    raw = client.generate_section(prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openrouter_raw.json").write_text(json.dumps(raw, indent=2))
    text = raw["choices"][0]["message"]["content"]
    section = GeminiSection.model_validate_json(text)
    return section
