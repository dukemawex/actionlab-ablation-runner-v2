from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from actionlab_ablation_runner.research.clients import (
    GeminiClient,
    TavilyClient,
    synthesize_research,
)
from actionlab_ablation_runner.schemas import Config, GeminiSection, SourceItem


def collect_sources(config: Config, out_dir: Path, client: TavilyClient) -> list[SourceItem]:
    sources: list[SourceItem] = []
    for topic in config.research.query_topics:
        sources.extend(client.search_topic(topic))
    unique: dict[str, SourceItem] = {str(item.url): item for item in sources}
    resolved = list(unique.values())
    if len(resolved) < config.research.min_sources:
        raise RuntimeError(
            f"Insufficient high-quality sources: "
            f"found {len(resolved)}, required {config.research.min_sources}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sources.json").write_text(
        json.dumps([s.model_dump() for s in resolved], indent=2, default=str)
    )
    return resolved


def generate_paper_sections(
    config: Config,
    ablations_frame: pd.DataFrame,
    out_dir: Path,
    tavily: TavilyClient,
    gemini: GeminiClient,
) -> GeminiSection:
    sources = collect_sources(config, out_dir, tavily)
    grouped = (
        ablations_frame.groupby("variant")[["accuracy", "f1"]]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    section = synthesize_research(sources, grouped.to_markdown(), out_dir, gemini)
    return section
