from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from actionlab_ablation_runner.engine.config import load_config
from actionlab_ablation_runner.engine.executor import run_experiments
from actionlab_ablation_runner.engine.generator import generate_variants
from actionlab_ablation_runner.engine.stats import compute_significance
from actionlab_ablation_runner.paper.typst_writer import write_typst
from actionlab_ablation_runner.research.clients import GeminiClient, TavilyClient
from actionlab_ablation_runner.research.pipeline import generate_paper_sections
from actionlab_ablation_runner.schemas import TelemetryEvent
from actionlab_ablation_runner.utils.logging import get_logger

logger = get_logger(__name__)


def run_pipeline(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    out_dir = Path(config.experiment.output_dir)
    variants = generate_variants(config.ablations)
    logger.info("generated_variants", extra={"extra_payload": {"count": len(variants)}})

    metrics_frame = run_experiments(config.experiment, variants, out_dir)
    summary = metrics_frame.groupby("variant")[["accuracy", "f1"]].mean().sort_values("accuracy", ascending=False)
    (out_dir / "metrics.json").write_text(summary.to_json(indent=2))

    baseline_variant = summary.index[0]
    stats = compute_significance(metrics_frame, baseline_variant=baseline_variant, out_dir=out_dir, seed=config.experiment.base_seed)

    tavily = TavilyClient()
    gemini = GeminiClient()
    section = generate_paper_sections(config, metrics_frame, out_dir, tavily, gemini)

    write_typst(section, out_dir / "paper.typ")

    events = [
        TelemetryEvent(timestamp=datetime.now(timezone.utc), event="run_completed", payload={"variants": len(variants)}),
        TelemetryEvent(timestamp=datetime.now(timezone.utc), event="stat_tests_generated", payload={"count": len(stats)}),
    ]
    (out_dir / "telemetry.json").write_text(json.dumps([e.model_dump(mode="json") for e in events], indent=2))


def summarize(config_path: str = "config.yaml") -> str:
    config = load_config(config_path)
    out_dir = Path(config.experiment.output_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    best_variant = max(metrics["accuracy"], key=metrics["accuracy"].get)
    return f"Best variant: {best_variant} with accuracy={metrics['accuracy'][best_variant]:.4f}"
