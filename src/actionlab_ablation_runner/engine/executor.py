from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from actionlab_ablation_runner.schemas import ExperimentConfig, RunMetric, Variant
from actionlab_ablation_runner.utils.determinism import seed_everything


def _variant_seed(base_seed: int, variant_hash: str, run_index: int) -> int:
    return base_seed + int(variant_hash[:8], 16) + run_index


def _single_run(config: ExperimentConfig, variant: Variant, run_index: int) -> RunMetric:
    seed = _variant_seed(config.base_seed, variant.hash_id, run_index)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    lr = float(variant.params.get("lr", 0.001))
    dropout = float(variant.params.get("dropout", 0.1))
    optimizer = str(variant.params.get("optimizer", "adam"))
    opt_bonus = 0.01 if optimizer == "adam" else -0.005
    accuracy = float(
        0.75 + opt_bonus + (0.01 - lr) + (0.25 - dropout) * 0.03 + rng.normal(0, 0.005)
    )
    f1 = float(accuracy - 0.02 + rng.normal(0, 0.004))
    return RunMetric(variant=variant.name, run_index=run_index, seed=seed, accuracy=accuracy, f1=f1)


def run_experiments(
    config: ExperimentConfig, variants: list[Variant], out_dir: Path
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variant, run_idx)
        for variant in variants
        for run_idx in range(config.runs_per_variant)
    ]
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        rows = list(executor.map(lambda args: _single_run(config, args[0], args[1]), jobs))
    frame = pd.DataFrame([row.model_dump() for row in rows])
    frame.to_csv(out_dir / "ablations.csv", index=False)
    return frame
