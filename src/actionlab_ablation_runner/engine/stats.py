from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from actionlab_ablation_runner.schemas import StatTestResult


def _bootstrap_effect_ci(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a_sample = rng.choice(a, size=a.size, replace=True)
        b_sample = rng.choice(b, size=b.size, replace=True)
        diffs.append(float(a_sample.mean() - b_sample.mean()))
    arr = np.array(diffs)
    effect = float(np.mean(arr))
    ci_low, ci_high = np.quantile(arr, [0.025, 0.975])
    return effect, float(ci_low), float(ci_high)


def compute_significance(df: pd.DataFrame, baseline_variant: str, out_dir: Path, seed: int) -> list[StatTestResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = df[df["variant"] == baseline_variant]
    results: list[StatTestResult] = []
    for variant in sorted(df["variant"].unique()):
        if variant == baseline_variant:
            continue
        current = df[df["variant"] == variant]
        for metric in ("accuracy", "f1"):
            a = current[metric].to_numpy()
            b = baseline[metric].to_numpy()
            _, p_val = ttest_ind(a, b, equal_var=False)
            effect, low, high = _bootstrap_effect_ci(a, b, seed=seed + len(results))
            results.append(
                StatTestResult(
                    variant=variant,
                    compared_to=baseline_variant,
                    metric=metric,
                    p_value=float(p_val),
                    effect_size=effect,
                    ci_low=low,
                    ci_high=high,
                    significant=bool(p_val < 0.05),
                )
            )
    payload = [r.model_dump() for r in results]
    (out_dir / "stat_tests.json").write_text(json.dumps(payload, indent=2))
    return results
