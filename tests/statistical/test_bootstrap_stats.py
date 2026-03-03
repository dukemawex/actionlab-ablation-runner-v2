import pandas as pd

from actionlab_ablation_runner.engine.stats import compute_significance


def test_statistical_outputs(tmp_path) -> None:
    rows = []
    for i in range(10):
        rows.append({"variant": "v1", "run_index": i, "seed": i, "accuracy": 0.8 + 0.001 * i, "f1": 0.78 + 0.001 * i})
        rows.append({"variant": "v0", "run_index": i, "seed": i, "accuracy": 0.7 + 0.001 * i, "f1": 0.68 + 0.001 * i})
    df = pd.DataFrame(rows)
    results = compute_significance(df, baseline_variant="v0", out_dir=tmp_path, seed=7)
    assert results
    assert all(r.metric in {"accuracy", "f1"} for r in results)
