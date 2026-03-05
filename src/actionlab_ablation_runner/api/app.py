from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

app = FastAPI(title="actionlab-ablation-runner")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/results/latest")
def latest_results() -> dict[str, Any]:
    base = Path("artifacts")
    metrics_file = base / "metrics.json"
    stats_file = base / "stat_tests.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="No experiments have been run yet.")
    metrics = json.loads(metrics_file.read_text())
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else []
    return {"metrics": metrics, "stat_tests": stats}
