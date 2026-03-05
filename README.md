# actionlab-ablation-runner

Production-grade deterministic ablation experimentation and research synthesis engine.

## Features
- Config-driven Cartesian ablation generation with stable hash variant IDs.
- Deterministic seeded runs with parallel-safe execution.
- Bootstrap + Welch t-test significance output.
- Tavily-backed source retrieval (minimum source gate).
- Gemini-backed paper section synthesis with strict JSON schema.
- CLI (`actionlab run|summarize|paper`) and FastAPI endpoint.
- CI with lint/test/pipeline/mkdocs/typst/security.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
actionlab run --config config.yaml
uvicorn actionlab_ablation_runner.api.app:app --reload
```

## Artifacts
Generated under `artifacts/`:
- `ablations.csv`
- `metrics.json`
- `stat_tests.json`
- `sources.json`
- `gemini_raw.json`
- `telemetry.json`
- `paper.typ`
