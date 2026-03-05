from __future__ import annotations

from pathlib import Path

import yaml

from actionlab_ablation_runner.schemas import Config


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    return Config.model_validate(raw)
