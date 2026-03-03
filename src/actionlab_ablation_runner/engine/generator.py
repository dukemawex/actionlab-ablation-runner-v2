from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

from actionlab_ablation_runner.schemas import Variant


def _hash_params(params: dict[str, Any]) -> str:
    normalized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]


def generate_variants(ablations: dict[str, list[Any]]) -> list[Variant]:
    keys = sorted(ablations.keys())
    combos = itertools.product(*(ablations[key] for key in keys))
    variants: list[Variant] = []
    for combo in combos:
        params = {k: v for k, v in zip(keys, combo, strict=True)}
        hash_id = _hash_params(params)
        name = "var_" + "_".join(f"{k}-{v}" for k, v in params.items()) + f"_{hash_id}"
        variants.append(Variant(name=name, params=params, hash_id=hash_id))
    return variants
