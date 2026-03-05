from actionlab_ablation_runner.engine.generator import generate_variants


def test_variant_generation_deterministic() -> None:
    ablations = {"optimizer": ["adam", "sgd"], "lr": [0.001, 0.01], "dropout": [0.1, 0.3]}
    a = generate_variants(ablations)
    b = generate_variants(ablations)
    assert [v.model_dump() for v in a] == [v.model_dump() for v in b]
    assert len(a) == 8
