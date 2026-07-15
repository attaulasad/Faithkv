import glob

import pytest
from pydantic import ValidationError

from kvcot.config import (
    load_lock_config,
    load_stage_config,
    config_identity,
    PROBE_FRACTIONS_ALL,
    PROBE_FRACTIONS_SCORED,
    StageConfig,
)

LOCK_PATH = "configs/lock.yaml"


def test_lock_config_loads():
    lock = load_lock_config(LOCK_PATH)
    assert lock.model.name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert lock.seeds == [13, 42, 2026]
    assert lock.generation.base_max_new_tokens == 6144
    assert lock.rkv.window_size == 8
    assert lock.rkv.mix_lambda == 0.1
    assert lock.rkv.retain_ratio == 0.2
    assert lock.rkv.retain_direction == "last"
    assert lock.probes.max_new_tokens == 48


def test_probe_fractions_frozen_values():
    assert PROBE_FRACTIONS_ALL == (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    assert PROBE_FRACTIONS_SCORED == (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    assert 0.0 not in PROBE_FRACTIONS_SCORED
    assert 1.0 not in PROBE_FRACTIONS_SCORED
    assert len(PROBE_FRACTIONS_SCORED) == 7


def test_wrong_seeds_rejected():
    import yaml

    with open(LOCK_PATH) as f:
        raw = yaml.safe_load(f)
    raw["seeds"] = [1, 2, 3]
    from kvcot.config import FrozenSettings

    with pytest.raises(ValidationError):
        FrozenSettings.model_validate(raw)


def test_wrong_probe_fractions_rejected():
    import yaml

    with open(LOCK_PATH) as f:
        raw = yaml.safe_load(f)
    raw["probes"]["fractions_scored"] = [0.1, 0.2, 0.3]
    from kvcot.config import FrozenSettings

    with pytest.raises(ValidationError):
        FrozenSettings.model_validate(raw)


@pytest.mark.parametrize(
    "path",
    sorted(glob.glob("configs/stage*.yaml")),
)
def test_all_stage_configs_load(path):
    stage, lock = load_stage_config(path)
    assert stage.stage_name
    assert lock.seeds == [13, 42, 2026]


def test_stage0_uses_single_seed():
    stage, lock = load_stage_config("configs/stage0_smoke.yaml")
    assert stage.resolve_seeds(lock) == [42]


def test_stage2_uses_all_three_seeds():
    stage, lock = load_stage_config("configs/stage2_main.yaml")
    assert stage.resolve_seeds(lock) == [13, 42, 2026]


def test_stage_config_rejects_percent_condition_name():
    with pytest.raises(ValidationError):
        StageConfig(
            stage_name="bad",
            dataset_manifest="x.jsonl",
            conditions=["rkv_10%"],
            output_dir="out",
        )


def test_config_identity_is_stable_sha256():
    h1 = config_identity(LOCK_PATH)
    h2 = config_identity(LOCK_PATH)
    assert h1 == h2
    assert len(h1) == 64


def test_seeds_override_must_be_subset_of_lock_seeds():
    stage, lock = load_stage_config("configs/stage0_smoke.yaml")
    stage.seeds_override = [999]
    with pytest.raises(ValueError):
        stage.resolve_seeds(lock)
