import glob

import pytest
from pydantic import ValidationError

from kvcot.config import (
    load_lock_config,
    load_stage_config,
    config_identity,
    FixedTraceSettings,
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


# --- FixedTraceSettings (§ Step 2/15) ---

def test_fixed_trace_settings_defaults():
    settings = FixedTraceSettings()
    assert settings.probe_max_new_tokens == 64
    assert settings.require_boxed_extraction is True
    assert settings.min_eligible_examples == 5
    assert settings.min_actual_compression_rate == 0.7
    assert settings.max_mean_f1_retention_ratio == 0.7


def test_fixed_trace_settings_rejects_zero_or_negative_probe_limit():
    with pytest.raises(ValidationError):
        FixedTraceSettings(probe_max_new_tokens=0)
    with pytest.raises(ValidationError):
        FixedTraceSettings(probe_max_new_tokens=-1)


def test_fixed_trace_settings_rejects_out_of_range_rates():
    with pytest.raises(ValidationError):
        FixedTraceSettings(min_actual_compression_rate=1.5)
    with pytest.raises(ValidationError):
        FixedTraceSettings(min_actual_compression_rate=-0.1)
    with pytest.raises(ValidationError):
        FixedTraceSettings(max_mean_f1_retention_ratio=0.0)
    with pytest.raises(ValidationError):
        FixedTraceSettings(max_mean_f1_retention_ratio=1.1)


def test_fixed_trace_settings_rejects_zero_min_eligible_examples():
    with pytest.raises(ValidationError):
        FixedTraceSettings(min_eligible_examples=0)


def test_early_gap_configs_load_fixed_trace_settings():
    stage, _lock = load_stage_config("configs/early_gap_b512.yaml")
    assert stage.fixed_trace is not None
    assert stage.fixed_trace.probe_max_new_tokens == 64


def test_stage_config_fixed_trace_defaults_to_none():
    stage, _lock = load_stage_config("configs/stage0_smoke.yaml")
    assert stage.fixed_trace is None


def test_primary_lock_probes_max_new_tokens_is_unchanged():
    # Frozen §4 value — the fixed-trace screen must never touch this;
    # its own decoding budget lives in FixedTraceSettings.probe_max_new_tokens.
    lock = load_lock_config(LOCK_PATH)
    assert lock.probes.max_new_tokens == 48


def test_early_gap_v2_b128_config_has_own_stage_identity():
    # External review 2026-07-16: b512/b1024 never exceed the budget on the
    # real GPU data collected (logs/b512_accuracy_compaction.log), and b256
    # is exceeded on at most ~6/10 traces -- structurally below
    # min_actual_compression_rate. b128 is a fresh stage_name/output_dir
    # (never a resumption of an early_gap_b*.yaml directory, which may hold
    # protocol-v1 data) chosen because every observed trace in that sample
    # exceeds it.
    stage, _lock = load_stage_config("configs/early_gap_v2_b128.yaml")
    assert stage.stage_name == "early_gap_v2_b128"
    assert stage.output_dir == "results/raw/protocol_v2/early_gap_b128"
    assert stage.rkv_budgets == [128]
    assert stage.conditions == ["full", "rkv_b128"]
    assert stage.fixed_trace is not None
    assert stage.fixed_trace.min_actual_compression_rate == 0.7
    assert stage.fixed_trace.max_mean_f1_retention_ratio == 0.7
