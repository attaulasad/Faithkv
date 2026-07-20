import copy

import pytest
from pydantic import ValidationError

from kvcot.discovery.discovery_config import (
    PINNED_RKV_UPSTREAM_REVISION,
    DiscoveryConfig,
    load_discovery_config,
    validate_full_revision,
)

VALID_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"


def _base_kwargs() -> dict:
    return dict(
        model=dict(
            name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            revision=VALID_REVISION,
            tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            tokenizer_revision=VALID_REVISION,
            model_type="llama",
            dtype="bfloat16",
        ),
        dataset=dict(name="MATH-500", revision=None),
        rkv=dict(budget=1024, upstream_revision=PINNED_RKV_UPSTREAM_REVISION),
    )


def test_valid_config_constructs():
    config = DiscoveryConfig(**_base_kwargs())
    assert config.model.model_type == "llama"
    assert config.dataset.revision_is_frozen is False


def test_real_config_file_loads_and_validates():
    config = load_discovery_config("configs/discovery/llama8b_math500_b1024.yaml")
    assert config.model.name == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert config.model.revision == VALID_REVISION
    assert config.rkv.budget == 1024
    # B1B-R2 §8: the MATH-500 dataset revision is now independently
    # verified and frozen -- no longer the pre-B1B-R2 unresolved gap.
    assert config.dataset.revision_is_frozen is True
    assert config.dataset.revision == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    assert config.dataset.config == "default"
    assert config.dataset.split == "test"
    assert config.generation.generation_mode == "greedy"
    assert config.generation.do_sample is False
    assert config.generation.max_new_tokens == 6144
    assert config.generation.batch_size == 1


@pytest.mark.parametrize("field_path", [("model", "revision"), ("model", "tokenizer_revision")])
def test_model_revision_rejects_main(field_path):
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs[field_path[0]][field_path[1]] = "main"
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


@pytest.mark.parametrize("field_path", [("model", "revision"), ("model", "tokenizer_revision")])
def test_model_revision_rejects_latest(field_path):
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs[field_path[0]][field_path[1]] = "latest"
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


@pytest.mark.parametrize("field_path", [("model", "revision"), ("model", "tokenizer_revision")])
def test_model_revision_rejects_null(field_path):
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs[field_path[0]][field_path[1]] = None
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


@pytest.mark.parametrize("field_path", [("model", "revision"), ("model", "tokenizer_revision")])
def test_model_revision_rejects_short_hash(field_path):
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs[field_path[0]][field_path[1]] = VALID_REVISION[:10]
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_rkv_upstream_revision_rejects_main():
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["rkv"]["upstream_revision"] = "main"
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_rkv_upstream_revision_rejects_short_hash():
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["rkv"]["upstream_revision"] = PINNED_RKV_UPSTREAM_REVISION[:10]
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_rkv_upstream_revision_must_match_pinned_submodule():
    kwargs = copy.deepcopy(_base_kwargs())
    # 40 valid hex chars, but NOT the actual pinned R-KV commit -- must
    # still be rejected (never allowed to silently drift).
    kwargs["rkv"]["upstream_revision"] = "a" * 40
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_model_type_must_have_verified_dispatch_entry():
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["model"]["model_type"] = "falcon"
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_dataset_revision_defaults_to_none_and_is_reported_unfrozen():
    kwargs = copy.deepcopy(_base_kwargs())
    config = DiscoveryConfig(**kwargs)
    assert config.dataset.revision is None
    assert config.dataset.revision_is_frozen is False


def test_dataset_name_must_be_math500_literal():
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["dataset"]["name"] = "gsm8k"
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_rkv_budget_must_be_positive():
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["rkv"]["budget"] = 0
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_validate_full_revision_helper_directly():
    assert validate_full_revision(VALID_REVISION, "x") == VALID_REVISION
    for bad in ("main", "latest", None, VALID_REVISION[:10], "", "G" * 40):
        with pytest.raises(ValueError):
            validate_full_revision(bad, "x")


@pytest.mark.parametrize("bad_revision", ["main", "latest", "G" * 40, "short"])
def test_dataset_revision_rejects_non_full_hash_when_present(bad_revision):
    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["dataset"]["revision"] = bad_revision
    with pytest.raises(ValidationError):
        DiscoveryConfig(**kwargs)


def test_generation_lock_defaults_are_frozen_and_deterministic():
    from kvcot.discovery.discovery_config import DiscoveryGenerationLock

    generation = DiscoveryGenerationLock()
    assert generation.generation_mode == "greedy"
    assert generation.do_sample is False
    assert generation.temperature is None
    assert generation.top_p is None
    assert generation.batch_size == 1
    assert generation.max_new_tokens == 6144
    assert generation.framework_seed == 13
    assert generation.no_offload_required is True


def test_generation_config_hash_is_stable_and_sensitive_to_changes():
    from kvcot.discovery.discovery_config import DiscoveryGenerationLock, generation_config_hash

    a = generation_config_hash(DiscoveryGenerationLock())
    b = generation_config_hash(DiscoveryGenerationLock())
    assert a == b
    changed = generation_config_hash(DiscoveryGenerationLock(max_new_tokens=100))
    assert changed != a


def test_rkv_config_hash_is_stable_and_sensitive_to_changes():
    from kvcot.discovery.discovery_config import rkv_config_hash

    kwargs = _base_kwargs()
    config_a = DiscoveryConfig(**kwargs)
    config_b = DiscoveryConfig(**copy.deepcopy(kwargs))
    assert rkv_config_hash(config_a.rkv) == rkv_config_hash(config_b.rkv)

    kwargs2 = copy.deepcopy(kwargs)
    kwargs2["rkv"]["budget"] = 2048
    config_c = DiscoveryConfig(**kwargs2)
    assert rkv_config_hash(config_c.rkv) != rkv_config_hash(config_a.rkv)


def test_prompt_template_hash_is_stable_and_pure_python():
    from kvcot.discovery.discovery_config import prompt_template_hash

    assert prompt_template_hash() == prompt_template_hash()
    assert len(prompt_template_hash()) == 64


def test_canonical_config_hash_changes_when_any_input_changes():
    from kvcot.discovery.discovery_config import canonical_config_hash

    config = load_discovery_config("configs/discovery/llama8b_math500_b1024.yaml")
    base_hash = canonical_config_hash(config)
    assert canonical_config_hash(config) == base_hash  # stable across repeated calls

    kwargs = copy.deepcopy(_base_kwargs())
    kwargs["dataset"]["revision"] = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    kwargs["rkv"]["budget"] = 2048  # actually differs from the real config's budget=1024
    changed_config = DiscoveryConfig(**kwargs)
    assert canonical_config_hash(changed_config) != base_hash
