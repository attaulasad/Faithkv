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
    assert config.dataset.revision_is_frozen is False


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
