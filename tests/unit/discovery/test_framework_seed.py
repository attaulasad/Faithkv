from kvcot.discovery.framework_seed import apply_framework_seed


def test_seeding_reports_python_and_torch_cpu_applied():
    policy = apply_framework_seed(13, "flash_attention_2", cuda_available=False)
    assert policy.framework_seed == 13
    assert policy.python_random_seeded is True
    assert policy.torch_cpu_seeded is True
    assert policy.torch_cuda_seeded is False
    assert policy.cudnn_deterministic_requested is False


def test_flash_attention_never_claims_bitwise_determinism():
    policy = apply_framework_seed(13, "flash_attention_2", cuda_available=False)
    assert policy.bitwise_determinism_guaranteed is False
    assert "not guaranteed bitwise-deterministic" in policy.tolerance_note


def test_sdpa_reports_no_known_nondeterministic_kernel():
    policy = apply_framework_seed(13, "sdpa", cuda_available=False)
    assert policy.bitwise_determinism_guaranteed is True


def test_seeding_is_reproducible():
    import random

    apply_framework_seed(13, "sdpa", cuda_available=False)
    a = random.random()
    apply_framework_seed(13, "sdpa", cuda_available=False)
    b = random.random()
    assert a == b
