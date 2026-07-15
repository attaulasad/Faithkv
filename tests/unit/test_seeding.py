from kvcot.utils.seeding import derive_seed


def test_deterministic_repeatable():
    a = derive_seed(42, "gsm8k", 7)
    b = derive_seed(42, "gsm8k", 7)
    assert a == b


def test_different_problem_index_gives_different_seed():
    a = derive_seed(42, "gsm8k", 7)
    b = derive_seed(42, "gsm8k", 8)
    assert a != b


def test_different_dataset_gives_different_seed():
    a = derive_seed(42, "gsm8k", 7)
    b = derive_seed(42, "math500", 7)
    assert a != b


def test_different_global_seed_gives_different_seed():
    a = derive_seed(42, "gsm8k", 7)
    b = derive_seed(13, "gsm8k", 7)
    assert a != b


def test_order_independent_not_a_simple_offset():
    # Guards against an accidental implementation like seed = global_seed + index,
    # which would make consecutive problems' seeds trivially correlated.
    seeds = [derive_seed(42, "gsm8k", i) for i in range(20)]
    diffs = [seeds[i + 1] - seeds[i] for i in range(len(seeds) - 1)]
    assert len(set(diffs)) > 1  # not a constant stride


def test_seed_is_nonnegative_and_fits_63_bits():
    for i in range(50):
        s = derive_seed(2026, "gsm8k", i)
        assert 0 <= s < (1 << 63)


def test_full_kv_and_rkv_share_seed_derivation():
    # §4: "FullKV and R-KV receive the identical derived seed" — this module
    # takes no condition/method parameter, so it is structurally impossible
    # for the two conditions to diverge. This test documents that guarantee.
    import inspect

    sig = inspect.signature(derive_seed)
    assert "condition" not in sig.parameters
    assert "method" not in sig.parameters
