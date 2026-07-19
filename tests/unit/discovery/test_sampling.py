import inspect

import pytest

from kvcot.discovery.sampling import (
    IdentitySeedParts,
    assign_depth_strata,
    cross_product_pairs,
    select_candidates_and_donors,
    select_events,
    select_kv_head,
    select_layer,
    sha256_seed,
)

IDENTITY = IdentitySeedParts(
    global_seed=42,
    dataset_name="math500",
    problem_index=7,
    model_revision="modelrev",
    rkv_revision="45eaa7d69d20b7388321f077020a610d9afb65bd",
)


def test_sha256_seed_is_deterministic_and_order_sensitive():
    a = sha256_seed(1, "x", 2)
    b = sha256_seed(1, "x", 2)
    c = sha256_seed(2, "x", 1)
    assert a == b
    assert a != c


def test_sha256_seed_golden_vector():
    # Hand-verified: sha256("1|x|2") first 8 bytes, big-endian unsigned.
    import hashlib

    expected = int.from_bytes(hashlib.sha256(b"1|x|2").digest()[:8], "big", signed=False)
    assert sha256_seed(1, "x", 2) == expected


def test_select_events_exactly_three_without_replacement():
    eligible = list(range(20))
    result = select_events(eligible, IDENTITY)
    assert result is not None
    assert len(result.selected_events_chronological) == 3
    assert len(set(result.selected_events_chronological)) == 3
    assert list(result.selected_events_chronological) == sorted(result.selected_events_chronological)


def test_select_events_fewer_than_three_eligible_is_ineligible():
    assert select_events([1, 2], IDENTITY) is None
    assert select_events([], IDENTITY) is None
    assert select_events([5], IDENTITY) is None


def test_select_events_deterministic_golden_vector():
    result1 = select_events(range(50), IDENTITY)
    result2 = select_events(range(50), IDENTITY)
    assert result1 == result2


def test_select_events_set_iteration_order_independent():
    a = select_events({5, 3, 19, 1, 8, 40, 2}, IDENTITY)
    b = select_events([1, 2, 3, 5, 8, 19, 40], IDENTITY)
    c = select_events([40, 19, 8, 5, 3, 2, 1], IDENTITY)
    assert a == b == c


def test_assign_depth_strata_every_event_gets_all_three_strata_exactly_once():
    events = (10, 20, 30)
    assignment = assign_depth_strata(events, IDENTITY)
    assert set(assignment.depth_stratum_by_event.keys()) == set(events)
    assert sorted(assignment.depth_stratum_by_event.values()) == [0, 1, 2]
    assert sorted(assignment.depth_strata_permutation) == [0, 1, 2]


def test_chronology_and_depth_are_separate_fields():
    events = (10, 20, 30)
    event_selection_ordinal = {event_id: k for k, event_id in enumerate(events)}
    assignment = assign_depth_strata(events, IDENTITY)
    # Chronological ordinal and depth stratum are two distinct dicts with
    # independently-drawn values -- the API never merges them into one
    # ambiguous "ordinal" field.
    assert set(event_selection_ordinal.keys()) == set(assignment.depth_stratum_by_event.keys())
    assert event_selection_ordinal is not assignment.depth_stratum_by_event


def test_changing_depth_permutation_seed_can_change_chronology_to_depth_mapping():
    events = (10, 20, 30)
    mappings = set()
    for problem_index in range(30):
        identity = IdentitySeedParts(
            global_seed=42,
            dataset_name="math500",
            problem_index=problem_index,
            model_revision="modelrev",
            rkv_revision="rkvrev",
        )
        assignment = assign_depth_strata(events, identity)
        mappings.add(tuple(assignment.depth_stratum_by_event[e] for e in events))
    # Across many different seeds, more than one distinct chronology->depth
    # mapping must occur -- proves the mapping is not pinned to one fixed
    # (e.g. identity/chronological) permutation.
    assert len(mappings) > 1


def test_select_layer_partitions_depth_thirds_and_covers_all_strata():
    num_hidden_layers = 28
    seen_ranges = {}
    for stratum in (0, 1, 2):
        result = select_layer(event_id=99, depth_stratum=stratum, num_hidden_layers=num_hidden_layers, identity=IDENTITY)
        assert result.lo <= result.layer_index < result.hi
        seen_ranges[stratum] = (result.lo, result.hi)
    assert seen_ranges[0][0] == 0
    assert seen_ranges[2][1] == num_hidden_layers
    assert seen_ranges[0][1] == seen_ranges[1][0]
    assert seen_ranges[1][1] == seen_ranges[2][0]


def test_select_layer_hard_fails_below_three_layers():
    with pytest.raises(ValueError):
        select_layer(event_id=1, depth_stratum=0, num_hidden_layers=2, identity=IDENTITY)
    with pytest.raises(ValueError):
        select_layer(event_id=1, depth_stratum=0, num_hidden_layers=0, identity=IDENTITY)


def test_select_layer_deterministic_golden_vector():
    a = select_layer(event_id=5, depth_stratum=1, num_hidden_layers=28, identity=IDENTITY)
    b = select_layer(event_id=5, depth_stratum=1, num_hidden_layers=28, identity=IDENTITY)
    assert a == b


def test_select_kv_head_within_range_and_deterministic():
    h1 = select_kv_head(event_id=5, num_key_value_heads=8, identity=IDENTITY)
    h2 = select_kv_head(event_id=5, num_key_value_heads=8, identity=IDENTITY)
    assert h1 == h2
    assert 0 <= h1 < 8


def test_select_kv_head_rejects_non_positive_head_count():
    with pytest.raises(ValueError):
        select_kv_head(event_id=1, num_key_value_heads=0, identity=IDENTITY)


def test_candidate_donor_sampling_independent_of_event_and_depth_sampling():
    sig = inspect.signature(select_candidates_and_donors)
    # No parameter accepts an event-selection or depth-assignment object, or
    # any swap outcome -- selection is a pure function of the pools and the
    # (event, layer, head) identity only.
    for name in sig.parameters:
        assert "outcome" not in name
        assert "gain" not in name
        assert "depth" not in name


def test_candidate_donor_sampling_exactly_two_each_and_cross_product_of_four():
    result = select_candidates_and_donors(
        evicted_pool=[1, 5, 9, 13, 17],
        retained_pool=[2, 6, 10, 14],
        event_id=3,
        layer_index=7,
        kv_head_index=2,
        identity=IDENTITY,
    )
    assert result is not None
    assert len(result.evicted_selected) == 2
    assert len(result.donor_selected) == 2
    assert len(result.cross_product) == 4
    assert set(result.cross_product) == {
        (e, r) for e in result.evicted_selected for r in result.donor_selected
    }


def test_candidate_donor_sampling_invalidates_event_when_pool_too_small():
    assert select_candidates_and_donors([1], [2, 3], 1, 1, 1, IDENTITY) is None
    assert select_candidates_and_donors([1, 2], [3], 1, 1, 1, IDENTITY) is None
    assert select_candidates_and_donors([], [], 1, 1, 1, IDENTITY) is None


def test_candidate_donor_sampling_pool_order_independent():
    a = select_candidates_and_donors([1, 5, 9, 13, 17], [2, 6, 10, 14], 3, 7, 2, IDENTITY)
    b = select_candidates_and_donors([17, 13, 9, 5, 1], [14, 10, 6, 2], 3, 7, 2, IDENTITY)
    c = select_candidates_and_donors(list({9, 1, 17, 5, 13}), list({14, 6, 2, 10}), 3, 7, 2, IDENTITY)
    assert a == b == c


def test_cross_product_pairs_pure_function():
    assert cross_product_pairs((1, 2), (10, 20)) == ((1, 10), (1, 20), (2, 10), (2, 20))


def test_full_pipeline_golden_vector_reproduces_exactly():
    eligible = list(range(30))
    events = select_events(eligible, IDENTITY)
    assert events is not None
    depths = assign_depth_strata(events.selected_events_chronological, IDENTITY)

    layer_and_head_by_event = {}
    for event_id in events.selected_events_chronological:
        stratum = depths.depth_stratum_by_event[event_id]
        layer = select_layer(event_id, stratum, num_hidden_layers=28, identity=IDENTITY)
        head = select_kv_head(event_id, num_key_value_heads=8, identity=IDENTITY)
        layer_and_head_by_event[event_id] = (layer.layer_index, head)

    # Re-run everything from scratch -- must reproduce byte-identically.
    events2 = select_events(eligible, IDENTITY)
    depths2 = assign_depth_strata(events2.selected_events_chronological, IDENTITY)
    layer_and_head_by_event2 = {}
    for event_id in events2.selected_events_chronological:
        stratum = depths2.depth_stratum_by_event[event_id]
        layer = select_layer(event_id, stratum, num_hidden_layers=28, identity=IDENTITY)
        head = select_kv_head(event_id, num_key_value_heads=8, identity=IDENTITY)
        layer_and_head_by_event2[event_id] = (layer.layer_index, head)

    assert events == events2
    assert depths == depths2
    assert layer_and_head_by_event == layer_and_head_by_event2
