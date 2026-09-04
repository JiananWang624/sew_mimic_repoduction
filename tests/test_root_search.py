import numpy as np
import pytest

from sew_mimic.exact.root_search import (
    EventAwareSearchConfig,
    FeasibleInterval,
    search_event_aware_fixed_slot_roots,
    solve_alignment_roots,
    RootSearchConfig,
    search_fixed_slot_roots,
    sp2_feasibility_margin,
    sp3_feasibility_margin,
    sp4_feasibility_margin,
)
from sew_mimic.geometry import sp4


def test_fixed_slot_roots_are_deterministic_and_slot_stable():
    config = RootSearchConfig(minimum=-1.0, maximum=1.0, samples=200, crossing_threshold=0.1)
    function = lambda angle: np.array([angle + .25, np.nan, angle - .5])
    first = search_fixed_slot_roots(function, config)
    second = search_fixed_slot_roots(function, config)
    assert [(root.slot, root.angle) for root in first.roots] == [(root.slot, root.angle) for root in second.roots]
    assert [(root.slot, root.angle) for root in first.roots] == [(0, pytest.approx(-.25)), (2, pytest.approx(.5))]
    assert first.slot_count == 3
    assert first.inactive_samples == 200


def test_sampled_zero_is_deduplicated_from_adjacent_brackets():
    result = search_fixed_slot_roots(lambda angle: np.array([angle]), RootSearchConfig(minimum=-1.0, maximum=1.0, samples=201, crossing_threshold=.1))
    assert len(result.roots) == 1
    assert result.roots[0].angle == 0.0
    assert result.sampled_exact_zeros == 1


def test_nan_and_discontinuity_crossing_are_rejected():
    config = RootSearchConfig(minimum=-1.0, maximum=1.0, samples=3, crossing_threshold=.1)
    result = search_fixed_slot_roots(lambda angle: np.array([-.2 if angle < 0 else .2, np.nan]), config)
    assert not result.roots
    assert result.rejected_crossings == 1


def test_small_endpoint_crossing_matches_official_threshold_intent():
    config = RootSearchConfig(minimum=-1.0, maximum=1.0, samples=3, crossing_threshold=.1)
    result = search_fixed_slot_roots(lambda angle: np.array([.09 * angle]), config)
    assert len(result.roots) == 1


def test_changed_slot_shape_is_invalid():
    with pytest.raises(ValueError, match="shape"):
        search_fixed_slot_roots(lambda angle: np.zeros(1 if angle < 0 else 2))


def test_feasibility_margins_have_expected_signs_and_degenerate_sp3():
    assert sp3_feasibility_margin([1, 0, 0], [1, 0, 0], [0, 0, 1], 1) > 0
    assert sp3_feasibility_margin([1, 0, 0], [1, 0, 0], [0, 0, 1], 3) < 0
    assert sp3_feasibility_margin([0, 0, 1], [0, 0, 1], [0, 0, 1], 0) == 0
    assert sp4_feasibility_margin([1, 0, 0], [0, 1, 0], [0, 0, 1], 0) > 0
    assert np.isfinite(sp2_feasibility_margin([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]))


def test_event_search_recovers_narrow_positive_branch_and_alignment_root():
    config = EventAwareSearchConfig(minimum=-1, maximum=1, initial_partitions=8, max_subdivision_depth=8)
    result = search_event_aware_fixed_slot_roots(
        lambda angle: np.array([1e-6 - (angle - .1234) ** 2, np.nan]),
        lambda angle: np.array([angle - .1234, np.nan]),
        config,
    )
    assert result.intervals
    assert [(root.slot, root.angle) for root in result.roots] == [(0, pytest.approx(.1234, abs=1e-8))]
    assert result.inactive_samples > 0


def test_event_search_handles_tangent_event_and_tangent_alignment_root():
    result = search_event_aware_fixed_slot_roots(
        lambda angle: np.array([-(angle - .2) ** 2]),
        lambda angle: np.array([(angle - .2) ** 2]),
        EventAwareSearchConfig(minimum=-1, maximum=1, initial_partitions=8),
    )
    assert len(result.roots) == 1
    assert result.roots[0].angle == pytest.approx(.2, abs=1e-7)


def test_event_search_is_repeatable_and_budget_bounded():
    config = EventAwareSearchConfig(minimum=-1, maximum=1, initial_partitions=16, maximum_event_evaluations=20)
    first = search_event_aware_fixed_slot_roots(lambda angle: np.array([1 - angle * angle]), lambda angle: np.array([angle]), config)
    second = search_event_aware_fixed_slot_roots(lambda angle: np.array([1 - angle * angle]), lambda angle: np.array([angle]), config)
    assert first.budget_exhausted and second.budget_exhausted
    assert first.roots == second.roots


def test_event_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        EventAwareSearchConfig(initial_partitions=0)
    with pytest.raises(ValueError):
        EventAwareSearchConfig(event_margin_tolerance=-1e-9)


@pytest.mark.parametrize(
    ("distance", "expected_count", "sign"),
    [(0.0, 2, 1), (1.0, 1, 0), (2.0, 1, -1)],
)
def test_sp4_margin_sign_matches_sp4_candidate_geometry(distance, expected_count, sign):
    candidates = sp4([1, 0, 0], [0, 1, 0], [0, 0, 1], distance)
    margin = sp4_feasibility_margin([1, 0, 0], [0, 1, 0], [0, 0, 1], distance)
    assert len(candidates) == expected_count
    assert np.sign(margin) == sign


def test_sp4_margin_preserves_nonunit_vector_and_distance_scale():
    vector = np.array([3.0, 0.0, 0.0])
    for distance, sign in ((2.0, 1), (3.0, 0), (4.0, -1)):
        candidates = sp4(vector, [0, 1, 0], [0, 0, 1], distance)
        margin = sp4_feasibility_margin(
            vector, [0, 1, 0], [0, 0, 1], distance
        )
        assert np.sign(margin) == sign
        assert len(candidates) == (2 if sign > 0 else 1)


def test_event_depth_and_tiny_budget_are_explicitly_bounded():
    shallow = search_event_aware_fixed_slot_roots(
        lambda angle: np.array([1e-6 - (angle - .1234) ** 2]),
        lambda angle: np.array([angle - .1234]),
        EventAwareSearchConfig(minimum=-1, maximum=1, initial_partitions=8, max_subdivision_depth=0),
    )
    deep = search_event_aware_fixed_slot_roots(
        lambda angle: np.array([1e-6 - (angle - .1234) ** 2]),
        lambda angle: np.array([angle - .1234]),
        EventAwareSearchConfig(minimum=-1, maximum=1, initial_partitions=8, max_subdivision_depth=24),
    )
    assert not shallow.roots
    assert deep.roots
    budget = search_event_aware_fixed_slot_roots(
        lambda angle: np.array([1 - angle * angle]),
        lambda angle: np.array([angle]),
        EventAwareSearchConfig(initial_partitions=64, maximum_event_evaluations=2),
    )
    assert budget.budget_exhausted


def test_separate_alignment_solver_recovers_tangent_minimum():
    result = solve_alignment_roots(
        (FeasibleInterval(0, -1.0, 1.0),),
        lambda angle: np.array([(angle - .2345) ** 2]),
        EventAwareSearchConfig(alignment_samples_per_interval=3),
    )
    assert len(result.roots) == 1
    assert result.roots[0].angle == pytest.approx(.2345, abs=1e-7)
    assert result.roots[0].residual <= 1e-9


def test_separate_alignment_solver_honors_callback_budget():
    result = solve_alignment_roots(
        (FeasibleInterval(0, -1.0, 1.0),),
        lambda angle: np.array([angle]),
        EventAwareSearchConfig(maximum_event_evaluations=2),
    )
    assert result.budget_exhausted
    assert result.alignment_evaluations == 2
