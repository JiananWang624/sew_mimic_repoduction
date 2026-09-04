import math

import numpy as np
import pytest

from sew_mimic.geometry import SP3_EXACT_TOL, rot, sp3


def test_sp3_returns_two_exact_roots_in_wrapped_order() -> None:
    result = sp3([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0)

    assert result.angles == pytest.approx((-math.pi / 3.0, math.pi / 3.0))
    assert result.is_exact == (True, True)
    assert result.residuals[0] <= SP3_EXACT_TOL
    assert result.residuals[1] <= SP3_EXACT_TOL
    assert not result.degenerate


def test_sp3_tangent_has_one_exact_root() -> None:
    result = sp3([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0)

    assert result.angles == pytest.approx((-math.pi,))
    assert result.is_exact == (True,)
    assert result.residuals[0] <= SP3_EXACT_TOL


def test_sp3_unreachable_returns_least_squares_extremum() -> None:
    result = sp3([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 3.0)

    assert result.angles == pytest.approx((-math.pi,))
    assert result.is_exact == (False,)
    assert result.residuals == pytest.approx((1.0,))
    assert "least-squares" in (result.message or "")


def test_sp3_is_invariant_to_axis_scaling() -> None:
    arguments = ([1.0, 2.0, 0.5], [-0.5, 1.0, 1.5], [0.2, -0.3, 0.7], 1.1)

    unit_axis = sp3(*arguments)
    scaled_axis = sp3(arguments[0], arguments[1], np.asarray(arguments[2]) * 17.0, arguments[3])

    assert scaled_axis == unit_axis


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0), "p1"),
        (([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0), "k"),
        (([np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0), "finite"),
        (([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.0), "nonnegative"),
        (([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], math.inf), "finite"),
    ],
)
def test_sp3_rejects_invalid_inputs(arguments: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sp3(*arguments)  # type: ignore[arg-type]


def test_sp3_parallel_p1_is_constant_distance_degenerate() -> None:
    result = sp3([0.0, 0.0, 2.0], [1.0, 0.0, 3.0], [0.0, 0.0, 4.0], math.sqrt(2.0))

    assert result.angles == (0.0,)
    assert result.is_exact == (True,)
    assert result.residuals == pytest.approx((0.0,))
    assert result.degenerate


def test_sp3_zero_p1_is_constant_distance_and_can_be_inexact() -> None:
    result = sp3([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0)

    assert result.angles == (0.0,)
    assert result.is_exact == (False,)
    assert result.residuals == pytest.approx((1.0,))
    assert result.degenerate


def test_sp3_parallel_p2_is_constant_distance_degenerate() -> None:
    result = sp3([1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], math.sqrt(5.0))

    assert result.angles == (0.0,)
    assert result.is_exact == (True,)
    assert result.residuals == pytest.approx((0.0,))
    assert result.degenerate


def test_sp3_repeated_calls_and_root_order_are_deterministic() -> None:
    arguments = ([0.2, -1.1, 0.7], [1.3, 0.1, -0.4], [0.3, 0.8, -0.2], 1.25)
    first = sp3(*arguments)
    second = sp3(*arguments)

    assert first == second
    assert list(first.angles) == sorted(first.angles)
    assert all(-math.pi <= angle < math.pi for angle in first.angles)


def test_sp3_near_boundary_roundoff_is_clipped_but_genuine_miss_is_not() -> None:
    near = sp3(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        np.nextafter(2.0, np.inf),
    )
    miss = sp3([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0 + 5e-11)

    assert near.angles == pytest.approx((-math.pi,))
    assert near.is_exact == (True,)
    assert near.residuals[0] == pytest.approx(np.nextafter(2.0, np.inf) - 2.0)
    assert miss.is_exact == (False,)
    assert miss.residuals[0] == pytest.approx(5e-11)


def test_sp3_below_minimum_distance_is_nonexact_least_squares() -> None:
    result = sp3([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0)

    assert result.angles == pytest.approx((0.0,))
    assert result.is_exact == (False,)
    assert result.residuals == pytest.approx((1.0,))
    assert "least-squares" in (result.message or "")


def test_sp3_mixed_scale_keeps_small_perpendicular_radius_nondegenerate() -> None:
    result = sp3([5e-5, 0.0, 0.0], [1e8, 0.0, 0.0], [0.0, 0.0, 1.0], 1e8)

    assert not result.degenerate
    assert result.angles == pytest.approx((-math.pi / 2.0, math.pi / 2.0), abs=2e-10)
    assert all(result.is_exact)


def test_sp3_large_finite_vectors_have_finite_outputs() -> None:
    result = sp3([1e308, 0.0, 0.0], [0.0, 1e308, 0.0], [0.0, 0.0, 1e308], 0.0)

    assert all(math.isfinite(angle) for angle in result.angles)
    assert all(math.isfinite(residual) for residual in result.residuals)
    assert any(result.is_exact)


def test_sp3_avoids_parallel_coordinate_cancellation() -> None:
    p1 = [1.0, 0.0, 1e8]
    p2 = [1.0, 0.0, 1e8]
    result = sp3(p1, p2, [0.0, 0.0, 1.0], math.sqrt(2.0))

    assert not result.degenerate
    assert result.angles == pytest.approx((-math.pi / 2.0, math.pi / 2.0), abs=2e-10)
    assert result.is_exact == (True, True)


def test_sp3_keeps_small_radius_with_large_common_axial_offset_nondegenerate() -> None:
    result = sp3([1.0, 0.0, 1e13], [1.0, 0.0, 1e13], [0.0, 0.0, 1.0], 1.0)

    assert not result.degenerate
    assert result.angles == pytest.approx((-math.pi / 3.0, math.pi / 3.0), abs=2e-10)
    assert result.is_exact == (True, True)


def test_sp3_huge_finite_axis_matches_scaled_axis() -> None:
    p1 = [0.8, -0.2, 0.5]
    p2 = [-0.1, 0.7, 0.3]
    axis = np.array([0.3, -0.8, 0.4])
    huge_axis = axis * (1e308 / np.max(np.abs(axis)))

    expected = sp3(p1, p2, axis, 0.9)
    actual = sp3(p1, p2, huge_axis, 0.9)

    assert actual == expected


def test_sp3_recomputes_residual_for_every_random_exact_case() -> None:
    rng = np.random.default_rng(20260904)
    exact_count = 0
    for _ in range(250):
        p1 = rng.normal(size=3)
        p2 = rng.normal(size=3)
        axis = rng.normal(size=3)
        theta_true = rng.uniform(-math.pi, math.pi)
        distance = float(np.linalg.norm(rot(axis, theta_true) @ p1 - p2))

        result = sp3(p1, p2, axis, distance)
        assert len(result.angles) in (1, 2)
        assert len(result.angles) == len(result.is_exact) == len(result.residuals)
        for angle, is_exact, residual in zip(result.angles, result.is_exact, result.residuals):
            recomputed = abs(float(np.linalg.norm(rot(axis, angle) @ p1 - p2)) - distance)
            assert residual == pytest.approx(recomputed, abs=2e-15)
            if is_exact:
                exact_count += 1
        assert any(result.is_exact)

    assert exact_count >= 250


def test_sp3_matches_official_ik_geo_sp3_sp4_reduction_case() -> None:
    result = sp3(
        [1.0, 2.0, 3.0],
        [-2.0, 0.5, 1.5],
        [0.0, 0.0, 2.0],
        2.689186715241201,
    )

    assert result.angles == pytest.approx((0.7, 2.8789305453376777), abs=2e-14)
    assert result.is_exact == (True, True)


def test_sp3_matches_official_ik_geo_reference_case() -> None:
    result = sp3(
        [-0.684773836644903, 0.941185563521231, 0.914333896485891],
        [-0.0292487025543176, 0.600560937777600, -0.716227322745569],
        [-0.152173233501273, 0.808599901340102, 0.568339253050978],
        1.64463116212829,
    )

    assert result.angles == pytest.approx(
        (-0.523658333853567, 2.88707606227219), abs=3e-14
    )
    assert result.is_exact == (True, True)


def test_sp3_warp_shaped_elbow_geometry() -> None:
    shoulder = np.array([0.0, 0.0, 0.0])
    wrist = np.array([0.4, 0.0, 0.0])
    e_sw_hat = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)
    l_se = 0.3
    l_ew = 0.25
    axis = np.array([0.0, 0.0, 1.0])

    result = sp3(l_se * e_sw_hat, wrist - shoulder, axis, l_ew)

    assert any(result.is_exact)
    for angle, is_exact in zip(result.angles, result.is_exact):
        if is_exact:
            elbow_from_shoulder = shoulder + rot(axis, angle) @ (l_se * e_sw_hat)
            assert np.linalg.norm(elbow_from_shoulder - shoulder) == pytest.approx(l_se)
            assert np.linalg.norm(elbow_from_shoulder - wrist) == pytest.approx(l_ew)
