import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sew_mimic.geometry import rot, sp1, sp2, sp4


RNG = np.random.default_rng(20260831)


def _random_unit() -> np.ndarray:
    vector = RNG.normal(size=3)
    return vector / np.linalg.norm(vector)


def _nonparallel_axes() -> tuple[np.ndarray, np.ndarray]:
    while True:
        axis1 = _random_unit()
        axis2 = _random_unit()
        if np.linalg.norm(np.cross(axis1, axis2)) > 0.2:
            return axis1, axis2


def test_rot_random_matches_scipy_rodrigues_rotation() -> None:
    for _ in range(200):
        axis = _random_unit()
        axis_scale = RNG.uniform(0.1, 10.0)
        theta = RNG.uniform(-4.0 * np.pi, 4.0 * np.pi)

        actual = rot(axis_scale * axis, theta)
        expected = Rotation.from_rotvec(axis * theta).as_matrix()

        np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=3e-15)


def test_sp1_random_exact_geometric_residual_is_machine_precision() -> None:
    for _ in range(200):
        axis = _random_unit()
        vector = RNG.normal(size=3)
        theta = RNG.uniform(-np.pi, np.pi)
        target = rot(axis, theta) @ vector

        recovered = sp1(vector, target, RNG.uniform(0.1, 10.0) * axis)
        residual = np.linalg.norm(rot(axis, recovered) @ vector - target)

        assert residual <= 2e-15 * max(1.0, np.linalg.norm(vector))


def test_sp2_random_exact_geometric_residual_is_machine_precision() -> None:
    for _ in range(200):
        while True:
            axis1, axis2 = _nonparallel_axes()
            common_target = RNG.normal(size=3)
            target_hat = common_target / np.linalg.norm(common_target)
            axis_dot = np.dot(axis1, axis2)
            amplitude1_squared = (1.0 - np.dot(axis1, target_hat) ** 2) * (
                1.0 - axis_dot**2
            )
            amplitude2_squared = (1.0 - np.dot(axis2, target_hat) ** 2) * (
                1.0 - axis_dot**2
            )
            b1 = np.dot(axis2, target_hat) - axis_dot * np.dot(axis1, target_hat)
            b2 = np.dot(axis1, target_hat) - axis_dot * np.dot(axis2, target_hat)
            if (
                amplitude1_squared - b1**2 > 0.05 * amplitude1_squared
                and amplitude2_squared - b2**2 > 0.05 * amplitude2_squared
            ):
                break
        theta1 = RNG.uniform(-np.pi, np.pi)
        theta2 = RNG.uniform(-np.pi, np.pi)
        p1 = rot(axis1, -theta1) @ common_target
        p2 = rot(axis2, -theta2) @ common_target

        solutions = sp2(
            p1,
            p2,
            RNG.uniform(0.1, 10.0) * axis1,
            RNG.uniform(0.1, 10.0) * axis2,
        )
        residuals = [
            np.linalg.norm(rot(axis1, pair[0]) @ p1 - rot(axis2, pair[1]) @ p2)
            for pair in solutions
        ]

        assert max(residuals) <= 2e-14 * max(1.0, np.linalg.norm(common_target))


def test_sp4_random_exact_geometric_residual_is_machine_precision() -> None:
    for _ in range(200):
        axis, plane_normal = _nonparallel_axes()
        vector = RNG.normal(size=3)
        theta = RNG.uniform(-np.pi, np.pi)
        distance = float(plane_normal @ (rot(axis, theta) @ vector))

        solutions = sp4(
            vector,
            plane_normal,
            RNG.uniform(0.1, 10.0) * axis,
            distance,
        )
        residuals = [
            abs(float(plane_normal @ (rot(axis, angle) @ vector)) - distance)
            for angle in solutions
        ]

        assert max(residuals) <= 4e-15 * max(1.0, np.linalg.norm(vector))


def test_sp4_random_unreachable_plane_returns_least_squares_solution() -> None:
    for _ in range(100):
        axis, plane_normal = _nonparallel_axes()
        vector = RNG.normal(size=3)
        offset = np.dot(plane_normal, axis) * np.dot(axis, vector)
        amplitude = np.linalg.norm(np.cross(axis, vector)) * np.linalg.norm(
            np.cross(axis, plane_normal)
        )
        distance = float(offset + amplitude + RNG.uniform(0.1, 2.0))

        solutions = sp4(vector, plane_normal, axis, distance)
        residual = abs(float(plane_normal @ (rot(axis, solutions[0]) @ vector)) - distance)
        expected_residual = distance - offset - amplitude

        assert solutions.shape == (1,)
        assert residual == pytest.approx(expected_residual, abs=3e-15)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (rot, ([0.0, 0.0, 0.0], 0.5)),
        (sp1, ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0])),
        (sp1, ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])),
        (sp2, ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0])),
        (sp4, ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0)),
    ],
)
def test_degenerate_inputs_raise_value_error(function: object, arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        function(*arguments)  # type: ignore[operator]
