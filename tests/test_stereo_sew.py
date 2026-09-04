"""Validation of the isolated stereographic SEW representation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sew_mimic.angles import angular_difference, wrap_to_pi
from sew_mimic.geometry import rot
from sew_mimic.sew import (
    StereoSew,
    StereoSewInverseResult,
    StereoSewReference,
    StereoSewSingularityError,
)


TUTORIAL_REFERENCE = StereoSewReference(
    e_t=np.array([0.0, 0.0, -1.0]),
    e_r=np.array([0.0, 1.0, 0.0]),
)
TUTORIAL_SEW = StereoSew(TUTORIAL_REFERENCE)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _sample_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([0.1, -0.2, 0.3]),
        np.array([0.4, 0.5, 0.2]),
        np.array([0.8, -0.1, 0.9]),
    )


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (-math.pi, -math.pi),
        (math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (0.0, 0.0),
    ],
)
def test_shared_angle_wrap_uses_half_open_interval(angle: float, expected: float) -> None:
    assert wrap_to_pi(angle) == pytest.approx(expected)
    assert -math.pi <= wrap_to_pi(angle) < math.pi


def test_angular_difference_is_signed_and_periodic() -> None:
    assert angular_difference(-math.pi + 0.1, math.pi - 0.1) == pytest.approx(0.2)
    assert angular_difference(0.7 + 2.0 * math.pi, 0.7) == pytest.approx(0.0)


def test_reference_validates_and_is_immutable() -> None:
    reference = StereoSewReference([1.0, 2e-12, 0.0], [0.0, 1.0, 0.0])
    assert np.linalg.norm(reference.e_t) == pytest.approx(1.0)
    assert abs(float(np.dot(reference.e_t, reference.e_r))) <= 1e-10
    assert not reference.e_t.flags.writeable
    with pytest.raises(ValueError):
        reference.e_t[0] = 0.0
    with pytest.raises(ValueError):
        reference.e_t.setflags(write=True)


@pytest.mark.parametrize(
    ("e_t", "e_r", "message"),
    [
        ([1.0, 0.0], [0.0, 1.0, 0.0], "shape"),
        ([math.nan, 0.0, 0.0], [0.0, 1.0, 0.0], "finite"),
        ([1.0 + 2e-10, 0.0, 0.0], [0.0, 1.0, 0.0], "unit"),
        ([1.0, 0.0, 0.0], [1e-9, 1.0, 0.0], "orthogonal"),
    ],
)
def test_reference_rejects_substantially_invalid_vectors(
    e_t: object, e_r: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StereoSewReference(e_t, e_r)  # type: ignore[arg-type]


def test_inverse_result_is_immutable() -> None:
    result = StereoSewInverseResult([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert not result.elbow_direction.flags.writeable
    assert not result.plane_normal.flags.writeable
    with pytest.raises(ValueError):
        result.plane_normal[0] = 1.0
    with pytest.raises(ValueError):
        result.plane_normal.setflags(write=True)


def test_official_tutorial_case_b_is_pinned() -> None:
    psi = TUTORIAL_SEW.forward(
        [0.0, 0.0, 0.0], [0.0, math.cos(0.7), math.sin(0.7)], [1.0, 0.0, 0.0]
    )
    inverse = TUTORIAL_SEW.inverse([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], psi)

    assert psi == pytest.approx(0.7, abs=3e-15)
    np.testing.assert_allclose(
        inverse.plane_normal,
        [0.0, -0.644217687237691, 0.7648421872844885],
        atol=3e-15,
    )


def test_official_tutorial_case_c_is_pinned() -> None:
    shoulder, elbow, wrist = _sample_geometry()
    psi = TUTORIAL_SEW.forward(shoulder, elbow, wrist)
    inverse = TUTORIAL_SEW.inverse(shoulder, wrist, psi)

    assert psi == pytest.approx(-0.3305384044883147, abs=3e-15)
    np.testing.assert_allclose(
        inverse.plane_normal,
        [-0.6346906679704016, 0.3690062023083731, 0.6789714122474063],
        atol=3e-15,
    )


def test_forward_inverse_round_trip_preserves_oriented_plane() -> None:
    shoulder, elbow, wrist = _sample_geometry()
    psi = TUTORIAL_SEW.forward(shoulder, elbow, wrist)
    recovered = TUTORIAL_SEW.inverse(shoulder, wrist, psi)
    expected_normal = _unit(np.cross(wrist - shoulder, elbow - shoulder))
    e_sw = _unit(wrist - shoulder)

    np.testing.assert_allclose(recovered.plane_normal, expected_normal, atol=3e-15)
    assert np.linalg.norm(recovered.elbow_direction) == pytest.approx(1.0)
    assert np.dot(recovered.elbow_direction, e_sw) == pytest.approx(0.0, abs=3e-15)
    np.testing.assert_allclose(
        recovered.plane_normal,
        np.cross(e_sw, recovered.elbow_direction),
        atol=3e-15,
    )


def test_forward_inverse_forward_preserves_signed_angle() -> None:
    shoulder, elbow, wrist = _sample_geometry()
    psi_1 = TUTORIAL_SEW.forward(shoulder, elbow, wrist)
    inverse = TUTORIAL_SEW.inverse(shoulder, wrist, psi_1)
    e_sw = _unit(wrist - shoulder)
    reconstructed_elbow = shoulder + 0.37 * e_sw + 0.52 * inverse.elbow_direction
    psi_2 = TUTORIAL_SEW.forward(shoulder, reconstructed_elbow, wrist)

    assert angular_difference(psi_2, psi_1) == pytest.approx(0.0, abs=3e-15)


def test_translation_and_positive_scale_invariance() -> None:
    shoulder, elbow, wrist = _sample_geometry()
    psi = TUTORIAL_SEW.forward(shoulder, elbow, wrist)
    translation = np.array([-3.1, 2.4, 8.7])
    scaled_elbow = shoulder + 5.2 * (elbow - shoulder)
    scaled_wrist = shoulder + 5.2 * (wrist - shoulder)

    assert TUTORIAL_SEW.forward(shoulder + translation, elbow + translation, wrist + translation) == pytest.approx(psi)
    assert TUTORIAL_SEW.forward(shoulder, scaled_elbow, scaled_wrist) == pytest.approx(psi)
    np.testing.assert_allclose(
        TUTORIAL_SEW.inverse(shoulder + translation, wrist + translation, psi).plane_normal,
        TUTORIAL_SEW.inverse(shoulder, wrist, psi).plane_normal,
        atol=3e-15,
    )


def test_rotation_covariance() -> None:
    shoulder, elbow, wrist = _sample_geometry()
    rotation = rot([0.3, -0.8, 0.4], 1.2)
    original_psi = TUTORIAL_SEW.forward(shoulder, elbow, wrist)
    original_normal = TUTORIAL_SEW.inverse(shoulder, wrist, original_psi).plane_normal
    rotated = StereoSew(
        StereoSewReference(rotation @ TUTORIAL_REFERENCE.e_t, rotation @ TUTORIAL_REFERENCE.e_r)
    )
    rotated_psi = rotated.forward(rotation @ shoulder, rotation @ elbow, rotation @ wrist)
    rotated_normal = rotated.inverse(rotation @ shoulder, rotation @ wrist, rotated_psi).plane_normal

    assert angular_difference(rotated_psi, original_psi) == pytest.approx(0.0, abs=5e-15)
    np.testing.assert_allclose(rotated_normal, rotation @ original_normal, atol=5e-15)


def test_inverse_is_periodic_and_deterministic() -> None:
    shoulder, elbow, wrist = _sample_geometry()
    psi = TUTORIAL_SEW.forward(shoulder, elbow, wrist)
    first = TUTORIAL_SEW.inverse(shoulder, wrist, psi)
    periodic = TUTORIAL_SEW.inverse(shoulder, wrist, psi + 2.0 * math.pi)
    repeated = TUTORIAL_SEW.inverse(shoulder, wrist, psi)

    np.testing.assert_allclose(periodic.plane_normal, first.plane_normal, atol=3e-15)
    np.testing.assert_allclose(periodic.elbow_direction, first.elbow_direction, atol=3e-15)
    np.testing.assert_allclose(repeated.plane_normal, first.plane_normal, atol=0.0)


@pytest.mark.parametrize(
    ("method", "arguments", "message"),
    [
        ("forward", ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]), "shoulder-to-wrist"),
        ("inverse", ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2), "shoulder-to-wrist"),
        ("forward", ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]), "collinear"),
    ],
)
def test_geometric_degeneracies_raise_distinct_singularity(
    method: str, arguments: tuple[object, ...], message: str
) -> None:
    with pytest.raises(StereoSewSingularityError, match=message):
        getattr(TUTORIAL_SEW, method)(*arguments)


def test_stereographic_half_line_and_nearby_direction() -> None:
    shoulder = np.zeros(3)
    wrist_on_half_line = TUTORIAL_REFERENCE.e_t
    elbow = np.array([1.0, 0.0, 0.0])
    with pytest.raises(StereoSewSingularityError, match="half-line"):
        TUTORIAL_SEW.forward(shoulder, elbow, wrist_on_half_line)
    with pytest.raises(StereoSewSingularityError, match="half-line"):
        TUTORIAL_SEW.inverse(shoulder, wrist_on_half_line, 0.0)

    # The inverse transverse cross product approaches the half-line at the
    # square of this perturbation; this remains safely above 64 eps.
    perturbation = 1e-6
    near_direction = _unit(np.array([perturbation, 0.0, -1.0]))
    psi = TUTORIAL_SEW.forward(shoulder, elbow, near_direction)
    inverse = TUTORIAL_SEW.inverse(shoulder, near_direction, psi)
    assert math.isfinite(psi)
    assert np.all(np.isfinite(inverse.plane_normal))

    # Stereo-SEW has a singular half-line, not a full line: the antipodal
    # shoulder-wrist direction must remain valid.
    opposite_direction = -TUTORIAL_REFERENCE.e_t
    opposite_psi = TUTORIAL_SEW.forward(shoulder, elbow, opposite_direction)
    opposite_inverse = TUTORIAL_SEW.inverse(shoulder, opposite_direction, opposite_psi)
    assert math.isfinite(opposite_psi)
    assert np.all(np.isfinite(opposite_inverse.plane_normal))


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("forward", ([math.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])),
        ("forward", ([0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])),
        ("inverse", ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], math.inf)),
        ("inverse", ([0.0, 0.0], [1.0, 0.0, 0.0], 0.0)),
    ],
)
def test_invalid_inputs_raise_value_error(method: str, arguments: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        getattr(TUTORIAL_SEW, method)(*arguments)


def test_fixed_seed_random_round_trips_have_machine_precision_error() -> None:
    rng = np.random.default_rng(20260905)
    maximum_plane_error = 0.0
    maximum_angle_error = 0.0
    cases = 500
    for _ in range(cases):
        e_t = _unit(rng.normal(size=3))
        provisional = rng.normal(size=3)
        e_r = _unit(provisional - e_t * np.dot(e_t, provisional))
        sew = StereoSew(StereoSewReference(e_t, e_r))
        shoulder = rng.normal(size=3)
        e_sw = _unit(rng.normal(size=3))
        transverse = _unit(np.cross(e_sw, rng.normal(size=3)))
        wrist = shoulder + rng.uniform(0.1, 4.0) * e_sw
        elbow = shoulder + rng.uniform(0.1, 4.0) * (0.31 * e_sw + transverse)
        psi = sew.forward(shoulder, elbow, wrist)
        inverse = sew.inverse(shoulder, wrist, psi)
        expected_normal = _unit(np.cross(wrist - shoulder, elbow - shoulder))
        maximum_plane_error = max(maximum_plane_error, float(np.linalg.norm(inverse.plane_normal - expected_normal)))
        reconstructed = shoulder + 0.43 * e_sw + 0.57 * inverse.elbow_direction
        maximum_angle_error = max(maximum_angle_error, abs(angular_difference(sew.forward(shoulder, reconstructed, wrist), psi)))

    deliberately_singular = 0
    singular_calls = (
        (TUTORIAL_SEW.forward, (np.zeros(3), np.array([0.0, 1.0, 0.0]), np.zeros(3))),
        (TUTORIAL_SEW.forward, (np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))),
        (TUTORIAL_SEW.inverse, (np.zeros(3), TUTORIAL_REFERENCE.e_t, 0.0)),
    )
    for function, arguments in singular_calls:
        with pytest.raises(StereoSewSingularityError):
            function(*arguments)
        deliberately_singular += 1

    assert cases == 500
    assert deliberately_singular == 3
    assert maximum_plane_error <= 2e-14
    assert maximum_angle_error <= 2e-14


def test_warp_style_cross_geometry_transfer_preserves_psi() -> None:
    human_shoulder, human_elbow, human_wrist = _sample_geometry()
    psi_human = TUTORIAL_SEW.forward(human_shoulder, human_elbow, human_wrist)
    robot_shoulder = np.array([-0.6, 0.8, -0.1])
    robot_wrist = np.array([0.5, 1.2, 0.7])
    inverse_robot = TUTORIAL_SEW.inverse(robot_shoulder, robot_wrist, psi_human)
    robot_axis = _unit(robot_wrist - robot_shoulder)
    robot_elbow = robot_shoulder + 0.4 * robot_axis + 0.3 * inverse_robot.elbow_direction

    assert angular_difference(
        TUTORIAL_SEW.forward(robot_shoulder, robot_elbow, robot_wrist), psi_human
    ) == pytest.approx(0.0, abs=4e-15)
