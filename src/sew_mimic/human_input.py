"""Human SEW keypoint and wrist-orientation preprocessing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

_DEGENERACY_TOL = 1e-12
_ROTATION_TOL = 1e-10


def _vector3(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _unit_difference(end: ArrayLike, start: ArrayLike, name: str) -> Vector:
    direction = _vector3(end, f"{name} end") - _vector3(start, f"{name} start")
    norm = float(np.linalg.norm(direction))
    if norm <= _DEGENERACY_TOL:
        raise ValueError(f"{name} endpoints must be distinct")
    return direction / norm


def _rotation_matrix(value: ArrayLike, name: str) -> Matrix:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=_ROTATION_TOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=_ROTATION_TOL, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return matrix


def compute_upper_arm_direction(shoulder: ArrayLike, elbow: ArrayLike) -> Vector:
    """Return ``unit(elbow - shoulder)`` as defined by Algorithm 1."""
    return _unit_difference(elbow, shoulder, "upper arm")


def compute_lower_arm_direction(elbow: ArrayLike, wrist: ArrayLike) -> Vector:
    """Return ``unit(wrist - elbow)`` as defined by Algorithm 1."""
    return _unit_difference(wrist, elbow, "lower arm")


def wrist_euler_to_rotation(
    angles: ArrayLike,
    *,
    order: str,
    degrees: bool,
    convention: Literal["intrinsic", "extrinsic"],
) -> Matrix:
    """Convert an explicitly specified Euler convention to ``H in SO(3)``.

    ``order`` uses lower-case axis letters such as ``"xyz"``. Intrinsic and
    extrinsic semantics are selected explicitly rather than inferred.
    """
    euler = np.asarray(angles, dtype=float)
    if euler.shape != (3,):
        raise ValueError(f"angles must have shape (3,), got {euler.shape}")
    if not np.all(np.isfinite(euler)):
        raise ValueError("angles must contain only finite values")
    if (
        not isinstance(order, str)
        or len(order) != 3
        or order != order.lower()
        or any(axis not in "xyz" for axis in order)
        or order[0] == order[1]
        or order[1] == order[2]
    ):
        raise ValueError("order must be a lower-case, valid three-axis Euler sequence")
    if not isinstance(degrees, (bool, np.bool_)):
        raise ValueError("degrees must be explicitly set to True or False")
    if convention not in ("intrinsic", "extrinsic"):
        raise ValueError("convention must be 'intrinsic' or 'extrinsic'")

    scipy_order = order.upper() if convention == "intrinsic" else order
    return Rotation.from_euler(scipy_order, euler, degrees=bool(degrees)).as_matrix()


def transform_human_to_robot_body_frame(
    shoulder: ArrayLike,
    elbow: ArrayLike,
    wrist: ArrayLike,
    hand_orientation: ArrayLike,
    *,
    rotation_robot_from_human: ArrayLike,
    translation_robot_from_human: ArrayLike,
) -> tuple[Vector, Vector, Vector, Matrix]:
    """Transform ``(s, e, w, H)`` from a human frame to robot frame 0.

    Points obey ``p_robot = R_robot_from_human @ p_human + t`` and hand
    orientation obeys ``H_robot = R_robot_from_human @ H_human``.
    """
    rotation = _rotation_matrix(rotation_robot_from_human, "rotation_robot_from_human")
    translation = _vector3(translation_robot_from_human, "translation_robot_from_human")
    hand = _rotation_matrix(hand_orientation, "hand_orientation")

    transformed_points = tuple(
        rotation @ _vector3(point, name) + translation
        for point, name in (
            (shoulder, "shoulder"),
            (elbow, "elbow"),
            (wrist, "wrist"),
        )
    )
    transformed_hand = rotation @ hand
    return (*transformed_points, transformed_hand)
