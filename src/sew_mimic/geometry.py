"""Closed-form rotation subproblems from the SEW-Mimic appendix.

The implementations follow Algorithms 5, 6, and 7 of SEW-Mimic and were
cross-checked against the independent IK-Geo reference implementation:
https://github.com/rpiRobotics/ik-geo

For Algorithm 7, ``x_tilde = A.T * b`` is the agreed interpretation of the
paper's inconsistent pseudocode, backed by the referenced IK-Geo implementation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

_DEGENERACY_TOL = 1e-12


def _vector3(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _unit(value: ArrayLike, name: str) -> Vector:
    vector = _vector3(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= _DEGENERACY_TOL:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _skew(vector: Vector) -> Matrix:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _unit_perpendicular(vector: Vector, axis: Vector, name: str) -> Vector:
    projection = vector - np.dot(vector, axis) * axis
    vector_norm = float(np.linalg.norm(vector))
    projection_norm = float(np.linalg.norm(projection))
    if vector_norm <= _DEGENERACY_TOL:
        raise ValueError(f"{name} must be nonzero")
    if projection_norm <= _DEGENERACY_TOL * vector_norm:
        raise ValueError(f"{name} must not be parallel to its rotation axis")
    return projection / projection_norm


def rot(axis: ArrayLike, theta: float) -> Matrix:
    """Return the Rodrigues rotation matrix for ``axis`` and ``theta``."""
    k = _unit(axis, "axis")
    angle = _scalar(theta, "theta")
    skew_k = _skew(k)
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return np.eye(3) + sine * skew_k + (1.0 - cosine) * (skew_k @ skew_k)


def sp1(p1: ArrayLike, p2: ArrayLike, k: ArrayLike) -> float:
    """Solve Appendix Algorithm 5 (circle and point)."""
    axis = _unit(k, "k")
    vector1 = _vector3(p1, "p1")
    vector2 = _vector3(p2, "p2")
    p1_hat = _unit_perpendicular(vector1, axis, "p1")
    p2_hat = _unit_perpendicular(vector2, axis, "p2")

    theta = 2.0 * math.atan2(
        float(np.linalg.norm(p1_hat - p2_hat)),
        float(np.linalg.norm(p1_hat + p2_hat)),
    )
    if np.dot(axis, np.cross(p1_hat, p2_hat)) < 0.0:
        theta = -theta
    return theta


def sp4(p: ArrayLike, h: ArrayLike, k: ArrayLike, d: float) -> NDArray[np.float64]:
    """Solve Appendix Algorithm 7 (circle and plane).

    Returns one least-squares/tangent solution or two exact solutions.
    ``h`` is normalized because the paper defines it as a unit plane normal
    and ``d`` as the signed plane distance.
    """
    vector = _vector3(p, "p")
    axis = _unit(k, "k")
    plane_normal = _unit(h, "h")
    distance = _scalar(d, "d")

    _unit_perpendicular(vector, axis, "p")
    _unit_perpendicular(plane_normal, axis, "h")

    skew_k = _skew(axis)
    basis = np.column_stack((skew_k @ vector, -(skew_k @ skew_k) @ vector))
    a = plane_normal @ basis
    b = distance - np.dot(plane_normal, axis) * np.dot(axis, vector)
    norm_a_squared = float(np.dot(a, a))
    if norm_a_squared <= (_DEGENERACY_TOL * np.linalg.norm(vector)) ** 2:
        raise ValueError("SP4 is degenerate because the circle-plane equation has no angle dependence")

    x_tilde = a * b
    if norm_a_squared > b * b:
        z = math.sqrt(norm_a_squared - b * b)
        null_vector = np.array([a[1], -a[0]])
        x_plus = x_tilde + z * null_vector
        x_minus = x_tilde - z * null_vector
        return np.array(
            [
                math.atan2(x_plus[0], x_plus[1]),
                math.atan2(x_minus[0], x_minus[1]),
            ]
        )

    return np.array([math.atan2(x_tilde[0], x_tilde[1])])


def sp2(
    p1: ArrayLike,
    p2: ArrayLike,
    k1: ArrayLike,
    k2: ArrayLike,
) -> NDArray[np.float64]:
    """Solve Appendix Algorithm 6 (two circles).

    Each returned row is one ``(theta1, theta2)`` solution pair.
    """
    axis1 = _unit(k1, "k1")
    axis2 = _unit(k2, "k2")
    if np.linalg.norm(np.cross(axis1, axis2)) <= _DEGENERACY_TOL:
        raise ValueError("k1 and k2 must not be parallel")

    vector1 = _unit(p1, "p1")
    vector2 = _unit(p2, "p2")
    _unit_perpendicular(vector1, axis1, "p1")
    _unit_perpendicular(vector2, axis2, "p2")

    theta1 = sp4(vector1, axis2, axis1, float(np.dot(axis2, vector2)))
    theta2 = sp4(vector2, axis1, axis2, float(np.dot(axis1, vector1)))

    if theta1.size > 1 or theta2.size > 1:
        theta1 = np.array([theta1[0], theta1[-1]])
        theta2 = np.array([theta2[-1], theta2[0]])

    return np.column_stack((theta1, theta2))
