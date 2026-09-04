"""Closed-form rotation subproblems from the SEW-Mimic appendix.

The implementations follow Algorithms 5, 6, and 7 of SEW-Mimic and were
cross-checked against the independent IK-Geo reference implementation:
https://github.com/rpiRobotics/ik-geo

For Algorithm 7, ``x_tilde = A.T * b`` is the agreed interpretation of the
paper's inconsistent pseudocode, backed by the referenced IK-Geo implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

_DEGENERACY_TOL = 1e-12
SP3_EXACT_TOL = 1e-10
_SP3_INTERSECTION_TOL = 64.0 * np.finfo(float).eps
_SP3_PROJECTION_TOL = 64.0 * np.finfo(float).eps


@dataclass(frozen=True)
class SP3Result:
    """Deterministic candidates returned by :func:`sp3`.

    ``residuals`` are the independently recomputed distance residuals,
    ``abs(norm(rot(k, theta) @ p1 - p2) - d)``.  A candidate is exact only
    when its residual is within the scale-aware SP3 tolerance.
    """

    angles: tuple[float, ...]
    is_exact: tuple[bool, ...]
    residuals: tuple[float, ...]
    degenerate: bool
    message: Optional[str] = None


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


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the half-open interval ``[-pi, pi)``."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    # Avoid returning positive zero from a platform-dependent modulo path.
    return 0.0 if wrapped == 0.0 else wrapped


def _stable_norm(vector: Vector) -> float:
    """Return a finite Euclidean norm without intermediate overflow."""
    values = tuple(float(component) for component in vector)
    if not all(math.isfinite(component) for component in values):
        raise ValueError("SP3 encountered a non-finite intermediate vector")
    return math.hypot(*values)


def _sp3_dot(vector1: Vector, vector2: Vector) -> float:
    """Compute a three-dimensional dot product with compensated summation."""
    return math.fsum(float(a) * float(b) for a, b in zip(vector1, vector2))


def _sp3_unit(value: ArrayLike, name: str) -> Vector:
    """Normalize an SP3 axis without overflowing its norm calculation."""
    vector = _vector3(value, name)
    component_scale = max(abs(float(component)) for component in vector)
    if component_scale == 0.0:
        raise ValueError(f"{name} must be nonzero")
    scaled = vector / component_scale
    scaled_norm = _stable_norm(scaled)
    if scaled_norm <= _DEGENERACY_TOL / component_scale:
        raise ValueError(f"{name} must be nonzero")
    return scaled / scaled_norm


def _sp3_residual(
    p1: Vector,
    p2: Vector,
    axis: Vector,
    distance: float,
    angle: float,
    length_scale: float,
) -> float:
    if not math.isfinite(angle):
        raise ValueError("SP3 produced a non-finite candidate angle")
    normalized_residual = abs(_stable_norm(rot(axis, angle) @ p1 - p2) - distance)
    residual = normalized_residual * length_scale
    if not math.isfinite(residual):
        raise ValueError("SP3 residual is outside the representable floating-point range")
    return residual


def sp3(
    p1: ArrayLike,
    p2: ArrayLike,
    k: ArrayLike,
    d: float,
    *,
    exact_tolerance: float = SP3_EXACT_TOL,
) -> SP3Result:
    """Solve IK-Geo Subproblem 3 (circle and sphere).

    The problem is

    ``||rot(k, theta) @ p1 - p2|| = d``.

    The perpendicular components of ``p1`` and ``p2`` to the normalized axis
    define a sinusoidal dot-product equation.  Its two analytic roots are
    returned in wrapped ascending angle order.  If the circle and sphere do
    not intersect, the deterministic extremum gives the continuous
    least-squares candidate.  Every result is classified from an independent
    Rodrigues/FK residual rather than from the analytic discriminant.

    This is the direct circle-sphere reduction used by IK-Geo Subproblem 3;
    it intentionally does not call :func:`sp4`, whose plane-normal
    normalization changes the meaning of this problem.
    """
    vector1 = _vector3(p1, "p1")
    vector2 = _vector3(p2, "p2")
    axis = _sp3_unit(k, "k")
    distance = _scalar(d, "d")
    if distance < 0.0:
        raise ValueError("d must be nonnegative")
    tolerance = _scalar(exact_tolerance, "exact_tolerance")
    if tolerance < 0.0:
        raise ValueError("exact_tolerance must be nonnegative")

    length_scale = max(
        1.0,
        *(abs(float(component)) for component in vector1),
        *(abs(float(component)) for component in vector2),
        distance,
    )
    scaled_vector1 = vector1 / length_scale
    scaled_vector2 = vector2 / length_scale
    scaled_distance = distance / length_scale
    norm1 = _stable_norm(scaled_vector1)
    norm2 = _stable_norm(scaled_vector2)
    axis_coordinate1 = _sp3_dot(axis, scaled_vector1)
    axis_coordinate2 = _sp3_dot(axis, scaled_vector2)
    p1_parallel = axis_coordinate1 * axis
    p2_parallel = axis_coordinate2 * axis
    p1_perpendicular = scaled_vector1 - p1_parallel
    p2_perpendicular = scaled_vector2 - p2_parallel
    radius1 = _stable_norm(p1_perpendicular)
    radius2 = _stable_norm(p2_perpendicular)
    axial_difference = axis_coordinate1 - axis_coordinate2
    effective_scale = max(
        radius1,
        radius2,
        abs(axial_difference),
        scaled_distance,
    )
    if effective_scale == 0.0:
        effective_scale = 1.0
    normalized_exact_tolerance = tolerance * max(
        1.0 / length_scale,
        effective_scale,
    )

    def make_result(
        candidate_angles: list[float],
        *,
        degenerate: bool,
        message: Optional[str] = None,
        force_inexact: bool = False,
    ) -> SP3Result:
        unique_angles = sorted(_wrap_angle(angle) for angle in candidate_angles)
        deduplicated: list[float] = []
        for angle in unique_angles:
            if not deduplicated:
                deduplicated.append(angle)
                continue
            cyclic_difference = min(
                abs(angle - deduplicated[-1]),
                2.0 * math.pi - abs(angle - deduplicated[-1]),
            )
            if cyclic_difference > _DEGENERACY_TOL:
                deduplicated.append(angle)
        if len(deduplicated) > 1:
            cyclic_difference = min(
                abs(deduplicated[0] - deduplicated[-1]),
                2.0 * math.pi - abs(deduplicated[0] - deduplicated[-1]),
            )
            if cyclic_difference <= _DEGENERACY_TOL:
                deduplicated.pop()

        residuals = tuple(
            _sp3_residual(
                scaled_vector1,
                scaled_vector2,
                axis,
                scaled_distance,
                angle,
                length_scale,
            )
            for angle in deduplicated
        )
        exact = tuple(
            residual / length_scale <= normalized_exact_tolerance and not force_inexact
            for residual in residuals
        )
        return SP3Result(
            angles=tuple(deduplicated),
            is_exact=exact,
            residuals=residuals,
            degenerate=degenerate,
            message=message,
        )

    # With either perpendicular radius numerically zero, the rotated endpoint
    # has no resolvable angular dependence.  Angle zero is the canonical
    # witness.
    if (
        radius1 <= _SP3_PROJECTION_TOL * norm1
        or radius2 <= _SP3_PROJECTION_TOL * norm2
    ):
        constant_residual = _sp3_residual(
            scaled_vector1,
            scaled_vector2,
            axis,
            scaled_distance,
            0.0,
            length_scale,
        )
        if constant_residual / length_scale <= normalized_exact_tolerance:
            message = (
                "perpendicular radius is numerically zero; "
                "rotation is underdetermined"
            )
        else:
            message = (
                "perpendicular radius is numerically zero; returned "
                "constant-distance least-squares representative"
            )
        return make_result([0.0], degenerate=True, message=message)

    unit_perpendicular1 = p1_perpendicular / radius1
    unit_perpendicular2 = p2_perpendicular / radius2
    alpha = _sp3_dot(unit_perpendicular1, unit_perpendicular2)
    beta = _sp3_dot(np.cross(axis, unit_perpendicular1), unit_perpendicular2)
    # Normalize the effective geometry once more before squaring.  This keeps
    # small perpendicular radii and axial differences representable after
    # the initial input-component scaling.
    effective_radius1 = radius1 / effective_scale
    effective_radius2 = radius2 / effective_scale
    effective_axial_difference = axial_difference / effective_scale
    effective_distance = scaled_distance / effective_scale
    target = 0.5 * math.fsum(
        (
            effective_radius1 * effective_radius1,
            effective_radius2 * effective_radius2,
            effective_axial_difference * effective_axial_difference,
            -(effective_distance * effective_distance),
        )
    )

    phase = math.atan2(beta, alpha)
    normalized_target = target / effective_radius1 / effective_radius2
    if normalized_target > 1.0 + _SP3_INTERSECTION_TOL:
        return make_result(
            [phase],
            degenerate=False,
            message="no circle-sphere intersection; returned least-squares extremum",
            force_inexact=True,
        )
    if normalized_target < -1.0 - _SP3_INTERSECTION_TOL:
        return make_result(
            [phase + math.pi],
            degenerate=False,
            message="no circle-sphere intersection; returned least-squares extremum",
            force_inexact=True,
        )

    # Clipping is restricted to the explicitly justified roundoff band.
    clipped_target = min(1.0, max(-1.0, normalized_target))
    delta = math.acos(clipped_target)
    return make_result([phase - delta, phase + delta], degenerate=False)
