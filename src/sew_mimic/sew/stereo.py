"""Stereographic shoulder-elbow-wrist (SEW) redundancy representation.

This is a small, explicit reimplementation of the construction in
``stereo-sew/IK_helpers/sew_stereo.m`` (commit d691747).  It intentionally
only represents the SEW half-plane; it does not solve an elbow position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..angles import wrap_to_pi
from ..geometry import rot


Vector = NDArray[np.float64]

# These are dimensionless: all geometric direction tests operate after unit
# normalization.  64 eps rejects only directions indistinguishable at normal
# floating-point precision, without embedding a dataset length scale.
REFERENCE_VALIDATION_TOL = 1e-10
_DIRECTION_DEGENERACY_TOL = 64.0 * np.finfo(float).eps


class StereoSewSingularityError(ValueError):
    """Raised when a valid input is at a geometric Stereo-SEW singularity."""


def _vector3(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _unit(vector: Vector, name: str) -> Vector:
    """Normalize a finite vector using its own scale, rejecting only zero."""
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise StereoSewSingularityError(f"{name} has zero length")
    scaled = vector / scale
    norm = math.hypot(*(float(component) for component in scaled))
    if norm == 0.0:  # Defensive: the finite, nonzero-scale case cannot reach this.
        raise StereoSewSingularityError(f"{name} has zero length")
    return scaled / norm


def _readonly(vector: Vector) -> Vector:
    # Back the array with immutable ``bytes`` so callers cannot undo the
    # read-only flag with ``setflags(write=True)`` after validation.
    return np.frombuffer(np.asarray(vector, dtype=np.float64).tobytes(), dtype=np.float64)


@dataclass(frozen=True)
class StereoSewReference:
    """Explicit stereographic reference vectors.

    ``e_t`` is the stereographic translation/pole direction (official MATLAB
    ``R``) and ``e_r`` is the reference direction (official MATLAB ``V``).
    Both must be unit and mutually orthogonal within
    :data:`REFERENCE_VALIDATION_TOL`; accepted vectors receive only a final
    small normalization to suppress roundoff.
    """

    e_t: Vector
    e_r: Vector

    def __post_init__(self) -> None:
        e_t = _vector3(self.e_t, "e_t")
        e_r = _vector3(self.e_r, "e_r")
        norm_t = float(np.linalg.norm(e_t))
        norm_r = float(np.linalg.norm(e_r))
        if abs(norm_t - 1.0) > REFERENCE_VALIDATION_TOL:
            raise ValueError("e_t must have unit length within 1e-10")
        if abs(norm_r - 1.0) > REFERENCE_VALIDATION_TOL:
            raise ValueError("e_r must have unit length within 1e-10")
        normalized_dot = float(np.dot(e_t / norm_t, e_r / norm_r))
        if abs(normalized_dot) > REFERENCE_VALIDATION_TOL:
            raise ValueError("e_t and e_r must be orthogonal within 1e-10")
        object.__setattr__(self, "e_t", _readonly(e_t / norm_t))
        object.__setattr__(self, "e_r", _readonly(e_r / norm_r))


@dataclass(frozen=True)
class StereoSewInverseResult:
    """A transverse elbow direction and the associated oriented SEW plane."""

    elbow_direction: Vector
    plane_normal: Vector

    def __post_init__(self) -> None:
        elbow_direction = _unit(_vector3(self.elbow_direction, "elbow_direction"), "elbow_direction")
        plane_normal = _unit(_vector3(self.plane_normal, "plane_normal"), "plane_normal")
        object.__setattr__(self, "elbow_direction", _readonly(elbow_direction))
        object.__setattr__(self, "plane_normal", _readonly(plane_normal))


@dataclass(frozen=True)
class StereoSew:
    """Forward and inverse stereographic SEW mappings for one reference pair."""

    reference: StereoSewReference

    def __post_init__(self) -> None:
        if not isinstance(self.reference, StereoSewReference):
            raise ValueError("reference must be a StereoSewReference")

    def _shoulder_wrist_direction(self, shoulder: ArrayLike, wrist: ArrayLike) -> tuple[Vector, Vector]:
        s = _vector3(shoulder, "shoulder")
        w = _vector3(wrist, "wrist")
        return s, _unit(w - s, "shoulder-to-wrist vector")

    def _reference_normal(self, e_sw: Vector) -> Vector:
        # For an orthonormal reference, this is zero precisely on the
        # stereographic half-line e_SW = e_t.
        k_r = np.cross(e_sw - self.reference.e_t, self.reference.e_r)
        magnitude = float(np.linalg.norm(k_r))
        if magnitude <= _DIRECTION_DEGENERACY_TOL:
            raise StereoSewSingularityError("stereographic reference half-line is singular")
        return k_r / magnitude

    def forward(self, shoulder: ArrayLike, elbow: ArrayLike, wrist: ArrayLike) -> float:
        """Return the signed, wrapped Stereo-SEW angle for ``S, E, W``."""
        s, e_sw = self._shoulder_wrist_direction(shoulder, wrist)
        e = _vector3(elbow, "elbow")
        e_se = _unit(e - s, "shoulder-to-elbow vector")
        n_sew_raw = np.cross(e_sw, e_se)
        n_sew_magnitude = float(np.linalg.norm(n_sew_raw))
        if n_sew_magnitude <= _DIRECTION_DEGENERACY_TOL:
            raise StereoSewSingularityError("shoulder, elbow, and wrist are collinear")
        n_sew = n_sew_raw / n_sew_magnitude
        n_ref = self._reference_normal(e_sw)
        sine = float(np.dot(n_sew, np.cross(e_sw, n_ref)))
        cosine = float(np.dot(n_sew, n_ref))
        return wrap_to_pi(math.atan2(sine, cosine))

    def inverse(self, shoulder: ArrayLike, wrist: ArrayLike, psi: float) -> StereoSewInverseResult:
        """Return the reference elbow direction and oriented SEW plane for ``psi``."""
        angle = float(psi)
        if not math.isfinite(angle):
            raise ValueError("psi must be finite")
        _, e_sw = self._shoulder_wrist_direction(shoulder, wrist)
        k_r = np.cross(e_sw - self.reference.e_t, self.reference.e_r)
        k_r_magnitude = float(np.linalg.norm(k_r))
        if k_r_magnitude <= _DIRECTION_DEGENERACY_TOL:
            raise StereoSewSingularityError("stereographic reference half-line is singular")
        # cross(k_r, p_SW) in the official code differs only by a positive
        # scale from cross(k_r, e_SW), so this avoids dimensional thresholds.
        e_x_raw = np.cross(k_r, e_sw)
        e_x_magnitude = float(np.linalg.norm(e_x_raw))
        if e_x_magnitude <= _DIRECTION_DEGENERACY_TOL:
            raise StereoSewSingularityError("inverse transverse direction is singular")
        e_x = e_x_raw / e_x_magnitude
        elbow_direction = rot(e_sw, angle) @ e_x
        plane_normal_raw = np.cross(e_sw, elbow_direction)
        plane_normal = _unit(plane_normal_raw, "inverse SEW plane normal")
        return StereoSewInverseResult(elbow_direction, plane_normal)
