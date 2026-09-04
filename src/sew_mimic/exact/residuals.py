"""Physical residuals for the real Gen3 pinch-site Exact-SEW task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ..angles import angular_difference
from ..common import ExactSewTarget, gen3_end_effector_pose
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, StereoSewSingularityError

Vector = NDArray[np.float64]
_SO3_TOL = 1e-10


def _rotation3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite with shape (3, 3)")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=_SO3_TOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=_SO3_TOL, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return matrix


def so3_log(rotation: ArrayLike) -> Vector:
    """Return the principal rotation vector of a proper rotation matrix.

    The sign convention is ``Log(R_target.T @ R_actual)``.  Its norm is the
    SO(3) geodesic error; SciPy's rotation-vector implementation is stable at
    both identity and the principal-angle pi boundary.
    """
    return Rotation.from_matrix(_rotation3(rotation, "rotation")).as_rotvec()


@dataclass(frozen=True)
class ExactSewResiduals:
    """Unscaled physical residual components in metres and radians."""

    position: Vector
    rotation: Vector
    sew: float | None
    actual_position: Vector
    actual_rotation: NDArray[np.float64]
    actual_psi: float | None

    def __post_init__(self) -> None:
        for name, shape in (("position", (3,)), ("rotation", (3,)),
                            ("actual_position", (3,)), ("actual_rotation", (3, 3))):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.sew is not None and not np.isfinite(self.sew):
            raise ValueError("sew must be finite when present")
        if self.actual_psi is not None and not np.isfinite(self.actual_psi):
            raise ValueError("actual_psi must be finite when present")

    @property
    def position_error_m(self) -> float:
        return float(np.linalg.norm(self.position))

    @property
    def orientation_error_rad(self) -> float:
        return float(np.linalg.norm(self.rotation))

    @property
    def sew_error_rad(self) -> float | None:
        return None if self.sew is None else abs(float(self.sew))


def robot_exact_sew_residuals(
    q: ArrayLike,
    target: ExactSewTarget,
    robot: Gen3Kinematics,
    geometry: Gen3StereoSewGeometry,
    stereo: StereoSew,
    *,
    include_sew: bool = True,
) -> ExactSewResiduals:
    """Evaluate only authoritative aligned pinch FK and validated SEW points."""
    if not isinstance(target, ExactSewTarget):
        raise ValueError("target must be an ExactSewTarget")
    position, rotation = gen3_end_effector_pose(q, robot)
    rotation_residual = so3_log(target.rotation.T @ rotation)
    if not include_sew:
        return ExactSewResiduals(position - target.position, rotation_residual, None,
                                 position, rotation, None)
    points = geometry.sew_points(q)
    psi = stereo.forward(points.shoulder, points.elbow, points.wrist)
    return ExactSewResiduals(
        position - target.position, rotation_residual,
        angular_difference(psi, target.psi), position, rotation, psi,
    )
