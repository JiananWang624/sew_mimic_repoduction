"""Physical end-effector evaluation using the MuJoCo-derived Gen3 FK."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..kinematics import Gen3Kinematics
from .types import HumanArmTarget, SolverDiagnostics, SolverResult


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

_ROTATION_MATRIX_TOL = 1e-10


@dataclass(frozen=True)
class EndEffectorMetrics:
    """True pinch-site errors and signed joint-limit margin."""

    ee_position_error_m: float
    ee_position_error_mm: float
    ee_orientation_error_rad: float
    ee_orientation_error_deg: float
    joint_limit_margin_rad: float
    joint_limit_margin_deg: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _vector3(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite array with shape (3,)")
    return vector


def _rotation3(value: ArrayLike, name: str) -> Matrix:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite array with shape (3, 3)")
    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=_ROTATION_MATRIX_TOL,
        rtol=0.0,
    ) or not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=_ROTATION_MATRIX_TOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return matrix


def compute_pose_errors(
    actual_position: ArrayLike,
    actual_rotation: ArrayLike,
    target_position: ArrayLike,
    target_rotation: ArrayLike,
) -> tuple[float, float]:
    """Return position and SO(3) errors for poses in one common frame."""
    actual_p = _vector3(actual_position, "actual_position")
    target_p = _vector3(target_position, "target_position")
    actual_R = _rotation3(actual_rotation, "actual_rotation")
    target_R = _rotation3(target_rotation, "target_rotation")
    position_error_m = float(np.linalg.norm(actual_p - target_p))
    cosine = float(
        np.clip((np.trace(target_R.T @ actual_R) - 1.0) / 2.0, -1.0, 1.0)
    )
    orientation_error_rad = float(np.arccos(cosine))
    return position_error_m, orientation_error_rad


def gen3_end_effector_pose(
    q: ArrayLike,
    robot: Gen3Kinematics,
) -> tuple[Vector, Matrix]:
    """Return physical pinch position and aligned hand rotation in base frame 0."""
    transform = robot.ee_transform(q)
    position = transform[:3, 3].copy()
    aligned_rotation = transform[:3, :3] @ robot.R_robot_align
    return position, aligned_rotation


def joint_limit_margin(q: ArrayLike, robot: Gen3Kinematics) -> float:
    """Return the minimum signed distance to any finite joint limit in radians."""
    configuration = np.asarray(q, dtype=float)
    if configuration.shape != (robot.dof,) or not np.all(np.isfinite(configuration)):
        raise ValueError(f"q must be a finite array with shape ({robot.dof},)")
    lower = robot.joint_limits[:, 0]
    upper = robot.joint_limits[:, 1]
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    margins = np.concatenate(
        (
            configuration[finite_lower] - lower[finite_lower],
            upper[finite_upper] - configuration[finite_upper],
        )
    )
    if margins.size == 0:
        raise ValueError("joint_limit_margin requires at least one finite joint limit")
    return float(np.min(margins))


def evaluate_end_effector(
    q: ArrayLike,
    target: HumanArmTarget,
    robot: Gen3Kinematics,
) -> EndEffectorMetrics:
    """Evaluate one configuration and target expressed in Gen3 base frame 0."""
    if not isinstance(target, HumanArmTarget):
        raise ValueError("target must be a HumanArmTarget")
    actual_position, actual_rotation = gen3_end_effector_pose(q, robot)
    position_error_m, orientation_error_rad = compute_pose_errors(
        actual_position,
        actual_rotation,
        target.task_point,
        target.hand_rotation,
    )
    margin_rad = joint_limit_margin(q, robot)
    return EndEffectorMetrics(
        ee_position_error_m=position_error_m,
        ee_position_error_mm=1000.0 * position_error_m,
        ee_orientation_error_rad=orientation_error_rad,
        ee_orientation_error_deg=float(np.degrees(orientation_error_rad)),
        joint_limit_margin_rad=margin_rad,
        joint_limit_margin_deg=float(np.degrees(margin_rad)),
    )


def evaluate_solver_result(
    result: SolverResult,
    target: HumanArmTarget,
    robot: Gen3Kinematics,
) -> SolverResult:
    """Attach true pinch-site metrics without changing solver status semantics."""
    if result.q is None:
        return result
    metrics = evaluate_end_effector(result.q, target, robot)
    previous = result.diagnostics
    diagnostics = SolverDiagnostics(
        position_error_m=metrics.ee_position_error_m,
        orientation_error_rad=metrics.ee_orientation_error_rad,
        sew_error_rad=previous.sew_error_rad,
        joint_limit_margin_rad=metrics.joint_limit_margin_rad,
        solve_time_ms=previous.solve_time_ms,
        branch_id=previous.branch_id,
        metadata={**previous.metadata, **metrics.to_dict()},
    )
    return SolverResult(
        method=result.method,
        status=result.status,
        q=result.q,
        diagnostics=diagnostics,
        message=result.message,
    )
