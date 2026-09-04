"""Shared targets, evaluation helpers, and solver contracts."""

from .evaluation import (
    EndEffectorMetrics,
    compute_pose_errors,
    evaluate_end_effector,
    evaluate_solver_result,
    gen3_end_effector_pose,
    joint_limit_margin,
)
from .status import SolverStatus
from .task_point import compute_human_task_point
from .types import HumanArmTarget, SolverDiagnostics, SolverResult

__all__ = [
    "EndEffectorMetrics",
    "HumanArmTarget",
    "SolverDiagnostics",
    "SolverResult",
    "SolverStatus",
    "compute_human_task_point",
    "compute_pose_errors",
    "evaluate_end_effector",
    "evaluate_solver_result",
    "gen3_end_effector_pose",
    "joint_limit_margin",
]
