"""Authoritative per-frame output rows, always measured using pinch-site FK."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import numpy as np

from ..angles import angular_difference
from ..common import HumanArmTarget, SolverResult, evaluate_end_effector
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, StereoSewSingularityError


_JOINT_LIMIT_EVALUATION_TOL_RAD = 1e-12


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class EvaluationRow:
    frame: int
    method: str
    status: str
    q: tuple[float, ...]
    ee_position_error_mm: float
    ee_orientation_error_deg: float
    sew_angle_error_deg: float
    joint_limit_valid: bool | None
    joint_limit_margin_deg: float
    branch_id: str | None
    solve_time_ms: float
    message: str | None = None
    diagnostics_json: str = "{}"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self); q = value.pop("q")
        value.update({f"q{i + 1}": q[i] for i in range(7)})
        return value


def evaluate_result(frame: int, method: str, result: SolverResult, target: HumanArmTarget,
                    robot: Gen3Kinematics, geometry: Gen3StereoSewGeometry,
                    stereo: StereoSew) -> EvaluationRow:
    """Preserve solver status, but independently calculate all reported physics."""
    nan = float("nan")
    elapsed = result.diagnostics.solve_time_ms if result.diagnostics.solve_time_ms is not None else nan
    diagnostics_json = json.dumps(
        result.diagnostics.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )
    if result.q is None:
        return EvaluationRow(frame, method, result.status.value, (nan,) * 7, nan, nan, nan,
                             None, nan, result.diagnostics.branch_id, elapsed, result.message,
                             diagnostics_json)
    metrics = evaluate_end_effector(result.q, target, robot)
    try:
        human_psi = stereo.forward(target.shoulder, target.elbow, target.wrist)
        points = geometry.sew_points(result.q)
        robot_psi = stereo.forward(points.shoulder, points.elbow, points.wrist)
        sew_error = math.degrees(abs(angular_difference(robot_psi, human_psi)))
    except StereoSewSingularityError:
        sew_error = nan
    return EvaluationRow(frame, method, result.status.value, tuple(float(x) for x in result.q),
                         metrics.ee_position_error_mm, metrics.ee_orientation_error_deg, sew_error,
                         metrics.joint_limit_margin_rad >= -_JOINT_LIMIT_EVALUATION_TOL_RAD,
                         metrics.joint_limit_margin_deg, result.diagnostics.branch_id, elapsed,
                         result.message, diagnostics_json)
