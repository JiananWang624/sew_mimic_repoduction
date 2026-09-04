import json

import numpy as np

from sew_mimic.common import HumanArmTarget, SolverDiagnostics, SolverResult, SolverStatus, gen3_end_effector_pose
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.pipeline import evaluate_result
from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference


def test_evaluator_uses_true_pinch_fk_and_preserves_status():
    robot = gen3_kinematics(); geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference()); q = np.zeros(7)
    p, R = gen3_end_effector_pose(q, robot)
    target = HumanArmTarget([0, 0, 0], [0, 1, 0], [1, 0, 0], R, p)
    result = SolverResult(
        "test",
        SolverStatus.SUCCESS_APPROX,
        q,
        SolverDiagnostics(position_error_m=99.0, solve_time_ms=2.0),
    )
    row = evaluate_result(3, "method", result, target, robot, geometry, stereo)
    assert row.status == SolverStatus.SUCCESS_APPROX.value
    assert row.ee_position_error_mm == 0.0
    assert row.ee_orientation_error_deg == 0.0
    assert row.joint_limit_valid is True
    assert json.loads(row.diagnostics_json)["position_error_m"] == 99.0


def test_failed_result_has_nan_joint_and_error_fields():
    robot = gen3_kinematics(); geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    target = HumanArmTarget([0, 0, 0], [0, 1, 0], [1, 0, 0], np.eye(3), [1, 0, 0])
    result = SolverResult(
        "x",
        SolverStatus.UNREACHABLE,
        None,
        SolverDiagnostics(metadata={"reason": "outside workspace"}),
    )
    row = evaluate_result(0, "method", result, target, robot, geometry, stereo)
    assert all(np.isnan(value) for value in row.q)
    assert np.isnan(row.ee_position_error_mm) and row.joint_limit_valid is None
    assert json.loads(row.diagnostics_json)["metadata"]["reason"] == "outside workspace"
