from types import MappingProxyType
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sew_mimic.common import ExactSewTarget, HumanArmTarget, SolverStatus, gen3_end_effector_pose
from sew_mimic.exact import (
    human_arm_to_exact_sew_target,
    retarget_exact_sew,
    solve_exact_sew,
)
from sew_mimic.exact.stereo_backend import ExactSewCandidate, ExactSewCandidateSet
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference


def _candidate_set(q: np.ndarray) -> ExactSewCandidateSet:
    candidate = ExactSewCandidate(q, -.2, 3, 0., 0., 0., True, .1, True, {})
    return ExactSewCandidateSet((candidate,), 1, 1, 1, MappingProxyType({}), .1)


def test_solver_enumerates_once_and_previous_does_not_reach_backend(monkeypatch):
    import sew_mimic.exact.solver as solver

    calls = []
    monkeypatch.setattr(solver, "enumerate_exact_sew_candidates", lambda *args, **kwargs: calls.append(kwargs) or _candidate_set(np.zeros(7)))
    target = ExactSewTarget(np.zeros(3), np.eye(3), 0.)
    result = solve_exact_sew(target, object(), object(), object(), q_previous=np.ones(7))
    assert result.status is SolverStatus.SUCCESS_EXACT
    assert len(calls) == 1
    assert "q_previous" not in calls[0]
    assert result.diagnostics.position_error_m == 0.
    assert result.diagnostics.branch_id is not None
    assert result.diagnostics.metadata["constraint_set"] == "pinch_pose_plus_stereo_sew"
    assert result.diagnostics.metadata["candidate_count"] == 1


def test_invalid_policy_and_backend_value_error_are_not_conflated(monkeypatch):
    import sew_mimic.exact.solver as solver

    target = ExactSewTarget(np.zeros(3), np.eye(3), 0.)
    invalid = solve_exact_sew(target, object(), object(), object(), branch_policy="bad")
    assert invalid.status is SolverStatus.INVALID_INPUT
    monkeypatch.setattr(solver, "enumerate_exact_sew_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("backend failure")))
    broken = solve_exact_sew(target, object(), object(), object())
    assert broken.status is SolverStatus.NUMERICAL_FAILURE


def test_human_conversion_and_singular_status():
    stereo = StereoSew(project_stereo_sew_reference())
    human = HumanArmTarget(
        np.zeros(3), np.array([0., 1., 0.]), np.array([1., 1., 0.]), np.eye(3), np.array([.1, .2, .3])
    )
    target = human_arm_to_exact_sew_target(human, stereo)
    assert np.array_equal(target.position, human.task_point)
    assert np.array_equal(target.rotation, human.hand_rotation)
    assert target.psi == pytest.approx(
        stereo.forward(human.shoulder, human.elbow, human.wrist)
    )

    class SingularStereo:
        def forward(self, *args):
            from sew_mimic.sew import StereoSewSingularityError
            raise StereoSewSingularityError("test singularity")

    result = retarget_exact_sew(human, object(), object(), SingularStereo())
    assert result.status is SolverStatus.SEW_SINGULAR


def test_narrow_event_aware_target_succeeds_once():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.array([.2, .3, -.4, .5, -.2, .3, .4])
    position, rotation = gen3_end_effector_pose(q, robot)
    points = geometry.sew_points(q)
    target = ExactSewTarget(position, rotation, stereo.forward(points.shoulder, points.elbow, points.wrist))
    result = solve_exact_sew(target, robot, geometry, stereo)
    assert result.status is SolverStatus.SUCCESS_EXACT
    assert result.diagnostics.position_error_m < 1e-6
    assert result.diagnostics.orientation_error_rad < 1e-6
    assert result.diagnostics.sew_error_rad < 1e-5


def test_method2_production_modules_have_no_numerical_fallback_dependency():
    exact_directory = Path(__file__).resolve().parents[1] / "src" / "sew_mimic" / "exact"
    production = "\n".join(
        (exact_directory / name).read_text(encoding="utf-8")
        for name in ("branch_selection.py", "solver.py")
    )
    for forbidden in (
        "numerical_oracle",
        "solve_pose(",
        "solve_pose_and_sew",
        "least_squares",
    ):
        assert forbidden not in production


def test_importing_method2_api_does_not_load_method3_oracle():
    source_directory = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_directory)
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from sew_mimic.exact import solve_exact_sew; "
                "assert 'sew_mimic.exact.numerical_oracle' not in sys.modules"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
