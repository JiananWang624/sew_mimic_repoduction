import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation

from sew_mimic.common import ExactSewTarget, gen3_end_effector_pose
from sew_mimic.exact import (
    ExactSewCandidate,
    ExactSewCandidateSet,
    R2R2R2RSearchConfig,
    NativeStereoSewTarget,
    enumerate_exact_sew_candidates,
    to_native_stereo_sew_target,
)
from sew_mimic.exact.stereo_backend import _represent, _valid_vector
from sew_mimic.exact.root_search import RootSearchConfig
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    project_stereo_sew_reference,
    sample_gen3_configurations,
)


def test_native_target_validates_and_owns_immutable_arrays():
    target = NativeStereoSewTarget(np.zeros(3), np.eye(3), 0.2)
    assert not target.position.flags.writeable
    assert not target.rotation_07.flags.writeable
    with pytest.raises(ValueError):
        NativeStereoSewTarget(np.zeros(3), np.diag([1.0, 1.0, -1.0]), 0.0)


def test_conversion_matches_native_poe_for_500_authoritative_fk_configurations():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for q in sample_gen3_configurations(robot, 500, 20260910):
        position, aligned_rotation = gen3_end_effector_pose(q, robot)
        points = geometry.sew_points(q)
        target = ExactSewTarget(
            position,
            aligned_rotation,
            stereo.forward(points.shoulder, points.elbow, points.wrist),
        )
        native = to_native_stereo_sew_target(target, robot, geometry)
        expected = geometry.forward(q)
        expected_rotation_07 = expected[:3, :3] @ geometry.R_7T.T
        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(native.position - expected[:3, 3])),
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            float(
                Rotation.from_matrix(
                    native.rotation_07.T @ expected_rotation_07
                ).magnitude()
            ),
        )
        assert native.psi == target.psi
    assert maximum_position_error < 1e-10
    assert maximum_rotation_error < 1e-10


def _known_problem():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.array([.2, .3, -.4, .5, -.2, .3, .4])
    position, rotation = gen3_end_effector_pose(q, robot)
    points = geometry.sew_points(q)
    return robot, geometry, stereo, q, ExactSewTarget(position, rotation, stereo.forward(points.shoulder, points.elbow, points.wrist))


def _target_from_q(robot, geometry, stereo, q):
    position, rotation = gen3_end_effector_pose(q, robot)
    points = geometry.sew_points(q)
    return ExactSewTarget(position, rotation, stereo.forward(points.shoulder, points.elbow, points.wrist))


def test_event_mode_recovers_pinned_exact_candidate_in_deterministic_order():
    robot, geometry, stereo, q, target = _known_problem()
    first = enumerate_exact_sew_candidates(target, robot, geometry, stereo)
    second = enumerate_exact_sew_candidates(target, robot, geometry, stereo)
    assert first.search_root_count >= 1
    assert first.exact_geometric_count >= 1
    assert any(candidate.wrist_search_angle == pytest.approx(-.365375158960394, abs=1e-10) for candidate in first.candidates)
    assert all(candidate.position_error_m < 1e-6 for candidate in first.candidates)
    assert all(candidate.orientation_error_rad < 1e-6 for candidate in first.candidates)
    assert all(candidate.sew_error_rad < 1e-5 for candidate in first.candidates)
    assert [(candidate.wrist_search_angle, candidate.search_branch, candidate.q.tolist()) for candidate in first.candidates] == sorted((candidate.wrist_search_angle, candidate.search_branch, candidate.q.tolist()) for candidate in first.candidates)
    assert [(c.wrist_search_angle, c.search_branch, c.q.tolist()) for c in first.candidates] == [(c.wrist_search_angle, c.search_branch, c.q.tolist()) for c in second.candidates]


@pytest.mark.parametrize("samples", [200, 400, 800])
def test_reference_fixed_grid_remains_empty_for_pinned_narrow_case(samples):
    robot, geometry, stereo, _, target = _known_problem()
    config = R2R2R2RSearchConfig(mode="reference_fixed_grid", reference_fixed_grid=RootSearchConfig(samples=samples))
    result = enumerate_exact_sew_candidates(target, robot, geometry, stereo, search_config=config)
    assert not result.candidates


def test_production_backend_does_not_depend_on_oracle_or_least_squares():
    source = (Path(__file__).parents[1] / "src" / "sew_mimic" / "exact" / "stereo_backend.py").read_text(encoding="utf-8")
    assert "numerical_oracle" not in source
    assert "least_squares" not in source


def test_candidate_contracts_and_periodic_representatives():
    robot, _, _, _, _ = _known_problem()
    q = np.zeros(7)
    represented, valid, _ = _represent(q + 8 * np.pi, robot)
    assert valid
    assert np.allclose(represented[[0, 2, 4, 6]], 0.0)
    outside = q.copy()
    outside[1] = np.pi
    represented, valid, _ = _represent(outside, robot)
    assert not valid
    assert represented[1] == pytest.approx(-np.pi)
    assert represented[1] != pytest.approx(robot.joint_limits[1, 0])
    candidate = ExactSewCandidate(q, 0.0, 0, 0.0, 0.0, 0.0, True, 0.0, True, {})
    assert not candidate.q.flags.writeable
    with pytest.raises(ValueError):
        ExactSewCandidate(np.zeros(6), 0.0, 0, 0.0, 0.0, 0.0, True, 0.0, True, {})
    with pytest.raises(ValueError):
        ExactSewCandidateSet((candidate,), -1, 1, 1, {}, 0.0)
    with pytest.raises(ValueError, match="booleans"):
        ExactSewCandidate(q, 0.0, 0, 0.0, 0.0, 0.0, 1, 0.0, True, {})


def test_subproblem_vector_postcheck_rejects_approximate_relation():
    expected = np.array([1.0, 0.0, 0.0])
    assert _valid_vector(expected, expected + np.array([1e-12, 0.0, 0.0]))
    assert not _valid_vector(expected, expected + np.array([1e-6, 0.0, 0.0]))


def test_far_target_never_emits_an_exact_candidate():
    robot, geometry, stereo, _, target = _known_problem()
    impossible = ExactSewTarget(
        target.position + np.array([100.0, 100.0, 100.0]),
        target.rotation,
        target.psi,
    )
    result = enumerate_exact_sew_candidates(impossible, robot, geometry, stereo)
    assert not result.candidates
    assert result.exact_geometric_count == 0


def test_tangent_q45_branch_keeps_companion_slot_inactive():
    robot, geometry, stereo, q, _ = _known_problem()
    q[4] = 0.0
    result = enumerate_exact_sew_candidates(
        _target_from_q(robot, geometry, stereo, q), robot, geometry, stereo
    )
    slots = {candidate.search_branch for candidate in result.candidates}
    assert 6 in slots
    assert 7 not in slots


def test_near_lower_limit_has_joint_valid_exact_candidate():
    robot, geometry, stereo, _, _ = _known_problem()
    q = np.where(np.isfinite(robot.joint_limits[:, 0]), robot.joint_limits[:, 0] + 1e-5, .1)
    target = _target_from_q(robot, geometry, stereo, q)
    result = enumerate_exact_sew_candidates(target, robot, geometry, stereo)
    valid = [candidate for candidate in result.candidates if candidate.joint_limit_valid]
    assert valid
    assert min(candidate.position_error_m for candidate in valid) < 1e-6
    assert min(candidate.orientation_error_rad for candidate in valid) < 1e-6
    assert min(candidate.sew_error_rad for candidate in valid) < 1e-5
