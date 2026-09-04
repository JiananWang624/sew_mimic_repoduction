"""Phase-3 validation of the independent Gen3 R-2R-2R-2R forward model."""

from __future__ import annotations

import numpy as np
import mujoco

from sew_mimic.config import CONFIG, project_path
from sew_mimic.csv_adapter import load_human_trajectory_csv
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.kinematics import GEN3_SCENE_PATH, Gen3Kinematics, load_mujoco_model
from sew_mimic.mounting import load_humanoid_mounted_gen3, world_trajectory_to_base
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    angular_margins,
    project_stereo_sew_reference,
    sample_gen3_configurations,
    select_project_reference,
)
from sew_mimic.sew.gen3_geometry import rotation_geodesic_error
from sew_mimic.geometry import rot
from sew_mimic.angles import angular_difference


def _geometry() -> Gen3StereoSewGeometry:
    return Gen3StereoSewGeometry.from_robot(gen3_kinematics())


def _human_directions() -> np.ndarray:
    trajectory = load_human_trajectory_csv(project_path(CONFIG["human_csv"]["input_path"]))
    robot, data = load_humanoid_mounted_gen3(trajectory.shoulders[0])
    points = np.stack((trajectory.shoulders, trajectory.elbows, trajectory.wrists), axis=1)
    base_points, _ = world_trajectory_to_base(
        points, trajectory.hand_orientations,
        data.xmat[int(robot.frame_body_ids[0])].reshape(3, 3),
        data.xpos[int(robot.frame_body_ids[0])],
    )
    vectors = base_points[:, 2] - base_points[:, 0]
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def test_model_extracts_expected_native_ik_geo_parameters() -> None:
    geometry = _geometry()
    np.testing.assert_allclose(
        geometry.H,
        [[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0], [-1, 0, -1, 0, -1, 0, -1]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        geometry.P,
        [[0]*8, [0, -.01175, 0, -.01275, 0, -.0003501, 0, 0], [.15643, .12838, 0, .42076, 0, .31436, 0, .167455]],
        atol=1e-12,
    )
    np.testing.assert_allclose(geometry.R_7T, np.eye(3), atol=1e-12)
    assert not geometry.H.flags.writeable

    with np.testing.assert_raises_regex(ValueError, "determinant"):
        Gen3StereoSewGeometry(
            geometry.H,
            geometry.P,
            np.diag([-1.0, 1.0, 1.0]),
            geometry.structural_residuals,
        )


def test_r_2r_2r_2r_intersections_are_model_exact() -> None:
    residuals = _geometry().structural_residuals
    assert residuals.axis_unit_error < 1e-14
    assert np.max(residuals.pair_intersection_m) < 1e-12
    assert residuals.odd_axis_parallel_error_rad < 1e-12
    assert residuals.even_axis_parallel_error_rad < 1e-12


def test_extraction_rejects_nonintersecting_axis_pair() -> None:
    model = load_mujoco_model(GEN3_SCENE_PATH)
    joint_3 = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_3"))
    model.jnt_pos[joint_3, 0] += 1e-4
    with np.testing.assert_raises_regex(ValueError, "R-2R-2R-2R"):
        Gen3StereoSewGeometry.from_robot(Gen3Kinematics(model))


def test_independent_poe_matches_pinch_fk_for_1000_random_plus_special() -> None:
    robot, geometry = gen3_kinematics(), _geometry()
    random = sample_gen3_configurations(robot, 1000, 20260906)
    near_limits = np.where(robot.joint_limited[:, None], robot.joint_limits, 0.0).T
    configurations = np.vstack((np.zeros(7), np.array([.2, -.3, .4, -.5, .6, -.7, .8]), near_limits, random))
    assert len(configurations) == 1004
    position_errors, rotation_errors = [], []
    for q in configurations:
        actual, expected = geometry.forward(q), robot.ee_transform(q)
        position_errors.append(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        rotation_errors.append(rotation_geodesic_error(actual[:3, :3], expected[:3, :3]))
    assert max(position_errors) < 1e-10
    assert max(rotation_errors) < 1e-10


def test_intermediate_axis_lines_match_mujoco_derived_chain() -> None:
    robot, geometry = gen3_kinematics(), _geometry()
    data = mujoco.MjData(robot.model)
    for q in sample_gen3_configurations(robot, 50, 20260906):
        points, directions = geometry.joint_axis_lines(q)
        for index, joint_id in enumerate(robot.joint_ids):
            data.qpos[robot.model.jnt_qposadr[joint_id]] = q[index]
        mujoco.mj_forward(robot.model, data)
        # The explicit virtual points must lie on both physical axis lines.
        base = int(robot.frame_body_ids[0])
        r_world_base = data.xmat[base].reshape(3, 3)
        p_world_base = data.xpos[base]
        physical_points = (r_world_base.T @ (data.xanchor[robot.joint_ids] - p_world_base).T).T
        physical_axes = (r_world_base.T @ data.xaxis[robot.joint_ids].T).T
        for index in range(7):
            assert np.linalg.norm(np.cross(points[index] - physical_points[index], physical_axes[index])) < 1e-10
            assert float(directions[index] @ physical_axes[index]) > 1.0 - 1e-12


def test_sew_points_are_joint1_and_pair_intersection_points() -> None:
    geometry = _geometry()
    points, directions = geometry.joint_axis_lines(np.array([.2, -.3, .4, -.5, .6, -.7, .8]))
    sew = geometry.sew_points(np.array([.2, -.3, .4, -.5, .6, -.7, .8]))
    np.testing.assert_allclose(sew.shoulder, points[0], atol=0.0)
    np.testing.assert_allclose(sew.elbow, points[3], atol=0.0)
    np.testing.assert_allclose(sew.wrist, points[5], atol=0.0)
    assert np.linalg.norm(np.cross(sew.elbow - points[4], directions[4])) < 1e-12
    assert np.linalg.norm(np.cross(sew.wrist - points[6], directions[6])) < 1e-12


def test_repeated_model_extraction_is_deterministic() -> None:
    first, second = _geometry(), _geometry()
    np.testing.assert_array_equal(first.H, second.H)
    np.testing.assert_array_equal(first.P, second.P)
    np.testing.assert_array_equal(first.R_7T, second.R_7T)


def test_reference_selection_is_deterministic_and_matches_config() -> None:
    robot, geometry = gen3_kinematics(), _geometry()
    robot_directions = np.array([
        (sew.wrist - sew.shoulder) / np.linalg.norm(sew.wrist - sew.shoulder)
        for sew in (geometry.sew_points(q) for q in sample_gen3_configurations(robot, 5000, 20260906))
    ])
    first = select_project_reference(robot_directions, _human_directions())
    second = select_project_reference(robot_directions, _human_directions())
    np.testing.assert_allclose(first.reference.e_t, [0, 0, -1])
    np.testing.assert_allclose(first.reference.e_r, [1, 0, 0])
    np.testing.assert_allclose(second.reference.e_t, first.reference.e_t)
    np.testing.assert_allclose(project_stereo_sew_reference().e_t, first.reference.e_t)
    np.testing.assert_allclose(project_stereo_sew_reference().e_r, first.reference.e_r)
    assert [name for name, _ in first.candidates] == ["+x", "-x", "+y", "-y", "+z", "-z"]


def test_shared_final_reference_evaluates_human_and_robot_geometry() -> None:
    robot, geometry, reference = gen3_kinematics(), _geometry(), project_stereo_sew_reference()
    sew = StereoSew(reference)
    robot_sew = [geometry.sew_points(q) for q in sample_gen3_configurations(robot, 100, 20260906)]
    robot_directions = np.array([(x.wrist-x.shoulder)/np.linalg.norm(x.wrist-x.shoulder) for x in robot_sew])
    assert angular_margins(robot_directions, reference.e_t).exact_singular == 0
    assert angular_margins(_human_directions(), reference.e_t).exact_singular == 0
    values = [sew.forward(x.shoulder, x.elbow, x.wrist) for x in robot_sew]
    assert np.all(np.isfinite(values))


def test_final_reference_has_deterministic_sign_wrap_and_rotation_covariance() -> None:
    geometry, reference = _geometry(), project_stereo_sew_reference()
    q = np.array([.2, -.3, .4, -.5, .6, -.7, .8])
    points = geometry.sew_points(q)
    sew = StereoSew(reference)
    psi = sew.forward(points.shoulder, points.elbow, points.wrist)
    assert sew.forward(points.shoulder, points.elbow, points.wrist) == psi
    assert -np.pi <= psi < np.pi
    perturbed = geometry.sew_points(q + np.array([1e-7, 0, 0, 0, 0, 0, 0]))
    assert abs(angular_difference(sew.forward(perturbed.shoulder, perturbed.elbow, perturbed.wrist), psi)) < 1e-5
    rotation = rot([.3, -.8, .4], 1.2)
    rotated = StereoSew(type(reference)(rotation @ reference.e_t, rotation @ reference.e_r))
    assert abs(rotated.forward(rotation @ points.shoulder, rotation @ points.elbow, rotation @ points.wrist) - psi) < 1e-12


def test_final_reference_wrap_boundary_round_trip() -> None:
    geometry, sew = _geometry(), StereoSew(project_stereo_sew_reference())
    points = geometry.sew_points(np.array([.2, -.3, .4, -.5, .6, -.7, .8]))
    e_sw = points.wrist - points.shoulder
    e_sw /= np.linalg.norm(e_sw)
    for expected in (np.pi - 1e-8, -np.pi + 1e-8):
        inverse = sew.inverse(points.shoulder, points.wrist, expected)
        elbow = points.shoulder + .3 * e_sw + .5 * inverse.elbow_direction
        assert abs(angular_difference(sew.forward(points.shoulder, elbow, points.wrist), expected)) < 1e-12


def test_angular_margins_rejects_empty_and_zero_rows() -> None:
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        angular_margins(np.empty((0, 3)), [0, 0, -1])
    with np.testing.assert_raises_regex(ValueError, "zero rows"):
        angular_margins([[0, 0, 0]], [0, 0, -1])


def test_reference_selection_requires_both_evidence_sources() -> None:
    directions = np.array([[1.0, 0.0, 0.0]])
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        select_project_reference(directions, np.empty((0, 3)))
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        select_project_reference(np.empty((0, 3)), directions)
