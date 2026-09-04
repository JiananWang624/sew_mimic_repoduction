import numpy as np
import mujoco

from sew_mimic.common import (
    HumanArmTarget,
    SolverStatus,
    compute_pose_errors,
    evaluate_end_effector,
    evaluate_solver_result,
    gen3_end_effector_pose,
    joint_limit_margin,
)
from sew_mimic.geometry import rot
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.mounting import (
    load_humanoid_mounted_gen3,
    world_trajectory_to_base,
)
from sew_mimic.retarget import sew_mimic
from sew_mimic.sew import solve_legacy_sew_mimic


def _target_with_pose(position: np.ndarray, rotation: np.ndarray) -> HumanArmTarget:
    return HumanArmTarget(
        shoulder=[0.0, 0.0, 0.0],
        elbow=[0.1, 0.0, 0.0],
        wrist=[0.2, 0.0, 0.0],
        hand_rotation=rotation,
        task_point=position,
    )


def test_q_generated_target_has_zero_true_pinch_pose_error() -> None:
    robot = gen3_kinematics()
    q = np.array([0.4, -0.6, 0.3, 0.8, -0.5, 0.7, -0.2])
    position, rotation = gen3_end_effector_pose(q, robot)

    metrics = evaluate_end_effector(q, _target_with_pose(position, rotation), robot)

    assert metrics.ee_position_error_m == 0.0
    assert metrics.ee_position_error_mm == 0.0
    assert metrics.ee_orientation_error_rad < 3e-8
    assert metrics.ee_orientation_error_deg < 2e-6


def test_artificial_task_displacement_has_expected_position_error() -> None:
    robot = gen3_kinematics()
    q = np.array([0.2, -0.4, 0.1, 0.7, -0.3, 0.5, -0.1])
    position, rotation = gen3_end_effector_pose(q, robot)
    displacement = np.array([0.03, -0.04, 0.12])

    metrics = evaluate_end_effector(
        q,
        _target_with_pose(position + displacement, rotation),
        robot,
    )

    np.testing.assert_allclose(
        metrics.ee_position_error_m, np.linalg.norm(displacement), atol=2e-16
    )
    np.testing.assert_allclose(
        metrics.ee_position_error_mm,
        1000.0 * np.linalg.norm(displacement),
        atol=2e-13,
    )


def test_pose_errors_are_invariant_under_nontrivial_common_rigid_transform() -> None:
    actual_position = np.array([0.2, -0.4, 0.8])
    target_position = np.array([-0.1, 0.3, 0.5])
    actual_rotation = rot([0.2, -0.3, 0.7], 0.8)
    target_rotation = rot([-0.6, 0.4, 0.1], -0.5)
    frame_rotation = rot([0.3, 0.8, -0.2], 1.1)
    frame_translation = np.array([1.2, -0.7, 0.4])

    errors = compute_pose_errors(
        actual_position, actual_rotation, target_position, target_rotation
    )
    transformed_errors = compute_pose_errors(
        frame_rotation @ actual_position + frame_translation,
        frame_rotation @ actual_rotation,
        frame_rotation @ target_position + frame_translation,
        frame_rotation @ target_rotation,
    )

    np.testing.assert_allclose(transformed_errors, errors, atol=5e-16)


def test_mounted_world_and_native_base_pose_errors_are_identical() -> None:
    robot, data = load_humanoid_mounted_gen3(
        [0.3, -0.2, 0.8], robot_world_offset=[0.12, -0.08, 0.30]
    )
    q = np.array([0.3, -0.5, 0.2, 0.9, -0.4, 0.6, -0.1])
    for index, joint_id in enumerate(robot.joint_ids):
        data.qpos[robot.model.jnt_qposadr[joint_id]] = q[index]
    mujoco.mj_forward(robot.model, data)
    site_id = mujoco.mj_name2id(
        robot.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"
    )
    actual_world_p = data.site_xpos[site_id].copy()
    actual_world_R = (
        data.site_xmat[site_id].reshape(3, 3) @ robot.R_robot_align
    )
    actual_base_p, actual_base_R = gen3_end_effector_pose(q, robot)
    base_body_id = int(robot.frame_body_ids[0])
    world_R_base = data.xmat[base_body_id].reshape(3, 3)
    world_p_base = data.xpos[base_body_id]
    np.testing.assert_allclose(
        world_R_base @ actual_base_p + world_p_base,
        actual_world_p,
        atol=4e-16,
    )
    np.testing.assert_allclose(
        world_R_base @ actual_base_R,
        actual_world_R,
        atol=7e-16,
    )

    target_world_p = actual_world_p + np.array([0.02, -0.01, 0.03])
    target_world_R = rot([0.1, 0.5, -0.2], 0.2) @ actual_world_R
    points_world = np.tile(target_world_p, (1, 3, 1))
    points_base, rotations_base = world_trajectory_to_base(
        points_world,
        target_world_R[None, :, :],
        world_R_base,
        world_p_base,
    )
    target_base_p = points_base[0, 0]
    target_base_R = rotations_base[0]

    base_errors = compute_pose_errors(
        actual_base_p, actual_base_R, target_base_p, target_base_R
    )
    world_errors = compute_pose_errors(
        actual_world_p,
        actual_world_R,
        target_world_p,
        target_world_R,
    )

    np.testing.assert_allclose(world_errors, base_errors, atol=5e-16)


def test_joint_limit_margin_is_signed_distance_to_nearest_finite_limit() -> None:
    robot = gen3_kinematics()

    assert joint_limit_margin(np.zeros(7), robot) == 2.09
    violating = np.zeros(7)
    violating[5] = 2.19
    np.testing.assert_allclose(joint_limit_margin(violating, robot), -0.1, atol=5e-16)


def test_method_zero_q_and_status_are_unchanged_by_external_position_evaluation() -> None:
    robot = gen3_kinematics()
    desired = np.array([0.4, -0.6, 0.3, 0.8, -0.5, 0.7, -0.2])
    q0 = desired + np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01])
    upper = robot.R_0_i(desired, 3) @ robot.arm_proxy_axis(3)
    lower = robot.R_0_i(desired, 5) @ robot.arm_proxy_axis(5)
    shoulder = np.array([0.1, -0.2, 0.3])
    elbow = shoulder + 0.31 * upper
    wrist = elbow + 0.27 * lower
    hand = robot.aligned_ee_rotation(desired)
    direct_q, _ = sew_mimic(q0, shoulder, elbow, wrist, hand)
    result = solve_legacy_sew_mimic(q0, shoulder, elbow, wrist, hand)
    target = HumanArmTarget(
        shoulder,
        elbow,
        wrist,
        hand,
        task_point=robot.ee_transform(direct_q)[:3, 3] + [0.1, 0.0, 0.0],
    )

    evaluated = evaluate_solver_result(result, target, robot)

    assert result.status is SolverStatus.SUCCESS_EXACT
    assert evaluated.status is SolverStatus.SUCCESS_EXACT
    assert evaluated.diagnostics.position_error_m == 0.1
    assert evaluated.q is not None
    np.testing.assert_array_equal(evaluated.q, direct_q)
