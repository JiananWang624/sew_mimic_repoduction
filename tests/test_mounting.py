import numpy as np

from sew_mimic.csv_adapter import SHOULDER_ANCHOR_WORLD
from sew_mimic.kinematics import GEN3_SCENE_PATH, Gen3Kinematics, load_mujoco_model
from sew_mimic.mounting import (
    GEN3_JOINT1_IN_BASE,
    HUMANOID_MOUNTING_NAME,
    evaluate_root_orientations,
    humanoid_root_rotation,
    load_humanoid_mounted_gen3,
    load_mounted_gen3,
    right_arm_base_position,
    root_orientation_candidates,
    select_humanoid_mounting,
    world_trajectory_to_base,
)


def test_mounting_candidates_change_only_the_gen3_root_world_pose() -> None:
    original = Gen3Kinematics(load_mujoco_model(GEN3_SCENE_PATH))
    original_body_positions = original.model.body_pos.copy()
    original_body_quaternions = original.model.body_quat.copy()

    for rotation in root_orientation_candidates().values():
        robot, data = load_mounted_gen3(rotation, SHOULDER_ANCHOR_WORLD)

        np.testing.assert_allclose(
            data.xanchor[int(robot.joint_ids[0])], SHOULDER_ANCHOR_WORLD, atol=1e-12
        )
        np.testing.assert_array_equal(
            robot.model.body_pos[2:], original_body_positions[2:]
        )
        np.testing.assert_array_equal(
            robot.model.body_quat[2:], original_body_quaternions[2:]
        )
        np.testing.assert_array_equal(
            robot.fixed_parent_to_child, original.fixed_parent_to_child
        )
        np.testing.assert_array_equal(robot.axes, original.axes)


def test_required_candidates_have_expected_world_proxy_directions() -> None:
    evaluations = {
        evaluation.name: evaluation
        for evaluation in evaluate_root_orientations(SHOULDER_ANCHOR_WORLD)
    }

    expected = {
        "identity": [0.0, 0.0, -1.0],
        "Rx(+90deg)": [0.0, 1.0, 0.0],
        "Rx(-90deg)": [0.0, -1.0, 0.0],
        "Ry(+90deg)": [-1.0, 0.0, 0.0],
        "Ry(-90deg)": [1.0, 0.0, 0.0],
    }
    for name, direction in expected.items():
        np.testing.assert_allclose(evaluations[name].h3_world, direction, atol=5e-16)
        np.testing.assert_allclose(evaluations[name].h5_world, direction, atol=5e-16)


def test_selection_uses_established_right_arm_side_mount() -> None:
    selected = select_humanoid_mounting(
        evaluate_root_orientations(SHOULDER_ANCHOR_WORLD)
    )

    assert selected.name == "Rx(+90deg)"
    assert selected.shoulder_to_wrist_direction[1] < -0.999
    assert abs(selected.shoulder_to_wrist_direction[2]) < 0.03


def test_fixed_humanoid_mounting_is_phase_one_selection() -> None:
    np.testing.assert_allclose(
        humanoid_root_rotation(),
        root_orientation_candidates()[HUMANOID_MOUNTING_NAME],
        atol=0.0,
    )
    assert HUMANOID_MOUNTING_NAME == "Rx(+90deg)"


def test_right_arm_root_pose_uses_explicit_joint1_offset() -> None:
    robot, data = load_humanoid_mounted_gen3(SHOULDER_ANCHOR_WORLD)
    base_body_id = int(robot.frame_body_ids[0])
    joint1_world = data.xanchor[int(robot.joint_ids[0])]
    expected_base = right_arm_base_position(SHOULDER_ANCHOR_WORLD)

    np.testing.assert_allclose(
        expected_base,
        [-0.448858765, 0.092386741, 0.342198364],
        atol=5e-16,
    )
    np.testing.assert_allclose(data.xpos[base_body_id], expected_base, atol=5e-16)
    np.testing.assert_allclose(joint1_world, SHOULDER_ANCHOR_WORLD, atol=5e-16)
    np.testing.assert_allclose(
        robot.model.body_quat[base_body_id],
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
        atol=5e-16,
    )
    recovered_offset = (
        data.xmat[base_body_id].reshape(3, 3).T
        @ (joint1_world - data.xpos[base_body_id])
    )
    np.testing.assert_allclose(recovered_offset, GEN3_JOINT1_IN_BASE, atol=5e-16)


def test_world_trajectory_to_base_round_trip() -> None:
    robot, data = load_humanoid_mounted_gen3(SHOULDER_ANCHOR_WORLD)
    base_body_id = int(robot.frame_body_ids[0])
    rotation = data.xmat[base_body_id].reshape(3, 3).copy()
    position = data.xpos[base_body_id].copy()
    points_world = np.array(
        [
            [[0.0, 0.0, 0.34], [0.1, -0.2, 0.4], [0.3, 0.1, 0.2]],
            [[0.01, 0.0, 0.35], [0.2, -0.1, 0.3], [0.4, 0.2, 0.1]],
        ]
    )
    orientations_world = np.array([np.eye(3), rotation])

    points_base, orientations_base = world_trajectory_to_base(
        points_world, orientations_world, rotation, position
    )

    reconstructed_points = points_base @ rotation.T + position
    reconstructed_orientations = np.einsum("ij,njk->nik", rotation, orientations_base)
    np.testing.assert_allclose(reconstructed_points, points_world, atol=2e-16)
    np.testing.assert_allclose(
        reconstructed_orientations, orientations_world, atol=5e-16
    )
