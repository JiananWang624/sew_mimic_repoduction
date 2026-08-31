from pathlib import Path

import mujoco
import numpy as np

from sew_mimic.kinematics import (
    Gen3Kinematics,
    R_0_i,
    T_0_i,
    ee_rotation,
    gen3_kinematics,
    load_mujoco_model,
    revolute_joint_ids,
    validate_gen3_arm,
)


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "kinova_gen3" / "scene.xml"


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    assert name is not None
    return name


def test_gen3_has_seven_controlled_revolute_joints() -> None:
    model = load_mujoco_model(MODEL_PATH)

    joint_ids = validate_gen3_arm(model)
    joint_names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in joint_ids]

    assert joint_names == [f"joint_{index}" for index in range(1, 8)]


def test_gen3_joint_axes_and_limits_match_menagerie_model() -> None:
    model = load_mujoco_model(MODEL_PATH)
    joint_ids = revolute_joint_ids(model)
    kinematics = Gen3Kinematics(model)

    np.testing.assert_allclose(kinematics.axes, model.jnt_axis[joint_ids], atol=0.0)
    np.testing.assert_array_equal(kinematics.joint_limited, [0, 1, 0, 1, 0, 1, 0])
    np.testing.assert_allclose(
        kinematics.joint_limits[[1, 3, 5]],
        [[-2.24, 2.24], [-2.57, 2.57], [-2.09, 2.09]],
    )
    np.testing.assert_array_equal(
        np.isinf(kinematics.joint_limits[[0, 2, 4, 6]]),
        np.ones((4, 2), dtype=bool),
    )


def test_fixed_parent_to_child_transforms_are_extracted_from_model() -> None:
    model = load_mujoco_model(MODEL_PATH)
    kinematics = Gen3Kinematics(model)

    for index, body_id in enumerate(kinematics.frame_body_ids[1:]):
        expected_rotation = np.empty(9)
        mujoco.mju_quat2Mat(expected_rotation, model.body_quat[body_id])
        np.testing.assert_allclose(
            kinematics.fixed_parent_to_child[index, :3, :3],
            expected_rotation.reshape(3, 3),
            atol=5e-16,
        )
        np.testing.assert_allclose(
            kinematics.fixed_parent_to_child[index, :3, 3],
            model.body_pos[body_id],
            atol=0.0,
        )


def test_gen3_body_names_match_menagerie_model() -> None:
    model = load_mujoco_model(MODEL_PATH)

    body_names = [_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in range(model.nbody)]

    assert body_names == [
        "world",
        "base_link",
        "shoulder_link",
        "half_arm_1_link",
        "half_arm_2_link",
        "forearm_link",
        "spherical_wrist_1_link",
        "spherical_wrist_2_link",
        "bracelet_link",
    ]


def _rotation_angle(rotation: np.ndarray) -> float:
    sine = 0.5 * np.linalg.norm(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    cosine = 0.5 * (np.trace(rotation) - 1.0)
    return float(np.arctan2(sine, cosine))


def test_custom_fk_matches_mujoco_for_200_random_configurations() -> None:
    model = load_mujoco_model(MODEL_PATH)
    data = mujoco.MjData(model)
    kinematics = Gen3Kinematics(model)
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    rng = np.random.default_rng(20260831)
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0

    for _ in range(200):
        q = np.empty(7)
        for index, limited in enumerate(kinematics.joint_limited):
            if limited:
                q[index] = rng.uniform(*kinematics.joint_limits[index])
            else:
                q[index] = rng.uniform(-np.pi, np.pi)

        for index, joint_id in enumerate(kinematics.joint_ids):
            data.qpos[model.jnt_qposadr[joint_id]] = q[index]
        mujoco.mj_forward(model, data)

        base_body_id = int(kinematics.frame_body_ids[0])
        world_R_base = data.xmat[base_body_id].reshape(3, 3)
        world_p_base = data.xpos[base_body_id]

        for frame_index, body_id in enumerate(kinematics.frame_body_ids):
            expected_rotation = world_R_base.T @ data.xmat[body_id].reshape(3, 3)
            expected_position = world_R_base.T @ (data.xpos[body_id] - world_p_base)
            actual_transform = kinematics.T_0_i(q, frame_index)
            position_error = np.linalg.norm(actual_transform[:3, 3] - expected_position)
            rotation_error = _rotation_angle(actual_transform[:3, :3].T @ expected_rotation)
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_rotation_error = max(maximum_rotation_error, rotation_error)

        expected_ee_rotation = world_R_base.T @ data.site_xmat[pinch_site_id].reshape(3, 3)
        expected_ee_position = world_R_base.T @ (data.site_xpos[pinch_site_id] - world_p_base)
        actual_ee_transform = kinematics.ee_transform(q)
        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(actual_ee_transform[:3, 3] - expected_ee_position)),
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            _rotation_angle(actual_ee_transform[:3, :3].T @ expected_ee_rotation),
        )

    print(f"maximum position error: {maximum_position_error:.3e} m")
    print(
        f"maximum rotation error: {maximum_rotation_error:.3e} rad "
        f"({np.degrees(maximum_rotation_error):.3e} deg)"
    )
    assert maximum_position_error < 1e-12
    assert maximum_rotation_error < 1e-12


def test_paper_notation_module_functions_use_model_derived_kinematics() -> None:
    q = np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8])
    kinematics = gen3_kinematics()

    np.testing.assert_allclose(R_0_i(q, 5), kinematics.R_0_i(q, 5), atol=0.0)
    np.testing.assert_allclose(T_0_i(q, 7), kinematics.T_0_i(q, 7), atol=0.0)
    np.testing.assert_allclose(ee_rotation(q), kinematics.ee_rotation(q), atol=0.0)
