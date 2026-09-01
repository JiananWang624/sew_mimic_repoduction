import numpy as np

from scripts.validate_first_frame import (
    _put_configuration,
    geometric_angle_diagnostics,
    robot_anchor_geometry,
)
from sew_mimic.csv_adapter import SHOULDER_ANCHOR_WORLD
from sew_mimic.kinematics import LOWER_ARM_PROXY_SIGN, UPPER_ARM_PROXY_SIGN
from sew_mimic.mounting import load_humanoid_mounted_gen3


def test_robot_anchor_geometry_uses_actual_mujoco_joint_anchors() -> None:
    robot, data = load_humanoid_mounted_gen3(SHOULDER_ANCHOR_WORLD)
    q = np.array([0.3, -0.7, 0.2, 1.0, -0.4, 0.8, -0.2])
    _put_configuration(robot, data, q)

    anchors, directions = robot_anchor_geometry(robot, data)

    for anchor, joint_index in zip(anchors, (0, 3, 5), strict=True):
        np.testing.assert_array_equal(
            anchor, data.xanchor[int(robot.joint_ids[joint_index])]
        )
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2e-16)
    np.testing.assert_allclose(
        directions[0],
        (anchors[1] - anchors[0]) / np.linalg.norm(anchors[1] - anchors[0]),
        atol=0.0,
    )
    np.testing.assert_allclose(
        directions[1],
        (anchors[2] - anchors[1]) / np.linalg.norm(anchors[2] - anchors[1]),
        atol=0.0,
    )


def test_geometric_angle_diagnostics_reports_proxy_and_position_angles() -> None:
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    minus_x = -x

    diagnostics = geometric_angle_diagnostics(x, y, x, y, minus_x, x)

    np.testing.assert_allclose(
        tuple(diagnostics.values()),
        (0.0, 0.0, 180.0, 90.0),
        atol=0.0,
    )


def test_gen3_proxy_signs_follow_physical_limb_geometry_over_1000_poses() -> None:
    robot, data = load_humanoid_mounted_gen3(SHOULDER_ANCHOR_WORLD)
    rng = np.random.default_rng(20260901)
    cosines = np.empty((1000, 4))

    for sample_index in range(1000):
        q = np.empty(robot.dof)
        for joint_index, limited in enumerate(robot.joint_limited):
            q[joint_index] = (
                rng.uniform(*robot.joint_limits[joint_index])
                if limited
                else rng.uniform(-np.pi, np.pi)
            )
        _put_configuration(robot, data, q)
        _, (upper_position_direction, lower_position_direction) = (
            robot_anchor_geometry(robot, data)
        )
        h3_native_body = data.xaxis[int(robot.joint_ids[2])]
        h5_native_body = data.xaxis[int(robot.joint_ids[4])]
        cosines[sample_index] = (
            h3_native_body @ upper_position_direction,
            UPPER_ARM_PROXY_SIGN * h3_native_body @ upper_position_direction,
            h5_native_body @ lower_position_direction,
            LOWER_ARM_PROXY_SIGN * h5_native_body @ lower_position_direction,
        )

    medians = np.median(cosines, axis=0)
    print(
        "proxy sign median cosines: "
        f"upper native={medians[0]:.9f}, upper signed={medians[1]:.9f}, "
        f"lower native={medians[2]:.9f}, lower signed={medians[3]:.9f}"
    )
    assert UPPER_ARM_PROXY_SIGN == -1.0
    assert LOWER_ARM_PROXY_SIGN == -1.0
    assert medians[0] < -0.9
    assert medians[1] > 0.9
    assert medians[2] < -0.99
    assert medians[3] > 0.99
