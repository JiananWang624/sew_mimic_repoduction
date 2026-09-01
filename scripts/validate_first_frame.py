"""Trace Algorithm 1 on CSV frame 0 in explicit body and native-base frames."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.csv_adapter import (  # noqa: E402
    R_BODY_FROM_CSV,
    R_INPUT_ALIGN,
    REQUIRED_COLUMNS,
    SHOULDER_ANCHOR_WORLD,
    HumanCSVAdapter,
)
from sew_mimic.human_input import (  # noqa: E402
    compute_lower_arm_direction,
    compute_upper_arm_direction,
    wrist_euler_to_rotation,
)
from sew_mimic.kinematics import (  # noqa: E402
    LOWER_ARM_PROXY_SIGN,
    UPPER_ARM_PROXY_SIGN,
    gen3_kinematics,
)
from sew_mimic.mounting import (  # noqa: E402
    humanoid_root_rotation,
    load_humanoid_mounted_gen3,
    world_trajectory_to_base,
)
from sew_mimic.retarget import align_axis, align_wrist, sew_mimic  # noqa: E402


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arctan2(np.linalg.norm(np.cross(first, second)), first @ second)
        )
    )


def _rotation_error_deg(actual: np.ndarray, desired: np.ndarray) -> float:
    residual = actual.T @ desired
    sine = 0.5 * np.linalg.norm(
        [
            residual[2, 1] - residual[1, 2],
            residual[0, 2] - residual[2, 0],
            residual[1, 0] - residual[0, 1],
        ]
    )
    cosine = 0.5 * (np.trace(residual) - 1.0)
    return float(np.degrees(np.arctan2(sine, cosine)))


def robot_anchor_geometry(robot, data: mujoco.MjData):
    """Return the joint_1/joint_4/joint_6 anchors and their segment directions."""
    expected_names = ("joint_1", "joint_4", "joint_6")
    indices = (0, 3, 5)
    actual_names = tuple(robot.joint_names[index] for index in indices)
    if actual_names != expected_names:
        raise ValueError(
            f"Expected anchor joints {expected_names}, found {actual_names}"
        )
    anchors = tuple(
        data.xanchor[int(robot.joint_ids[index])].copy() for index in indices
    )
    upper = anchors[1] - anchors[0]
    lower = anchors[2] - anchors[1]
    upper_norm = float(np.linalg.norm(upper))
    lower_norm = float(np.linalg.norm(lower))
    if upper_norm <= 1e-12 or lower_norm <= 1e-12:
        raise ValueError("Robot joint-anchor segments must be nonzero")
    return anchors, (upper / upper_norm, lower / lower_norm)


def geometric_angle_diagnostics(
    u_human_body: np.ndarray,
    l_human_body: np.ndarray,
    signed_h3_proxy_body: np.ndarray,
    signed_h5_proxy_body: np.ndarray,
    u_robot_pos: np.ndarray,
    l_robot_pos: np.ndarray,
) -> dict[str, float]:
    """Compute signed-proxy and physical-anchor alignment errors."""
    return {
        "angle(u_human_body, signed_h3_proxy_body)": _angle_deg(
            u_human_body, signed_h3_proxy_body
        ),
        "angle(l_human_body, signed_h5_proxy_body)": _angle_deg(
            l_human_body, signed_h5_proxy_body
        ),
        "angle(u_human_body, u_robot_pos)": _angle_deg(
            u_human_body, u_robot_pos
        ),
        "angle(l_human_body, l_robot_pos)": _angle_deg(
            l_human_body, l_robot_pos
        ),
    }


def _put_configuration(robot, data: mujoco.MjData, q: np.ndarray) -> np.ndarray:
    for index, joint_id in enumerate(robot.joint_ids):
        data.qpos[robot.model.jnt_qposadr[joint_id]] = q[index]
    mujoco.mj_forward(robot.model, data)
    return np.array(
        [data.qpos[robot.model.jnt_qposadr[joint_id]] for joint_id in robot.joint_ids]
    )


def _set_sphere(geom, position: np.ndarray, radius: float, color) -> None:
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        position,
        np.eye(3).ravel(),
        np.asarray(color),
    )


def _set_arrow(geom, start: np.ndarray, end: np.ndarray, width: float, color) -> None:
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    geom.rgba[:] = color
    geom.emission = 1.0


def _next_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo diagnostic scene has no free geometry slots")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _populate_diagnostic_geoms(
    scene,
    human_positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    robot_positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    human_directions: tuple[np.ndarray, np.ndarray],
    robot_directions: tuple[np.ndarray, np.ndarray],
    robot_position_directions: tuple[np.ndarray, np.ndarray],
    tool_origin: np.ndarray,
    desired_tool_rotation: np.ndarray,
    actual_tool_rotation: np.ndarray,
) -> None:
    human_colors = (
        (1.0, 0.25, 0.15, 1.0),
        (1.0, 0.65, 0.05, 1.0),
        (1.0, 0.95, 0.1, 1.0),
    )
    robot_colors = (
        (0.1, 0.3, 1.0, 1.0),
        (0.0, 0.85, 1.0, 1.0),
        (0.7, 0.2, 1.0, 1.0),
    )
    axis_colors = (
        (1.0, 0.1, 0.1, 1.0),
        (0.1, 1.0, 0.1, 1.0),
        (0.1, 0.35, 1.0, 1.0),
    )

    for position, color in zip(human_positions, human_colors, strict=True):
        _set_sphere(_next_geom(scene), position, 0.025, color)
    for position, color in zip(robot_positions, robot_colors, strict=True):
        _set_sphere(_next_geom(scene), position, 0.018, color)
    for index, color in enumerate(axis_colors):
        _set_arrow(
            _next_geom(scene),
            np.zeros(3),
            np.eye(3)[index] * 0.25,
            0.008,
            color,
        )

    shoulder_origin = robot_positions[0]
    elbow_origin = robot_positions[1]
    for start, end, color, width in (
        (
            shoulder_origin,
            shoulder_origin + 0.26 * human_directions[0],
            human_colors[0],
            0.016,
        ),
        (
            shoulder_origin,
            shoulder_origin + 0.20 * robot_directions[0],
            robot_colors[0],
            0.006,
        ),
        (shoulder_origin, robot_positions[1], (1.0, 0.1, 0.85, 1.0), 0.005),
        (
            elbow_origin,
            elbow_origin + 0.26 * human_directions[1],
            human_colors[1],
            0.016,
        ),
        (
            elbow_origin,
            elbow_origin + 0.20 * robot_directions[1],
            robot_colors[1],
            0.006,
        ),
        (elbow_origin, robot_positions[2], (0.2, 1.0, 0.25, 1.0), 0.005),
    ):
        _set_arrow(_next_geom(scene), start, end, width, color)

    desired_colors = axis_colors
    actual_colors = (
        (1.0, 0.2, 1.0, 1.0),
        (1.0, 0.9, 0.1, 1.0),
        (0.1, 1.0, 1.0, 1.0),
    )
    for axis_index in range(3):
        _set_arrow(
            _next_geom(scene),
            tool_origin,
            tool_origin + 0.16 * desired_tool_rotation[:, axis_index],
            0.011,
            desired_colors[axis_index],
        )
        _set_arrow(
            _next_geom(scene),
            tool_origin,
            tool_origin + 0.12 * actual_tool_rotation[:, axis_index],
            0.005,
            actual_colors[axis_index],
        )


def save_first_frame_screenshot(
    path: Path,
    robot,
    data: mujoco.MjData,
    human_positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    human_directions: tuple[np.ndarray, np.ndarray],
    desired_tool_rotation: np.ndarray,
) -> None:
    """Render q_final with markers and all six comparison arrows."""
    robot_positions, robot_position_directions = robot_anchor_geometry(robot, data)
    robot_directions = (
        UPPER_ARM_PROXY_SIGN * data.xaxis[int(robot.joint_ids[2])].copy(),
        LOWER_ARM_PROXY_SIGN * data.xaxis[int(robot.joint_ids[4])].copy(),
    )
    site_id = mujoco.mj_name2id(
        robot.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"
    )
    tool_origin = data.site_xpos[site_id].copy()
    actual_tool_rotation = (
        data.site_xmat[site_id].reshape(3, 3) @ robot.R_robot_align
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.mean(
        np.vstack((*human_positions, *robot_positions)), axis=0
    )
    camera.distance = 1.35
    camera.azimuth = 90.0
    camera.elevation = -15.0

    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        _populate_diagnostic_geoms(
            renderer.scene,
            human_positions,
            robot_positions,
            human_directions,
            robot_directions,
            robot_position_directions,
            tool_origin,
            desired_tool_rotation,
            actual_tool_rotation,
        )
        image = renderer.render()
    finally:
        renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def show_first_frame(
    robot,
    data: mujoco.MjData,
    q_final: np.ndarray,
    human_positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    human_directions: tuple[np.ndarray, np.ndarray],
    desired_tool_rotation: np.ndarray,
    q0_seconds: float,
    solution_seconds: float,
) -> None:
    """Show q0, then q_final, with the six geometric comparison arrows."""
    model = robot.model
    q0 = np.zeros(7)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = human_positions[0] + np.array([0.0, -0.2, 0.0])
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -18.0
        start_time = time.perf_counter()

        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            showing_solution = elapsed >= q0_seconds
            if solution_seconds > 0.0 and elapsed >= q0_seconds + solution_seconds:
                break
            written_q = _put_configuration(
                robot, data, q_final if showing_solution else q0
            )
            expected_q = q_final if showing_solution else q0
            if not np.allclose(written_q, expected_q, atol=0.0, rtol=0.0):
                raise RuntimeError("MuJoCo qpos changed after mj_forward")

            robot_positions, robot_position_directions = robot_anchor_geometry(
                robot, data
            )
            robot_directions = (
                UPPER_ARM_PROXY_SIGN
                * data.xaxis[int(robot.joint_ids[2])].copy(),
                LOWER_ARM_PROXY_SIGN
                * data.xaxis[int(robot.joint_ids[4])].copy(),
            )
            tool_origin = data.site_xpos[site_id].copy()
            actual_tool_rotation = (
                data.site_xmat[site_id].reshape(3, 3) @ robot.R_robot_align
            )
            viewer.user_scn.ngeom = 0
            _populate_diagnostic_geoms(
                viewer.user_scn,
                human_positions,
                robot_positions,
                human_directions,
                robot_directions,
                robot_position_directions,
                tool_origin,
                desired_tool_rotation,
                actual_tool_rotation,
            )
            viewer.sync()
            time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "test.csv")
    parser.add_argument("--q0-seconds", type=float, default=3.0)
    parser.add_argument(
        "--solution-seconds",
        type=float,
        default=0.0,
        help="0 keeps q_final visible until the viewer is closed",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=PROJECT_ROOT / "output" / "first_frame_geometric_diagnostic.png",
    )
    arguments = parser.parse_args()

    table = pd.read_csv(arguments.input, nrows=1)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if table.empty:
        raise ValueError("CSV must contain at least one frame")
    row = table.loc[0, REQUIRED_COLUMNS].to_numpy(float)
    if not np.all(np.isfinite(row)):
        raise ValueError("CSV frame 0 contains non-finite required values")
    shoulder_csv, elbow_csv, wrist_csv = row[0:3], row[3:6], row[6:9]

    np.testing.assert_allclose(
        R_BODY_FROM_CSV.T @ R_BODY_FROM_CSV, np.eye(3), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(np.linalg.det(R_BODY_FROM_CSV), 1.0, atol=0.0, rtol=0.0)
    adapter = HumanCSVAdapter()
    wrist_euler_deg = row[9:12]
    rotation_wrist_csv = wrist_euler_to_rotation(
        wrist_euler_deg,
        order="xyz",
        degrees=True,
        convention="extrinsic",
    )
    hand_body_raw = R_BODY_FROM_CSV @ rotation_wrist_csv
    shoulder_body, elbow_body, wrist_body, hand_body = adapter.adapt_frame(
        shoulder_csv, elbow_csv, wrist_csv, wrist_euler_deg
    )
    np.testing.assert_allclose(hand_body, hand_body_raw @ R_INPUT_ALIGN, atol=5e-16)
    np.testing.assert_allclose(
        shoulder_body, SHOULDER_ANCHOR_WORLD, atol=5e-10, rtol=0.0
    )
    u_body = compute_upper_arm_direction(shoulder_body, elbow_body)
    l_body = compute_lower_arm_direction(elbow_body, wrist_body)

    mounted_robot, data = load_humanoid_mounted_gen3(shoulder_body)
    base_body_id = int(mounted_robot.frame_body_ids[0])
    rotation_body_from_base = data.xmat[base_body_id].reshape(3, 3).copy()
    np.testing.assert_allclose(
        rotation_body_from_base, humanoid_root_rotation(), atol=5e-16, rtol=0.0
    )
    points_base, orientations_base = world_trajectory_to_base(
        np.array([[shoulder_body, elbow_body, wrist_body]]),
        hand_body[None, :, :],
        rotation_body_from_base,
        data.xpos[base_body_id],
    )
    shoulder_base, elbow_base, wrist_base = points_base[0]
    hand_base = orientations_base[0]
    u_base = rotation_body_from_base.T @ u_body
    l_base = rotation_body_from_base.T @ l_body
    np.testing.assert_allclose(
        u_base, compute_upper_arm_direction(shoulder_base, elbow_base), atol=5e-16
    )
    np.testing.assert_allclose(
        l_base, compute_lower_arm_direction(elbow_base, wrist_base), atol=5e-16
    )

    native_robot = gen3_kinematics()
    q0 = np.zeros(7)
    q_after_upper = q0.copy()
    q_after_upper[0:2] = align_axis(3, q_after_upper, u_base, native_robot)
    q_after_lower = q_after_upper.copy()
    q_after_lower[2:4] = align_axis(5, q_after_lower, l_base, native_robot)
    q_after_wrist = q_after_lower.copy()
    q_after_wrist[4:7] = align_wrist(q_after_wrist, hand_base, native_robot)
    q_final, diagnostics = sew_mimic(
        q0, shoulder_base, elbow_base, wrist_base, hand_base
    )
    np.testing.assert_allclose(q_final, q_after_wrist, atol=5e-15, rtol=0.0)

    h3_native_base = native_robot.R_0_i(q_final, 3) @ native_robot.axes[2]
    h5_native_base = native_robot.R_0_i(q_final, 5) @ native_robot.axes[4]
    signed_h3_proxy_body = (
        rotation_body_from_base @ (UPPER_ARM_PROXY_SIGN * h3_native_base)
    )
    signed_h5_proxy_body = (
        rotation_body_from_base @ (LOWER_ARM_PROXY_SIGN * h5_native_base)
    )
    upper_error = _angle_deg(u_body, signed_h3_proxy_body)
    lower_error = _angle_deg(l_body, signed_h5_proxy_body)
    actual_tool_body = (
        rotation_body_from_base @ native_robot.aligned_ee_rotation(q_final)
    )
    wrist_rotation_error = _rotation_error_deg(actual_tool_body, hand_body)
    wrist_axis_errors = np.array(
        [
            _angle_deg(hand_body[:, axis], actual_tool_body[:, axis])
            for axis in range(3)
        ]
    )

    written_q = _put_configuration(mounted_robot, data, q_final)
    np.testing.assert_allclose(written_q, q_final, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        data.xaxis[int(mounted_robot.joint_ids[2])],
        rotation_body_from_base @ h3_native_base,
        atol=8e-16,
    )
    np.testing.assert_allclose(
        data.xaxis[int(mounted_robot.joint_ids[4])],
        rotation_body_from_base @ h5_native_base,
        atol=8e-16,
    )
    site_id = mujoco.mj_name2id(
        mounted_robot.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"
    )
    np.testing.assert_allclose(
        data.site_xmat[site_id].reshape(3, 3) @ mounted_robot.R_robot_align,
        actual_tool_body,
        atol=8e-16,
    )
    robot_positions, robot_position_directions = robot_anchor_geometry(
        mounted_robot, data
    )
    s_robot, e_robot, w_robot = robot_positions
    u_robot_pos, l_robot_pos = robot_position_directions
    angle_diagnostics = geometric_angle_diagnostics(
        u_body,
        l_body,
        signed_h3_proxy_body,
        signed_h5_proxy_body,
        u_robot_pos,
        l_robot_pos,
    )

    print("R_body_from_csv:")
    print(R_BODY_FROM_CSV)
    print(f"det(R_body_from_csv) = {np.linalg.det(R_BODY_FROM_CSV):.1f}")
    print("R_body_from_base = Rx(+90deg):")
    print(np.array2string(rotation_body_from_base, precision=9, suppress_small=True))
    print("root position:", np.array2string(data.xpos[base_body_id], precision=9))
    print("joint1 world:", np.array2string(data.xanchor[int(mounted_robot.joint_ids[0])], precision=9))
    print("\nshoulder_csv:", np.array2string(shoulder_csv, precision=9))
    print("elbow_csv:   ", np.array2string(elbow_csv, precision=9))
    print("wrist_csv:   ", np.array2string(wrist_csv, precision=9))
    print("\nshoulder_body:", np.array2string(shoulder_body, precision=9))
    print("elbow_body:   ", np.array2string(elbow_body, precision=9))
    print("wrist_body:   ", np.array2string(wrist_body, precision=9))
    print("\nu_body:", np.array2string(u_body, precision=9))
    print("l_body:", np.array2string(l_body, precision=9))
    print(f"angle_between_u_and_l_deg: {_angle_deg(u_body, l_body):.9f}")
    print("u_base:", np.array2string(u_base, precision=9))
    print("l_base:", np.array2string(l_base, precision=9))
    print("\nq0:           ", np.array2string(q0, precision=9))
    print("q_after_upper:", np.array2string(q_after_upper, precision=9))
    print("q_after_lower:", np.array2string(q_after_lower, precision=9))
    print("q_after_wrist:", np.array2string(q_after_wrist, precision=9))
    print("q_final:      ", np.array2string(q_final, precision=9))
    print("MuJoCo qpos:  ", np.array2string(written_q, precision=9))
    print("q3/q4 changed:", not np.allclose(q_after_lower[2:4], q0[2:4]))
    print("\nh3_native_base:", np.array2string(h3_native_base, precision=9))
    print("h5_native_base:", np.array2string(h5_native_base, precision=9))
    print(
        "signed upper_proxy_body:",
        np.array2string(signed_h3_proxy_body, precision=9),
    )
    print(
        "signed lower_proxy_body:",
        np.array2string(signed_h5_proxy_body, precision=9),
    )
    print(f"upper_error_deg: {upper_error:.6e}")
    print(f"lower_error_deg: {lower_error:.6e}")
    print(f"joint_limit_valid: {diagnostics['joint_limit_valid']}")
    print("\nraw Wrist_Rx/Ry/Rz (deg):", np.array2string(wrist_euler_deg, precision=9))
    print("detected Motive convention: normalized XYZW quaternion -> extrinsic XYZ degrees")
    print("R_wrist_csv = Rz(rz) @ Ry(ry) @ Rx(rx):")
    print(np.array2string(rotation_wrist_csv, precision=9, suppress_small=True))
    print("H_body_raw = R_body_from_csv @ R_wrist_csv:")
    print(np.array2string(hand_body_raw, precision=9, suppress_small=True))
    print("R_input_align (-Y_device, +X_device, +Z_device):")
    print(np.array2string(R_INPUT_ALIGN, precision=9, suppress_small=True))
    print("H_body = H_body_raw @ R_input_align:")
    print(np.array2string(hand_body, precision=9, suppress_small=True))
    print("H_base = R_body_from_base.T @ H_body:")
    print(np.array2string(hand_base, precision=9, suppress_small=True))
    print("R_robot_align (Gen3 EE frame -> canonical tool basis):")
    print(np.array2string(native_robot.R_robot_align, precision=9, suppress_small=True))
    print("desired hand/tool axes in body frame [X Y Z]:")
    print(np.array2string(hand_body, precision=9, suppress_small=True))
    print("actual Gen3 tool axes in body frame [X Y Z]:")
    print(np.array2string(actual_tool_body, precision=9, suppress_small=True))
    print(f"total SO(3) orientation error_deg: {wrist_rotation_error:.9e}")
    print(f"X-axis pointing error_deg: {wrist_axis_errors[0]:.9e}")
    print(f"Y-axis error_deg: {wrist_axis_errors[1]:.9e}")
    print(f"Z-axis error_deg: {wrist_axis_errors[2]:.9e}")
    print("\nS_robot (joint_1):", np.array2string(s_robot, precision=9))
    print("E_robot (joint_4):", np.array2string(e_robot, precision=9))
    print("W_robot (joint_6):", np.array2string(w_robot, precision=9))
    print("u_robot_pos:", np.array2string(u_robot_pos, precision=9))
    print("l_robot_pos:", np.array2string(l_robot_pos, precision=9))
    print("\nupper-arm comparison in body frame:")
    print("human u_body:          ", np.array2string(u_body, precision=9))
    print(
        "signed h3 proxy_body:  ",
        np.array2string(signed_h3_proxy_body, precision=9),
    )
    print("joint1->joint4:        ", np.array2string(u_robot_pos, precision=9))
    print("\nlower-arm comparison in body frame:")
    print("human l_body:          ", np.array2string(l_body, precision=9))
    print(
        "signed h5 proxy_body:  ",
        np.array2string(signed_h5_proxy_body, precision=9),
    )
    print("joint4->joint6:        ", np.array2string(l_robot_pos, precision=9))
    print("\ngeometric angle diagnostics (deg):")
    for name, value in angle_diagnostics.items():
        print(f"{name}: {value:.9f}")

    if not arguments.no_viewer:
        print("\nshowing q0 first (expected nearly straight), then q_final")
        print(
            "arrows: human u=red, signed h3 proxy=blue, "
            "joint1->joint4=magenta; human l=orange, signed h5 proxy=cyan, "
            "joint4->joint6=green"
        )
        show_first_frame(
            mounted_robot,
            data,
            q_final,
            (shoulder_body, elbow_body, wrist_body),
            (u_body, l_body),
            hand_body,
            arguments.q0_seconds,
            arguments.solution_seconds,
        )
    _put_configuration(mounted_robot, data, q_final)
    save_first_frame_screenshot(
        arguments.screenshot,
        mounted_robot,
        data,
        (shoulder_body, elbow_body, wrist_body),
        (u_body, l_body),
        hand_body,
    )
    print(f"screenshot: {arguments.screenshot.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
