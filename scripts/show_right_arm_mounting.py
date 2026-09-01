"""Verify and display the fixed q=0 humanoid right-arm Gen3 mounting."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.config import CONFIG, project_path  # noqa: E402
from sew_mimic.csv_adapter import (  # noqa: E402
    R_BODY_FROM_CSV,
    HumanCSVAdapter,
)
from sew_mimic.mounting import (  # noqa: E402
    DEFAULT_ROBOT_WORLD_OFFSET,
    GEN3_JOINT1_IN_BASE,
    humanoid_root_rotation,
    load_humanoid_mounted_gen3,
    right_arm_base_position,
)


SHOULDER_COLUMNS = ("Shoulder_X", "Shoulder_Y", "Shoulder_Z")
_HUMAN_CSV_CONFIG = CONFIG["human_csv"]
_MOUNTING_CONFIG = CONFIG["mounting_validation"]


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
    geom.rgba = color


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=project_path(_HUMAN_CSV_CONFIG["input_path"])
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=float(_MOUNTING_CONFIG["seconds"]),
        help="0 keeps the viewer open until it is closed",
    )
    parser.add_argument("--no-viewer", action="store_true")
    arguments = parser.parse_args()
    if arguments.seconds < 0.0:
        parser.error("--seconds must be nonnegative")

    first_row = pd.read_csv(arguments.input, nrows=1)
    missing = [column for column in SHOULDER_COLUMNS if column not in first_row]
    if missing:
        raise ValueError(f"CSV is missing shoulder columns: {missing}")
    if first_row.empty:
        raise ValueError("CSV must contain at least one frame")
    shoulder_csv = first_row.loc[0, SHOULDER_COLUMNS].to_numpy(float)
    adapter = HumanCSVAdapter()
    human_shoulder_world = adapter.position_to_world(shoulder_csv)
    robot_shoulder_world = human_shoulder_world + np.asarray(
        DEFAULT_ROBOT_WORLD_OFFSET, dtype=float
    )

    root_rotation = humanoid_root_rotation()
    expected_base_position = right_arm_base_position(robot_shoulder_world)
    robot, data = load_humanoid_mounted_gen3(human_shoulder_world)
    model = robot.model
    base_body_id = int(robot.frame_body_ids[0])
    joint1_id = int(robot.joint_ids[0])
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    base_world = data.xpos[base_body_id].copy()
    joint1_world = data.xanchor[joint1_id].copy()
    joint_error = joint1_world - robot_shoulder_world
    offset_world = joint1_world - base_world
    recovered_offset_base = root_rotation.T @ offset_world
    robot_wrist = data.xanchor[int(robot.joint_ids[5])].copy()
    shoulder_to_wrist = robot_wrist - joint1_world
    shoulder_to_wrist /= np.linalg.norm(shoulder_to_wrist)

    print("R_body_from_csv:")
    print(R_BODY_FROM_CSV)
    print("shoulder CSV (mm):", np.array2string(shoulder_csv, precision=9))
    print("human shoulder world (m):", np.array2string(human_shoulder_world, precision=9))
    print("robot world offset (m):", np.array2string(np.asarray(DEFAULT_ROBOT_WORLD_OFFSET), precision=9))
    print("joint1_in_base (m):", np.array2string(GEN3_JOINT1_IN_BASE, precision=9))
    print("root rotation Rx(+90deg):")
    print(np.array2string(root_rotation, precision=9, suppress_small=True))
    print("root quaternion WXYZ:", np.array2string(model.body_quat[base_body_id], precision=9))
    print("computed base position (m):", np.array2string(expected_base_position, precision=9))
    print("MuJoCo base origin (m):", np.array2string(base_world, precision=9))
    print("joint1 world anchor (m):", np.array2string(joint1_world, precision=9))
    print("joint1 target error vector (m):", np.array2string(joint_error, precision=12))
    print(f"joint1 target error norm (m): {np.linalg.norm(joint_error):.3e}")
    print("recovered joint1 offset in base (m):", np.array2string(recovered_offset_base, precision=9))
    print("q=0 shoulder->wrist world direction:", np.array2string(shoulder_to_wrist, precision=9))

    if arguments.no_viewer:
        return 0

    print("viewer markers: human shoulder=red, robot joint1=blue, base origin=green")
    print("base-to-joint1 offset arrow=yellow; world +X/+Y/+Z=red/green/blue")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.user_scn.ngeom = 7
        _set_sphere(
            viewer.user_scn.geoms[0],
            human_shoulder_world,
            0.032,
            (1.0, 0.2, 0.1, 0.55),
        )
        _set_sphere(
            viewer.user_scn.geoms[1],
            joint1_world,
            0.018,
            (0.1, 0.35, 1.0, 1.0),
        )
        _set_sphere(
            viewer.user_scn.geoms[2],
            base_world,
            0.022,
            (0.1, 1.0, 0.25, 1.0),
        )
        _set_arrow(
            viewer.user_scn.geoms[3],
            base_world,
            joint1_world,
            0.009,
            (1.0, 0.8, 0.05, 1.0),
        )
        for index, color in enumerate(
            ((1.0, 0.1, 0.1, 1.0), (0.1, 1.0, 0.1, 1.0), (0.1, 0.35, 1.0, 1.0))
        ):
            _set_arrow(
                viewer.user_scn.geoms[index + 4],
                np.zeros(3),
                np.eye(3)[index] * 0.25,
                0.007,
                color,
            )

        viewer.cam.lookat[:] = human_shoulder_world + np.asarray(
            _MOUNTING_CONFIG["camera_lookat_offset_m"], dtype=float
        )
        viewer.cam.distance = float(_MOUNTING_CONFIG["camera_distance_m"])
        viewer.cam.azimuth = float(_MOUNTING_CONFIG["camera_azimuth_deg"])
        viewer.cam.elevation = float(_MOUNTING_CONFIG["camera_elevation_deg"])
        start = time.perf_counter()
        while viewer.is_running():
            if arguments.seconds and time.perf_counter() - start >= arguments.seconds:
                break
            viewer.sync()
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
