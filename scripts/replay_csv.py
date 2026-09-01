"""Retarget the complete human CSV and replay it on the mounted Gen3."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.config import CONFIG, project_path  # noqa: E402
from sew_mimic.csv_adapter import (  # noqa: E402
    R_BODY_FROM_CSV,
    WRIST_EULER_CONVENTION,
    WRIST_EULER_DEGREES,
    WRIST_EULER_ORDER,
    HumanCSVAdapter,
    HumanTrajectory,
    load_human_trajectory_csv,
)
from sew_mimic.kinematics import (  # noqa: E402
    LOWER_ARM_PROXY_SIGN,
    UPPER_ARM_PROXY_SIGN,
)
from sew_mimic.mounting import (  # noqa: E402
    HUMANOID_MOUNTING_NAME,
    load_humanoid_mounted_gen3,
    world_trajectory_to_base,
)
from sew_mimic.retarget import sew_mimic  # noqa: E402


OUTPUT_COLUMNS = (
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "upper_arm_error_deg",
    "lower_arm_error_deg",
    "wrist_error_deg",
)
SEGMENT_COLUMNS = ("bite_id", "motive_frame", "event", "event_frame_index")
_HUMAN_CSV_CONFIG = CONFIG["human_csv"]
_REPLAY_CONFIG = CONFIG["replay_csv"]


def load_segment_boundaries(path: Path) -> np.ndarray:
    """Mark transitions that are not adjacent samples of one labeled event."""
    table = pd.read_csv(path, usecols=list(SEGMENT_COLUMNS))
    if len(table) < 2:
        return np.zeros(0, dtype=bool)
    return (
        (table["bite_id"].to_numpy()[1:] != table["bite_id"].to_numpy()[:-1])
        | (table["event"].to_numpy()[1:] != table["event"].to_numpy()[:-1])
        | (np.diff(table["motive_frame"].to_numpy()) != 1)
        | (np.diff(table["event_frame_index"].to_numpy()) != 1)
    )


def trajectory_in_mounted_base(
    trajectory_world: HumanTrajectory,
    robot,
    data: mujoco.MjData,
) -> HumanTrajectory:
    """Convert adapted world data into the fixed Gen3 base frame."""
    base_body_id = int(robot.frame_body_ids[0])
    points_world = np.stack(
        (
            trajectory_world.shoulders,
            trajectory_world.elbows,
            trajectory_world.wrists,
        ),
        axis=1,
    )
    points_base, orientations_base = world_trajectory_to_base(
        points_world,
        trajectory_world.hand_orientations,
        data.xmat[base_body_id].reshape(3, 3),
        data.xpos[base_body_id],
    )
    return HumanTrajectory(
        points_base[:, 0],
        points_base[:, 1],
        points_base[:, 2],
        orientations_base,
    )


def retarget_trajectory(trajectory_base: HumanTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """Run unchanged Algorithm 1 sequentially with the previous frame as q0."""
    configurations = np.empty((len(trajectory_base), 7))
    errors_deg = np.empty((len(trajectory_base), 3))
    q_previous = np.zeros(7)

    for frame in range(len(trajectory_base)):
        shoulder = trajectory_base.shoulders[frame]
        elbow = trajectory_base.elbows[frame]
        wrist = trajectory_base.wrists[frame]
        upper = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
        lower = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        try:
            q_current, diagnostics = sew_mimic(
                q_previous,
                shoulder,
                elbow,
                wrist,
                trajectory_base.hand_orientations[frame],
            )
        except ValueError as error:
            raise ValueError(
                f"SEW-Mimic failed at CSV row {frame}: {error}; "
                f"q_prev={np.array2string(q_previous, precision=8)}, "
                f"human_u_base={np.array2string(upper, precision=8)}, "
                f"human_l_base={np.array2string(lower, precision=8)}"
            ) from error

        if not diagnostics["joint_limit_valid"]:
            raise RuntimeError(f"joint-limit violation at CSV row {frame}")
        configurations[frame] = q_current
        errors_deg[frame] = [
            diagnostics["upper_arm_error_deg"],
            diagnostics["lower_arm_error_deg"],
            diagnostics["wrist_rotation_error_deg"],
        ]
        q_previous = q_current

    return configurations, errors_deg


def save_retargeted_csv(
    output_path: Path,
    configurations: np.ndarray,
    errors_deg: np.ndarray,
) -> None:
    """Save one machine-readable row per input frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.column_stack((configurations, errors_deg))
    pd.DataFrame(values, columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def report_trajectory(
    configurations: np.ndarray,
    errors_deg: np.ndarray,
    fps: float,
    jump_threshold_rad: float,
    segment_boundaries: np.ndarray | None = None,
) -> bool:
    """Print alignment distributions and continuity diagnostics."""
    labels = ("upper arm", "lower arm", "wrist")
    print(f"frames: {len(configurations)}")
    print(f"duration at {fps:g} fps: {(len(configurations) - 1) / fps:.3f} s")
    print("orientation error distribution (degrees):")
    for index, label in enumerate(labels):
        values = errors_deg[:, index]
        print(
            f"  {label:9s} median={np.median(values):.3e}, "
            f"p95={np.percentile(values, 95):.3e}, max={np.max(values):.3e}"
        )

    if len(configurations) < 2:
        print("continuity: only one frame")
        return False

    steps = np.diff(configurations, axis=0)
    boundaries = (
        np.zeros(len(steps), dtype=bool)
        if segment_boundaries is None
        else np.asarray(segment_boundaries, dtype=bool)
    )
    if boundaries.shape != (len(steps),):
        raise ValueError("segment_boundaries must have shape (frames - 1,)")
    continuous_steps = steps[~boundaries]
    velocities = continuous_steps * fps
    print(f"labeled trajectory segments: {int(np.sum(boundaries)) + 1}")
    print("maximum continuous |delta q| per joint (rad/frame):")
    print(np.array2string(np.max(np.abs(continuous_steps), axis=0), precision=6))
    print("maximum continuous |joint velocity| per joint (rad/s):")
    print(np.array2string(np.max(np.abs(velocities), axis=0), precision=6))

    jump_locations = np.argwhere(
        (np.abs(steps) > jump_threshold_rad) & ~boundaries[:, None]
    )
    boundary_jump_count = int(
        np.sum((np.abs(steps) > jump_threshold_rad) & boundaries[:, None])
    )
    print(
        f"excluded {boundary_jump_count} threshold crossings at "
        f"{int(np.sum(boundaries))} segment boundaries"
    )
    if not len(jump_locations):
        print(f"no joint step exceeded {jump_threshold_rad:.3f} rad/frame")
        return False

    print(
        f"warning: {len(jump_locations)} joint steps exceeded "
        f"{jump_threshold_rad:.3f} rad/frame"
    )
    for frame, joint in jump_locations[:10]:
        print(
            f"  row {frame}->{frame + 1}, q{joint + 1}, "
            f"delta={steps[frame, joint]:+.6f} rad"
        )
    return True


def plot_trajectory(
    configurations: np.ndarray,
    errors_deg: np.ndarray,
    fps: float,
    output_path: Path,
    segment_boundaries: np.ndarray | None = None,
) -> None:
    """Plot errors, joint angles, and finite-difference joint velocities."""
    sample_times = np.arange(len(configurations), dtype=float) / fps
    velocities = np.zeros_like(configurations)
    if len(configurations) > 1:
        velocities[1:] = np.diff(configurations, axis=0) * fps
    boundaries = (
        np.zeros(max(len(configurations) - 1, 0), dtype=bool)
        if segment_boundaries is None
        else np.asarray(segment_boundaries, dtype=bool)
    )
    if boundaries.shape != (max(len(configurations) - 1, 0),):
        raise ValueError("segment_boundaries must have shape (frames - 1,)")
    segment_starts = np.concatenate(([False], boundaries))
    velocities[segment_starts] = np.nan
    plotted_configurations = configurations.copy()
    plotted_configurations[segment_starts] = np.nan

    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for index, label in enumerate(("upper arm", "lower arm", "wrist")):
        axes[0].plot(sample_times, np.maximum(errors_deg[:, index], 1e-16), label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("orientation error (deg)")
    axes[0].legend(ncols=3)

    for joint in range(7):
        axes[1].plot(
            sample_times, plotted_configurations[:, joint], label=f"q{joint + 1}"
        )
        axes[2].plot(sample_times, velocities[:, joint], label=f"q{joint + 1}")
    axes[1].set_ylabel("joint angle (rad)")
    axes[2].set_ylabel("joint velocity (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[1].legend(ncols=4)
    axes[2].legend(ncols=4)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"SEW-Mimic CSV trajectory — {HUMANOID_MOUNTING_NAME}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


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


def replay_in_mujoco(
    robot,
    data: mujoco.MjData,
    configurations: np.ndarray,
    trajectory_world: HumanTrajectory,
    fps: float,
    max_frames: int = 0,
) -> None:
    """Play the mounted trajectory once with human/robot comparison markers."""
    model = robot.model
    frame_period = 1.0 / fps
    frame_count = len(configurations) if max_frames == 0 else min(max_frames, len(configurations))
    axis_ends = np.eye(3) * 0.25
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
    actual_axis_colors = (
        (1.0, 0.2, 1.0, 1.0),
        (1.0, 0.9, 0.1, 1.0),
        (0.1, 1.0, 1.0, 1.0),
    )
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.user_scn.ngeom = 19
        viewer.cam.lookat[:] = trajectory_world.shoulders[0] + np.asarray(
            _REPLAY_CONFIG["camera_lookat_offset_m"], dtype=float
        )
        viewer.cam.distance = float(_REPLAY_CONFIG["camera_distance_m"])
        viewer.cam.azimuth = float(_REPLAY_CONFIG["camera_azimuth_deg"])
        viewer.cam.elevation = float(_REPLAY_CONFIG["camera_elevation_deg"])
        start_time = time.perf_counter()

        for frame in range(frame_count):
            if not viewer.is_running():
                break
            configuration = configurations[frame]
            for index, joint_id in enumerate(robot.joint_ids):
                data.qpos[model.jnt_qposadr[joint_id]] = configuration[index]
            mujoco.mj_forward(model, data)

            human_positions = (
                trajectory_world.shoulders[frame],
                trajectory_world.elbows[frame],
                trajectory_world.wrists[frame],
            )
            robot_positions = tuple(
                data.xanchor[int(robot.joint_ids[index])].copy() for index in (0, 3, 5)
            )
            human_upper = human_positions[1] - human_positions[0]
            human_upper /= np.linalg.norm(human_upper)
            human_lower = human_positions[2] - human_positions[1]
            human_lower /= np.linalg.norm(human_lower)
            robot_directions = (
                UPPER_ARM_PROXY_SIGN
                * data.xaxis[int(robot.joint_ids[2])].copy(),
                LOWER_ARM_PROXY_SIGN
                * data.xaxis[int(robot.joint_ids[4])].copy(),
            )

            for index, position in enumerate(human_positions):
                _set_sphere(viewer.user_scn.geoms[index], position, 0.025, human_colors[index])
            for index, position in enumerate(robot_positions):
                _set_sphere(
                    viewer.user_scn.geoms[index + 3], position, 0.018, robot_colors[index]
                )
            for index in range(3):
                _set_arrow(
                    viewer.user_scn.geoms[index + 6],
                    np.zeros(3),
                    axis_ends[index],
                    0.008,
                    axis_colors[index],
                )
            for geom_index, start, direction, color, width in (
                (9, human_positions[0], human_upper, human_colors[0], 0.009),
                (10, human_positions[1], human_lower, human_colors[1], 0.009),
                (11, robot_positions[0], robot_directions[0], robot_colors[0], 0.007),
                (12, robot_positions[1], robot_directions[1], robot_colors[1], 0.007),
            ):
                _set_arrow(
                    viewer.user_scn.geoms[geom_index],
                    start,
                    start + 0.22 * direction,
                    width,
                    color,
                )

            tool_origin = data.site_xpos[site_id]
            desired_tool = trajectory_world.hand_orientations[frame]
            actual_tool = data.site_xmat[site_id].reshape(3, 3) @ robot.R_robot_align
            for axis_index in range(3):
                _set_arrow(
                    viewer.user_scn.geoms[13 + 2 * axis_index],
                    tool_origin,
                    tool_origin + 0.16 * desired_tool[:, axis_index],
                    0.011,
                    axis_colors[axis_index],
                )
                _set_arrow(
                    viewer.user_scn.geoms[14 + 2 * axis_index],
                    tool_origin,
                    tool_origin + 0.12 * actual_tool[:, axis_index],
                    0.005,
                    actual_axis_colors[axis_index],
                )

            viewer.sync()
            remaining = start_time + (frame + 1) * frame_period - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=project_path(_HUMAN_CSV_CONFIG["input_path"])
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_path(_REPLAY_CONFIG["output_path"]),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=project_path(_REPLAY_CONFIG["plot_path"]),
    )
    parser.add_argument("--fps", type=float, default=float(_REPLAY_CONFIG["fps"]))
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=float(_REPLAY_CONFIG["jump_threshold_rad"]),
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=float(_HUMAN_CSV_CONFIG["position_scale_to_m"]),
    )
    parser.add_argument(
        "--rotation-body-from-csv",
        type=float,
        nargs=9,
        default=R_BODY_FROM_CSV.ravel(),
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--viewer-max-frames",
        type=int,
        default=int(_REPLAY_CONFIG["viewer_max_frames"]),
        help="0 plays every frame; a positive value is useful for a viewer smoke test",
    )
    arguments = parser.parse_args()
    if arguments.fps <= 0.0:
        parser.error("--fps must be positive")
    if arguments.jump_threshold <= 0.0:
        parser.error("--jump-threshold must be positive")
    if arguments.viewer_max_frames < 0:
        parser.error("--viewer-max-frames must be nonnegative")

    adapter = HumanCSVAdapter(
        rotation_body_from_csv=np.asarray(arguments.rotation_body_from_csv).reshape(3, 3),
        position_scale=arguments.position_scale,
    )
    trajectory_world = load_human_trajectory_csv(arguments.input, adapter)
    segment_boundaries = load_segment_boundaries(arguments.input)
    robot, data = load_humanoid_mounted_gen3(trajectory_world.shoulders[0])
    base_body_id = int(robot.frame_body_ids[0])
    print("R_body_from_csv:")
    print(adapter.rotation_body_from_csv)
    wrist_units = "degrees" if WRIST_EULER_DEGREES else "radians"
    print(
        f"wrist convention: {WRIST_EULER_CONVENTION} "
        f"{WRIST_EULER_ORDER.upper()} {wrist_units}; "
        "H_body=R_body_from_csv@R_wrist_csv@R_input_align"
    )
    print("R_input_align:")
    print(adapter.rotation_input_align)
    print(f"fixed mounting: {HUMANOID_MOUNTING_NAME}")
    print("root rotation:")
    print(data.xmat[base_body_id].reshape(3, 3))
    print("root position:", data.xpos[base_body_id])
    print("joint1 world position:", data.xanchor[int(robot.joint_ids[0])])
    print("q=0 native h3 world axis:", data.xaxis[int(robot.joint_ids[2])])
    print("q=0 native h5 world axis:", data.xaxis[int(robot.joint_ids[4])])
    print(
        "q=0 signed upper proxy:",
        UPPER_ARM_PROXY_SIGN * data.xaxis[int(robot.joint_ids[2])],
    )
    print(
        "q=0 signed lower proxy:",
        LOWER_ARM_PROXY_SIGN * data.xaxis[int(robot.joint_ids[4])],
    )

    trajectory_base = trajectory_in_mounted_base(trajectory_world, robot, data)
    configurations, errors_deg = retarget_trajectory(trajectory_base)
    save_retargeted_csv(arguments.output, configurations, errors_deg)
    plot_trajectory(
        configurations,
        errors_deg,
        arguments.fps,
        arguments.plot,
        segment_boundaries,
    )
    jump_detected = report_trajectory(
        configurations,
        errors_deg,
        arguments.fps,
        arguments.jump_threshold,
        segment_boundaries,
    )
    print(f"joint limits valid: all {len(configurations)} frames")
    print(f"saved CSV: {arguments.output.resolve()}")
    print(f"saved diagnostics plot: {arguments.plot.resolve()}")
    if jump_detected:
        print("review the reported rows before treating the motion as continuous")

    if not arguments.no_viewer:
        print("viewer: human=red/orange/yellow, robot=blue/cyan/purple")
        print("world axes: +X=red forward, +Y=green left, +Z=blue up")
        print(
            "tool triads: desired=red/green/blue, "
            "actual=magenta/yellow/cyan"
        )
        replay_in_mujoco(
            robot,
            data,
            configurations,
            trajectory_world,
            arguments.fps,
            arguments.viewer_max_frames,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
