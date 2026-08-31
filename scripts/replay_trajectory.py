"""Generate, retarget, plot, and replay a smooth synthetic human trajectory."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.kinematics import Gen3Kinematics, gen3_kinematics  # noqa: E402
from sew_mimic.retarget import sew_mimic  # noqa: E402


def make_synthetic_trajectory(
    robot: Gen3Kinematics,
    duration: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create smooth, reachable ``(s, e, w, H)`` samples using Gen3 FK."""
    if duration <= 0.0 or fps <= 0.0:
        raise ValueError("duration and fps must be positive")

    frame_count = int(round(duration * fps)) + 1
    sample_times = np.linspace(0.0, duration, frame_count)
    phase = 2.0 * np.pi * sample_times / duration

    # FK is used only to create the synthetic directions and H. The reference
    # joint angles are not passed to SEW-Mimic.
    reference_q = np.column_stack(
        (
            0.60 * np.sin(0.55 * phase),
            0.55 + 0.28 * np.sin(0.70 * phase + 0.25),
            0.50 * np.sin(0.80 * phase + 0.60),
            -1.15 + 0.32 * np.sin(0.65 * phase - 0.30),
            0.55 * np.sin(0.90 * phase + 0.35),
            0.50 + 0.24 * np.sin(0.75 * phase - 0.20),
            0.65 * np.sin(0.85 * phase + 0.45),
        )
    )

    shoulders = np.zeros((frame_count, 3))
    elbows = np.empty((frame_count, 3))
    wrists = np.empty((frame_count, 3))
    hand_orientations = np.empty((frame_count, 3, 3))

    for frame, q_reference in enumerate(reference_q):
        upper_direction = robot.R_0_i(q_reference, 3) @ robot.axes[2]
        lower_direction = robot.R_0_i(q_reference, 5) @ robot.axes[4]
        elbows[frame] = shoulders[frame] + upper_direction
        wrists[frame] = elbows[frame] + lower_direction
        hand_orientations[frame] = robot.aligned_ee_rotation(q_reference)

    return sample_times, shoulders, elbows, wrists, hand_orientations


def retarget_trajectory(
    robot: Gen3Kinematics,
    shoulders: np.ndarray,
    elbows: np.ndarray,
    wrists: np.ndarray,
    hand_orientations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Algorithm 1 frame-by-frame, seeding each solve with the prior q."""
    frame_count = len(shoulders)
    configurations = np.empty((frame_count, robot.dof))
    errors_deg = np.empty((frame_count, 3))
    q_previous = np.zeros(robot.dof)

    for frame in range(frame_count):
        q_current, diagnostics = sew_mimic(
            q_previous,
            shoulders[frame],
            elbows[frame],
            wrists[frame],
            hand_orientations[frame],
        )
        configurations[frame] = q_current
        errors_deg[frame] = [
            diagnostics["upper_arm_error_deg"],
            diagnostics["lower_arm_error_deg"],
            diagnostics["wrist_rotation_error_deg"],
        ]
        if not diagnostics["joint_limit_valid"]:
            raise RuntimeError(f"joint-limit violation at frame {frame}")
        q_previous = q_current

    return configurations, errors_deg


def plot_diagnostics(
    sample_times: np.ndarray,
    configurations: np.ndarray,
    errors_deg: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    """Plot orientation errors, joint angles, and finite-difference velocities."""
    velocities = np.gradient(configurations, sample_times, axis=0)
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    plotted_errors = np.maximum(errors_deg, 1e-16)
    for index, label in enumerate(("upper arm", "lower arm", "wrist")):
        axes[0].plot(sample_times, plotted_errors[:, index], label=label)
    axes[0].set_ylabel("orientation error (deg)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncols=3)

    for joint in range(configurations.shape[1]):
        axes[1].plot(sample_times, configurations[:, joint], label=f"q{joint + 1}")
    axes[1].set_ylabel("joint angle (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncols=4)

    for joint in range(velocities.shape[1]):
        axes[2].plot(sample_times, velocities[:, joint], label=f"q{joint + 1}")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("joint velocity (rad/s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncols=4)

    figure.suptitle("SEW-Mimic synthetic trajectory diagnostics")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    return figure


def report_continuity(
    sample_times: np.ndarray,
    configurations: np.ndarray,
    errors_deg: np.ndarray,
    jump_threshold_rad: float,
) -> bool:
    """Print tracking and continuity extrema; return whether a jump was found."""
    steps = np.diff(configurations, axis=0)
    velocities = steps / np.diff(sample_times)[:, None]
    maximum_steps = np.max(np.abs(steps), axis=0)
    maximum_velocities = np.max(np.abs(velocities), axis=0)

    print(f"frames={len(sample_times)}  duration={sample_times[-1]:.3f} s")
    print(
        "maximum orientation errors: "
        f"upper={np.max(errors_deg[:, 0]):.3e} deg, "
        f"lower={np.max(errors_deg[:, 1]):.3e} deg, "
        f"wrist={np.max(errors_deg[:, 2]):.3e} deg"
    )
    print("per-joint maximum |delta q| (rad/frame)")
    print(np.array2string(maximum_steps, precision=6, suppress_small=False))
    print("per-joint maximum |velocity| (rad/s)")
    print(np.array2string(maximum_velocities, precision=6, suppress_small=False))

    jump_locations = np.argwhere(np.abs(steps) > jump_threshold_rad)
    if jump_locations.size:
        print(f"detected {len(jump_locations)} possible branch switches/jumps")
        for frame, joint in jump_locations[:10]:
            print(
                f"  frame {frame}->{frame + 1}, q{joint + 1}, "
                f"delta={steps[frame, joint]:+.6f} rad"
            )
        return True

    print(f"no joint step exceeded the {jump_threshold_rad:.3f} rad jump threshold")
    return False


def replay_in_mujoco(
    robot: Gen3Kinematics,
    configurations: np.ndarray,
    fps: float,
) -> None:
    """Play joint configurations once in the passive MuJoCo viewer."""
    model = robot.model
    data = mujoco.MjData(model)
    frame_period = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.perf_counter()
        for frame, configuration in enumerate(configurations):
            if not viewer.is_running():
                break
            for index, joint_id in enumerate(robot.joint_ids):
                data.qpos[model.jnt_qposadr[joint_id]] = configuration[index]
            mujoco.mj_forward(model, data)
            viewer.sync()
            remaining = start_time + (frame + 1) * frame_period - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--jump-threshold", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "synthetic_trajectory_diagnostics.png",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args()

    robot = gen3_kinematics()
    sample_times, shoulders, elbows, wrists, hand_orientations = make_synthetic_trajectory(
        robot, arguments.duration, arguments.fps
    )
    configurations, errors_deg = retarget_trajectory(
        robot, shoulders, elbows, wrists, hand_orientations
    )
    jump_detected = report_continuity(
        sample_times, configurations, errors_deg, arguments.jump_threshold
    )
    figure = plot_diagnostics(sample_times, configurations, errors_deg, arguments.output)
    print(f"saved diagnostic plot: {arguments.output.resolve()}")

    if not arguments.no_show:
        plt.show(block=False)
    if not arguments.no_viewer:
        replay_in_mujoco(robot, configurations, arguments.fps)
    if not arguments.no_show:
        plt.show()
    else:
        plt.close(figure)

    return int(jump_detected)


if __name__ == "__main__":
    raise SystemExit(main())
