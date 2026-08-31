"""Retarget a human trajectory CSV, save results, and replay it in MuJoCo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from replay_trajectory import replay_in_mujoco  # noqa: E402
from sew_mimic.csv_adapter import (  # noqa: E402
    MOTIVE_TO_GEN3_BODY_ROTATION,
    HumanCSVAdapter,
    load_human_trajectory_csv,
)
from sew_mimic.kinematics import gen3_kinematics  # noqa: E402
from sew_mimic.retarget import sew_mimic  # noqa: E402


OUTPUT_COLUMNS = (
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "upper_error_deg",
    "lower_error_deg",
    "wrist_error_deg",
)


def retarget_csv(
    input_path: Path,
    output_path: Path,
    adapter: HumanCSVAdapter,
) -> np.ndarray:
    """Adapt and retarget a complete CSV, then write the requested columns."""
    trajectory = load_human_trajectory_csv(input_path, adapter)
    configurations = np.empty((len(trajectory), 7))
    output_values = np.empty((len(trajectory), len(OUTPUT_COLUMNS)))
    q_previous = np.zeros(7)

    for frame in range(len(trajectory)):
        try:
            q_current, diagnostics = sew_mimic(
                q_previous,
                trajectory.shoulders[frame],
                trajectory.elbows[frame],
                trajectory.wrists[frame],
                trajectory.hand_orientations[frame],
            )
        except ValueError as error:
            raise ValueError(f"SEW-Mimic failed at CSV row {frame}: {error}") from error
        configurations[frame] = q_current
        output_values[frame] = [
            *q_current,
            diagnostics["upper_arm_error_deg"],
            diagnostics["lower_arm_error_deg"],
            diagnostics["wrist_rotation_error_deg"],
        ]
        if not diagnostics["joint_limit_valid"]:
            raise RuntimeError(f"joint-limit violation at CSV row {frame}")
        q_previous = q_current

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_values, columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)
    return configurations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "test.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "test_retargeted.csv",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument(
        "--rotation-robot-from-csv",
        type=float,
        nargs=9,
        default=MOTIVE_TO_GEN3_BODY_ROTATION.ravel(),
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument(
        "--translation-robot-from-csv",
        type=float,
        nargs=3,
        default=np.zeros(3),
        metavar=("TX", "TY", "TZ"),
    )
    parser.add_argument("--no-viewer", action="store_true")
    arguments = parser.parse_args()
    if arguments.fps <= 0.0:
        parser.error("--fps must be positive")

    adapter = HumanCSVAdapter(
        rotation_robot_from_csv=np.asarray(arguments.rotation_robot_from_csv).reshape(3, 3),
        translation_robot_from_csv=arguments.translation_robot_from_csv,
        position_scale=arguments.position_scale,
    )
    configurations = retarget_csv(arguments.input, arguments.output, adapter)
    steps = np.abs(np.diff(configurations, axis=0))
    print(f"retargeted {len(configurations)} frames")
    print(f"saved: {arguments.output.resolve()}")
    if len(steps):
        print(
            "maximum |delta q| per frame: "
            + np.array2string(np.max(steps, axis=0), precision=6)
        )

    if not arguments.no_viewer:
        replay_in_mujoco(gen3_kinematics(), configurations, arguments.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
