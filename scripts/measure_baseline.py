"""Measure Method 0 against the configured human task point using pinch-site FK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.common import (  # noqa: E402
    HumanArmTarget,
    SolverStatus,
    compute_human_task_point,
    evaluate_solver_result,
)
from sew_mimic.common.task_point import (  # noqa: E402
    DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
    DEFAULT_TASK_POINT_MODE,
)
from sew_mimic.config import CONFIG, project_path  # noqa: E402
from sew_mimic.csv_adapter import HumanCSVAdapter, load_human_trajectory_csv  # noqa: E402
from sew_mimic.mounting import (  # noqa: E402
    DEFAULT_ROBOT_WORLD_OFFSET,
    load_humanoid_mounted_gen3,
    world_trajectory_to_base,
)
from sew_mimic.sew import solve_legacy_sew_mimic  # noqa: E402


OUTPUT_COLUMNS = (
    "frame",
    "solver_status",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "ee_position_error_mm",
    "ee_orientation_error_deg",
    "upper_arm_error_deg",
    "lower_arm_error_deg",
    "wrist_rotation_error_deg",
    "joint_limit_valid",
    "joint_limit_margin_deg",
    "message",
)

_HUMAN_CSV_CONFIG = CONFIG["human_csv"]
DEFAULT_OUTPUT_PATH = project_path("output/baseline_metrics.csv")
_SUCCESS_STATUSES = {SolverStatus.SUCCESS_EXACT, SolverStatus.SUCCESS_APPROX}


def _trajectory_in_base(trajectory_world, robot, data):
    base_body_id = int(robot.frame_body_ids[0])
    points_world = np.stack(
        (
            trajectory_world.shoulders,
            trajectory_world.elbows,
            trajectory_world.wrists,
        ),
        axis=1,
    )
    return world_trajectory_to_base(
        points_world,
        trajectory_world.hand_orientations,
        data.xmat[base_body_id].reshape(3, 3),
        data.xpos[base_body_id],
    )


def measure_baseline(
    input_path: Path,
    output_path: Path,
    adapter: HumanCSVAdapter | None = None,
) -> pd.DataFrame:
    """Measure every input frame and save one result row per frame."""
    trajectory_world = load_human_trajectory_csv(input_path, adapter)
    robot, data = load_humanoid_mounted_gen3(trajectory_world.shoulders[0])
    points_base, orientations_base = _trajectory_in_base(
        trajectory_world, robot, data
    )

    rows: list[dict[str, object]] = []
    q_previous = np.zeros(7)
    for frame in range(len(trajectory_world)):
        shoulder, elbow, wrist = points_base[frame]
        hand_rotation = orientations_base[frame]
        task_point = compute_human_task_point(
            wrist,
            hand_rotation,
            mode=DEFAULT_TASK_POINT_MODE,
            human_wrist_to_task_offset_m=DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
        )
        target = HumanArmTarget(
            shoulder=shoulder,
            elbow=elbow,
            wrist=wrist,
            hand_rotation=hand_rotation,
            task_point=task_point,
        )
        result = solve_legacy_sew_mimic(
            q_previous,
            shoulder,
            elbow,
            wrist,
            hand_rotation,
        )
        if result.status in _SUCCESS_STATUSES:
            result = evaluate_solver_result(result, target, robot)
            assert result.q is not None
            q_previous = result.q

        metadata = result.diagnostics.metadata
        q_values = result.q if result.q is not None else np.full(7, np.nan)
        row: dict[str, object] = {
            "frame": frame,
            "solver_status": result.status.value,
            **{f"q{index + 1}": q_values[index] for index in range(7)},
            "ee_position_error_mm": metadata.get("ee_position_error_mm", np.nan),
            "ee_orientation_error_deg": metadata.get(
                "ee_orientation_error_deg", np.nan
            ),
            "upper_arm_error_deg": metadata.get("upper_arm_error_deg", np.nan),
            "lower_arm_error_deg": metadata.get("lower_arm_error_deg", np.nan),
            "wrist_rotation_error_deg": metadata.get(
                "wrist_rotation_error_deg", np.nan
            ),
            "joint_limit_valid": metadata.get("joint_limit_valid", None),
            "joint_limit_margin_deg": metadata.get(
                "joint_limit_margin_deg", np.nan
            ),
            "message": result.message,
        }
        rows.append(row)

    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def summarize_baseline(table: pd.DataFrame) -> dict[str, float | int]:
    """Return the required Method-0 trajectory summary."""
    successful = table["solver_status"].isin(
        [status.value for status in _SUCCESS_STATUSES]
    )
    valid = table.loc[successful]
    position = valid["ee_position_error_mm"].to_numpy(dtype=float)
    orientation = valid["ee_orientation_error_deg"].to_numpy(dtype=float)
    margin = valid["joint_limit_margin_deg"].to_numpy(dtype=float)
    if not len(valid):
        raise ValueError("baseline summary requires at least one successful frame")

    violation_count = int(np.sum(margin < 0.0))
    return {
        "valid_frame_count": int(np.sum(successful)),
        "failure_count": int(np.sum(~successful)),
        "ee_position_mean_mm": float(np.mean(position)),
        "ee_position_median_mm": float(np.median(position)),
        "ee_position_p95_mm": float(np.percentile(position, 95)),
        "ee_position_p99_mm": float(np.percentile(position, 99)),
        "ee_position_max_mm": float(np.max(position)),
        "ee_orientation_mean_deg": float(np.mean(orientation)),
        "ee_orientation_median_deg": float(np.median(orientation)),
        "ee_orientation_p95_deg": float(np.percentile(orientation, 95)),
        "ee_orientation_max_deg": float(np.max(orientation)),
        "joint_limit_violation_count": violation_count,
        "joint_limit_violation_fraction": violation_count / len(valid),
        "minimum_joint_limit_margin_deg": float(np.min(margin)),
    }


def print_summary(summary: dict[str, float | int]) -> None:
    print(f"valid frames: {summary['valid_frame_count']}")
    print(f"failures: {summary['failure_count']}")
    print(
        "EE position error (mm): "
        f"mean={summary['ee_position_mean_mm']:.6f}, "
        f"median={summary['ee_position_median_mm']:.6f}, "
        f"p95={summary['ee_position_p95_mm']:.6f}, "
        f"p99={summary['ee_position_p99_mm']:.6f}, "
        f"max={summary['ee_position_max_mm']:.6f}"
    )
    print(
        "EE orientation error (deg): "
        f"mean={summary['ee_orientation_mean_deg']:.6e}, "
        f"median={summary['ee_orientation_median_deg']:.6e}, "
        f"p95={summary['ee_orientation_p95_deg']:.6e}, "
        f"max={summary['ee_orientation_max_deg']:.6e}"
    )
    print(
        "joint limits: "
        f"violations={summary['joint_limit_violation_count']}, "
        f"fraction={summary['joint_limit_violation_fraction']:.6f}, "
        f"minimum_margin_deg={summary['minimum_joint_limit_margin_deg']:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_path(_HUMAN_CSV_CONFIG["input_path"]),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    print(f"task point mode: {DEFAULT_TASK_POINT_MODE}")
    print(
        "human wrist-to-task offset in canonical hand frame (m):",
        DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
    )
    print("robot world offset (m):", np.asarray(DEFAULT_ROBOT_WORLD_OFFSET))
    table = measure_baseline(arguments.input, arguments.output)
    print_summary(summarize_baseline(table))
    print(f"saved CSV: {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
