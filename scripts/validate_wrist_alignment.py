"""Validate Gen3 AlignWrist and tool-axis conventions without human CSV data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.kinematics import Gen3Kinematics, gen3_kinematics  # noqa: E402
from sew_mimic.retarget import align_wrist  # noqa: E402


FIRST_FRAME_Q_AFTER_LOWER = np.array(
    [1.046125445, 1.485342602, -0.788082060, 1.366690019], dtype=float
)


@dataclass(frozen=True)
class WristSelfConsistencyReport:
    errors_deg: np.ndarray
    failure_count: int
    joint_limit_failure_count: int

    @property
    def median_error_deg(self) -> float:
        return float(np.median(self.errors_deg)) if self.errors_deg.size else np.nan

    @property
    def max_error_deg(self) -> float:
        return float(np.max(self.errors_deg)) if self.errors_deg.size else np.nan


@dataclass(frozen=True)
class ToolAxisReport:
    positive_dots: np.ndarray
    negative_dots: np.ndarray
    maximum_fixed_transform_error_deg: float
    rotation_7_to_tool_local: np.ndarray
    robot_alignment: np.ndarray


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


def _sample_valid_configuration(
    robot: Gen3Kinematics, rng: np.random.Generator
) -> np.ndarray:
    q = np.empty(robot.dof)
    for index, limited in enumerate(robot.joint_limited):
        q[index] = (
            rng.uniform(*robot.joint_limits[index])
            if limited
            else rng.uniform(-np.pi, np.pi)
        )
    return q


def _write_qpos(robot: Gen3Kinematics, data: mujoco.MjData, q: np.ndarray) -> None:
    for index, joint_id in enumerate(robot.joint_ids):
        data.qpos[robot.model.jnt_qposadr[joint_id]] = q[index]
    mujoco.mj_forward(robot.model, data)


def validate_align_wrist(
    samples: int,
    seed: int,
    fixed_q1_to_q4: np.ndarray = FIRST_FRAME_Q_AFTER_LOWER,
) -> WristSelfConsistencyReport:
    """Run robot-only AlignWrist self-consistency with q1:q4 held fixed."""
    if samples < 1:
        raise ValueError("samples must be positive")
    robot = gen3_kinematics()
    fixed = np.asarray(fixed_q1_to_q4, dtype=float)
    if fixed.shape != (4,):
        raise ValueError("fixed_q1_to_q4 must have shape (4,)")
    if np.any(fixed < robot.joint_limits[:4, 0]) or np.any(
        fixed > robot.joint_limits[:4, 1]
    ):
        raise ValueError("fixed_q1_to_q4 violates Gen3 joint limits")

    rng = np.random.default_rng(seed)
    errors: list[float] = []
    failure_count = 0
    joint_limit_failure_count = 0
    for _ in range(samples):
        q_ref = _sample_valid_configuration(robot, rng)
        q_ref[:4] = fixed
        target = robot.aligned_ee_rotation(q_ref)

        q_init = _sample_valid_configuration(robot, rng)
        q_init[:4] = fixed
        try:
            wrist_solution = align_wrist(q_init, target, robot)
        except ValueError:
            failure_count += 1
            continue

        q_solution = q_init.copy()
        q_solution[4:7] = wrist_solution
        within_limits = np.all(q_solution >= robot.joint_limits[:, 0] - 1e-12) and np.all(
            q_solution <= robot.joint_limits[:, 1] + 1e-12
        )
        if not within_limits:
            joint_limit_failure_count += 1
            failure_count += 1
            continue
        if not np.array_equal(q_solution[:4], fixed):
            failure_count += 1
            continue

        error = _rotation_error_deg(robot.aligned_ee_rotation(q_solution), target)
        errors.append(error)
        if error > 1e-9:
            failure_count += 1

    return WristSelfConsistencyReport(
        errors_deg=np.asarray(errors),
        failure_count=failure_count,
        joint_limit_failure_count=joint_limit_failure_count,
    )


def validate_tool_axis_convention(
    samples: int,
    seed: int,
    fixed_q1_to_q4: np.ndarray = FIRST_FRAME_Q_AFTER_LOWER,
) -> ToolAxisReport:
    """Compare native h7 with T(q)'s +X and recover R_7_T from MuJoCo FK."""
    if samples < 1:
        raise ValueError("samples must be positive")
    robot = gen3_kinematics()
    data = mujoco.MjData(robot.model)
    rng = np.random.default_rng(seed)
    fixed = np.asarray(fixed_q1_to_q4, dtype=float)
    site_id = mujoco.mj_name2id(
        robot.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site"
    )
    frame_7_body_id = int(robot.frame_body_ids[7])
    positive_dots = np.empty(samples)
    negative_dots = np.empty(samples)
    fixed_transform_errors = np.empty(samples)

    for sample_index in range(samples):
        q = _sample_valid_configuration(robot, rng)
        q[:4] = fixed
        _write_qpos(robot, data, q)

        rotation_0_to_7_mujoco = data.xmat[frame_7_body_id].reshape(3, 3)
        rotation_0_to_tool_mujoco = data.site_xmat[site_id].reshape(3, 3)
        recovered_rotation_7_to_tool = (
            rotation_0_to_7_mujoco.T @ rotation_0_to_tool_mujoco
        )
        fixed_transform_errors[sample_index] = _rotation_error_deg(
            recovered_rotation_7_to_tool, robot.ee_rotation_in_7
        )

        h7_world = rotation_0_to_7_mujoco @ robot.axes[6]
        tool_x_world = robot.aligned_ee_rotation(q)[:, 0]
        positive_dots[sample_index] = h7_world @ tool_x_world
        negative_dots[sample_index] = -h7_world @ tool_x_world

    return ToolAxisReport(
        positive_dots=positive_dots,
        negative_dots=negative_dots,
        maximum_fixed_transform_error_deg=float(np.max(fixed_transform_errors)),
        rotation_7_to_tool_local=robot.ee_rotation_in_7.copy(),
        robot_alignment=robot.R_robot_align.copy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260901)
    arguments = parser.parse_args()

    wrist = validate_align_wrist(arguments.samples, arguments.seed)
    print("Phase A: robot-side AlignWrist self-consistency")
    print("fixed q1:q4:", np.array2string(FIRST_FRAME_Q_AFTER_LOWER, precision=9))
    print(f"samples: {arguments.samples}")
    print(f"median wrist orientation error: {wrist.median_error_deg:.9e} deg")
    print(f"max wrist orientation error: {wrist.max_error_deg:.9e} deg")
    print(f"failure count: {wrist.failure_count}")
    print(f"joint-limit failure count: {wrist.joint_limit_failure_count}")

    tool = validate_tool_axis_convention(arguments.samples, arguments.seed + 1)
    print("\nPhase B: Gen3 final-axis/tool-frame convention")
    print("R_7_T_local from MuJoCo model/FK:")
    print(np.array2string(tool.rotation_7_to_tool_local, precision=9, suppress_small=True))
    print("existing R_robot_align used by T(q):")
    print(np.array2string(tool.robot_alignment, precision=9, suppress_small=True))
    print(
        "dot(+h7_world, tool_x_world): "
        f"median={np.median(tool.positive_dots):.12f}, "
        f"min={np.min(tool.positive_dots):.12f}, "
        f"max={np.max(tool.positive_dots):.12f}"
    )
    print(
        "dot(-h7_world, tool_x_world): "
        f"median={np.median(tool.negative_dots):.12f}, "
        f"min={np.min(tool.negative_dots):.12f}, "
        f"max={np.max(tool.negative_dots):.12f}"
    )
    print(
        "maximum FK-vs-model R_7_T_local error: "
        f"{tool.maximum_fixed_transform_error_deg:.9e} deg"
    )

    return int(
        wrist.failure_count > 0
        or wrist.joint_limit_failure_count > 0
        or wrist.max_error_deg > 1e-9
        or np.median(tool.negative_dots) < 1.0 - 1e-12
        or tool.maximum_fixed_transform_error_deg > 1e-10
    )


if __name__ == "__main__":
    raise SystemExit(main())
