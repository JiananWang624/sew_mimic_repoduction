"""Bounded fixed-base diagnostics for the Method-3 numerical oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.common import (
    ExactSewTarget,
    HumanArmTarget,
    SolverResult,
    SolverStatus,
    compute_human_task_point,
    gen3_end_effector_pose,
)
from sew_mimic.csv_adapter import load_human_trajectory_csv
from sew_mimic.exact import NumericalExactSewOracle
from sew_mimic.mounting import load_humanoid_mounted_gen3, world_trajectory_to_base
from sew_mimic.sew import sample_gen3_configurations


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def _report(label: str, results: list[SolverResult]) -> None:
    exact = sum(result.status is SolverStatus.SUCCESS_EXACT for result in results)
    approximate = sum(result.status is SolverStatus.SUCCESS_APPROX for result in results)
    failures = sum(result.status is SolverStatus.NUMERICAL_FAILURE for result in results)
    positions = [
        result.diagnostics.position_error_m
        for result in results
        if result.diagnostics.position_error_m is not None
    ]
    rotations = [
        result.diagnostics.orientation_error_rad
        for result in results
        if result.diagnostics.orientation_error_rad is not None
    ]
    sews = [
        result.diagnostics.sew_error_rad
        for result in results
        if result.diagnostics.sew_error_rad is not None
    ]
    times = [
        result.diagnostics.solve_time_ms
        for result in results
        if result.diagnostics.solve_time_ms is not None
    ]
    count = len(results)
    print(
        f"{label}: count={count} exact={exact}/{count} "
        f"approx={approximate}/{count} numerical_failure={failures}/{count}"
    )
    print(
        f"{label}: position_m median={_percentile(positions, 50):.6g} "
        f"p95={_percentile(positions, 95):.6g}; orientation_rad "
        f"median={_percentile(rotations, 50):.6g} "
        f"p95={_percentile(rotations, 95):.6g}"
    )
    if sews:
        print(f"{label}: sew_rad median={_percentile(sews, 50):.6g} p95={_percentile(sews, 95):.6g}")
    print(
        f"{label}: solve_ms mean={np.mean(times):.3f} "
        f"median={_percentile(times, 50):.3f} "
        f"p95={_percentile(times, 95):.3f}"
    )


def _target_from_configuration(oracle: NumericalExactSewOracle, q: np.ndarray) -> ExactSewTarget:
    position, rotation = gen3_end_effector_pose(q, oracle.robot)
    points = oracle.geometry.sew_points(q)
    psi = oracle.stereo.forward(points.shoulder, points.elbow, points.wrist)
    return ExactSewTarget(position, rotation, psi)


def _synthetic(oracle: NumericalExactSewOracle, count: int) -> None:
    configurations = sample_gen3_configurations(oracle.robot, count, 20260909)
    pose_results, sew_results = [], []
    for q_true in configurations:  # q_true constructs a target only; never a default seed.
        target = _target_from_configuration(oracle, q_true)
        pose_results.append(oracle.solve_pose(target.position, target.rotation))
        sew_results.append(oracle.solve_pose_and_sew(target))
    _report("synthetic_pose", pose_results)
    _report("synthetic_pose_and_sew", sew_results)
    for label, results in (
        ("synthetic_pose", pose_results),
        ("synthetic_pose_and_sew", sew_results),
    ):
        exact = [result for result in results if result.status is SolverStatus.SUCCESS_EXACT]
        if exact:
            print(
                label,
                "max_exact_errors",
                max(r.diagnostics.position_error_m or 0.0 for r in exact),
                max(r.diagnostics.orientation_error_rad or 0.0 for r in exact),
                max(r.diagnostics.sew_error_rad or 0.0 for r in exact),
            )
    local = oracle.solve_pose_and_sew(
        _target_from_configuration(oracle, configurations[0]),
        q_seed=configurations[0] + 1e-4,
    )
    print(
        "local_perturbed_seed",
        local.status.value,
        local.diagnostics.position_error_m,
        local.diagnostics.orientation_error_rad,
        local.diagnostics.sew_error_rad,
    )
    q_limit = np.where(
        np.isfinite(oracle.robot.joint_limits[:, 0]),
        oracle.robot.joint_limits[:, 0] + 1e-5,
        0.1,
    )
    near = oracle.solve_pose_and_sew(
        _target_from_configuration(oracle, q_limit), q_seed=q_limit + 1e-7
    )
    print("near_limit", near.status.value, "margin_rad", near.diagnostics.joint_limit_margin_rad)


def _csv(path: Path, maximum: int, stride: int, all_frames: bool) -> None:
    trajectory = load_human_trajectory_csv(path)
    robot, data = load_humanoid_mounted_gen3(trajectory.shoulders[0])
    oracle = NumericalExactSewOracle(robot=robot)
    base = int(robot.frame_body_ids[0])
    world_points = np.stack(
        (trajectory.shoulders, trajectory.elbows, trajectory.wrists), axis=1
    )
    points, rotations = world_trajectory_to_base(
        world_points,
        trajectory.hand_orientations,
        data.xmat[base].reshape(3, 3),
        data.xpos[base].copy(),
    )
    indices = list(range(0, len(points), stride))
    if not all_frames:
        indices = indices[:maximum]
    pose_results, sew_results = [], []
    for frame in indices:
        shoulder, elbow, wrist = points[frame]
        hand = rotations[frame]
        human = HumanArmTarget(
            shoulder,
            elbow,
            wrist,
            hand,
            compute_human_task_point(wrist, hand),
        )
        target = ExactSewTarget(
            human.task_point,
            human.hand_rotation,
            oracle.stereo.forward(human.shoulder, human.elbow, human.wrist),
        )
        pose = oracle.solve_pose(target.position, target.rotation)
        sew = oracle.solve_pose_and_sew(target)
        pose_results.append(pose)
        sew_results.append(sew)
        position_mm = (
            None
            if sew.diagnostics.position_error_m is None
            else 1000.0 * sew.diagnostics.position_error_m
        )
        orientation_deg = (
            None
            if sew.diagnostics.orientation_error_rad is None
            else np.degrees(sew.diagnostics.orientation_error_rad)
        )
        sew_deg = (
            None
            if sew.diagnostics.sew_error_rad is None
            else np.degrees(sew.diagnostics.sew_error_rad)
        )
        margin_deg = (
            None
            if sew.diagnostics.joint_limit_margin_rad is None
            else np.degrees(sew.diagnostics.joint_limit_margin_rad)
        )
        print(
            "frame", frame,
            "pose_status", pose.status.value,
            "exact_sew_status", sew.status.value,
            "position_error_mm", position_mm,
            "orientation_error_deg", orientation_deg,
            "sew_error_deg", sew_deg,
            "joint_limit_margin_deg", margin_deg,
            "solve_time_ms", sew.diagnostics.solve_time_ms,
            "n_starts", sew.diagnostics.metadata.get("n_starts"),
            "best_seed", sew.diagnostics.metadata.get("best_seed"),
            "nfev", sew.diagnostics.metadata.get("best_nfev"),
        )
    _report("csv_pose", pose_results)
    _report("csv_pose_and_sew", sew_results)
    mismatch = sum(
        pose.status is SolverStatus.SUCCESS_EXACT
        and sew.status is not SolverStatus.SUCCESS_EXACT
        for pose, sew in zip(pose_results, sew_results)
    )
    print("pose_exact_sew_nonexact", mismatch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "test.csv")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--synthetic", type=int, default=100)
    args = parser.parse_args()
    if args.max_frames < 1 or args.stride < 1 or args.synthetic < 1:
        raise SystemExit("counts and stride must be positive")
    _synthetic(NumericalExactSewOracle(), args.synthetic)
    _csv(args.input, args.max_frames, args.stride, args.all)


if __name__ == "__main__":
    main()
