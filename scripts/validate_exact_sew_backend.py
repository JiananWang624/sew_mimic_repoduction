"""Deterministic Phase-5A production Exact-SEW candidate coverage diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.common import ExactSewTarget, SolverStatus, gen3_end_effector_pose
from sew_mimic.exact import (
    NumericalExactSewOracle,
    R2R2R2RSearchConfig,
    enumerate_exact_sew_candidates,
    to_native_stereo_sew_target,
)
from sew_mimic.exact.root_search import RootSearchConfig
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    project_stereo_sew_reference,
    sample_gen3_configurations,
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def _target(robot, geometry, stereo, q: np.ndarray) -> ExactSewTarget:
    position, rotation = gen3_end_effector_pose(q, robot)
    points = geometry.sew_points(q)
    return ExactSewTarget(position, rotation, stereo.forward(points.shoulder, points.elbow, points.wrist))


def _stats(label: str, values: list[float]) -> None:
    print(f"{label}: mean={float(np.mean(values)):.6g} median={_percentile(values, 50):.6g} p95={_percentile(values, 95):.6g} max={max(values) if values else float('nan'):.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")

    oracle = NumericalExactSewOracle()
    robot, geometry, stereo = oracle.robot, Gen3StereoSewGeometry.from_robot(oracle.robot), StereoSew(project_stereo_sew_reference())
    configurations = sample_gen3_configurations(robot, args.count, args.seed)
    coverage = {"oracle_exact_production_exact": 0, "oracle_exact_production_miss": 0, "both_nonexact": 0, "production_exact_oracle_nonexact": 0}
    exact_counts: list[float] = []
    valid_counts: list[float] = []
    production_times: list[float] = []
    oracle_times: list[float] = []
    event_evaluations: list[float] = []
    refinements: list[float] = []
    intervals: list[float] = []
    accepted_position: list[float] = []
    accepted_rotation: list[float] = []
    accepted_sew: list[float] = []
    misses: list[tuple[int, np.ndarray, ExactSewTarget, dict]] = []

    for index, q_true in enumerate(configurations):
        target = _target(robot, geometry, stereo, q_true)
        started = time.perf_counter()
        candidates = enumerate_exact_sew_candidates(target, robot, geometry, stereo)
        production_times.append(1000.0 * (time.perf_counter() - started))
        oracle_result = oracle.solve_pose_and_sew(target)
        oracle_times.append(oracle_result.diagnostics.solve_time_ms or float("nan"))
        production_exact = candidates.exact_geometric_count > 0
        oracle_exact = oracle_result.status is SolverStatus.SUCCESS_EXACT
        if oracle_exact and production_exact:
            coverage["oracle_exact_production_exact"] += 1
        elif oracle_exact:
            coverage["oracle_exact_production_miss"] += 1
            misses.append((index, q_true, target, dict(candidates.rejection_counts)))
        elif production_exact:
            coverage["production_exact_oracle_nonexact"] += 1
        else:
            coverage["both_nonexact"] += 1
        exact_counts.append(float(candidates.exact_geometric_count))
        valid_counts.append(float(candidates.joint_limit_valid_count))
        event_evaluations.append(float(candidates.rejection_counts.get("event_margin_evaluations", 0) + candidates.rejection_counts.get("event_alignment_evaluations", 0)))
        refinements.append(float(candidates.rejection_counts.get("event_refinements", 0)))
        intervals.append(float(candidates.rejection_counts.get("event_intervals", 0)))
        for candidate in candidates.candidates:
            accepted_position.append(candidate.position_error_m)
            accepted_rotation.append(candidate.orientation_error_rad)
            accepted_sew.append(candidate.sew_error_rad)

    print(f"TARGETS total={args.count} seed={args.seed}")
    print("FOUR_WAY", coverage)
    print(f"PRODUCTION coverage_exact={sum(value > 0 for value in exact_counts)}/{args.count} coverage_joint_valid={sum(value > 0 for value in valid_counts)}/{args.count}")
    _stats("exact_candidate_count", exact_counts)
    _stats("joint_valid_candidate_count", valid_counts)
    _stats("production_solve_ms", production_times)
    _stats("oracle_solve_ms", [value for value in oracle_times if np.isfinite(value)])
    _stats("event_evaluations", event_evaluations)
    _stats("event_refinements", refinements)
    _stats("event_intervals", intervals)
    print("MAX_ACCEPTED", {"position_m": max(accepted_position, default=float("nan")), "orientation_rad": max(accepted_rotation, default=float("nan")), "sew_rad": max(accepted_sew, default=float("nan"))})
    for index, q_true, target, diagnostics in misses:
        print("MISS", {"index": index, "q_true": q_true.tolist(), "position": target.position.tolist(), "rotation": target.rotation.tolist(), "psi": target.psi, "diagnostics": diagnostics})

    pinned_q = np.array([.2, .3, -.4, .5, -.2, .3, .4])
    policy_target = _target(robot, geometry, stereo, pinned_q)
    pinned_points = geometry.sew_points(pinned_q)
    wrist = policy_target.position - to_native_stereo_sew_target(policy_target, robot, geometry).rotation_07 @ geometry.P[:, 7]
    reference = stereo.inverse(geometry.P[:, 0], wrist, policy_target.psi)
    pwe = pinned_points.elbow - pinned_points.wrist
    wa_true = float(np.arctan2(reference.plane_normal @ np.cross(-(wrist - geometry.P[:, 0]) / np.linalg.norm(wrist - geometry.P[:, 0]), pwe), (-(wrist - geometry.P[:, 0]) / np.linalg.norm(wrist - geometry.P[:, 0])) @ pwe))
    for samples in (200, 400, 800):
        config = R2R2R2RSearchConfig(mode="reference_fixed_grid", reference_fixed_grid=RootSearchConfig(samples=samples))
        started = time.perf_counter()
        result = enumerate_exact_sew_candidates(policy_target, robot, geometry, stereo, search_config=config)
        print("POLICY", {"mode": f"reference_{samples}", "exact": result.exact_geometric_count, "roots": result.search_root_count, "time_ms": 1000 * (time.perf_counter() - started)})
    started = time.perf_counter()
    event = enumerate_exact_sew_candidates(policy_target, robot, geometry, stereo)
    closest = min(event.candidates, key=lambda candidate: abs(candidate.wrist_search_angle - wa_true), default=None)
    print("POLICY", {"mode": "event", "wa_true": wa_true, "wa_closest": None if closest is None else closest.wrist_search_angle, "exact": event.exact_geometric_count, "joint_valid": event.joint_limit_valid_count, "roots": event.search_root_count, "residual": None if closest is None else (closest.position_error_m, closest.orientation_error_rad, closest.sew_error_rad), "evaluations": event.rejection_counts.get("event_margin_evaluations", 0) + event.rejection_counts.get("event_alignment_evaluations", 0), "time_ms": 1000 * (time.perf_counter() - started)})

    q_limit = np.where(np.isfinite(robot.joint_limits[:, 0]), robot.joint_limits[:, 0] + 1e-5, 0.1)
    near = enumerate_exact_sew_candidates(_target(robot, geometry, stereo, q_limit), robot, geometry, stereo)
    near_oracle = oracle.solve_pose_and_sew(
        _target(robot, geometry, stereo, q_limit),
        q_seed=q_limit + 1e-7,
    )
    impossible = _target(robot, geometry, stereo, configurations[0])
    impossible = ExactSewTarget(impossible.position + np.array([100.0, 100.0, 100.0]), impossible.rotation, impossible.psi)
    far = enumerate_exact_sew_candidates(impossible, robot, geometry, stereo)
    print("NEAR_LIMIT", {"production_exact": near.exact_geometric_count, "production_joint_valid": near.joint_limit_valid_count, "oracle_status": near_oracle.status.value, "margins": [candidate.joint_limit_margin_rad for candidate in near.candidates]})
    print("IMPOSSIBLE", {"exact": far.exact_geometric_count, "roots": far.search_root_count, "diagnostics": dict(far.rejection_counts)})


if __name__ == "__main__":
    main()
