"""Validate Method 2 branch policies on consecutive mounted human CSV frames."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.angles import angular_difference  # noqa: E402
from sew_mimic.common import (  # noqa: E402
    HumanArmTarget,
    SolverStatus,
    compute_human_task_point,
)
from sew_mimic.common.task_point import (  # noqa: E402
    DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
    DEFAULT_TASK_POINT_MODE,
)
from sew_mimic.config import CONFIG, project_path  # noqa: E402
from sew_mimic.csv_adapter import load_human_trajectory_csv  # noqa: E402
from sew_mimic.exact import (  # noqa: E402
    BranchSelectionOutcome,
    NumericalExactSewOracle,
    enumerate_exact_sew_candidates,
    human_arm_to_exact_sew_target,
    normalized_authoritative_residual,
    select_exact_sew_branch,
)
from sew_mimic.mounting import (  # noqa: E402
    load_humanoid_mounted_gen3,
    world_trajectory_to_base,
)
from sew_mimic.sew import (  # noqa: E402
    Gen3StereoSewGeometry,
    StereoSew,
    StereoSewSingularityError,
    project_stereo_sew_reference,
)


_INPUT_PATH = project_path(CONFIG["human_csv"]["input_path"])
_EXACT = SolverStatus.SUCCESS_EXACT


@dataclass
class PolicyMeasurements:
    statuses: Counter[str] = field(default_factory=Counter)
    positions: list[float] = field(default_factory=list)
    orientations: list[float] = field(default_factory=list)
    sew_errors: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    solve_times_ms: list[float] = field(default_factory=list)
    branch_ids: list[str | None] = field(default_factory=list)
    configurations: list[np.ndarray | None] = field(default_factory=list)

    def record(self, outcome, solve_time_ms: float) -> None:
        self.statuses[outcome.status.value] += 1
        self.solve_times_ms.append(float(solve_time_ms))
        self.branch_ids.append(outcome.branch_id)
        if outcome.candidate is None:
            self.configurations.append(None)
            return
        candidate = outcome.candidate
        self.configurations.append(candidate.q.copy())
        self.positions.append(candidate.position_error_m)
        self.orientations.append(candidate.orientation_error_rad)
        self.sew_errors.append(candidate.sew_error_rad)
        self.margins.append(candidate.joint_limit_margin_rad)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _mean_percentiles(values: list[float]) -> dict[str, float]:
    summary = _percentiles(values)
    return {"mean": float(np.mean(values)) if values else float("nan"), **summary}


def _jump_summary(measurements: PolicyMeasurements) -> tuple[int, dict[str, float]]:
    switches = 0
    jumps: list[float] = []
    previous_branch: str | None = None
    previous_q: np.ndarray | None = None
    for branch, q in zip(
        measurements.branch_ids, measurements.configurations, strict=True
    ):
        if branch is None or q is None:
            continue
        if previous_branch is not None and branch != previous_branch:
            switches += 1
        if previous_q is not None:
            difference = np.array(
                [
                    angular_difference(float(current), float(previous))
                    for current, previous in zip(q, previous_q, strict=True)
                ]
            )
            jumps.append(float(np.linalg.norm(difference)))
        previous_branch = branch
        previous_q = q
    return switches, _percentiles(jumps)


def _selectable(candidate) -> bool:
    return bool(
        candidate.exact
        and candidate.joint_limit_valid
        and candidate.position_error_m < 1e-6
        and candidate.orientation_error_rad < 1e-6
        and candidate.sew_error_rad < 1e-5
    )


def _independent_continuous_index(candidate_set, previous: np.ndarray | None) -> int | None:
    selectable = [
        (index, candidate)
        for index, candidate in enumerate(candidate_set.candidates)
        if _selectable(candidate)
    ]
    if not selectable:
        return None
    if previous is None:
        return min(
            selectable,
            key=lambda item: (
                -item[1].joint_limit_margin_rad,
                normalized_authoritative_residual(item[1]),
                item[0],
            ),
        )[0]
    return min(
        selectable,
        key=lambda item: (
            sum(
                angular_difference(float(current), float(old)) ** 2
                for current, old in zip(item[1].q, previous, strict=True)
            ),
            -item[1].joint_limit_margin_rad,
            normalized_authoritative_residual(item[1]),
            item[0],
        ),
    )[0]


def _human_targets(path: Path, start: int, count: int):
    trajectory = load_human_trajectory_csv(path)
    if start < 0 or count < 1 or start + count > len(trajectory):
        raise ValueError(
            f"requested frame range [{start}, {start + count}) outside "
            f"trajectory length {len(trajectory)}"
        )
    robot, data = load_humanoid_mounted_gen3(trajectory.shoulders[0])
    points_world = np.stack(
        (trajectory.shoulders, trajectory.elbows, trajectory.wrists), axis=1
    )
    base_body_id = int(robot.frame_body_ids[0])
    points_base, rotations_base = world_trajectory_to_base(
        points_world,
        trajectory.hand_orientations,
        data.xmat[base_body_id].reshape(3, 3),
        data.xpos[base_body_id],
    )
    targets: list[HumanArmTarget] = []
    for frame in range(start, start + count):
        shoulder, elbow, wrist = points_base[frame]
        rotation = rotations_base[frame]
        targets.append(
            HumanArmTarget(
                shoulder,
                elbow,
                wrist,
                rotation,
                compute_human_task_point(
                    wrist,
                    rotation,
                    mode=DEFAULT_TASK_POINT_MODE,
                    human_wrist_to_task_offset_m=DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
                ),
            )
        )
    return robot, targets


def _print_policy(label: str, values: PolicyMeasurements, frame_count: int) -> None:
    switches, jumps = _jump_summary(values)
    failures = {
        key: count for key, count in sorted(values.statuses.items()) if key != _EXACT.value
    }
    print(
        label,
        {
            "success_exact_fraction": values.statuses[_EXACT.value] / frame_count,
            "failure_status_counts": failures,
            "position_m": _percentiles(values.positions),
            "orientation_rad": _percentiles(values.orientations),
            "sew_rad": _percentiles(values.sew_errors),
            "joint_limit_margin_rad": {
                "minimum": min(values.margins, default=float("nan")),
                "median": float(np.median(values.margins)) if values.margins else float("nan"),
            },
            "branch_switch_count": switches,
            "wrapped_joint_jump_rad": jumps,
            "solve_time_ms": _mean_percentiles(values.solve_times_ms),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_INPUT_PATH)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--oracle-count", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.oracle_count < 0 or arguments.oracle_count > arguments.count:
        raise SystemExit("--oracle-count must be between zero and --count")

    robot, human_targets = _human_targets(
        arguments.input, arguments.start, arguments.count
    )
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    oracle = NumericalExactSewOracle() if arguments.oracle_count else None
    policy_data = {
        "canonical": PolicyMeasurements(),
        "continuous": PolicyMeasurements(),
    }
    candidate_counts: list[float] = []
    oracle_counts: Counter[str] = Counter()
    oracle_evaluated = 0
    q_previous: np.ndarray | None = None
    verified_continuous_frames = 0

    for offset, human_target in enumerate(human_targets):
        frame = arguments.start + offset
        frame_started = time.perf_counter()
        try:
            target = human_arm_to_exact_sew_target(human_target, stereo)
            candidate_set = enumerate_exact_sew_candidates(
                target, robot, geometry, stereo
            )
        except StereoSewSingularityError:
            failure = BranchSelectionOutcome(
                SolverStatus.SEW_SINGULAR, None, None, None
            )
            elapsed = 1000.0 * (time.perf_counter() - frame_started)
            for measurements in policy_data.values():
                measurements.record(failure, elapsed)
            candidate_counts.append(0.0)
            continue
        except Exception as error:
            raise RuntimeError(
                f"Method-2 backend failed abnormally at CSV frame {frame}: {error}"
            ) from error

        enumeration_ms = 1000.0 * (time.perf_counter() - frame_started)
        candidate_counts.append(float(len(candidate_set.candidates)))

        selection_started = time.perf_counter()
        canonical = select_exact_sew_branch(candidate_set, branch_policy="canonical")
        canonical_ms = enumeration_ms + 1000.0 * (time.perf_counter() - selection_started)
        policy_data["canonical"].record(canonical, canonical_ms)

        previous_for_frame = None if q_previous is None else q_previous.copy()
        selection_started = time.perf_counter()
        continuous = select_exact_sew_branch(
            candidate_set,
            branch_policy="continuous",
            q_previous=previous_for_frame,
        )
        continuous_ms = enumeration_ms + 1000.0 * (time.perf_counter() - selection_started)
        expected_index = _independent_continuous_index(
            candidate_set, previous_for_frame
        )
        if continuous.candidate_index != expected_index:
            raise RuntimeError(
                f"continuous local-minimum rule failed at CSV frame {frame}: "
                f"selected {continuous.candidate_index}, expected {expected_index}"
            )
        verified_continuous_frames += 1
        policy_data["continuous"].record(continuous, continuous_ms)
        if continuous.candidate is not None:
            q_previous = continuous.candidate.q.copy()

        if oracle_evaluated < arguments.oracle_count:
            assert oracle is not None
            oracle_result = oracle.solve_pose_and_sew(target)
            oracle_evaluated += 1
            comparison = f"oracle_{oracle_result.status.value}__method2_{canonical.status.value}"
            oracle_counts[comparison] += 1
            if oracle_result.status is _EXACT and canonical.status in (
                SolverStatus.NO_VALID_BRANCH,
                SolverStatus.NUMERICAL_FAILURE,
            ):
                raise RuntimeError(
                    f"oracle exact / Method-2 {canonical.status.value} at CSV frame "
                    f"{frame}; rejection diagnostics={dict(candidate_set.rejection_counts)}"
                )

    print(
        "VALIDATION",
        {
            "input": str(arguments.input.resolve()),
            "start": arguments.start,
            "frames": arguments.count,
            "search_mode": "event_aware",
            "single_enumeration_per_frame": True,
            "continuous_local_minimum_verified_frames": verified_continuous_frames,
        },
    )
    _print_policy("CANONICAL", policy_data["canonical"], arguments.count)
    _print_policy("CONTINUOUS", policy_data["continuous"], arguments.count)
    print("CANDIDATE_COUNT", _mean_percentiles(candidate_counts))
    print(
        "ORACLE_SUBSET",
        {"frames": oracle_evaluated, "comparisons": dict(sorted(oracle_counts.items()))},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
