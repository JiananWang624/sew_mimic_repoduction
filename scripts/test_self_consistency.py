"""Run a 1000-pose FK-to-SEW-Mimic self-consistency check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.kinematics import Gen3Kinematics, gen3_kinematics  # noqa: E402
from sew_mimic.retarget import sew_mimic  # noqa: E402


def _random_configuration(
    robot: Gen3Kinematics,
    rng: np.random.Generator,
) -> np.ndarray:
    q = np.empty(robot.dof)
    for index, limited in enumerate(robot.joint_limited):
        if limited:
            q[index] = rng.uniform(*robot.joint_limits[index])
        else:
            q[index] = rng.uniform(-np.pi, np.pi)
    return q


def _vector_error_deg(first: np.ndarray, second: np.ndarray) -> float:
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


def _print_distribution(name: str, values: np.ndarray) -> None:
    percentiles = np.percentile(values, [0, 50, 90, 95, 99, 100])
    print(
        f"{name:<12} "
        f"min={percentiles[0]:.3e}  "
        f"median={percentiles[1]:.3e}  "
        f"p90={percentiles[2]:.3e}  "
        f"p95={percentiles[3]:.3e}  "
        f"p99={percentiles[4]:.3e}  "
        f"max={percentiles[5]:.3e}  "
        f"mean={np.mean(values):.3e} deg"
    )


def run_self_consistency(samples: int, seed: int) -> int:
    if samples <= 0:
        raise ValueError("samples must be positive")

    robot = gen3_kinematics()
    rng = np.random.default_rng(seed)
    errors = np.empty((samples, 3))
    failures: list[tuple[int, np.ndarray, np.ndarray, str]] = []

    for sample_index in range(samples):
        q_target = _random_configuration(robot, rng)
        q0 = _random_configuration(robot, rng)

        upper_direction = robot.R_0_i(q_target, 3) @ robot.axes[2]
        lower_direction = robot.R_0_i(q_target, 5) @ robot.axes[4]
        hand_orientation = robot.aligned_ee_rotation(q_target)

        shoulder = np.zeros(3)
        elbow = shoulder + upper_direction
        wrist = elbow + lower_direction

        try:
            q_result, diagnostics = sew_mimic(
                q0,
                shoulder,
                elbow,
                wrist,
                hand_orientation,
            )
        except ValueError as error:
            failures.append((sample_index, q_target, q0, str(error)))
            errors[sample_index] = np.nan
            continue

        result_upper = robot.R_0_i(q_result, 3) @ robot.axes[2]
        result_lower = robot.R_0_i(q_result, 5) @ robot.axes[4]
        result_hand = robot.aligned_ee_rotation(q_result)
        errors[sample_index] = [
            _vector_error_deg(result_upper, upper_direction),
            _vector_error_deg(result_lower, lower_direction),
            _rotation_error_deg(result_hand, hand_orientation),
        ]

        independent_diagnostics = np.array(
            [
                diagnostics["upper_arm_error_deg"],
                diagnostics["lower_arm_error_deg"],
                diagnostics["wrist_rotation_error_deg"],
            ]
        )
        np.testing.assert_allclose(errors[sample_index], independent_diagnostics, atol=1e-12)
        if not diagnostics["joint_limit_valid"]:
            failures.append((sample_index, q_target, q0, "result violates joint limits"))

    successful = errors[~np.isnan(errors).any(axis=1)]
    print(f"samples={samples}  successful={len(successful)}  failures={len(failures)}  seed={seed}")
    if len(successful):
        print("orientation error distribution (degrees)")
        _print_distribution("upper arm", successful[:, 0])
        _print_distribution("lower arm", successful[:, 1])
        _print_distribution("wrist", successful[:, 2])

    if failures:
        print("\nfirst failed samples (joint/frame diagnostic inputs)")
        for index, q_target, q0, message in failures[:5]:
            print(f"sample {index}: {message}")
            print(f"  q_target={np.array2string(q_target, precision=8)}")
            print(f"  q0={np.array2string(q0, precision=8)}")
        return 1

    error_threshold_deg = 1e-8
    worst_flat_index = int(np.argmax(successful))
    worst_sample, worst_component = np.unravel_index(worst_flat_index, successful.shape)
    if successful[worst_sample, worst_component] > error_threshold_deg:
        labels = ("upper arm", "lower arm", "wrist")
        print(
            f"large {labels[worst_component]} error at successful sample "
            f"{worst_sample}: {successful[worst_sample, worst_component]:.6e} deg"
        )
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260831)
    arguments = parser.parse_args()
    return run_self_consistency(arguments.samples, arguments.seed)


if __name__ == "__main__":
    raise SystemExit(main())
