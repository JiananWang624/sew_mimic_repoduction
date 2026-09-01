from pathlib import Path

import numpy as np
import pandas as pd

import scripts.replay_csv as replay_csv
from sew_mimic.csv_adapter import HumanTrajectory, SHOULDER_ANCHOR_WORLD
from sew_mimic.mounting import load_humanoid_mounted_gen3


def _trajectory(frame_count: int = 3) -> HumanTrajectory:
    shoulders = np.zeros((frame_count, 3))
    elbows = np.tile([1.0, 0.0, 0.0], (frame_count, 1))
    wrists = elbows + np.tile([0.0, 1.0, 0.0], (frame_count, 1))
    orientations = np.tile(np.eye(3), (frame_count, 1, 1))
    return HumanTrajectory(shoulders, elbows, wrists, orientations)


def test_trajectory_in_mounted_base_matches_rigid_world_transform() -> None:
    robot, data = load_humanoid_mounted_gen3(SHOULDER_ANCHOR_WORLD)
    trajectory_world = _trajectory(2)
    base_body_id = int(robot.frame_body_ids[0])
    rotation = data.xmat[base_body_id].reshape(3, 3).copy()
    position = data.xpos[base_body_id].copy()

    trajectory_base = replay_csv.trajectory_in_mounted_base(
        trajectory_world, robot, data
    )

    np.testing.assert_allclose(
        trajectory_base.shoulders,
        (trajectory_world.shoulders - position) @ rotation,
        atol=2e-16,
    )
    np.testing.assert_allclose(
        trajectory_base.hand_orientations,
        np.einsum("ij,njk->nik", rotation.T, trajectory_world.hand_orientations),
        atol=5e-16,
    )


def test_retarget_trajectory_seeds_each_frame_with_previous_solution(monkeypatch) -> None:
    seen_q0: list[np.ndarray] = []

    def fake_sew_mimic(q0, shoulder, elbow, wrist, hand):
        seen_q0.append(np.asarray(q0).copy())
        q = np.asarray(q0) + 0.01
        return q, {
            "upper_arm_error_deg": 1.0,
            "lower_arm_error_deg": 2.0,
            "wrist_rotation_error_deg": 3.0,
            "joint_limit_valid": True,
        }

    monkeypatch.setattr(replay_csv, "sew_mimic", fake_sew_mimic)

    configurations, errors = replay_csv.retarget_trajectory(_trajectory())

    np.testing.assert_allclose(seen_q0[0], np.zeros(7), atol=0.0)
    np.testing.assert_allclose(seen_q0[1], np.full(7, 0.01), atol=0.0)
    np.testing.assert_allclose(seen_q0[2], np.full(7, 0.02), atol=0.0)
    np.testing.assert_allclose(configurations[-1], np.full(7, 0.03), atol=1e-16)
    np.testing.assert_allclose(errors, np.tile([1.0, 2.0, 3.0], (3, 1)), atol=0.0)


def test_save_retargeted_csv_has_requested_columns(tmp_path: Path) -> None:
    configurations = np.arange(14, dtype=float).reshape(2, 7) / 10.0
    errors = np.array([[1e-12, 2e-12, 3e-12], [4e-12, 5e-12, 6e-12]])
    output_path = tmp_path / "retargeted.csv"

    replay_csv.save_retargeted_csv(output_path, configurations, errors)
    table = pd.read_csv(output_path)

    assert tuple(table.columns) == replay_csv.OUTPUT_COLUMNS
    np.testing.assert_allclose(table.iloc[:, :7], configurations, atol=0.0)
    np.testing.assert_allclose(table.iloc[:, 7:], errors, rtol=1e-15)


def test_load_segment_boundaries_separates_labeled_events(tmp_path: Path) -> None:
    path = tmp_path / "segments.csv"
    pd.DataFrame(
        {
            "bite_id": [1, 1, 1, 1, 2],
            "motive_frame": [10, 11, 20, 21, 30],
            "event": ["transfer", "transfer", "withdrawal", "withdrawal", "transfer"],
            "event_frame_index": [0, 1, 0, 1, 0],
        }
    ).to_csv(path, index=False)

    np.testing.assert_array_equal(
        replay_csv.load_segment_boundaries(path),
        [False, True, False, True],
    )
