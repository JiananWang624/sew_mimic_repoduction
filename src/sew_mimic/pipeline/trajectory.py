"""Shared, once-per-frame human preprocessing for comparative retargeting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..common import HumanArmTarget, compute_human_task_point
from ..common.task_point import DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M, DEFAULT_TASK_POINT_MODE
from ..csv_adapter import HumanCSVAdapter, load_human_trajectory_csv
from ..mounting import load_humanoid_mounted_gen3, world_trajectory_to_base
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference


@dataclass(frozen=True)
class TrajectoryFrame:
    frame: int
    target: HumanArmTarget


@dataclass(frozen=True)
class PreparedTrajectory:
    robot: Gen3Kinematics
    geometry: Gen3StereoSewGeometry
    stereo: StereoSew
    frames: tuple[TrajectoryFrame, ...]


def sample_frame_indices(
    total_frames: int,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[int, ...]:
    """Return the deterministic input-frame selection used by every method."""
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError("start_frame is outside the input trajectory")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1 when provided")
    indices = tuple(range(start_frame, total_frames, stride))
    return indices if max_frames is None else indices[:max_frames]


def prepare_trajectory(input_path: str | Path, *, start_frame: int = 0,
                       max_frames: int | None = None, stride: int = 1,
                       adapter: HumanCSVAdapter | None = None) -> PreparedTrajectory:
    """Mount once, transform once, and create the shared target objects once."""
    world = load_human_trajectory_csv(input_path, adapter)
    if not len(world):
        raise ValueError("human trajectory is empty")
    robot, data = load_humanoid_mounted_gen3(world.shoulders[0])
    base_id = int(robot.frame_body_ids[0])
    points = np.stack((world.shoulders, world.elbows, world.wrists), axis=1)
    points, rotations = world_trajectory_to_base(
        points, world.hand_orientations, data.xmat[base_id].reshape(3, 3), data.xpos[base_id]
    )
    indices = sample_frame_indices(
        len(world),
        start_frame=start_frame,
        max_frames=max_frames,
        stride=stride,
    )
    frames = []
    for index in indices:
        shoulder, elbow, wrist = points[index]
        hand = rotations[index]
        frames.append(TrajectoryFrame(index, HumanArmTarget(
            shoulder, elbow, wrist, hand,
            compute_human_task_point(wrist, hand, mode=DEFAULT_TASK_POINT_MODE,
                                     human_wrist_to_task_offset_m=DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M),
        )))
    return PreparedTrajectory(robot, Gen3StereoSewGeometry.from_robot(robot),
                              StereoSew(project_stereo_sew_reference()), tuple(frames))
