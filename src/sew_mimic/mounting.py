"""World-pose mounting utilities for the fixed Kinova Gen3 base."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import CONFIG
from .geometry import rot
from .kinematics import Gen3Kinematics, GEN3_SCENE_PATH, load_mujoco_model


Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]

_ROBOT_CONFIG = CONFIG["robot"]
HUMANOID_MOUNTING_NAME = str(_ROBOT_CONFIG["mounting_name"])
GEN3_JOINT1_IN_BASE = np.asarray(
    _ROBOT_CONFIG["joint1_in_base_m"], dtype=float
)
DEFAULT_ROBOT_WORLD_OFFSET = tuple(
    float(value) for value in _ROBOT_CONFIG["world_offset_m"]
)


@dataclass(frozen=True)
class MountingEvaluation:
    name: str
    root_rotation: Matrix
    root_position: Vector
    joint_positions: Matrix
    h3_world: Vector
    h5_world: Vector
    shoulder_to_wrist_direction: Vector
    morphology_score: float


def root_orientation_candidates() -> dict[str, Matrix]:
    """Return the required simple root-orientation candidates."""
    half_pi = 0.5 * np.pi
    return {
        "identity": np.eye(3),
        "Rx(+90deg)": rot([1.0, 0.0, 0.0], half_pi),
        "Rx(-90deg)": rot([1.0, 0.0, 0.0], -half_pi),
        "Ry(+90deg)": rot([0.0, 1.0, 0.0], half_pi),
        "Ry(-90deg)": rot([0.0, 1.0, 0.0], -half_pi),
    }


def humanoid_root_rotation() -> Matrix:
    """Return the fixed right-arm mounting selected from the Gen3 geometry."""
    return root_orientation_candidates()[HUMANOID_MOUNTING_NAME].copy()


def right_arm_base_position(human_shoulder_world: ArrayLike) -> Vector:
    """Place the base so its rotated joint-1 offset ends at the human shoulder."""
    shoulder = np.asarray(human_shoulder_world, dtype=float)
    if shoulder.shape != (3,) or not np.all(np.isfinite(shoulder)):
        raise ValueError("human_shoulder_world must be a finite length-3 vector")
    return shoulder - humanoid_root_rotation() @ GEN3_JOINT1_IN_BASE


def _matrix_to_quaternion(rotation: ArrayLike) -> Vector:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).ravel())
    return quaternion


def mount_gen3_root(
    robot: Gen3Kinematics,
    root_rotation: ArrayLike,
    shoulder_anchor_world: ArrayLike,
) -> mujoco.MjData:
    """Set only base_link's world pose and align joint 1 with the anchor."""
    model = robot.model
    base_body_id = int(robot.frame_body_ids[0])
    candidate_rotation = np.asarray(root_rotation, dtype=float)
    anchor = np.asarray(shoulder_anchor_world, dtype=float)
    if candidate_rotation.shape != (3, 3):
        raise ValueError("root_rotation must have shape (3, 3)")
    if anchor.shape != (3,):
        raise ValueError("shoulder_anchor_world must have shape (3,)")

    original_root_rotation = np.empty(9)
    mujoco.mju_quat2Mat(original_root_rotation, model.body_quat[base_body_id])
    mounted_rotation = candidate_rotation @ original_root_rotation.reshape(3, 3)
    model.body_quat[base_body_id] = _matrix_to_quaternion(mounted_rotation)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint1_id = int(robot.joint_ids[0])
    model.body_pos[base_body_id] += anchor - data.xanchor[joint1_id]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not np.allclose(data.xanchor[joint1_id], anchor, atol=1e-12, rtol=0.0):
        raise RuntimeError("failed to align Gen3 joint 1 with shoulder anchor")
    return data


def evaluate_root_orientation(
    name: str,
    root_rotation: ArrayLike,
    shoulder_anchor_world: ArrayLike,
) -> MountingEvaluation:
    """Evaluate one q=0 mounting using MuJoCo forward kinematics."""
    robot = Gen3Kinematics(load_mujoco_model(GEN3_SCENE_PATH))
    data = mount_gen3_root(robot, root_rotation, shoulder_anchor_world)
    model = robot.model
    base_body_id = int(robot.frame_body_ids[0])
    joint_positions = np.asarray(data.xanchor[robot.joint_ids], dtype=float).copy()
    shoulder_to_wrist = joint_positions[5] - joint_positions[0]
    shoulder_to_wrist /= np.linalg.norm(shoulder_to_wrist)
    horizontal_fraction = float(np.linalg.norm(shoulder_to_wrist[:2]))
    forward_fraction = max(float(shoulder_to_wrist[0]), 0.0)

    return MountingEvaluation(
        name=name,
        root_rotation=np.asarray(data.xmat[base_body_id], dtype=float).reshape(3, 3).copy(),
        root_position=np.asarray(data.xpos[base_body_id], dtype=float).copy(),
        joint_positions=joint_positions,
        h3_world=np.asarray(data.xaxis[int(robot.joint_ids[2])], dtype=float).copy(),
        h5_world=np.asarray(data.xaxis[int(robot.joint_ids[4])], dtype=float).copy(),
        shoulder_to_wrist_direction=shoulder_to_wrist,
        morphology_score=horizontal_fraction + 0.25 * forward_fraction,
    )


def evaluate_root_orientations(
    shoulder_anchor_world: ArrayLike,
) -> list[MountingEvaluation]:
    """Evaluate all required candidates without using a human target pose."""
    return [
        evaluate_root_orientation(name, rotation, shoulder_anchor_world)
        for name, rotation in root_orientation_candidates().items()
    ]


def select_humanoid_mounting(
    evaluations: list[MountingEvaluation],
) -> MountingEvaluation:
    """Select the numerically established fixed right-arm mounting."""
    for evaluation in evaluations:
        if evaluation.name == HUMANOID_MOUNTING_NAME:
            return evaluation
    raise ValueError(f"evaluations must include {HUMANOID_MOUNTING_NAME}")


def load_mounted_gen3(
    root_rotation: ArrayLike,
    shoulder_anchor_world: ArrayLike,
) -> tuple[Gen3Kinematics, mujoco.MjData]:
    """Load a fresh Gen3 and apply only its root world pose."""
    robot = Gen3Kinematics(load_mujoco_model(GEN3_SCENE_PATH))
    data = mount_gen3_root(robot, root_rotation, shoulder_anchor_world)
    return robot, data


def load_humanoid_mounted_gen3(
    human_shoulder_world: ArrayLike,
    robot_world_offset: ArrayLike = DEFAULT_ROBOT_WORLD_OFFSET,
) -> tuple[Gen3Kinematics, mujoco.MjData]:
    """Load Gen3 with the fixed right-arm pose plus an XYZ world offset."""
    shoulder = np.asarray(human_shoulder_world, dtype=float)
    offset = np.asarray(robot_world_offset, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("robot_world_offset must be a finite length-3 vector")
    robot_shoulder = shoulder + offset
    robot = Gen3Kinematics(load_mujoco_model(GEN3_SCENE_PATH))
    model = robot.model
    base_body_id = int(robot.frame_body_ids[0])
    model.body_quat[base_body_id] = _matrix_to_quaternion(humanoid_root_rotation())
    model.body_pos[base_body_id] = right_arm_base_position(robot_shoulder)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint1_world = data.xanchor[int(robot.joint_ids[0])]
    if not np.allclose(joint1_world, robot_shoulder, atol=1e-12, rtol=0.0):
        raise RuntimeError(
            "right-arm root pose did not place Gen3 joint_1 at the requested position"
        )
    return robot, data


def world_trajectory_to_base(
    points_world: ArrayLike,
    orientations_world: ArrayLike,
    rotation_world_from_base: ArrayLike,
    position_world_of_base: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Express world-frame trajectory points and orientations in the mounted base."""
    points = np.asarray(points_world, dtype=float)
    orientations = np.asarray(orientations_world, dtype=float)
    rotation = np.asarray(rotation_world_from_base, dtype=float)
    position = np.asarray(position_world_of_base, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (3, 3):
        raise ValueError("points_world must have shape (frames, 3 points, 3 coordinates)")
    if orientations.shape != (len(points), 3, 3):
        raise ValueError("orientations_world must have shape (frames, 3, 3)")
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("mounted base pose must contain a 3x3 rotation and length-3 position")

    points_base = (points - position) @ rotation
    orientations_base = np.einsum("ij,njk->nik", rotation.T, orientations)
    return points_base, orientations_base
