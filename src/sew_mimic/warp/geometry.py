"""Fixed-link geometry contract for the generic WARP c-SEW construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.types import HumanArmTarget

Vector = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str) -> Vector:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return array.copy()


def _readonly(value: Vector) -> Vector:
    return np.frombuffer(np.asarray(value, dtype=np.float64).tobytes(), dtype=np.float64)


def _unit(value: Vector, name: str) -> Vector:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


@dataclass(frozen=True)
class WarpArmGeometry:
    """Configuration-invariant generic WARP arm geometry, in one robot frame."""

    shoulder: Vector
    upper_arm_length: float
    forearm_length: float
    wrist_to_task: Vector

    def __post_init__(self) -> None:
        shoulder = _vector3(self.shoulder, "shoulder")
        wrist_to_task = _vector3(self.wrist_to_task, "wrist_to_task")
        upper = float(self.upper_arm_length)
        forearm = float(self.forearm_length)
        if not math.isfinite(upper) or upper <= 0.0:
            raise ValueError("upper_arm_length must be positive and finite")
        if not math.isfinite(forearm) or forearm <= 0.0:
            raise ValueError("forearm_length must be positive and finite")
        object.__setattr__(self, "shoulder", _readonly(shoulder))
        object.__setattr__(self, "wrist_to_task", _readonly(wrist_to_task))
        object.__setattr__(self, "upper_arm_length", upper)
        object.__setattr__(self, "forearm_length", forearm)


def compute_adaptive_offset(
    human_targets: tuple[HumanArmTarget, ...] | list[HumanArmTarget],
    robot_geometries: tuple[WarpArmGeometry, ...] | list[WarpArmGeometry],
) -> Vector:
    """Return WARP's centroid placement offset for one or more arms.

    For one arm this is a documented single-arm adaptation: its task-point
    difference is the centroid difference.  The returned vector never mutates
    a geometry or a robot base. Human points/orientations and robot shoulders
    must already use the same orientation-aligned coordinate frame; this
    function deliberately has no hidden frame transform.
    """
    humans, robots = tuple(human_targets), tuple(robot_geometries)
    if not humans or len(humans) != len(robots):
        raise ValueError("human_targets and robot_geometries must have equal nonzero length")
    predicted: list[Vector] = []
    tasks: list[Vector] = []
    for human, robot in zip(humans, robots, strict=True):
        if not isinstance(human, HumanArmTarget) or not isinstance(robot, WarpArmGeometry):
            raise ValueError("inputs must contain HumanArmTarget and WarpArmGeometry")
        upper = _unit(human.elbow - human.shoulder, "human upper-arm vector")
        lower = _unit(human.wrist - human.elbow, "human forearm vector")
        predicted.append(robot.shoulder + robot.upper_arm_length * upper + robot.forearm_length * lower + human.hand_rotation @ robot.wrist_to_task)
        tasks.append(human.task_point)
    return _readonly(
        np.mean(np.asarray(tasks), axis=0)
        - np.mean(np.asarray(predicted), axis=0)
    )
