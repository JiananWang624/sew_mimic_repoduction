"""Model-derived forward kinematics for the MuJoCo Menagerie Kinova Gen3."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import rot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEN3_SCENE_PATH = PROJECT_ROOT / "assets" / "kinova_gen3" / "scene.xml"

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


def load_mujoco_model(xml_path: str | Path) -> mujoco.MjModel:
    """Load an MJCF model from disk."""
    return mujoco.MjModel.from_xml_path(str(Path(xml_path)))


def revolute_joint_ids(model: mujoco.MjModel) -> list[int]:
    """Return the IDs of all revolute (MuJoCo hinge) joints."""
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    return [joint_id for joint_id in range(model.njnt) if model.jnt_type[joint_id] == hinge]


def controlled_revolute_joint_ids(model: mujoco.MjModel) -> list[int]:
    """Return unique revolute joints targeted by joint actuators."""
    joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
    revolute = set(revolute_joint_ids(model))
    controlled: list[int] = []

    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != joint_transmission:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id in revolute and joint_id not in controlled:
            controlled.append(joint_id)

    return controlled


def validate_gen3_arm(model: mujoco.MjModel) -> list[int]:
    """Validate that the model is a seven-joint, fully actuated revolute arm."""
    revolute = revolute_joint_ids(model)
    controlled = controlled_revolute_joint_ids(model)

    if len(revolute) != 7:
        raise ValueError(f"Expected 7 revolute joints, found {len(revolute)}")
    if controlled != revolute:
        raise ValueError(
            "Expected every revolute joint to have one joint actuator; "
            f"revolute IDs={revolute}, controlled IDs={controlled}"
        )
    if model.nu != 7:
        raise ValueError(f"Expected 7 actuators, found {model.nu}")

    return controlled


def _quat_to_matrix(quaternion: ArrayLike) -> Matrix:
    quat = np.asarray(quaternion, dtype=float)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must be a finite array with shape (4,)")
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _align_x_axis_with(direction: ArrayLike) -> Matrix:
    """Return the minimum-angle rotation taking +X to ``direction``."""
    target = np.asarray(direction, dtype=float)
    target /= np.linalg.norm(target)
    x_axis = np.array([1.0, 0.0, 0.0])
    cross = np.cross(x_axis, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(x_axis @ target, -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        return rot([0.0, 1.0, 0.0], np.pi)
    return rot(cross / sine, np.arctan2(sine, cosine))


class Gen3Kinematics:
    """Seven-joint Kinova kinematics extracted directly from a MuJoCo model.

    Frame 0 is ``base_link``. Frames 1 through 7 are the child-body frames
    containing ``joint_1`` through ``joint_7`` respectively.
    """

    dof = 7

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.joint_ids = np.asarray(validate_gen3_arm(model), dtype=int)
        self.joint_names = tuple(
            self._name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in self.joint_ids
        )
        joint_body_ids = np.asarray(model.jnt_bodyid[self.joint_ids], dtype=int)
        base_body_id = int(model.body_parentid[joint_body_ids[0]])
        self.frame_body_ids = np.concatenate(([base_body_id], joint_body_ids))
        self.frame_names = tuple(
            self._name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in self.frame_body_ids
        )
        self._validate_serial_chain(model)

        self.axes = np.asarray(model.jnt_axis[self.joint_ids], dtype=float).copy()
        axis_norms = np.linalg.norm(self.axes, axis=1)
        if np.any(axis_norms <= 1e-12):
            raise ValueError("all seven model joint axes must be nonzero")
        self.axes /= axis_norms[:, None]
        self.joint_positions = np.asarray(model.jnt_pos[self.joint_ids], dtype=float).copy()

        self.fixed_parent_to_child = np.repeat(np.eye(4)[None, :, :], self.dof, axis=0)
        for index, body_id in enumerate(joint_body_ids):
            self.fixed_parent_to_child[index, :3, :3] = _quat_to_matrix(model.body_quat[body_id])
            self.fixed_parent_to_child[index, :3, 3] = model.body_pos[body_id]

        self.joint_limited = np.asarray(model.jnt_limited[self.joint_ids], dtype=bool).copy()
        raw_limits = np.asarray(model.jnt_range[self.joint_ids], dtype=float)
        self.joint_limits = np.column_stack(
            (
                np.where(self.joint_limited, raw_limits[:, 0], -np.inf),
                np.where(self.joint_limited, raw_limits[:, 1], np.inf),
            )
        )

        pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        if pinch_site_id < 0:
            raise ValueError("Kinova model must contain the pinch_site end-effector frame")
        if int(model.site_bodyid[pinch_site_id]) != joint_body_ids[-1]:
            raise ValueError("pinch_site must be attached to the joint_7 body")
        self.ee_position_in_7 = np.asarray(model.site_pos[pinch_site_id], dtype=float).copy()
        self.ee_rotation_in_7 = _quat_to_matrix(model.site_quat[pinch_site_id])

        # The paper's R_align makes the tool +X pointing direction parallel
        # to h7. Menagerie's pinch_site uses a different axis convention, so
        # derive the smallest fixed convention rotation instead of guessing
        # an axis permutation.
        pointing_axis_in_tool = self.ee_rotation_in_7.T @ self.axes[-1]
        self.ee_alignment = _align_x_axis_with(pointing_axis_in_tool)

    @staticmethod
    def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
        name = mujoco.mj_id2name(model, object_type, int(object_id))
        if name is None:
            raise ValueError(f"model object {object_type} with ID {object_id} is unnamed")
        return name

    def _validate_serial_chain(self, model: mujoco.MjModel) -> None:
        for index in range(1, self.frame_body_ids.size):
            child_id = int(self.frame_body_ids[index])
            expected_parent_id = int(self.frame_body_ids[index - 1])
            if int(model.body_parentid[child_id]) != expected_parent_id:
                raise ValueError("the seven Gen3 joint bodies must form one serial chain")

    @staticmethod
    def _configuration(q: ArrayLike) -> Vector:
        configuration = np.asarray(q, dtype=float)
        if configuration.shape != (7,):
            raise ValueError(f"q must have shape (7,), got {configuration.shape}")
        if not np.all(np.isfinite(configuration)):
            raise ValueError("q must contain only finite values")
        return configuration

    @staticmethod
    def _frame_index(i: int) -> int:
        if not isinstance(i, (int, np.integer)) or not 0 <= int(i) <= 7:
            raise ValueError("i must be an integer in [0, 7]")
        return int(i)

    def T_0_i(self, q: ArrayLike, i: int) -> Matrix:
        """Return the homogeneous transform from frame ``i`` to frame 0."""
        configuration = self._configuration(q)
        frame_index = self._frame_index(i)
        transform = np.eye(4)

        for joint_index in range(frame_index):
            fixed = self.fixed_parent_to_child[joint_index]
            joint_rotation = rot(self.axes[joint_index], configuration[joint_index])
            joint_position = self.joint_positions[joint_index]

            parent_to_child = np.eye(4)
            parent_to_child[:3, :3] = fixed[:3, :3] @ joint_rotation
            parent_to_child[:3, 3] = fixed[:3, 3] + fixed[:3, :3] @ (
                joint_position - joint_rotation @ joint_position
            )
            transform = transform @ parent_to_child

        return transform

    def R_0_i(self, q: ArrayLike, i: int) -> Matrix:
        """Return the rotation from frame ``i`` to frame 0."""
        return self.T_0_i(q, i)[:3, :3]

    def ee_transform(self, q: ArrayLike) -> Matrix:
        """Return the homogeneous transform from ``pinch_site`` to frame 0."""
        transform = self.T_0_i(q, 7)
        local = np.eye(4)
        local[:3, :3] = self.ee_rotation_in_7
        local[:3, 3] = self.ee_position_in_7
        return transform @ local

    def ee_rotation(self, q: ArrayLike) -> Matrix:
        """Return the ``pinch_site`` end-effector rotation in frame 0."""
        return self.ee_transform(q)[:3, :3]

    def aligned_ee_rotation(self, q: ArrayLike) -> Matrix:
        """Return the paper-convention end-effector orientation ``T(q)``."""
        return self.ee_rotation(q) @ self.ee_alignment


@lru_cache(maxsize=1)
def gen3_kinematics() -> Gen3Kinematics:
    """Load and cache kinematic parameters from the Menagerie model."""
    return Gen3Kinematics(load_mujoco_model(GEN3_SCENE_PATH))


def R_0_i(q: ArrayLike, i: int) -> Matrix:
    """Return the Kinova frame-``i`` rotation in base frame 0."""
    return gen3_kinematics().R_0_i(q, i)


def T_0_i(q: ArrayLike, i: int) -> Matrix:
    """Return the Kinova frame-``i`` homogeneous transform in base frame 0."""
    return gen3_kinematics().T_0_i(q, i)


def ee_rotation(q: ArrayLike) -> Matrix:
    """Return the Kinova ``pinch_site`` rotation in base frame 0."""
    return gen3_kinematics().ee_rotation(q)
