"""Canonical human task-point definition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import CONFIG


TaskPointMode = Literal["wrist", "wrist_plus_hand_offset"]
Vector = NDArray[np.float64]

_ROTATION_MATRIX_TOL = 1e-10


def validate_task_point_config(
    config: object,
) -> tuple[TaskPointMode, Vector]:
    """Validate and return the task-point mode and canonical-hand offset."""
    if not isinstance(config, Mapping):
        raise ValueError("task_point config must be a mapping")
    required = ("mode", "human_wrist_to_task_offset_m")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"task_point config is missing required keys: {missing}")
    mode = config["mode"]
    if mode not in ("wrist", "wrist_plus_hand_offset"):
        raise ValueError(
            "task_point.mode must be 'wrist' or 'wrist_plus_hand_offset'"
        )
    try:
        offset = np.asarray(config["human_wrist_to_task_offset_m"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "task_point.human_wrist_to_task_offset_m must contain numeric values"
        ) from error
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(
            "task_point.human_wrist_to_task_offset_m must be a finite array "
            "with shape (3,)"
        )
    return cast(TaskPointMode, mode), offset.copy()


DEFAULT_TASK_POINT_MODE, DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M = (
    validate_task_point_config(CONFIG.get("task_point"))
)


def compute_human_task_point(
    wrist: ArrayLike,
    hand_rotation: ArrayLike,
    *,
    mode: TaskPointMode | str = DEFAULT_TASK_POINT_MODE,
    human_wrist_to_task_offset_m: ArrayLike = DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
) -> Vector:
    """Return ``t_h`` in the same frame as ``wrist`` and ``hand_rotation``.

    For ``wrist``, ``t_h = w_h``. For ``wrist_plus_hand_offset``,
    ``t_h = w_h + H_h @ p_human_WT``, where the offset is expressed in the
    established canonical human hand frame.
    """
    wrist_vector = np.asarray(wrist, dtype=float)
    if wrist_vector.shape != (3,) or not np.all(np.isfinite(wrist_vector)):
        raise ValueError("wrist must be a finite array with shape (3,)")
    rotation = np.asarray(hand_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("hand_rotation must be a finite array with shape (3, 3)")
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=_ROTATION_MATRIX_TOL,
        rtol=0.0,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=_ROTATION_MATRIX_TOL,
        rtol=0.0,
    ):
        raise ValueError("hand_rotation must be a proper rotation matrix")
    offset = np.asarray(human_wrist_to_task_offset_m, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(
            "human_wrist_to_task_offset_m must be a finite array with shape (3,)"
        )

    if mode == "wrist":
        return wrist_vector.copy()
    if mode == "wrist_plus_hand_offset":
        return wrist_vector + rotation @ offset
    raise ValueError("mode must be 'wrist' or 'wrist_plus_hand_offset'")
