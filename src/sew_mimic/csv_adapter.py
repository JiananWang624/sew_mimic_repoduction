"""Adapter from the human trajectory CSV schema to SEW-Mimic inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .human_input import transform_human_to_robot_body_frame, wrist_euler_to_rotation


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

POSITION_COLUMNS = (
    "Shoulder_X",
    "Shoulder_Y",
    "Shoulder_Z",
    "Elbow_X",
    "Elbow_Y",
    "Elbow_Z",
    "Wrist_X",
    "Wrist_Y",
    "Wrist_Z",
)
EULER_COLUMNS = ("Wrist_Rx", "Wrist_Ry", "Wrist_Rz")
REQUIRED_COLUMNS = POSITION_COLUMNS + EULER_COLUMNS

# OptiTrack Motive uses X-right, Y-up, Z-back for this recording. Gen3 frame
# 0 uses X-forward, Y-left, Z-up, giving (x, y, z)_robot = (-z, -x, y)_csv.
MOTIVE_TO_GEN3_BODY_ROTATION = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)


@dataclass(frozen=True)
class HumanTrajectory:
    shoulders: NDArray[np.float64]
    elbows: NDArray[np.float64]
    wrists: NDArray[np.float64]
    hand_orientations: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.shoulders)


@dataclass(frozen=True)
class HumanCSVAdapter:
    """Apply every CSV-to-robot coordinate conversion in one place.

    Wrist angles are fixed to the supplied convention: intrinsic XYZ Euler
    angles in radians. The default frame transform is the recorded Motive
    frame to the Gen3 body frame.
    """

    rotation_robot_from_csv: ArrayLike = field(
        default_factory=lambda: MOTIVE_TO_GEN3_BODY_ROTATION.copy()
    )
    translation_robot_from_csv: ArrayLike = field(default_factory=lambda: np.zeros(3))
    position_scale: float = 1.0

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_robot_from_csv, dtype=float)
        translation = np.asarray(self.translation_robot_from_csv, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_robot_from_csv must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
            raise ValueError("rotation_robot_from_csv must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("rotation_robot_from_csv must have determinant +1")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_robot_from_csv must be a finite length-3 vector")
        if not np.isfinite(self.position_scale) or self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive and finite")
        object.__setattr__(self, "rotation_robot_from_csv", rotation.copy())
        object.__setattr__(self, "translation_robot_from_csv", translation.copy())

    def adapt_frame(
        self,
        shoulder: ArrayLike,
        elbow: ArrayLike,
        wrist: ArrayLike,
        wrist_euler: ArrayLike,
    ) -> tuple[Vector, Vector, Vector, Matrix]:
        """Convert one CSV frame into robot-body-frame ``(s, e, w, H)``."""
        hand_orientation_csv = wrist_euler_to_rotation(
            wrist_euler,
            order="xyz",
            degrees=False,
            convention="intrinsic",
        )
        return transform_human_to_robot_body_frame(
            self.position_scale * np.asarray(shoulder, dtype=float),
            self.position_scale * np.asarray(elbow, dtype=float),
            self.position_scale * np.asarray(wrist, dtype=float),
            hand_orientation_csv,
            rotation_robot_from_human=self.rotation_robot_from_csv,
            translation_robot_from_human=self.translation_robot_from_csv,
        )


def load_human_trajectory_csv(
    path: str | Path,
    adapter: HumanCSVAdapter | None = None,
) -> HumanTrajectory:
    """Load and adapt every row of a human trajectory CSV."""
    table = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    values = table.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=1))
        raise ValueError(f"CSV contains non-finite required values in rows {bad_rows.tolist()}")

    converter = adapter if adapter is not None else HumanCSVAdapter()
    frame_count = len(table)
    shoulders = np.empty((frame_count, 3))
    elbows = np.empty((frame_count, 3))
    wrists = np.empty((frame_count, 3))
    hand_orientations = np.empty((frame_count, 3, 3))

    for frame, row in enumerate(values):
        shoulder, elbow, wrist, hand = converter.adapt_frame(
            row[0:3], row[3:6], row[6:9], row[9:12]
        )
        shoulders[frame] = shoulder
        elbows[frame] = elbow
        wrists[frame] = wrist
        hand_orientations[frame] = hand

    return HumanTrajectory(shoulders, elbows, wrists, hand_orientations)
