import numpy as np
import pytest

from sew_mimic.common import HumanArmTarget, compute_human_task_point
from sew_mimic.common.task_point import validate_task_point_config
from sew_mimic.geometry import rot


def test_wrist_task_point_mode_returns_wrist_without_aliasing() -> None:
    wrist = np.array([0.2, -0.3, 0.4])

    task_point = compute_human_task_point(
        wrist,
        np.eye(3),
        mode="wrist",
        human_wrist_to_task_offset_m=[0.1, 0.2, 0.3],
    )
    wrist[0] = 99.0

    np.testing.assert_array_equal(task_point, [0.2, -0.3, 0.4])


def test_wrist_plus_hand_offset_uses_canonical_hand_frame() -> None:
    wrist = np.array([0.2, -0.3, 0.4])
    hand = rot([0.3, -0.4, 0.5], 0.8)
    offset = np.array([0.05, -0.02, 0.1])

    task_point = compute_human_task_point(
        wrist,
        hand,
        mode="wrist_plus_hand_offset",
        human_wrist_to_task_offset_m=offset,
    )

    np.testing.assert_allclose(task_point, wrist + hand @ offset, atol=2e-16)


@pytest.mark.parametrize(
    "offset",
    [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4], [0.1, np.nan, 0.3]],
)
def test_task_point_rejects_invalid_offsets(offset: list[float]) -> None:
    with pytest.raises(ValueError, match="human_wrist_to_task_offset_m"):
        compute_human_task_point(
            np.zeros(3),
            np.eye(3),
            mode="wrist_plus_hand_offset",
            human_wrist_to_task_offset_m=offset,
        )


def test_task_point_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        compute_human_task_point(np.zeros(3), np.eye(3), mode="palm")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (None, "must be a mapping"),
        ({"mode": "wrist"}, "missing required keys"),
        (
            {"mode": "palm", "human_wrist_to_task_offset_m": [0.0, 0.0, 0.0]},
            "task_point.mode",
        ),
        (
            {
                "mode": "wrist",
                "human_wrist_to_task_offset_m": [0.0, np.nan, 0.0],
            },
            "finite array",
        ),
    ],
)
def test_task_point_config_validation_is_actionable(
    config: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_task_point_config(config)


def test_human_arm_target_copies_valid_arrays() -> None:
    shoulder = np.array([0.1, 0.2, 0.3])
    target = HumanArmTarget(
        shoulder=shoulder,
        elbow=[0.2, 0.3, 0.4],
        wrist=[0.3, 0.4, 0.5],
        hand_rotation=np.eye(3),
        task_point=[0.35, 0.45, 0.55],
    )
    shoulder[0] = 99.0

    np.testing.assert_array_equal(target.shoulder, [0.1, 0.2, 0.3])
    assert target.hand_rotation.shape == (3, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shoulder", [0.0, 0.0]),
        ("elbow", [0.0, np.inf, 0.0]),
        ("wrist", [0.0, 0.0, np.nan]),
        ("task_point", [0.0, 0.0, 0.0, 0.0]),
        ("hand_rotation", np.diag([1.0, 1.0, 2.0])),
        ("hand_rotation", np.diag([1.0, 1.0, -1.0])),
    ],
)
def test_human_arm_target_rejects_invalid_fields(field: str, value: object) -> None:
    arguments = {
        "shoulder": np.zeros(3),
        "elbow": np.ones(3),
        "wrist": 2.0 * np.ones(3),
        "hand_rotation": np.eye(3),
        "task_point": 3.0 * np.ones(3),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        HumanArmTarget(**arguments)
