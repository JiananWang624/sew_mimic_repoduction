from __future__ import annotations

import numpy as np
import pytest

from sew_mimic.common.types import HumanArmTarget
from sew_mimic.warp.geometry import WarpArmGeometry, compute_adaptive_offset


def _human(task: np.ndarray) -> HumanArmTarget:
    return HumanArmTarget(np.zeros(3), np.array([.2, 0, .1]), np.array([.4, 0, .2]), np.eye(3), task)


def test_geometry_is_validated_and_immutable() -> None:
    geometry = WarpArmGeometry(np.zeros(3), .3, .25, np.array([.1, 0, 0]))
    assert not geometry.shoulder.flags.writeable
    with pytest.raises(ValueError):
        WarpArmGeometry(np.zeros(2), .3, .25, np.zeros(3))
    with pytest.raises(ValueError):
        WarpArmGeometry(np.zeros(3), 0, .25, np.zeros(3))


def test_adaptive_offset_supports_single_and_two_arm_centroids() -> None:
    geometry = WarpArmGeometry(np.zeros(3), .3, .25, np.array([.1, 0, 0]))
    human = _human(np.array([1., 2., 3.]))
    predicted = .3 * (human.elbow - human.shoulder) / np.linalg.norm(human.elbow - human.shoulder) + .25 * (human.wrist - human.elbow) / np.linalg.norm(human.wrist - human.elbow) + np.array([.1, 0, 0])
    np.testing.assert_allclose(compute_adaptive_offset([human], [geometry]), human.task_point - predicted)
    second = _human(np.array([3., 4., 5.]))
    np.testing.assert_allclose(compute_adaptive_offset([human, second], [geometry, geometry]), .5 * (human.task_point + second.task_point) - predicted)
