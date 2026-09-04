from __future__ import annotations

import numpy as np
import pytest

from sew_mimic.geometry import SP3Result, rot
import sew_mimic.warp.skeleton as skeleton_module

from sew_mimic.common.types import HumanArmTarget
from sew_mimic.sew.stereo import StereoSew, StereoSewReference
from sew_mimic.warp import WarpArmGeometry, WarpSkeletonStatus, construct_warp_skeleton


def _stereo() -> StereoSew:
    return StereoSew(StereoSewReference(np.array([0., 0., 1.]), np.array([1., 0., 0.])))


def _case() -> tuple[HumanArmTarget, WarpArmGeometry, np.ndarray]:
    # This exact chain was selected so the WARP positive SP3 root is present.
    shoulder, elbow, wrist = np.zeros(3), np.array([.2, .2, .1]), np.array([.45, 0., .2])
    task = wrist + np.array([.1, 0., 0.])
    human = HumanArmTarget(shoulder, elbow, wrist, np.eye(3), task)
    return human, WarpArmGeometry(shoulder, np.linalg.norm(elbow), np.linalg.norm(wrist - elbow), np.array([.1, 0., 0.])), task


def test_exact_skeleton_obeys_all_invariants_deterministically() -> None:
    human, geometry, target = _case()
    first = construct_warp_skeleton(human, target, geometry, _stereo())
    second = construct_warp_skeleton(human, target, geometry, _stereo())
    assert first.status is WarpSkeletonStatus.SUCCESS_EXACT
    assert first.exact and first.theta_sew is not None and first.theta_sew > 1e-12
    assert first.palm_error_m <= 1e-12 and first.upper_length_error_m <= 1e-12
    assert first.forearm_length_error_m <= 1e-12 and first.sew_error_rad <= 1e-10
    np.testing.assert_array_equal(first.elbow, second.elbow)


def test_cross_morphology_and_nonidentity_common_frame_are_exact() -> None:
    stereo = _stereo()
    robot_geometry = WarpArmGeometry(
        np.array([.1, -.2, .3]), .42, .31, np.array([.08, -.01, .02])
    )
    upper_direction = np.array([1., 0., 0.])
    forearm_direction = np.array([.3, np.sqrt(1. - .3**2), 0.])
    robot_wrist = (
        robot_geometry.shoulder
        + robot_geometry.upper_arm_length * upper_direction
        + robot_geometry.forearm_length * forearm_direction
    )
    human_shoulder = robot_wrist - .27 * upper_direction - .48 * forearm_direction
    human_elbow = human_shoulder + .27 * upper_direction
    hand = rot([.2, -.4, .7], .6)
    task = robot_wrist + hand @ robot_geometry.wrist_to_task
    human = HumanArmTarget(human_shoulder, human_elbow, robot_wrist, hand, task)
    base = construct_warp_skeleton(human, task, robot_geometry, stereo)
    assert base.status is WarpSkeletonStatus.SUCCESS_EXACT

    frame_rotation = rot([-.3, .8, .1], .9)
    frame_translation = np.array([.4, -.1, .2])
    transformed_human = HumanArmTarget(
        frame_rotation @ human.shoulder + frame_translation,
        frame_rotation @ human.elbow + frame_translation,
        frame_rotation @ human.wrist + frame_translation,
        frame_rotation @ human.hand_rotation,
        frame_rotation @ human.task_point + frame_translation,
    )
    transformed_geometry = WarpArmGeometry(
        frame_rotation @ robot_geometry.shoulder + frame_translation,
        robot_geometry.upper_arm_length,
        robot_geometry.forearm_length,
        robot_geometry.wrist_to_task,
    )
    transformed_stereo = StereoSew(
        StereoSewReference(
            frame_rotation @ stereo.reference.e_t,
            frame_rotation @ stereo.reference.e_r,
        )
    )
    transformed = construct_warp_skeleton(
        transformed_human,
        frame_rotation @ task + frame_translation,
        transformed_geometry,
        transformed_stereo,
    )
    assert transformed.status is WarpSkeletonStatus.SUCCESS_EXACT
    np.testing.assert_allclose(
        transformed.elbow,
        frame_rotation @ base.elbow + frame_translation,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        transformed.hand_rotation, frame_rotation @ base.hand_rotation, atol=2e-15
    )


def test_success_result_contract_rejects_incomplete_or_inexact_values() -> None:
    with pytest.raises(ValueError, match="exact must"):
        skeleton_module.WarpSkeletonResult(
            WarpSkeletonStatus.SUCCESS_EXACT, "invalid", exact=False
        )
    with pytest.raises(ValueError, match="complete skeleton"):
        skeleton_module.WarpSkeletonResult(
            WarpSkeletonStatus.SUCCESS_EXACT, "invalid", exact=True
        )


def test_no_exact_or_positive_root_and_singular_are_explicit() -> None:
    human, geometry, target = _case()
    unreachable = construct_warp_skeleton(human, np.array([10., 0., 0.]), geometry, _stereo())
    assert unreachable.status is WarpSkeletonStatus.NO_EXACT_SP3_ROOT
    singular_human = HumanArmTarget(np.zeros(3), np.array([.1, 0, .1]), np.array([0, 0, .4]), np.eye(3), np.array([.1, 0, .4]))
    singular = construct_warp_skeleton(singular_human, singular_human.task_point, geometry, _stereo())
    assert singular.status is WarpSkeletonStatus.SEW_SINGULAR


def test_least_squares_sp3_and_nonpositive_exact_roots_are_rejected(monkeypatch) -> None:
    human, geometry, target = _case()
    monkeypatch.setattr(
        skeleton_module, "sp3",
        lambda *args: SP3Result((.4,), (False,), (.1,), False, "least squares"),
    )
    assert construct_warp_skeleton(human, target, geometry, _stereo()).status is WarpSkeletonStatus.NO_EXACT_SP3_ROOT
    monkeypatch.setattr(
        skeleton_module, "sp3",
        lambda *args: SP3Result((0.0, -0.2), (True, True), (0.0, 0.0), False),
    )
    assert construct_warp_skeleton(human, target, geometry, _stereo()).status is WarpSkeletonStatus.NO_POSITIVE_EXACT_ROOT
