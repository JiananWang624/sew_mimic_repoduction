import numpy as np
import pytest

import sew_mimic.sew.legacy_adapter as legacy_adapter
from sew_mimic.common import SolverStatus
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.retarget import sew_mimic
from sew_mimic.sew import solve_legacy_sew_mimic


def _reachable_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    robot = gen3_kinematics()
    desired = np.array([0.4, -0.6, 0.3, 0.8, -0.5, 0.7, -0.2])
    q0 = desired + np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01])
    upper_arm = robot.R_0_i(desired, 3) @ robot.arm_proxy_axis(3)
    lower_arm = robot.R_0_i(desired, 5) @ robot.arm_proxy_axis(5)
    shoulder = np.array([0.1, -0.2, 0.3])
    elbow = shoulder + 0.31 * upper_arm
    wrist = elbow + 0.27 * lower_arm
    hand = robot.aligned_ee_rotation(desired)
    return q0, shoulder, elbow, wrist, hand


def test_legacy_adapter_returns_exactly_the_direct_legacy_configuration() -> None:
    inputs = _reachable_case()
    q_direct, diagnostics_direct = sew_mimic(*inputs)

    result = solve_legacy_sew_mimic(*inputs)

    assert result.status is SolverStatus.SUCCESS_EXACT
    assert result.q is not None
    np.testing.assert_array_equal(result.q, q_direct)
    assert result.diagnostics.metadata["upper_arm_error_deg"] == diagnostics_direct[
        "upper_arm_error_deg"
    ]
    assert result.diagnostics.metadata["lower_arm_error_deg"] == diagnostics_direct[
        "lower_arm_error_deg"
    ]
    assert result.diagnostics.metadata["wrist_rotation_error_deg"] == diagnostics_direct[
        "wrist_rotation_error_deg"
    ]
    assert result.diagnostics.position_error_m is None
    assert (
        result.diagnostics.metadata["constraint_set"]
        == "legacy_sew_direction_orientation"
    )


def test_legacy_adapter_is_deterministic_for_identical_input() -> None:
    inputs = _reachable_case()

    first = solve_legacy_sew_mimic(*inputs)
    second = solve_legacy_sew_mimic(*inputs)

    assert first.status is second.status
    assert first.q is not None and second.q is not None
    np.testing.assert_array_equal(first.q, second.q)


def test_legacy_adapter_success_has_finite_seven_joint_contract() -> None:
    result = solve_legacy_sew_mimic(*_reachable_case())

    assert result.status is SolverStatus.SUCCESS_EXACT
    assert result.q is not None
    assert result.q.shape == (7,)
    assert np.all(np.isfinite(result.q))


def test_legacy_adapter_maps_degenerate_input_without_changing_direct_api() -> None:
    q0 = np.zeros(7)
    shoulder = np.zeros(3)
    elbow = np.zeros(3)
    wrist = np.array([1.0, 0.0, 0.0])
    hand = np.eye(3)

    with pytest.raises(ValueError, match="upper arm"):
        sew_mimic(q0, shoulder, elbow, wrist, hand)

    result = solve_legacy_sew_mimic(q0, shoulder, elbow, wrist, hand)

    assert result.status is SolverStatus.INVALID_INPUT
    assert result.q is None
    assert result.message is not None and "upper arm" in result.message


@pytest.mark.parametrize(
    ("shoulder", "elbow", "wrist"),
    [
        ([0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        ([0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 1.0, 0.0]),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.inf, 1.0, 0.0]),
    ],
)
def test_legacy_adapter_maps_malformed_landmarks_to_invalid_input(
    shoulder: list[float], elbow: list[float], wrist: list[float]
) -> None:
    result = solve_legacy_sew_mimic(
        np.zeros(7), shoulder, elbow, wrist, np.eye(3)
    )

    assert result.status is SolverStatus.INVALID_INPUT
    assert result.q is None


def test_legacy_adapter_never_labels_nonexact_legacy_residuals_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        legacy_adapter,
        "sew_mimic",
        lambda *args: (
            np.zeros(7),
            {
                "upper_arm_error_deg": 0.1,
                "lower_arm_error_deg": 0.0,
                "wrist_rotation_error_deg": 0.0,
                "joint_limit_valid": True,
            },
        ),
    )

    result = legacy_adapter.solve_legacy_sew_mimic(
        np.zeros(7), np.zeros(3), np.ones(3), 2.0 * np.ones(3), np.eye(3)
    )

    assert result.status is SolverStatus.SUCCESS_APPROX
    assert result.q is not None
