import numpy as np
import pytest

import sew_mimic.retarget as retarget_module
from sew_mimic.kinematics import Gen3Kinematics, gen3_kinematics
from sew_mimic.retarget import align_axis, align_wrist, sew_mimic


RNG = np.random.default_rng(20260831)


def _random_configuration(robot: Gen3Kinematics) -> np.ndarray:
    q = np.empty(robot.dof)
    for index, limited in enumerate(robot.joint_limited):
        if limited:
            lower, upper = robot.joint_limits[index]
            margin = 0.05 * (upper - lower)
            q[index] = RNG.uniform(lower + margin, upper - margin)
        else:
            q[index] = RNG.uniform(-np.pi, np.pi)
    return q


def _rotation_angle(rotation: np.ndarray) -> float:
    sine = 0.5 * np.linalg.norm(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    cosine = 0.5 * (np.trace(rotation) - 1.0)
    return float(np.arctan2(sine, cosine))


@pytest.mark.parametrize("axis_index", [3, 5, 7])
def test_align_axis_random_targets_are_aligned_to_machine_precision(axis_index: int) -> None:
    robot = gen3_kinematics()
    first_q_index = axis_index - 3
    second_q_index = axis_index - 2

    for _ in range(100):
        q0 = _random_configuration(robot)
        desired = q0.copy()
        desired[first_q_index] = RNG.uniform(-np.pi, np.pi)
        lower, upper = robot.joint_limits[second_q_index]
        margin = 0.05 * (upper - lower)
        desired[second_q_index] = RNG.uniform(lower + margin, upper - margin)
        axis_to_align = (
            robot.arm_proxy_axis(axis_index)
            if axis_index in (3, 5)
            else robot.axes[axis_index - 1]
        )
        target = robot.R_0_i(desired, axis_index) @ axis_to_align

        solution = align_axis(axis_index, q0, target, robot)
        result = q0.copy()
        result[[first_q_index, second_q_index]] = solution
        aligned_axis = robot.R_0_i(result, axis_index) @ axis_to_align

        assert np.linalg.norm(aligned_axis - target) < 2e-12


def test_align_axis_selects_solution_closest_to_q0(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = gen3_kinematics()
    q0 = np.zeros(7)
    q0[1] = 0.4
    monkeypatch.setattr(
        retarget_module,
        "sp2",
        lambda *args: np.array([[0.05, -0.1], [1.0, -0.8]]),
    )

    solution = align_axis(3, q0, [1.0, 0.0, 0.0], robot)

    np.testing.assert_allclose(solution, [0.05, 0.3], atol=0.0)


def test_align_axis_filters_joint_limit_violations(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = gen3_kinematics()
    q0 = np.zeros(7)
    q0[1] = 2.0
    monkeypatch.setattr(
        retarget_module,
        "sp2",
        lambda *args: np.array([[0.01, 0.5], [0.8, -0.5]]),
    )

    solution = align_axis(3, q0, [1.0, 0.0, 0.0], robot)

    np.testing.assert_allclose(solution, [0.8, 1.5], atol=0.0)


def test_align_axis_raises_when_all_solutions_violate_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = gen3_kinematics()
    q0 = np.zeros(7)
    monkeypatch.setattr(
        retarget_module,
        "sp2",
        lambda *args: np.array([[0.1, 3.0], [-0.1, -3.0]]),
    )

    with pytest.raises(ValueError, match="joint limits"):
        align_axis(3, q0, [1.0, 0.0, 0.0], robot)


def test_lower_align_uses_signed_proxy_but_native_predecessor_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = gen3_kinematics()
    q0 = np.zeros(7)
    captured: tuple[np.ndarray, ...] | None = None

    def capture_sp2(*args: np.ndarray) -> np.ndarray:
        nonlocal captured
        captured = tuple(np.asarray(argument).copy() for argument in args)
        return np.zeros((1, 2))

    monkeypatch.setattr(retarget_module, "sp2", capture_sp2)
    align_axis(5, q0, robot.arm_proxy_axis(5), robot)

    assert captured is not None
    rotation_0_to_3 = robot.R_0_i(q0, 3)
    expected_proxy_in_3 = (
        rotation_0_to_3.T
        @ robot.R_0_i(q0, 5)
        @ robot.arm_proxy_axis(5)
    )
    expected_h4_in_3 = (
        rotation_0_to_3.T @ robot.R_0_i(q0, 4) @ robot.axes[3]
    )
    np.testing.assert_allclose(captured[1], expected_proxy_in_3, atol=0.0)
    np.testing.assert_allclose(captured[2], -robot.axes[2], atol=0.0)
    np.testing.assert_allclose(captured[3], expected_h4_in_3, atol=0.0)


def test_gen3_alignment_makes_tool_x_follow_physical_mount_direction() -> None:
    robot = gen3_kinematics()

    aligned_tool_x_in_7 = (
        robot.ee_rotation_in_7
        @ robot.R_robot_align
        @ np.array([1.0, 0.0, 0.0])
    )

    mount_direction_in_7 = robot.ee_position_in_7 / np.linalg.norm(
        robot.ee_position_in_7
    )
    np.testing.assert_allclose(aligned_tool_x_in_7, mount_direction_in_7, atol=2e-16)
    np.testing.assert_allclose(mount_direction_in_7, -robot.axes[6], atol=0.0)


def test_align_wrist_matches_desired_hand_rotation_to_machine_precision() -> None:
    robot = gen3_kinematics()
    maximum_rotation_error = 0.0

    for _ in range(200):
        q0 = _random_configuration(robot)
        desired = q0.copy()
        desired[4] = RNG.uniform(-np.pi, np.pi)
        lower, upper = robot.joint_limits[5]
        desired[5] = RNG.uniform(lower + 0.05, upper - 0.05)
        desired[6] = RNG.uniform(-np.pi, np.pi)
        H = robot.aligned_ee_rotation(desired)

        solution = align_wrist(q0, H, robot)
        result = q0.copy()
        result[4:7] = solution
        rotation_error = _rotation_angle(robot.aligned_ee_rotation(result).T @ H)
        maximum_rotation_error = max(maximum_rotation_error, rotation_error)

    print(f"maximum wrist rotation error: {maximum_rotation_error:.3e} rad")
    assert maximum_rotation_error < 2e-12


def test_align_wrist_enforces_q7_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = gen3_kinematics()
    monkeypatch.setattr(robot, "joint_limits", robot.joint_limits.copy())
    robot.joint_limits[6] = [-0.25, 0.25]
    monkeypatch.setattr(retarget_module, "align_axis", lambda *args: np.array([0.0, 0.0]))
    monkeypatch.setattr(retarget_module, "sp1", lambda *args: 1.0)

    with pytest.raises(ValueError, match="q7.*joint limits"):
        align_wrist(np.zeros(7), np.eye(3), robot)


def test_align_wrist_rejects_non_rotation_matrix() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        align_wrist(np.zeros(7), np.diag([1.0, 1.0, 2.0]), gen3_kinematics())


def test_sew_mimic_matches_all_three_orientations_to_machine_precision() -> None:
    robot = gen3_kinematics()
    maximum_errors = np.zeros(3)

    for _ in range(100):
        desired = _random_configuration(robot)
        q0 = desired.copy()
        q0[[0, 2, 4, 6]] += RNG.uniform(-0.05, 0.05, size=4)
        q0[[1, 3, 5]] += RNG.uniform(-0.03, 0.03, size=3)
        q0[[1, 3, 5]] = np.clip(
            q0[[1, 3, 5]],
            robot.joint_limits[[1, 3, 5], 0] + 0.01,
            robot.joint_limits[[1, 3, 5], 1] - 0.01,
        )

        upper_arm = robot.R_0_i(desired, 3) @ robot.arm_proxy_axis(3)
        lower_arm = robot.R_0_i(desired, 5) @ robot.arm_proxy_axis(5)
        shoulder = RNG.normal(size=3)
        elbow = shoulder + 0.31 * upper_arm
        wrist = elbow + 0.27 * lower_arm
        H = robot.aligned_ee_rotation(desired)

        q0_before = q0.copy()
        result, diagnostics = sew_mimic(q0, shoulder, elbow, wrist, H)

        np.testing.assert_array_equal(q0, q0_before)
        np.testing.assert_allclose(
            robot.R_0_i(result, 3) @ robot.arm_proxy_axis(3),
            upper_arm,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            robot.R_0_i(result, 5) @ robot.arm_proxy_axis(5),
            lower_arm,
            atol=2e-12,
        )
        assert _rotation_angle(robot.aligned_ee_rotation(result).T @ H) < 2e-12
        assert diagnostics["joint_limit_valid"] is True
        maximum_errors = np.maximum(
            maximum_errors,
            [
                diagnostics["upper_arm_error_deg"],
                diagnostics["lower_arm_error_deg"],
                diagnostics["wrist_rotation_error_deg"],
            ],
        )

    print(
        "maximum SEW diagnostic errors: "
        f"upper={maximum_errors[0]:.3e} deg, "
        f"lower={maximum_errors[1]:.3e} deg, "
        f"wrist={maximum_errors[2]:.3e} deg"
    )
    assert np.all(maximum_errors < 2e-10)


def test_sew_mimic_updates_configuration_in_algorithm_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, np.ndarray]] = []

    def fake_align_axis(
        index: int,
        q: np.ndarray,
        target: np.ndarray,
        robot: Gen3Kinematics,
    ) -> np.ndarray:
        calls.append((f"axis_{index}", q.copy()))
        return np.array([0.1, 0.2]) if index == 3 else np.array([0.3, 0.4])

    def fake_align_wrist(
        q: np.ndarray,
        H: np.ndarray,
        robot: Gen3Kinematics,
    ) -> np.ndarray:
        calls.append(("wrist", q.copy()))
        return np.array([0.5, 0.6, 0.7])

    monkeypatch.setattr(retarget_module, "align_axis", fake_align_axis)
    monkeypatch.setattr(retarget_module, "align_wrist", fake_align_wrist)

    result, diagnostics = sew_mimic(
        np.zeros(7),
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        np.eye(3),
    )

    assert [name for name, _ in calls] == ["axis_3", "axis_5", "wrist"]
    np.testing.assert_allclose(calls[0][1], np.zeros(7), atol=0.0)
    np.testing.assert_allclose(calls[1][1][:2], [0.1, 0.2], atol=0.0)
    np.testing.assert_allclose(calls[2][1][:4], [0.1, 0.2, 0.3, 0.4], atol=0.0)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], atol=0.0)
    assert set(diagnostics) == {
        "upper_arm_error_deg",
        "lower_arm_error_deg",
        "wrist_rotation_error_deg",
        "joint_limit_valid",
    }


@pytest.mark.parametrize(
    ("shoulder", "elbow", "wrist", "message"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "upper arm"),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], "lower arm"),
    ],
)
def test_sew_mimic_rejects_degenerate_human_arm_segments(
    shoulder: list[float],
    elbow: list[float],
    wrist: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sew_mimic(np.zeros(7), shoulder, elbow, wrist, np.eye(3))
