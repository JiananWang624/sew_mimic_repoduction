import numpy as np

from scripts.validate_wrist_alignment import (
    validate_align_wrist,
    validate_tool_axis_convention,
)


def test_align_wrist_self_consistency_over_1000_valid_wrist_poses() -> None:
    report = validate_align_wrist(samples=1000, seed=20260901)

    assert report.failure_count == 0
    assert report.joint_limit_failure_count == 0
    assert report.errors_deg.size == 1000
    assert report.median_error_deg < 1e-12
    assert report.max_error_deg < 1e-10


def test_gen3_native_h7_is_antiparallel_to_physical_tool_x_over_1000_poses() -> None:
    report = validate_tool_axis_convention(samples=1000, seed=20260902)

    np.testing.assert_allclose(report.positive_dots, -1.0, atol=2e-15)
    np.testing.assert_allclose(report.negative_dots, 1.0, atol=2e-15)
    assert report.maximum_fixed_transform_error_deg < 1e-12
    np.testing.assert_allclose(
        report.rotation_7_to_tool_local,
        np.diag([1.0, -1.0, -1.0]),
        atol=0.0,
    )
