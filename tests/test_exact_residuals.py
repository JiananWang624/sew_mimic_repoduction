import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sew_mimic.common import ExactSewTarget, gen3_end_effector_pose
from sew_mimic.exact import robot_exact_sew_residuals, so3_log
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference


def test_exact_target_validates_and_wraps_psi():
    target = ExactSewTarget(np.zeros(3), np.eye(3), 3 * np.pi)
    assert target.psi == -np.pi
    with pytest.raises(ValueError):
        ExactSewTarget(np.zeros(2), np.eye(3), 0)
    with pytest.raises(ValueError):
        ExactSewTarget(np.zeros(3), np.eye(3) * 2, 0)
    with pytest.raises(ValueError):
        ExactSewTarget(np.zeros(3), np.diag([1.0, 1.0, -1.0]), 0)
    with pytest.raises(ValueError):
        ExactSewTarget(np.zeros(3), np.eye(3), np.nan)


@pytest.mark.parametrize(
    "rotation_vector",
    [
        np.zeros(3),
        np.array([0.2, -0.3, 0.6]),
        (np.pi - 1e-12) * np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0),
    ],
)
def test_so3_log_is_stable_and_geodesic(rotation_vector):
    value = so3_log(Rotation.from_rotvec(rotation_vector).as_matrix())
    assert np.all(np.isfinite(value))
    assert np.linalg.norm(value) == pytest.approx(
        np.linalg.norm(rotation_vector), abs=1e-11
    )


def test_exact_residual_is_zero_at_authoritative_fk():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.array([.2, .3, -.4, .5, -.2, .3, .4])
    p, R = gen3_end_effector_pose(q, robot)
    points = geometry.sew_points(q)
    target = ExactSewTarget(p, R, stereo.forward(points.shoulder, points.elbow, points.wrist))
    residual = robot_exact_sew_residuals(q, target, robot, geometry, stereo)
    assert residual.position_error_m < 1e-14
    assert residual.orientation_error_rad < 1e-14
    assert residual.sew_error_rad < 1e-14
