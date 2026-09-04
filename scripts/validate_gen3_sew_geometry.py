"""Deterministic Phase-3 report for Gen3 Stereo-SEW forward geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.config import CONFIG, project_path  # noqa: E402
from sew_mimic.csv_adapter import load_human_trajectory_csv  # noqa: E402
from sew_mimic.kinematics import gen3_kinematics  # noqa: E402
from sew_mimic.mounting import load_humanoid_mounted_gen3, world_trajectory_to_base  # noqa: E402
from sew_mimic.sew import (  # noqa: E402
    Gen3StereoSewGeometry,
    StereoSew,
    angular_margins,
    project_stereo_sew_reference,
    sample_gen3_configurations,
    select_project_reference,
)
from sew_mimic.sew.gen3_geometry import rotation_geodesic_error  # noqa: E402


def _human_base_points() -> np.ndarray:
    trajectory = load_human_trajectory_csv(project_path(CONFIG["human_csv"]["input_path"]))
    robot, data = load_humanoid_mounted_gen3(trajectory.shoulders[0])
    points = np.stack((trajectory.shoulders, trajectory.elbows, trajectory.wrists), axis=1)
    base_points, _ = world_trajectory_to_base(
        points, trajectory.hand_orientations,
        data.xmat[int(robot.frame_body_ids[0])].reshape(3, 3),
        data.xpos[int(robot.frame_body_ids[0])],
    )
    return base_points


def _human_directions(points: np.ndarray) -> np.ndarray:
    vectors = points[:, 2] - points[:, 0]
    norms = np.linalg.norm(vectors, axis=1)
    invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= 1e-12))
    if len(invalid):
        raise ValueError(
            "human shoulder-wrist direction is invalid in rows "
            f"{invalid.tolist()}"
        )
    return vectors / norms[:, None]


def _format_margin(label: str, margin) -> None:
    print(f"{label}: samples={margin.samples} exact={margin.exact_singular} near(<=5deg)={margin.near_singular}")
    print(f"  min/P1/P5/median rad = {margin.minimum_rad:.12g} / {margin.p1_rad:.12g} / {margin.p5_rad:.12g} / {margin.median_rad:.12g}")


def main() -> None:
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    print("GEN3 FAMILY VALIDATION")
    print("H =")
    print(np.array2string(geometry.H, precision=10))
    print("P =")
    print(np.array2string(geometry.P, precision=10))
    print(f"R_7T =\n{np.array2string(geometry.R_7T, precision=10)}")
    print(f"axis-unit max error: {geometry.structural_residuals.axis_unit_error:.3e}")
    print(
        "odd/even axis parallel angular error rad: "
        f"{geometry.structural_residuals.odd_axis_parallel_error_rad:.3e}/"
        f"{geometry.structural_residuals.even_axis_parallel_error_rad:.3e}"
    )
    print("(2,3)/(4,5)/(6,7) closest-line residual m:", np.array2string(geometry.structural_residuals.pair_intersection_m, precision=3))

    q_random = sample_gen3_configurations(robot, 1000, 20260906)
    q_limits = np.where(robot.joint_limited[:, None], robot.joint_limits, 0.0).T
    q_values = np.vstack((np.zeros(7), np.array([.2, -.3, .4, -.5, .6, -.7, .8]), q_limits, q_random))
    position_errors, rotation_errors = [], []
    for q in q_values:
        actual, expected = geometry.forward(q), robot.ee_transform(q)
        position_errors.append(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        rotation_errors.append(rotation_geodesic_error(actual[:3, :3], expected[:3, :3]))
    print("FK VALIDATION")
    print("native joint7->pinch local p =", np.array2string(robot.ee_position_in_7))
    print("native joint7->pinch local R =\n", np.array2string(robot.ee_rotation_in_7))
    print("R_robot_align =\n", np.array2string(robot.R_robot_align))
    print("aligned pinch rotation = R_pinch @ R_robot_align")
    print(f"samples={len(q_values)} position mean/max m={np.mean(position_errors):.3e}/{np.max(position_errors):.3e}")
    print(f"samples={len(q_values)} rotation mean/max rad={np.mean(rotation_errors):.3e}/{np.max(rotation_errors):.3e}")
    print("SEW POINT VALIDATION")
    print("S=joint-1 axis point; E=closest-line intersection axes 4/5; W=closest-line intersection axes 6/7.")

    robot_q = sample_gen3_configurations(robot, 5000, 20260906)
    robot_directions = np.array([
        (x.wrist - x.shoulder) / np.linalg.norm(x.wrist - x.shoulder)
        for x in (geometry.sew_points(q) for q in robot_q)
    ])
    human_points = _human_base_points()
    human_directions = _human_directions(human_points)
    search = select_project_reference(robot_directions, human_directions)
    print("REFERENCE SEARCH")
    for name, score in search.candidates:
        print(f"{name}: combined minimum={score.minimum_rad:.12g} rad P1={score.p1_rad:.12g} P5={score.p5_rad:.12g} median={score.median_rad:.12g}")
    print("selected e_t =", np.array2string(search.reference.e_t, precision=10))
    print("selected e_r =", np.array2string(search.reference.e_r, precision=10))
    configured = project_stereo_sew_reference()
    if not (np.array_equal(configured.e_t, search.reference.e_t) and np.array_equal(configured.e_r, search.reference.e_r)):
        raise RuntimeError("config.yaml stereo_sew reference differs from deterministic selection")
    print("SINGULARITY MARGINS")
    _format_margin("Robot", angular_margins(robot_directions, configured.e_t))
    _format_margin("Human", angular_margins(human_directions, configured.e_t))
    sew = StereoSew(configured)
    angles = [sew.forward(x.shoulder, x.elbow, x.wrist) for x in (geometry.sew_points(q) for q in robot_q)]
    print(f"Stereo-SEW robot evaluations={len(angles)} finite={bool(np.all(np.isfinite(angles)))}")
    human_angles = [sew.forward(*points) for points in human_points]
    print(f"Stereo-SEW human evaluations={len(human_angles)} finite={bool(np.all(np.isfinite(human_angles)))}")


if __name__ == "__main__":
    main()
