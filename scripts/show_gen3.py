"""Inspect and display the MuJoCo Menagerie Kinova Gen3 model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from sew_mimic.kinematics import (  # noqa: E402
    GEN3_SCENE_PATH,
    load_mujoco_model,
    revolute_joint_ids,
    validate_gen3_arm,
)


MODEL_PATH = GEN3_SCENE_PATH


def _object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else "<unnamed>"


def print_model_summary(model: mujoco.MjModel) -> None:
    """Print joints, axes, limits, actuation status, and body names."""
    joint_ids = revolute_joint_ids(model)
    controlled_ids = validate_gen3_arm(model)

    print("Model: MuJoCo Menagerie Kinova Gen3")
    print(f"MJCF: {MODEL_PATH}")
    print(f"Revolute joints ({len(joint_ids)}):")
    for joint_id in joint_ids:
        name = _object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        axis = model.jnt_axis[joint_id]
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            limits = f"[{lower:.8f}, {upper:.8f}] rad"
        else:
            limits = "unlimited"
        print(
            f"  [{joint_id}] {name}: "
            f"axis(local)=[{axis[0]:.8f}, {axis[1]:.8f}, {axis[2]:.8f}], "
            f"limits={limits}"
        )

    controlled_names = [
        _object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in controlled_ids
    ]
    print(f"Controlled arm joints ({len(controlled_ids)}): {', '.join(controlled_names)}")
    print("Confirmed: Kinova Gen3 arm has 7 controlled revolute joints.")

    print(f"Bodies/links ({model.nbody} including world):")
    for body_id in range(model.nbody):
        name = _object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"  [{body_id}] {name}")


def show_viewer(model: mujoco.MjModel) -> None:
    """Open the passive MuJoCo viewer until its window is closed."""
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    print("Opening MuJoCo viewer. Close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="print and validate the model without opening a window",
    )
    args = parser.parse_args()

    model = load_mujoco_model(MODEL_PATH)
    print_model_summary(model)
    if not args.no_viewer:
        try:
            show_viewer(model)
        except KeyboardInterrupt:
            print("\nViewer closed.")


if __name__ == "__main__":
    main()
