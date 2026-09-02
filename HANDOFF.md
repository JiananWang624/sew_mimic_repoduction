# SEW-Mimic Reproduction - Project Handoff

> Source of truth: current code and tests, with `paper/sew_mimic.pdf` used to
> check the paper notation. This file is intentionally implementation-focused;
> `README.md` is the user-facing guide.

## 1. Project overview

This project reproduces the closed-form SEW-Mimic retargeting core from **A
Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot
Teleoperation** for a MuJoCo Menagerie Kinova Gen3 7-DoF arm. It maps a human
pose `(shoulder, elbow, wrist, hand orientation)` to seven robot joint angles.
It matches upper- and lower-arm **directions**, not absolute human elbow/wrist
positions.

The implemented paper scope is Algorithm 1 (`SEW-Mimic`), Algorithm 2
(`AlignAxis`), Algorithm 3 (`AlignWrist`, parallel-wrist case), and Appendix
Algorithms 5/6/7 (`SP1`, `SP2`, `SP4`). There is no optimizer/Jacobian IK,
smoothing, collision avoidance, Safety Filter, training, ROS, or real-robot
control.

Main data path:

```text
processed human CSV
-> HumanCSVAdapter: CSV frame -> body/MuJoCo world frame
-> fixed Gen3 root mounting and world -> native-base transform
-> framewise sew_mimic(q_previous, s, e, w, H)
-> q1..q7 + angular diagnostics
-> output CSV/plots/MuJoCo playback
```

Core array types are NumPy `float64`. Positions are metres, joint angles are
radians, reported alignment errors are degrees, orientations are 3x3 proper
rotation matrices, and a joint vector has shape `(7,)`.

## 2. Core algorithm

### 2.1 Paper objective and actual solver behavior

The paper poses orientation retargeting conceptually as

```text
min_q  mu_c(u, R_0_3(q) h3)^2
     + mu_c(l, R_0_5(q) h5)^2
     + mu_m(T(q), H)^2
```

subject to joint limits, where

```text
u = unit(e - s)              human upper-arm direction
l = unit(w - e)              human lower-arm direction
H                            desired canonical hand orientation
T(q)                         canonical robot tool orientation
```

The code does not numerically minimize this objective and does not expose a
training-style loss/update rule. It solves the three terms sequentially with
closed-form subproblems. `metrics.py` reports geometric angles in degrees; it
does **not** reproduce the paper's normalized `mu_c`/`mu_m` values.

### 2.2 Algorithm 1: SEW-Mimic

**Paper step -> code:** `src/sew_mimic/retarget.py -> sew_mimic()`.

```text
1. q <- copy(q0)
2. u <- unit(elbow - shoulder)
   l <- unit(wrist - elbow)
3. q[0:2] <- align_axis(3, q, u, robot)       # paper q1,q2
4. q[2:4] <- align_axis(5, q, l, robot)       # paper q3,q4
5. q[4:7] <- align_wrist(q, H, robot)          # paper q5,q6,q7
6. return q and diagnostics
```

`q0` is copied and never mutated. Each later step sees the joints solved by the
earlier step. The public `sew_mimic()` currently hardcodes the cached native
Gen3 instance from `gen3_kinematics()`; unlike `align_axis()` and
`align_wrist()`, it does not accept an arbitrary robot object.

For a trajectory, `scripts/replay_csv.py -> retarget_trajectory()` initializes
`q_previous = zeros(7)` and uses every solved `q_current` as the next frame's
initial configuration. This nearest-branch continuation is the only continuity
mechanism.

### 2.3 Algorithm 2: AlignAxis

**Paper step -> code:** `src/sew_mimic/retarget.py -> align_axis(i, q0, v,
robot)`.

For 1-based paper joint index `i`, the function solves joints `i-2` and `i-1`;
their Python indices are `i-3` and `i-2`.

1. Validate `q0`, normalize target `v`, and select frame `i-2`.
2. Transform the target into that frame:
   `target_in_frame = R_0_(i-2)(q0).T @ v`.
3. Express the axis to align in the same frame. For `i=3` and `i=5`, this is
   the signed anatomical proxy returned by `robot.arm_proxy_axis(i)`. For
   `i=7`, it remains native `h7`.
4. Express predecessor axis `h_(i-1)` in frame `i-2`. Actual rotation axes are
   always native axes; proxy signs must never alter FK rotation axes.
5. Call
   `sp2(target, axis_to_align, -native_h_(i-2), native_h_(i-1))`.
6. Treat SP2 results as angle **deltas**, add them to the current pair, reject
   candidates outside MuJoCo-derived limits, then select the candidate
   minimizing `abs(delta_q_a) + abs(delta_q_b)`.
7. Raise `ValueError` when no closed-form candidate satisfies limits; there is
   no fallback.

Model-specific difference from the paper's plain `h3/h5` notation:

```text
upper proxy = UPPER_ARM_PROXY_SIGN * h3_native = -h3_native
lower proxy = LOWER_ARM_PROXY_SIGN * h5_native = -h5_native
```

These signs were validated against MuJoCo joint-anchor geometry over 1000
configurations. Do not globally negate `robot.axes[2]` or `robot.axes[4]`.

### 2.4 Algorithm 3: AlignWrist

**Paper step -> code:** `src/sew_mimic/retarget.py -> align_wrist(q0, H,
robot)`.

The implementation is specifically the paper's **parallel-wrist** formulation:

1. Read fixed `R_7_T_local` from `robot.ee_rotation_in_7`.
2. Compute desired joint-7 orientation:
   `R_0_7_des = H @ R_robot_align.T @ R_7_T_local.T`.
3. Align native `h7` using `align_axis(7, ...)`, producing `q5,q6`.
   The paper writes a desired matrix column; code uses the coordinate-invariant
   `desired_axis_7 = R_0_7_des @ robot.axes[6]` because the Menagerie h7/tool
   convention is not paper-canonical.
4. Express actual and desired h6 directions in frame 7.
5. Solve `q7 = SP1(h6_actual_in_7, h6_desired_in_7, -h7)`.
6. `_bound_angle()` chooses the periodic q7 representative nearest current q7
   that satisfies limits.

`Gen3Kinematics` derives `R_robot_align` from the `pinch_site` position and
orientation. Tests establish that native `+h7` is anti-parallel to canonical
tool `+X`; the native joint axis itself is not negated. The canonical tool
orientation used everywhere is `aligned_ee_rotation(q)`, not raw
`ee_rotation(q)`.

### 2.5 Appendix closed-form subproblems

**File:** `src/sew_mimic/geometry.py`.

- `rot(axis, theta)`: normalizes the axis and uses Rodrigues' formula
  `I + sin(theta)[k]x + (1-cos(theta))[k]x^2`.
- `sp1(p1,p2,k)` / Algorithm 5: projects and normalizes both vectors in the
  plane perpendicular to `k`, computes
  `2 atan2(||p1_hat-p2_hat||, ||p1_hat+p2_hat||)`, and fixes the sign with
  `k dot (p1_hat cross p2_hat)`.
- `sp4(p,h,k,d)` / Algorithm 7: constructs
  `F=[sk(k)p, -sk(k)^2p]`, `a=h.T@F`, and
  `b=d-(h dot k)(k dot p)`. It returns two exact circle-plane intersections
  when `||a||^2>b^2`, otherwise one tangent/least-squares direction.
- `sp2(p1,p2,k1,k2)` / Algorithm 6: normalizes inputs, calls SP4 twice, and
  cross-pairs the returned roots into one or two `(theta1,theta2)` solutions.

Important paper/code difference: Algorithm 7 prints `x=A^dagger b`, but this
project intentionally follows the agreed official-reference interpretation
`x_tilde=A.T*b`, implemented as `x_tilde = a * b`. This choice was explicitly
accepted for this reproduction and must not be "corrected" casually.

The code explicitly rejects zero axes, parallel SP2 axes, vectors parallel to
their rotation axes, and angle-independent SP4 geometry. Exact feasible tests
check final geometric residuals near machine precision, not raw angle equality.

### 2.6 Kinematics equations

**File/class:** `src/sew_mimic/kinematics.py -> Gen3Kinematics`.

The constructor reads native joint axes, `jnt_pos`, body transforms, limits,
joint names, and the `pinch_site` transform directly from the MJCF. It validates
one serial chain with exactly seven controlled hinge joints.

`T_0_i(q,i)` multiplies each model-derived fixed parent-to-child transform with
a rotation about the MuJoCo joint anchor. For one joint, its translated term is
implemented as

```text
t_fixed + R_fixed @ (p_joint - R(h, q) @ p_joint)
```

so rotation occurs about `jnt_pos`, not the child-body origin. `R_0_i()` returns
the 3x3 block. `ee_transform()` appends the model-derived `pinch_site` local
pose. Tests compare every link and tool pose against `mujoco.mj_forward()` over
200 random configurations with `<1e-12` m/rad thresholds.

### 2.7 Deliberate paper-to-code adaptations

- Paper Algorithm 1 says to sync human/robot frames first; this repository does
  that outside `sew_mimic()` through the CSV adapter and mounting conversion.
  Paper Algorithm 8 (`MakeFrame`) is not implemented.
- Paper notation directly aligns `h3/h5`; Gen3 requires separately signed
  anatomical proxies while retaining native axes for rotation.
- Paper Algorithm 3's desired matrix-column expression is implemented as
  `R_0_7_des @ h7_native`, together with a model-derived `R_robot_align`.
- Paper Algorithm 7 line 5 uses pseudoinverse notation; code intentionally uses
  the authorized official-reference `A.T @ b` interpretation.
- Code enforces actual MuJoCo joint limits and selects a branch near q0. The
  paper's global-optimality proposition assumes no joint limits or collision.
- Code reports angular diagnostics instead of the paper's normalized objective
  terms.

## 3. Project structure

```text
sew_mimic_repro/
├── config.yaml                    # single runtime/configuration source
├── README.md                      # user-facing English guide
├── HANDOFF.md                     # this developer/agent handoff
├── requirements.txt
├── src/sew_mimic/
│   ├── config.py                  # YAML loader and project-relative paths
│   ├── geometry.py                # Rodrigues, SP1, SP2, SP4
│   ├── kinematics.py              # model-derived Gen3 FK/tool convention
│   ├── human_input.py             # directions, Euler conversion, frame helper
│   ├── csv_adapter.py             # project-specific CSV -> body/world adapter
│   ├── mounting.py                # fixed Gen3 world root pose and frame mapping
│   ├── retarget.py                # Algorithms 1-3
│   └── metrics.py                 # angular diagnostics and limit validity
├── scripts/
│   ├── replay_csv.py              # production-like full CSV path
│   ├── validate_first_frame.py    # detailed frame/mount/proxy/tool diagnostic
│   ├── validate_wrist_alignment.py
│   ├── test_self_consistency.py
│   ├── replay_trajectory.py       # synthetic native-base trajectory
│   ├── show_right_arm_mounting.py
│   └── show_gen3.py
├── tests/                         # geometry/FK/retarget/adapter/replay tests
├── data/test.csv                  # current processed 4344-row trajectory
├── assets/kinova_gen3/            # Menagerie MJCF, meshes, license
├── paper/sew_mimic.pdf            # reference paper
└── output/                        # generated CSV/plots/screenshots
```

`scripts/test_single_pose.py` and `scripts/benchmark.py` currently contain only
module docstrings; they are placeholders, not functioning entry points.

## 4. Key functions and classes

| Location | Symbol | Algorithmic role |
|---|---|---|
| `geometry.py` | `rot()` | Rodrigues rotation used by FK and subproblems. |
| `geometry.py` | `sp1()` | One-axis vector alignment, Appendix Algorithm 5. |
| `geometry.py` | `sp2()` | Two-axis alignment, Appendix Algorithm 6. |
| `geometry.py` | `sp4()` | Circle-plane solve used by SP2, Appendix Algorithm 7. |
| `kinematics.py` | `Gen3Kinematics` | Extracts and evaluates the native Gen3 serial chain. |
| `kinematics.py` | `T_0_i()`, `R_0_i()` | Paper frame transforms in native base frame 0. |
| `kinematics.py` | `aligned_ee_rotation()` | Paper-canonical `T(q)`. |
| `kinematics.py` | `arm_proxy_axis()` | Keeps anatomical h3/h5 proxy signs separate from native axes. |
| `human_input.py` | `compute_*_arm_direction()` | Implements `unit(e-s)` and `unit(w-e)`. |
| `human_input.py` | `wrist_euler_to_rotation()` | Explicit SciPy intrinsic/extrinsic Euler conversion. |
| `csv_adapter.py` | `HumanCSVAdapter` | Validates/configures CSV scale, frames, and wrist alignment. |
| `csv_adapter.py` | `load_human_trajectory_csv()` | Produces batched `HumanTrajectory` arrays. |
| `mounting.py` | `load_humanoid_mounted_gen3()` | Loads a fresh visual model and sets only root pose. |
| `mounting.py` | `world_trajectory_to_base()` | Converts `(N,3,3)` points and `(N,3,3)` orientations to native base. |
| `retarget.py` | `align_axis()` | Paper Algorithm 2 plus limits/nearest branch. |
| `retarget.py` | `align_wrist()` | Paper Algorithm 3 for Gen3. |
| `retarget.py` | `sew_mimic()` | Paper Algorithm 1 orchestration. |
| `metrics.py` | `compute_retarget_diagnostics()` | Upper/lower angular error, SO(3) wrist angle, limit check. |

There is no `loss`, `Model.forward()`, gradient update, training class, or model
checkpoint in this repository.

## 5. Real execution flow

### Full CSV path

```text
scripts/replay_csv.py:main()
-> CONFIG defaults from config.yaml
-> HumanCSVAdapter(...)
-> load_human_trajectory_csv()
   -> HumanCSVAdapter.adapt_frame() per row
-> load_segment_boundaries()
-> load_humanoid_mounted_gen3(first shoulder)
-> trajectory_in_mounted_base()
   -> mounting.world_trajectory_to_base()
-> retarget_trajectory()
   -> sew_mimic(q_previous, ...) per row
      -> align_axis(3) -> sp2() -> sp4()
      -> align_axis(5) -> sp2() -> sp4()
      -> align_wrist() -> align_axis(7)/sp2()/sp4() -> sp1()
      -> compute_retarget_diagnostics()
-> save_retargeted_csv()
-> plot_trajectory() and report_trajectory()
-> replay_in_mujoco() unless --no-viewer
   -> write all seven q values to data.qpos
   -> mujoco.mj_forward()
   -> draw human/robot markers, limb arrows, and tool triads
```

Output columns are `q1..q7`, `upper_arm_error_deg`,
`lower_arm_error_deg`, and `wrist_error_deg`.

### Validation paths

- `validate_first_frame.py`: traces `q0 -> q_after_upper -> q_after_lower ->
  q_after_wrist`, compares proxies and actual joint-anchor limb directions, and
  renders desired/actual tool triads.
- `validate_wrist_alignment.py`: fixes q1:q4, samples valid q5:q7 targets, then
  validates reconstructed tool orientation and h7/tool-X convention over 1000
  samples.
- `test_self_consistency.py`: creates `(u,l,H)` from random Gen3 FK and solves
  from a different q0 over the configured sample count.
- `replay_trajectory.py`: creates a smooth, reachable trajectory from native
  Gen3 FK and re-solves it. It uses the cached native model; it is not the
  humanoid-world-mounted CSV visualization path.

## 6. Implementation details and pitfalls

### Frames and wrist conventions

- Current CSV axes: `+X=human left`, `+Y=up`, `+Z=forward`.
- Body/world axes: `+X=forward`, `+Y=left`, `+Z=up`.
- `R_BODY_FROM_CSV=[[0,0,1],[1,0,0],[0,1,0]]`; it is orthogonal with
  determinant `+1`. Do not restore the old reflected `-Z` mapping.
- Positions use absolute conversion
  `p_world = R_BODY_FROM_CSV @ (0.001 * p_csv_mm)`. The configured
  `reference_shoulder_world_m` is a diagnostic expectation, not a translation
  applied by the adapter.
- Current processed wrist columns are three Euler values, configured as
  extrinsic XYZ in degrees. The adapter does not directly parse a four-column
  quaternion.
- `R_wrist_csv` becomes
  `H_body = R_BODY_FROM_CSV @ R_wrist_csv @ R_INPUT_ALIGN`; then mounting uses
  `H_base = R_body_from_base.T @ H_body`.
- Canonical hand axes are X=index/pointing, Y=palm normal, Z=thumb. Keep
  `R_INPUT_ALIGN` (tracking device -> hand convention) separate from
  `R_robot_align` (Gen3 tool -> canonical tool convention).

### Mounting

- Fixed right-arm rotation is `Rx(+90deg)`; native Gen3 +Z maps to world -Y.
- `joint_1`, not `base_link`, is the robot shoulder. Its configured native
  offset is `[0,0,0.15643]` m.
- Root position is
  `human_shoulder + robot_world_offset - R_root @ joint1_in_base`.
- Current `config.yaml` value is `world_offset_m: [0,0,0.2]`, so joint1 is 20
  cm above the human shoulder marker. The adjacent YAML comment still says the
  default is no offset; the **numeric value controls actual behavior**.
- World translation leaves direction targets and joint solutions unchanged,
  but intentionally separates human and robot absolute markers.
- Never compensate a frame error by editing MJCF internal body offsets or
  native axes.

### Configuration

- `src/sew_mimic/config.py` loads root `config.yaml` once at import time.
  Restart Python after editing YAML.
- Relative configured paths resolve from project root.
- YAML validation is shallow: malformed top-level documents fail clearly, but
  missing keys generally raise `KeyError` at module import and some bad shapes
  fail later in the consuming module.
- Replay CLI can override input/output/plot/FPS/jump threshold/position scale
  and CSV rotation. Robot world offset, wrist convention, and input alignment
  are intentionally YAML/core settings, not replay CLI flags.

### Limits, branches, and degeneracy

- MuJoCo marks joints 2/4/6 limited and 1/3/5/7 unlimited in this MJCF.
- `align_axis()` treats SP2 angles as deltas around q0, filters limits, and uses
  L1 distance to q0 for branch selection. Changing this affects continuity.
- `_bound_angle()` wraps q7 by `2*pi` toward current q7 and enforces limits.
- There is no smoothing/unwrapping pass after solving.
- A coincident shoulder-elbow or elbow-wrist pair is rejected.
- SP functions normalize axes/vectors and deliberately reject degenerate cases.

### Trajectory behavior

- Full replay starts at `q=zeros(7)` and then uses the previous CSV row's
  solution, including across labeled event boundaries.
- Event boundaries only suppress discontinuities in plots/statistics; they do
  not reset `q_previous`.
- `replay_csv.py` additionally requires `bite_id`, `motive_frame`, `event`, and
  `event_frame_index`. `HumanCSVAdapter` alone requires only the 12 pose fields.
- Any unsatisfied closed-form/limit case aborts at the first failing row with
  q0/u/l diagnostics. There is no silent recovery.
- Joint velocities are finite differences times configured FPS. Boundary rows
  become NaN for plotting/reporting.

### MuJoCo details

- MuJoCo quaternions are WXYZ. Root orientation is written to
  `model.body_quat[base_body_id]`.
- After every qpos/root update, call `mujoco.mj_forward(model,data)` before
  reading anchors, axes, sites, or rendering.
- Joint qpos addresses come from `model.jnt_qposadr`; do not assume direct IDs
  are qpos indices in a different model.
- `gen3_kinematics()` is `lru_cache(maxsize=1)` and represents native-base
  kinematics. Mounted visualization models are fresh loads.

### Reproducibility

- Core randomized test seeds are fixed (mostly `20260831`; wrist convention
  tests use `20260901/20260902`). Script defaults are in YAML.
- No explicit `TODO` or `FIXME` markers were found.
- The worktree was not clean when this handoff was generated. Preserve existing
  user changes and inspect `git status`/`git diff` before editing.

## 7. Quick modification guide

| Desired change | Start here | Also verify |
|---|---|---|
| SP1/SP2/SP4 or Rodrigues math | `geometry.py` | `tests/test_geometry.py`, paper Appendix A; preserve `A.T*b` decision. |
| Algorithm 1 order | `retarget.py:sew_mimic()` | `tests/test_retarget.py`, trajectory previous-q behavior. |
| Axis alignment/branch selection | `retarget.py:align_axis()` | proxy/native distinction, limits, `test_retarget.py`. |
| Wrist solve/tool convention | `retarget.py:align_wrist()` | `Gen3Kinematics.ee_*`, `R_robot_align`, wrist validation tests. |
| FK/model/tool site | `kinematics.py` or MJCF | `tests/test_kinematics.py`, `tests/test_wrist_alignment.py`; do not hand-edit derived axes. |
| Robot root XYZ/rotation | `config.yaml`, `mounting.py` | first-frame and mounting diagnostics; internal MJCF must remain unchanged. |
| CSV units/axes/Euler convention | `config.yaml`, `csv_adapter.py` | human-input tests and first-frame numeric output. |
| New CSV schema | `csv_adapter.py` constants/loader | `replay_csv.py:load_segment_boundaries()` and tests. |
| Diagnostics/evaluation | `metrics.py`, validation scripts | output column naming in `replay_csv.py`. |
| Trajectory continuity policy | both replay scripts | branch-nearest logic and segment-boundary semantics. |
| Runtime/default parameters | root `config.yaml` | restart process; `tests/test_config.py`. |
| Training/loss | Not present | requires a new, explicitly scoped subsystem; do not confuse diagnostics with paper loss. |

Dependency chains to keep in mind:

```text
config.yaml -> config.py -> csv_adapter.py / kinematics.py / mounting.py / scripts
kinematics.py -> geometry.rot
retarget.py -> geometry + human_input + kinematics + metrics
replay_csv.py -> csv_adapter + mounting + retarget
```

After algorithm/frame changes, minimally run the directly affected tests plus
`validate_first_frame.py --no-viewer`. Before accepting broader changes, run
the full test suite and a headless CSV replay.

## 8. Current implementation status

Implemented and covered by tests:

- Closed-form Algorithms 1-3 and Appendix SP1/SP2/SP4.
- Gen3 model extraction and custom FK vs MuJoCo FK.
- Gen3-specific h3/h5 proxy signs and h7/tool alignment.
- Proper CSV/body frame mapping, fixed wrist input alignment, and Euler parsing.
- Fixed humanoid right-arm root mounting plus configurable world XYZ offset.
- First-frame, 1000-pose proxy/wrist/self-consistency diagnostics.
- Sequential full-CSV retargeting, CSV output, plots, and MuJoCo playback.

Relative to the full paper, this is deliberately a **partial reproduction**:
Safety Filter, Algorithm 8 automatic body-frame construction, perpendicular
wrist Euler special case, bimanual support, user studies, learned policies,
hardware/control, and paper experiments are not implemented.

Known gaps / possible future work:

- `benchmark.py` and `test_single_pose.py` are placeholders.
- No package metadata (`pyproject.toml`) or installed console entry points;
  scripts prepend `src` to `sys.path`.
- Minimal pose-only CSV files work with the adapter but not full replay unless
  segment metadata columns are added or boundary loading is made optional.
- Direct quaternion input is not supported by the current adapter.
- No graceful frame-level failure policy, smoothing, velocity limiting,
  collision checking, or safety filter.
- Config schema/type validation could be made stricter without changing
  algorithm behavior.
- Generic-robot support is incomplete because `sew_mimic()` selects the cached
  Gen3 internally and wrist logic is the Gen3 parallel-wrist path.
- Current output quality on arbitrary future datasets is unconfirmed; rerun
  first-frame and full-trajectory diagnostics whenever input conventions change.

## Commands for the next agent

```powershell
# Install
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Full tests
.venv\Scripts\python.exe -m pytest -q

# Frame/mount/wrist diagnostics without interactive viewer
.venv\Scripts\python.exe scripts\show_right_arm_mounting.py --no-viewer
.venv\Scripts\python.exe scripts\validate_first_frame.py --no-viewer
.venv\Scripts\python.exe scripts\validate_wrist_alignment.py

# Full trajectory without viewer
.venv\Scripts\python.exe scripts\replay_csv.py --no-viewer
```
