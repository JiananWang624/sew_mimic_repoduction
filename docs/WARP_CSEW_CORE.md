# Generic WARP c-SEW Core

This package reproduces the corrected-skeleton portion of WARP for a generic
robot satisfying WARP's fixed-link arm assumptions. It intentionally stops at
the corrected `(shoulder, elbow, wrist, hand rotation)` geometry and does not
call SEW-Mimic or produce robot joint angles.

Reference: [WARP: Whole-Body Retargeting for Learning from Offline Human
Demonstrations](https://arxiv.org/html/2606.29940v2), equations 2–3.

## Fixed geometry

`WarpArmGeometry` contains one shoulder position, fixed positive upper-arm and
forearm lengths, and a fixed wrist-to-task vector expressed in the hand frame.
The values are immutable and are never recomputed from a candidate robot
configuration.

For each human arm, the scale-independent directions are

```text
u_h = unit(E_h - S_h)
l_h = unit(W_h - E_h)
```

and the predicted robot task point is

```text
t_hat = S_r + L_SE u_h + L_EW l_h + H_h p_WT.
```

For `N >= 1`, `compute_adaptive_offset()` returns

```text
offset = centroid(t_h) - centroid(t_hat).
```

`N=2` is the paper's bimanual equation. `N=1` is an explicit single-arm
adaptation. The function returns the offset and never moves a robot or mutates
geometry.

## Corrected skeleton

`construct_warp_skeleton()` receives the target task point already expressed
in the desired robot frame. Human `S/E/W`, human hand orientation, the robot
shoulder, the target point, and the supplied Stereo-SEW reference must all use
that same orientation-aligned coordinate frame. It applies no hidden transform
or adaptive offset. The caller is responsible for transforming all inputs
consistently and applying any virtual placement exactly once.

Hand orientation transfers directly, `H_r = H_h`, and wrist position follows
from the hard palm constraint:

```text
W_r = t_target_robot - H_r p_WT
W_r + H_r p_WT = t_target_robot.
```

The supplied `StereoSew` instance transfers the human redundancy parameter:

```text
psi_h = stereo.forward(S_h, E_h, W_h)
n_hat = stereo.inverse(S_r, W_r, psi_h).plane_normal.
```

The elbow is reconstructed through the validated SP3 implementation:

```text
e_SW = unit(W_r - S_r)
theta = sp3(L_SE e_SW, W_r - S_r, n_hat, L_EW)
E_r = S_r + R(n_hat, theta) (L_SE e_SW).
```

Only SP3 roots already classified exact are considered. The WARP branch rule
is `theta > 1e-12 rad`; least-squares roots and exact nonpositive roots produce
distinct non-success statuses. Exact results are independently checked at
`1e-12 m` for palm and link lengths and `1e-10 rad` for Stereo-SEW transfer.

The validation script generated 1,000 deterministic exact fixed-link cases
whose human upper/forearm lengths differ from the robot geometry. All 1,000
succeeded. Maximum observed errors were:

- palm: `0 m`
- upper-arm length: `1.665e-16 m`
- forearm length: `2.776e-16 m`
- Stereo-SEW: `8.882e-16 rad`

## Kinova Gen3 compatibility

Exact WARP assumes configuration-invariant `L_SE`, `L_EW`, and `p_WT`.
`check_warp_fixed_geometry_compatibility()` evaluates these quantities over
deterministic mechanically valid configurations using the validated Gen3
`S/E/W` points and aligned MuJoCo pinch-site FK.

For 1,000 configurations with seed `20260912`:

```text
L_SE min/max       0.3545435742 / 0.5496802542 m
L_SE std/variation 0.0597750822 / 0.1951366800 m
L_EW variation     2.776e-16 m
p_WT mean          [0.167455, 0, 0] m
p_WT max deviation 1.976e-15 m
```

With the explicit `1e-10 m` fixed-geometry tolerance, the current Gen3 is not
WARP-compatible because `L_SE` is configuration-dependent. No approximate or
configuration-dependent `L_SE` is substituted, and no validated Gen3 geometry
is redefined. Method 2 Exact-SEW remains the recommended real Gen3 retargeter.

Integration of the generic corrected skeleton with legacy SEW-Mimic is outside
this phase and is not justified for the incompatible Gen3 representation.
