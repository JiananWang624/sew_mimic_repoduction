# Retargeting Architecture

The repository keeps four retargeting methods separate. A status is exact only
with respect to the constraints declared by that method.

## Method 0 — Legacy SEW-Mimic

This is the verified regression baseline implemented by
`sew_mimic.retarget.sew_mimic`. It matches human upper-arm direction,
lower-arm direction, and hand orientation. It does not constrain the actual
Gen3 `pinch_site` position. The shared-result adapter delegates to the legacy
implementation without changing its inputs, joint solution, subproblems, or
failure behavior for direct callers.

## Method 1 — WARP-cSEW + Legacy SEW-Mimic

This paper-reproduction and comparison method is not implemented yet. A future
virtual-placement mode may model WARP torso/base motion, but it must be clearly
labelled and kept separate from physically executable fixed-base results.

## Method 2 — Exact Pose + Stereo-SEW Gen3 IK

This planned method is the recommended fixed-base Kinova Gen3 solver. It will
match the target pinch position, aligned pinch orientation, and human
Stereo-SEW parameter without moving the Gen3 root per frame. Unreachable
fixed-base targets must be reported explicitly.

## Method 3 — Numerical Exact Pose + SEW

This planned numerical method is a validation oracle only. It must not silently
replace Method 2, and least-squares solutions must never be labelled exact.

## Validation authority

For every future method, final end-effector position and orientation errors
must be evaluated using the real MuJoCo-derived `Gen3Kinematics` `pinch_site`
forward kinematics. Visualization offsets must never enter mathematical
calculations.
