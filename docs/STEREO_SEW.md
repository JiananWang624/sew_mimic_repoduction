# Stereographic SEW representation

## Project reference

The project-wide reference is deliberately configured, never implicit:
`config.yaml: stereo_sew.e_t = [0, 0, -1]` and `e_r = [1, 0, 0]`.  It is used
identically for human and robot geometry after both are expressed in the native
Gen3-base evaluation frame.  The Phase-3 selection and workspace evidence are
recorded in `GEN3_STEREO_SEW_GEOMETRY.md`; changing this pair changes the
numerical zero/sign convention for psi.

## Purpose

A seven-revolute-joint arm has one degree of kinematic redundancy after an
end-effector pose is fixed.  The shoulder-elbow-wrist (SEW) angle describes
the oriented half-plane containing the shoulder, elbow, and wrist.  This
phase implements that representation only: it does not solve an elbow point
or any robot inverse kinematics.

## Stereographic convention

Conventional SEW definitions need a chosen reference direction around the
shoulder-wrist axis.  Stereo-SEW instead uses an explicit orthonormal pair
`(e_t, e_r)`.  `e_t` is the stereographic translation/pole direction and
sets the singular half-line; `e_r` is the reference direction.  These map to
the official Stereo-SEW MATLAB names `(R, V)`, respectively.

The reference is always explicit:

```python
reference = StereoSewReference(e_t=..., e_r=...)
sew = StereoSew(reference)
psi = sew.forward(S, E, W)
inverse = sew.inverse(S, W, psi)
```

`e_t` and `e_r` must be finite three-vectors, unit length within `1e-10`, and
orthogonal within `1e-10`. Accepted vectors are then minimally normalized to
remove roundoff. The Phase-2B tutorial pair `[0, 0, -1]`, `[0, 1, 0]` is not
the final project reference; Phase 3 selected the configured pair stated above.

## Forward map

Let `p_SE = E-S`, `p_SW = W-S`, and `e_SW = unit(p_SW)`.  The oriented SEW
normal and stereographic reference normal are

```text
n_SEW = unit(cross(p_SW, p_SE))
n_ref = unit(cross(e_SW - e_t, e_r))
```

The signed angle is

```text
psi = atan2(dot(n_SEW, cross(e_SW, n_ref)), dot(n_SEW, n_ref))
```

and is wrapped to `[-pi, pi)`.  The cross-product order is intentional and
matches the cited official implementation.

## Inverse map

For an explicit `psi`, the inverse returns a unit transverse reference
`elbow_direction` and oriented `plane_normal`, not an elbow XYZ point:

```text
k_r  = cross(e_SW - e_t, e_r)
e_x  = unit(cross(k_r, p_SW))
e_CE = Rot(e_SW, psi) e_x
n_SEW = unit(cross(e_SW, e_CE))
```

Using `cross(k_r, e_SW)` is algebraically identical to the `p_SW` expression
after normalization and avoids a dimensional threshold. The generic WARP core
uses the returned half-plane with fixed-link geometry; Exact-SEW uses the same
forward angle to constrain a robot configuration.

## Singularities and tolerances

`StereoSewSingularityError` distinguishes geometric degeneracy from invalid
input (`ValueError`).  It is raised for a zero shoulder-wrist vector, a
collinear shoulder-elbow-wrist triple, the stereographic half-line, or an
undefined inverse transverse direction.  The half-line is `e_SW = e_t`, where
`cross(e_SW-e_t, e_r)` vanishes.  Direction and collinearity tests occur after
normalization and use `64 * machine_epsilon`, so they are scale-aware and not
tied to CSV units.  Wrong shape, NaN/Inf, non-finite angle, and invalid
reference vectors are ordinary invalid inputs.

## Scope and attribution

This module deliberately contains only the Stereo-SEW representation, not an
elbow-position solver or Gen3 IK. Phase 3 selected the final reference pair
relative to the actual Gen3 workspace and canonical frame; the downstream
WARP and Exact-SEW packages consume this representation without redefining it.

The equations are reimplemented (not copied) from the official
[Stereo-SEW repository](https://github.com/rpiRobotics/stereo-sew), especially
`IK_helpers/sew_stereo.m` at commit `d691747`, and are attributed to Elias and
Wen, *Redundancy parameterization and inverse kinematics of 7-DOF revolute
manipulators*.
