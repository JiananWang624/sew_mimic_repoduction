# Geometric subproblems

## SP3: circle and sphere

`sew_mimic.geometry.sp3(p1, p2, k, d)` solves the one-angle constraint

```text
||R(k, theta) p1 - p2|| = d,     d >= 0,
```

where `k` is normalized internally and `R` is the repository's Rodrigues
rotation convention. Decompose each vector into components parallel and
perpendicular to `k`. The distance equation reduces to

```text
alpha cos(theta) + beta sin(theta) = target,
```

with `alpha = p1_perp dot p2_perp`, `beta = (k cross p1_perp) dot p2_perp`.
The implementation solves this circle/sphere reduction directly; it does not
call SP4, whose plane-normal normalization has different semantics.

The return value is the frozen `SP3Result` contract:

```text
angles, is_exact, residuals, degenerate, message
```

Each residual is independently recomputed as
`abs(norm(R(k, theta) @ p1 - p2) - d)`. Internally, all lengths share the
common scale `max(1, abs(p1 components), abs(p2 components), d)` to avoid
overflow. The effective geometry scale is then
`max(radius1, radius2, abs(axis-coordinate difference), scaled d)` so the
target equation remains well-conditioned when common axial coordinates are
very large. The returned residual is converted back to physical units.
Exactness uses the physical residual threshold

```text
SP3_EXACT_TOL * max(1, radius1, radius2,
                    abs(k dot (p1 - p2)), d),
```

where `SP3_EXACT_TOL = 1e-10`. Thus an analytic candidate is never labelled
exact solely from an intermediate discriminant. A genuine no-intersection
case is always marked non-exact, even if a caller supplies a looser residual
tolerance.

Regular roots are wrapped to `[-pi, pi)` and sorted in ascending order.
Tangent roots and cyclic duplicates are returned once. If either perpendicular
radius is numerically zero, the distance has no resolvable angle dependence:
angle `0` is returned as a deterministic representative and marked exact only
when its recomputed residual meets the same tolerance. A radius is treated as
numerically zero below `64 * machine_epsilon` times its vector norm. The input
axis is rejected when its norm is at most `1e-12`; intersection-boundary
clipping is limited to `64 * machine_epsilon` in the normalized scalar
equation; periodic roots within `1e-12` radians are deduplicated. If the circle
and sphere do not intersect, the nearest circle extremum is returned as a
non-exact continuous least-squares candidate.

The reduction follows Alexander J. Elias and John T. Wen, *IK-Geo: Unified
Robot Inverse Kinematics Using Subproblem Decomposition* (arXiv:2211.05737)
and its official BSD-3-Clause reference implementation:
[rpiRobotics/ik-geo](https://github.com/rpiRobotics/ik-geo). This repository
implements the equation independently rather than copying the third-party
source.
The SP3 primitive is retained independently for later WARP elbow reconstruction;
this document does not define or implement WARP.
