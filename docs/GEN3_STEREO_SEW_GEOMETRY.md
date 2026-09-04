# Gen3 Stereo-SEW Geometry

Phase 3 validates a forward-only, native-base representation of the MuJoCo
Menagerie Kinova Gen3 for the official R-2R-2R-2R Stereo-SEW/IK-Geo family.
No inverse kinematics is implemented here.

## Convention and extraction

`Gen3StereoSewGeometry.from_robot()` extracts every quantity at `q=0` from the
repository MuJoCo model.  The independent POE calculation starts with
`p=P[:,0]`, `R=I`; for joint `i`, it applies `R=R Rot(H[:,i],q[i])` and then
`p=p+R P[:,i+1]`.  The final physical pinch transform is `(R R_7T,p)`.

`H` is the seven native base-frame joint axes. `P` contains `p01, p12, ...,
p7T`. The virtual intersections, rather than raw anchors, define `p12`,
`p34`, and `p56`; the paired zero columns represent intersecting axes.
`R_7T` is the fixed physical `pinch_site` rotation in this POE terminal frame.
For this model it is identity. This must not be confused with
`R_robot_align`, which remains the separate canonical-hand alignment used by
the existing SEW-Mimic implementation.

The extracted values (insignificant MuJoCo roundoff omitted) are:

```text
H = [[0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 1, 0],
     [-1,0,-1,0,-1,0,-1]]
P = [[0, 0,       0, 0,       0, 0,         0, 0],
     [0,-.01175,  0,-.01275,  0,-.0003501, 0, 0],
     [.15643,.12838,0,.42076, 0,.31436,     0,.167455]]
R_7T = I
```

Each `H` column is the q=0 native-base direction of its same-numbered MuJoCo
hinge: `h1=joint_1` through `h7=joint_7`. `P` is not copied from external
dimensions; its columns are extracted as follows.

| Column | Physical or virtual displacement |
| --- | --- |
| `p01` | base origin to joint-1 axis point |
| `p12` | joint-1 point to virtual axes-(2,3) intersection |
| `p23` | zero: axes 2 and 3 share that virtual point |
| `p34` | axes-(2,3) intersection to virtual axes-(4,5) intersection |
| `p45` | zero: axes 4 and 5 share that virtual point |
| `p56` | axes-(4,5) intersection to virtual axes-(6,7) intersection |
| `p67` | zero: axes 6 and 7 share that virtual point |
| `p7T` | virtual axes-(6,7) intersection to physical pinch site |

The direct MuJoCo joint-7-to-pinch local transform is
`p=[0,0,-0.061525]` m and `R=diag(1,-1,-1)`. After the q=0 virtual-frame
construction, the corresponding POE terminal rotation is `R_7T=I`; these are
different descriptions of the same physical site. `R_robot_align` is also
separate: exactly (up to model roundoff)
`[[0,0,-1],[0,1,0],[1,0,0]]`, and the established aligned orientation remains
`R_pinch(q) @ R_robot_align`. Neither is used to alter native axes or POE FK.

The three pair closest-line residuals are checked during extraction; the
validated model has intersecting `(2,3)`, `(4,5)`, `(6,7)` pairs. SEW points
are respectively the joint-1 axis point, axes-(4,5) virtual intersection, and
axes-(6,7) virtual intersection.

## Shared project reference

The final shared native-Gen3-base reference is in `config.yaml`:

```text
e_t = [0, 0, -1]
e_r = [1, 0, 0]
```

Robot directions are from 5,000 deterministic valid configurations (seed
`20260906`). Human directions are every adapted `data/test.csv` frame after
the established `Rx(+90)` mounted-base conversion. Candidates are ordered
`+x,-x,+y,-y,+z,-z`; the largest combined minimum angular margin wins, with
that order breaking exact ties. `e_r` projects the canonical axis with greatest
perpendicular magnitude, breaking ties `+x,+y,+z`.

The diagnostic script reports the exact current margin statistics and FK
residuals. A near singularity is a non-exact margin no greater than five
degrees; exact uses the Phase-2B `64*eps` direction criterion.

At validation time the selected reference gave no exact or near singularities.
The robot min/P1/P5/median margins were
`0.173286704481 / 0.675365376207 / 1.04738081226 / 2.17551714181` rad; human
values were `0.762501138025 / 0.799902994525 / 0.850105645932 /
1.11375679368` rad. The 1,004-pose POE/pinch comparison (1,000 random plus
zero, known, and two limit configurations) had mean/max position errors
`5.314e-16/9.883e-16` m and rotation errors `6.147e-16/1.451e-15` rad.

Changing this configured pair changes the numerical zero/sign convention of
Stereo-SEW psi. Human and robot use it identically only after expression in
the shared native base frame; visualization/root offsets never enter these
calculations.

## Sources and limitation

The family/parameter convention is attributed to [Stereo-SEW, pinned commit
`d691747`](https://github.com/rpiRobotics/stereo-sew/tree/d6917478037b924e1292e65a8f52398da3948851)
and [IK-Geo, pinned commit `a3a1675`](https://github.com/rpiRobotics/ik-geo/tree/a3a1675).
The source equations are reimplemented, not copied. This phase proves only the
forward model and reference selection; it does not establish reachability,
branch selection, or an Exact-SEW inverse solver.
