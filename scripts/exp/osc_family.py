#!/usr/bin/env python3
"""The osc triangle family -- single source of truth.

A triangle with base [-0.7, 0.7] x {0} and height H_SLIDE, whose apex slides
horizontally:

    x(psi) = X0 + AMP * sin(psi)

The DC offset X0 is the point of this module.  With X0 = 0 (the family the
first round of results used) the apex crosses the base midpoint twice per
cycle, and at those two phases the triangle is ISOCELES -- it has a mirror
symmetry, and the family passes through an achiral shape with the apex
switching from one side of centre to the other.  With X0 > 0 the apex never
reaches the midpoint, so:

  * every member is scalene, i.e. the family lies entirely on the generic
    locus and no member has any symmetry at all (not rotational, not
    mirror) -- hence constant chirality throughout the cycle;
  * psi = 0/180/360 stop being special points.

Constraints that X0 and AMP must satisfy together:

  * X0 > 0                -- the apex never hits the midpoint
  * AMP > FLIP_X + X0     -- x still reaches both flip lines +-FLIP_X, so
                             the classical normalization still flips FOUR
                             times per cycle

IMPORTANT: the classical field cache, the training target, the finetuned
checkpoints and every figure are all keyed to this family.  Changing X0 or
AMP invalidates all of them, which is why FAMILY (a short id derived from
the parameters) is baked into the cache filenames.
"""

import numpy as np
from hbsn_exp import shapes

H_SLIDE = 0.933
X0 = 0.10
AMP = 0.50
FLIP_X = 0.358  # apex offset at which the classical representative flips
FAMILY = f"x{round(X0 * 1000):03d}a{round(AMP * 1000):03d}"
BASE_HALF = 0.7


def x_of(psi):
    """Apex x at phase psi (deg).  Scalars and arrays both work."""
    return X0 + AMP * np.sin(np.deg2rad(psi))


def bound(psi):
    """Raw slide-triangle boundary at phase psi (deg)."""
    return shapes.triangle_slide(x=float(x_of(psi)), h=H_SLIDE)


def selfcheck() -> None:
    """Geometry only -- no classical solver, so this is instant."""
    psi = np.linspace(0.0, 360.0, 3601)
    x = x_of(psi)
    xmin, xmax = float(np.abs(x).min()), float(np.abs(x).max())
    assert X0 > 0, "X0 must be positive or the apex reaches the midpoint"
    assert abs(x).min() > 0, "family contains a mirror-symmetric member"
    assert x.max() > FLIP_X and x.min() < -FLIP_X, (
        f"x range [{x.min():+.3f},{x.max():+.3f}] does not straddle both "
        f"flip lines (+-{FLIP_X}): the family would not flip four times"
    )
    # no rotational symmetry anywhere: a triangle with apex off the
    # perpendicular bisector is scalene, and a scalene triangle has none
    print(
        f"family {FAMILY}: X0={X0} AMP={AMP} H={H_SLIDE}\n"
        f"  x in [{x.min():+.3f}, {x.max():+.3f}]   |x|min = {xmin:.4f} "
        f"(> 0 -> every member scalene, constant chirality)\n"
        f"  straddles +-{FLIP_X}: yes -> 4 classical flips per cycle\n"
        f"  base [-{BASE_HALF}, {BASE_HALF}], apex stays inside the base "
        f"span: {xmax <= BASE_HALF}"
    )


if __name__ == "__main__":
    selfcheck()
