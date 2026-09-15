#!/usr/bin/env python3
"""Build the training gt: the classical HBS with the arcs relabelled.

The target is the classical field carried around the four seams by a
180 deg rotation inside the two arcs between them -- see classical_arcs
for how the arcs are MEASURED rather than assumed.  Measured result: max
adjacent-d 0.3031 -> 0.0136, pairs above 0.02 from 4 to 0, the flip never
increasing any pair, and the 360 -> 0 wrap exactly 0.0000.  So the target
is continuous by construction and needs no flip-window exclusion, no
theta* planning and no smoothing.

`valid` carries TWO exclusions and both are skipped by the trainer
(`finite & gvalid` in finetune_osc.py):

  * psi where the classical solver returns NaN -- there is no target
    there to fit.  The count is a property of the family (see the cache
    builder's report), not a constant;
  * (removed) an earlier revision also dropped psi outside 30-330 deg.
    That window existed only to avoid the wrap where psi=0 and psi=360 are
    the same shape sitting on opposite sides of a symmetric member; this
    family has no symmetric member (see osc_family), so the wrap is
    unambiguous and the whole cycle is now trained on.

Writes runs/cache_osc/gt_arcs_<family>.npz in the key layout finetune_osc.py
already expects from --gt-cache (gt / valid / width).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HBS_PYTHON_DIR"] = "/nonexistent"

import classical_arcs as CA
import eval_osc_standard as S
import osc_family as F
from hbsn_exp import grid

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
CACHE = os.path.join(REPO, "runs/cache_osc")
SRC = os.path.join(CACHE, f"fields_osc_{F.FAMILY}.npz")
OUT = os.path.join(CACHE, f"gt_arcs_{F.FAMILY}.npz")


def main() -> None:
    d = np.load(SRC)
    ths, cf = d["ths"], d["cf"]
    finite = np.array([not np.all(np.isnan(c)) for c in cf])
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)

    cfz = np.where(finite[:, None, None], cf, 0).astype(complex)
    states, seams = CA.flip_states(cfz, finite, w, z, mask)
    gt = CA.apply_flip(cfz, states, z, mask).astype(np.complex64)

    valid = finite

    def adj(F):
        return np.array(
            [
                S._wrms(np.abs(F[i]), np.abs(F[i + 1]), w)
                if valid[i] and valid[i + 1]
                else np.nan
                for i in range(len(F) - 1)
            ]
        )

    a_raw, a_gt = adj(cfz), adj(gt)
    print(f"source  {SRC}")
    print(f"seams measured at psi {[round(float(ths[i]), 2) for i in seams]}")
    arcs, i = [], 0
    while i < len(states):
        if states[i] == 1:
            j = i
            while j + 1 < len(states) and states[j + 1] == 1:
                j += 1
            arcs.append(f"{float(ths[i]):.2f}-{float(ths[j]):.2f}")
            i = j
        i += 1
    print(
        f"flipped arcs: {', '.join(arcs)}  ({int((states == 1).sum())} samples)"
    )
    print(
        f"valid psi {int(valid.sum())}/{len(ths)}  "
        f"(classical NaN {int((~finite).sum())}, no range exclusion)"
    )
    print(
        f"adjacent-d  raw classical max {np.nanmax(a_raw):.4f} "
        f"({int((a_raw > 0.02).sum())} pairs > 0.02)   "
        f"arc gt max {np.nanmax(a_gt):.4f} "
        f"({int((a_gt > 0.02).sum())} pairs > 0.02)"
    )
    i0, i1 = 0, len(ths) - 1
    print(f"wrap 360->0 wRMS {S._wrms(np.abs(gt[i0]), np.abs(gt[i1]), w):.4f}")

    np.savez_compressed(
        OUT,
        gt=gt,
        valid=valid,
        width=np.array(0),
        ths=ths,
        states=states,
        seams=ths[np.array(seams, dtype=int)],
        frame_ref=np.array("classical+arc-flip"),
    )
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
