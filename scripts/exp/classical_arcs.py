#!/usr/bin/env python3
"""Classical HBS made continuous by a per-arc 180 deg relabelling.

The classical HBS of the osc family is discontinuous at four seams, and
across a seam the field jumps ~0.30 raw but only ~0.009 after R_180: the
SHAPE is continuous and only the branch LABEL flips.  So the family is
continuous if each arc between consecutive seams carries its neighbour's
label, i.e. if every sample in the flipped arcs is physically rotated by
180 deg.

WHICH arcs is measured, never assumed.  For every link between
consecutive valid samples -- including the links that straddle a NaN gap,
where two of the four seams actually live -- compare

    d_same = wRMS(|cf(psi)|,       |cf(psi')|)
    d_flip = wRMS(|R_180 cf(psi)|, |cf(psi')|)

A link that prefers the flipped side is a seam.  The flip state is then
carried forward from psi = 0, toggling at each seam; anchored that way the
state closes after four toggles, so the construction is consistent around
the whole circle including the 360 -> 0 wrap.

This touches nothing but the classical field: no theta*, no base model.
"""

import eval_osc_standard as S
import numpy as np


def links(valid):
    """(i, j) pairs, both defined, adjacent in psi.

    Consecutive samples inside a run, PLUS the pair straddling each NaN
    gap.  Measured on this family the gap links turn out not to be needed
    -- all four seams are interior to a run, the dead zones sit beside
    them rather than on them -- but a family whose seam fell inside an
    uncomputable stretch would otherwise be silently missed, and the
    extra links cost nothing.
    """
    out = [
        (i, i + 1) for i in range(len(valid) - 1) if valid[i] and valid[i + 1]
    ]
    k = 0
    while k < len(valid) - 1:
        if valid[k] and not valid[k + 1]:
            j = k + 1
            while j < len(valid) and not valid[j]:
                j += 1
            if j < len(valid):
                out.append((k, j))
        k += 1
    return sorted(out)


def rotate180(field, z, mask):
    from hbsn_exp import ops

    return ops.rotate_mu(field, np.pi, z, mask)


def seam_links(cf, valid, w, z, mask):
    """(i, j, d_same, d_flip, prefers_flip) for every link."""
    res = []
    for i, j in links(valid):
        d_same = S._wrms(np.abs(cf[i]), np.abs(cf[j]), w)
        d_flip = S._wrms(np.abs(rotate180(cf[i], z, mask)), np.abs(cf[j]), w)
        res.append((i, j, d_same, d_flip, d_flip < d_same))
    return res


def flip_states(cf, valid, w, z, mask):
    """Per-sample 0/1 flip state, anchored at 0 for psi = 0."""
    seamed = [i for i, _j, _a, _b, f in seam_links(cf, valid, w, z, mask) if f]
    s = np.zeros(len(cf), dtype=int)
    ptr, state = 0, 0
    for k in range(len(cf)):
        while ptr < len(seamed) and seamed[ptr] < k:
            state ^= 1
            ptr += 1
        s[k] = state
    return s, seamed


def apply_flip(cf, states, z, mask):
    """R_180 on the samples whose state is 1."""
    out = cf.copy()
    for i in np.where(states == 1)[0]:
        out[i] = rotate180(cf[i], z, mask)
    return out
