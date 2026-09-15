#!/usr/bin/env python3
"""Build the "ideal continuous HBS" ground truth for the osc triangle family.

Motivation.  The classical HBS of this family is discontinuous at four
seams: measured, across a seam the raw field jumps ~0.30 but only ~0.009
after R_180, i.e. the SHAPE is continuous and only the branch LABEL flips.
So a continuous version of the family does exist: classical fields carried
around with the label flipped after each seam.

Construction.
  1. frame: for each valid psi, theta*(psi) = argmin_theta
     wRMS_w(|h_base|, |R_theta cf|) -- the classical field expressed in the
     base model's orientation frame.  B(psi) = R_{theta*(psi)} cf(psi).
  2. branch anchoring (k): theta* is ill-conditioned wherever the field
     is near-symmetric -- the objective then has two minima ~180 deg
     apart and theta* hops between them.  Per psi pick k in {-1,0,1}
     minimising the wRMS distance of R_{180k} B to BASE, which pins the
     ABSOLUTE branch.  base, not the previous sample, must be the
     reference: wRMS is over modulus fields, where the two branches are
     near-tied off a seam, so a previous-sample chain has no signal to
     latch onto and simply inherits its (arbitrary) seed -- measured, that
     version emitted a gt rotated 180 deg from base across the family.
     Measured, k=0 wins at 1224/1224 valid psi, i.e. the seam absorption
     is theta* doing the work, not k.
  3. smoothing: whatever roughness is left is per-sample theta* estimation
     noise.  Smooth B along psi as COMPLEX fields (moving average over
     valid samples only, within a valid run) -- `--width` in samples,
     1 = off.
  4. psi where the classical field is NaN are skipped entirely; nothing is
     interpolated across them (the seams live in or beside those gaps, so
     interpolating across would invent the one thing we cannot verify).

KNOWN LIMIT -- four breaks remain and cannot be removed by this
construction.  theta* is discontinuous at psi 16.25, 163.50, 195.50 and
344.25 (118-119 deg), at none of which the classical branch flips
(classical adjacent-d there 0.003-0.010, base's own 0.0020 -- both
fields are smooth).  They sit either side of psi = 0 and 180, where the
triangle is mirror-symmetric (x = AMP sin psi = 0) and the orientation is
genuinely unidentifiable: the objective has two shallow basins 118 deg
apart whose gap is only 2-4% at the flop, widening to 100%+ over the
next 14 deg (16.25: 50% by 21.5 deg; 163.50: 100% by 171.5 deg).  So no
per-sample branch or tie-break rule removes them, it only relocates one
sample -- measured, a 5% tie-break left max and mean adjacent-d
bit-identical.  This is the symmetry obstruction the paper proves for
the classical 180 deg selection, reappearing in the orientation
estimate.

TWO FURTHER BREAKS ARE FIXABLE, AND --frame-cache FIXES THEM.
With base as the reference, theta* additionally snaps 155.5 deg at psi
218 and 322, where base's residual blows up to 0.157-0.179 against 0.05
typical: base is mid-morph and no rotation of cf fits it (its own
adjacent-d there is 0.033 against 0.0027 typical, the worst in the
family).  A finetuned ckpt does not share that breakdown -- measured,
igtA16's residual there is 0.051 and its plan moves 3.0 deg, not 155.5.
Pointing --frame-cache at its cached fields halves the gt's max
adjacent-d (width 9: 0.0528 -> 0.0258, now under base's own 0.0519),
drops the two worst flip windows from 0.0528 to 0.0131 (4x under base),
and takes F(gt -> classical) from 0.0288 to 0.0090, i.e. the gt becomes
a genuinely faithful continuous carry of the classical family.

CAUTION -- the reference also anchors the absolute BRANCH (see
_apply_branch), so switching it re-frames the gt.  Measured, base and
igtA16 differ by an exact 180 deg that alternates at the four classical
seams (0-40 same, 40-140 flipped, 140-220 same, 220-319 flipped,
319-360 same; 51/51, zero ambiguous, over 102 probes).  C is blind to
that -- R_180 is an isometry -- but the O gate is not: igtA16's own |R|
against base measures 0.011, far under FRAME_R_MIN = 0.25.  So a gt
built this way asks the finetune for a frame O will reject.  Decide
which of the two is the reference BEFORE training on it.

Reports the resulting adjacent-d profile so the smoothing width can be
chosen against evidence, and writes the field figure + theta(psi) overlay.

Usage:
    ideal_gt_osc.py --sweep              # report widths 1..21, no figure
    ideal_gt_osc.py --width 9 --figure   # build, save npz + figure
    ideal_gt_osc.py --width 9 --figure --frame-cache <ft fields.npz>
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_osc_standard as S
from finetune_osc import FLIPS, _images
from hbsn_exp import grid, ops

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
FIELD_CACHE = os.path.join(REPO, "runs/cache_osc/fields_osc.npz")
OUT_NPZ = os.path.join(REPO, "runs/cache_osc/ideal_gt_osc.npz")
COLS = [40.3, 90.0, 139.5, 180.0, 220.8, 289.0, 0.0, 60.0, 120.0, 200.0, 300.0]


def _theta_star(cf, hf, valid, w, z, mask):
    """Per-psi GLOBAL argmin of the base-anchored objective."""
    th = np.full(len(cf), np.nan)
    for i in np.where(valid)[0]:
        th[i] = S._best_theta(
            cf[i], np.abs(hf[i]).astype(np.float32), w, z, mask
        )[0]
    return th


def _runs(valid):
    out, s = [], None
    for i in range(len(valid)):
        if valid[i] and s is None:
            s = i
        if not valid[i] and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(valid) - 1))
    return out


def _apply_branch(cf, hf, valid, w, z, mask, th):
    """Stage 2: apply R_theta, then pick the ABSOLUTE branch k in {-1,0,1}.

    base is the EXTERNAL reference that fixes the absolute branch; the
    previous-sample chain is not, because it only sees modulus fields and
    so cannot tell the two branches apart off a seam.  Measured: base
    prefers k=0 at 1224/1224 valid psi (psi=90, wRMS(|B|,|base|)=0.107
    vs wRMS(|R180 B|,|base|)=0.322), i.e. the seam absorption attributed
    to k is really theta* doing the work.
    """
    G = np.zeros_like(cf)
    ks = np.zeros(len(cf), dtype=int)
    for i in np.where(valid)[0]:
        B = ops.rotate_mu(cf[i], np.deg2rad(th[i]), z, mask)
        cands = {
            k: (
                B
                if k == 0
                else ops.rotate_mu(B, np.deg2rad(180.0 * k), z, mask)
            )
            for k in (-1, 0, 1)
        }
        best = min(
            (
                (S._wrms(np.abs(hf[i]), np.abs(c), w), k, c)
                for k, c in cands.items()
            )
        )
        G[i], ks[i] = best[2], best[1]
    return G, ks


def branch_fix(cf, hf, valid, w, z, mask):
    """Stages 1-2: base frame + branch anchoring.

    theta* alone carries the continuity: it is the GLOBAL argmin, so it
    follows a real seam (the classical branch flips there and the global
    min moves with it), and `_apply_branch`'s k stays 0 at every valid
    psi.  Where theta* is discontinuous WITHOUT a classical branch flip
    the gt inherits a break -- see the module docstring's KNOWN LIMIT;
    no per-sample rule removes it.
    """
    th = _theta_star(cf, hf, valid, w, z, mask)
    G, ks = _apply_branch(cf, hf, valid, w, z, mask, th)
    return G, th, ks


def smooth_gt(G, valid, width):
    """Stage 3 -- moving average along psi within each valid run.

    The residual roughness is per-sample theta* estimation noise, which no
    branch choice can fix.  Averaging happens over valid samples only and
    never across a NaN gap; the run's two endpoints are kept from the
    average so the seam-adjacent values do not drift.
    """
    if width <= 1:
        return G
    out = G.copy()
    half = width // 2
    for a, b in _runs(valid):
        seg = G[a : b + 1]
        n = len(seg)
        sm = np.empty_like(seg)
        for j in range(n):
            lo, hi = max(0, j - half), min(n, j + half + 1)
            sm[j] = seg[lo:hi].mean(axis=0)
        sm[0], sm[-1] = seg[0], seg[-1]
        out[a : b + 1] = sm
    return out


def build_gt(cf, hf, valid, w, z, mask, width=1):
    """Branch-consistent + smoothed ideal gt."""
    G, th, ks = branch_fix(cf, hf, valid, w, z, mask)
    return smooth_gt(G, valid, width), th, ks


def _adj(F, valid, w):
    return np.array(
        [
            S._wrms(np.abs(F[i]), np.abs(F[i + 1]), w)
            if (valid[i] and valid[i + 1])
            else np.nan
            for i in range(len(F) - 1)
        ]
    )


def report(F, valid, w, ths, label):
    a = _adj(F, valid, w)
    mid = 0.5 * (ths[:-1] + ths[1:])
    pk = [float(np.nanmax(a[np.abs(mid - f) <= S.FLIP_WIN])) for f in FLIPS]
    print(
        f"{label:<26} max {np.nanmax(a):.4f}  mean {np.nanmean(a):.4f}  "
        f"flip peaks {[round(p, 4) for p in pk]}"
    )
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description="build ideal continuous gt")
    ap.add_argument("--width", type=int, default=9, help="smoothing window")
    ap.add_argument(
        "--frame-cache",
        default=None,
        help="npz with an `hf` to use as the ORIENTATION reference "
        "instead of the pretrained base model. Base's own plan snaps "
        "155.5 deg at psi 218/322 (where its theta* residual blows up to "
        "0.16 against 0.05 typical) and the gt inherits that break; a "
        "finetuned ckpt's cached fields do not. CAUTION: the reference "
        "also anchors the absolute branch, so switching it changes the "
        "gt's frame -- base and igtA16 differ by an exact 180 deg "
        "alternating at the four classical seams, which C cannot see but "
        "the O gate will.",
    )
    ap.add_argument("--sweep", action="store_true", help="report widths only")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--out", default=OUT_NPZ)
    args = ap.parse_args()

    d = np.load(FIELD_CACHE)
    ths, cf, hf = d["ths"], d["cf"], d["hf"].astype(complex)
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)
    valid = np.array([not np.all(np.isnan(c)) for c in cf])
    cf = np.where(valid[:, None, None], cf, 0).astype(complex)
    print(f"valid {int(valid.sum())}/{len(ths)} psi; {len(_runs(valid))} runs")

    ref = hf
    if args.frame_cache:
        ref = np.load(args.frame_cache)["hf"].astype(complex)
        print(f"orientation reference: {args.frame_cache}")

    report(cf, valid, w, ths, "classical (raw)")
    report(hf, valid, w, ths, "base")
    report(ref, valid, w, ths, "reference")
    report(
        _rot_only(
            cf, ref, valid, w, z, mask, _theta_star(cf, ref, valid, w, z, mask)
        ),
        valid,
        w,
        ths,
        "R_theta* (k applied)",
    )

    if args.sweep:
        # front stage once, then only the cheap smoothing is swept
        G0, _, ks = branch_fix(cf, ref, valid, w, z, mask)
        nsw = int((ks[np.where(valid)[0]] != 0).sum())
        print(f"branch switches: {nsw}")
        for width in (1, 3, 5, 7, 9, 13, 21):
            report(
                smooth_gt(G0, valid, width),
                valid,
                w,
                ths,
                f"gt width={width}",
            )
        return

    G0, th, ks = branch_fix(cf, ref, valid, w, z, mask)
    G = smooth_gt(G0, valid, args.width)
    a = report(G, valid, w, ths, f"gt width={args.width}")
    # how far did smoothing move the fields?  must stay below the family's
    # own step scale (classical mean adjacent-d), or we are inventing shape
    dev = np.array(
        [S._wrms(np.abs(G0[i]), np.abs(G[i]), w) for i in np.where(valid)[0]]
    )
    scales = np.nanmean(a)
    print(
        f"shape moved by smoothing: mean {dev.mean():.4f} max {dev.max():.4f} "
        f"(family step scale {scales:.4f}; ratio {dev.mean() / scales:.2f})"
    )
    sw = np.where(valid)[0][ks[np.where(valid)[0]] != 0]
    print(
        f"branch switches: {len(sw)} at psi {[round(float(ths[i]), 2) for i in sw[:16]]}"
    )
    np.savez_compressed(
        args.out,
        ths=ths,
        gt=G.astype(np.complex64),
        theta_star=th,
        branch_k=ks,
        valid=valid,
        adj=a,
        width=args.width,
        frame_ref=np.array(args.frame_cache or "base"),
    )
    print(f"saved: {args.out}")
    if args.figure:
        make_figure(ths, cf, hf, G, th, ks, valid, a, args.out)


def _rot_only(cf, hf, valid, w, z, mask, th):
    B = np.zeros_like(cf)
    for i in np.where(valid)[0]:
        B[i] = ops.rotate_mu(cf[i], np.deg2rad(th[i]), z, mask)
    return B


def make_figure(ths, cf, hf, gt, th, ks, valid, adj, out_npz):
    """Fields for the preview columns + theta(psi) overlay underneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _z, _mask = grid.get_ghbs_grid()
    fig = plt.figure(figsize=(2.1 * len(COLS), 7.4))
    gs = fig.add_gridspec(
        3, len(COLS), height_ratios=[1, 1, 1.5], hspace=0.28, wspace=0.06
    )
    rows = [("input shape", None), ("classical HBS", cf), ("ideal gt", gt)]
    imgs = _images(ths)
    for r, (label, F) in enumerate(rows):
        for c, ps in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            k = int(np.argmin(np.abs(ths - ps)))
            if F is None:
                ax.imshow(imgs[k], cmap="gray")
            elif not valid[k]:
                ax.text(0.5, 0.5, "NaN", ha="center", va="center", fontsize=8)
            else:
                ax.imshow(np.abs(F[k]), cmap="jet", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{ps:.1f}$^\\circ$", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(ths, th, ".", ms=2, color="tab:blue", label=r"$\theta^*$ (raw)")
    thb = th + 180.0 * ks
    ax.plot(
        ths,
        thb,
        "-",
        lw=1.2,
        color="tab:red",
        label=r"$\theta^* + 180k$ (branch-fixed)",
    )
    for f in FLIPS:
        ax.axvline(f, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\psi$ (deg)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        f"orientation plan; max adjacent-d {np.nanmax(adj):.4f} "
        f"(classical 0.3031, base 0.0519)",
        fontsize=9,
    )
    png = os.path.join(REPO, "figures/hbsn/candidates/ideal_gt_osc.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
