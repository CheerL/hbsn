#!/usr/bin/env python3
"""6-row column comparison: input / classical / classical+arc-flip / gt /
base HBSN / ft.

Rows, top to bottom:
    input shape      the boundary triangle (common scale)
    classical HBS    raw hbs 1.0.1 field, jumps ~0.30 at each of the 4 seams
    classical + R180 per arc
                     the same field with the arcs between seams physically
                     rotated 180 deg, which is what makes it continuous
                     (scripts/exp/classical_arcs.py; arcs measured, not
                     assumed)
    rotated HBS (gt) R_{theta*} cf after branch-consistency + smoothing
                     (scripts/exp/ideal_gt_osc.py)
    base HBSN        the pretrained model
    ft HBSN          the finetuned checkpoint (--ckpt)

The arc-flip row is the point of the figure: it shows that the classical
family's discontinuity is purely a branch LABEL, since a 180 deg rotation
inside the two arcs removes all four seams (max adjacent-d 0.3031 -> 0.0136,
zero pairs above 0.02, against base's own 0.0519).  The gt row is the
target the finetune was actually given.  Identical rendering to
e1_deform_osc_preview (|field|*mask, jet, 0-0.8), with the 139-180 and
180-220 arcs densified so a break inside either shows up as a trend
rather than as one suspicious pair.  A column whose psi is classical-NaN
is rendered from the nearest valid psi and the title reports the psi
actually used.
Pinned hbs==1.0.1 (HBS_PYTHON_DIR) before importing.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import classical_arcs as CA
import eval_osc_standard as S
import osc_family as OF
from hbsn_exp import grid, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
FIG_W, FIG_H = 26.8, 8.9
H_SLIDE, AMP = OF.H_SLIDE, OF.AMP
FLIPS = [30.88, 149.12, 246.88, 293.12]
# 139-180 and 180-220 densified: base and the gt both break inside those
# arcs while the finetuned row does not, and the existing grid only had the
# two ends.  Columns are the ones this family's classical field can
# actually be computed at.
COLS = [
    40.3,
    40.5,
    90.0,
    139.5,
    139.7,
    150.0,
    160.0,
    170.0,
    175.0,
    180.0,
    185.0,
    190.0,
    200.0,
    210.0,
    220.8,
    221.0,
    289.0,
    319.0,
    319.2,
]
FIELD_CACHE = (
    os.path.join(HERE, "../../runs/cache_osc", f"fields_osc_{OF.FAMILY}.npz")
)
GT_CACHE = (
    os.path.join(HERE, "../../runs/cache_osc", f"gt_arcs_{OF.FAMILY}.npz")
)
PNG_PATH = os.path.join(OUT_DIR, "deformation_osc_gt_compare.png")
FONTSIZE = 13


def _nearest_valid(ths, valid, ps):
    """Closest psi whose classical field is computable.

    The classical HBS is NaN on ~15% of the sweep (dead zones beside the
    seams).  A requested column that lands in one is rendered from the
    nearest valid psi instead, and the panel title reports the psi actually
    used so the substitution is visible rather than silent.
    """
    for k in np.argsort(np.abs(ths - ps)):
        if valid[k]:
            return int(k)
    raise ValueError(f"no valid psi anywhere near {ps}")


def _adj_profile(F, valid, w):
    """adjacent-d between consecutive psi, NaN across a dead zone."""
    return np.array(
        [
            S._wrms(F[i], F[i + 1], w)
            if (valid[i] and valid[i + 1])
            else np.nan
            for i in range(len(F) - 1)
        ]
    )


def _raw_slide(psi):
    """Common-scale input triangle (per-shape rescale removed)."""
    x = float(OF.x_of(psi))
    a = np.array([[-OF.BASE_HALF, 0.0], [OF.BASE_HALF, 0.0], [x, H_SLIDE]])
    edges = [
        a[i]
        + (a[(i + 1) % 3] - a[i])
        * np.linspace(0, 1, 101, endpoint=False)[:, None]
        for i in range(3)
    ]
    bd = np.concatenate(edges)
    return bd - bd.mean(axis=0)


def _infer_sel(net, ths, sel):
    """Fields for the selected psi indices, inferred live."""
    from hbsn_exp import nets

    out = []
    for k in sel:
        img = shapes.shape_to_image(
            shapes.triangle_slide(x=float(OF.x_of(ths[k])), h=H_SLIDE)
        )[..., 0]
        out.append(nets.field_to_complex(nets.infer(net, img)))
    return np.array(out, dtype=np.complex64)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="6-row osc comparison")
    ap.add_argument("--ckpt", required=True, help="finetuned checkpoint")
    ap.add_argument("--tag", default=None, help="row label + filename suffix")
    ap.add_argument(
        "--gt-cache", default=GT_CACHE, help="npz from ideal_gt_osc.py"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))
    png_path = args.out or os.path.join(
        OUT_DIR,
        "deformation_osc_gt_compare" + (f"_{tag}" if tag else "") + ".png",
    )

    d = np.load(FIELD_CACHE)
    ths, cf, hf = d["ths"], d["cf"], d["hf"].astype(complex)
    g = np.load(args.gt_cache)
    gt = g["gt"].astype(complex)
    gvalid = g["valid"]
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)
    sel = [_nearest_valid(ths, gvalid, ps) for ps in COLS]
    used = [float(ths[k]) for k in sel]
    print("columns used:", [round(u, 2) for u in used])
    for want, got in zip(COLS, used, strict=True):
        if abs(want - got) > 0.2:
            print(f"  psi {want:.1f} is classical NaN -> substituted {got:.2f}")

    # classical carried around with the 180 deg relabelling per arc, so the
    # four seams disappear; see classical_arcs for how the arcs are measured
    cfz = np.where(gvalid[:, None, None], cf, 0).astype(complex)
    states, seamed = CA.flip_states(cfz, gvalid, w, z, mask)
    cf_arc = CA.apply_flip(cfz, states, z, mask)
    print(
        f"arc flip: {len(seamed)} measured seams at psi "
        f"{[round(float(ths[i]), 2) for i in seamed]}"
    )
    a_raw = _adj_profile(np.abs(cfz), gvalid, w)
    a_arc = _adj_profile(np.abs(cf_arc), gvalid, w)
    print(
        f"  adjacent-d  raw classical max {np.nanmax(a_raw):.4f} "
        f"({int((a_raw > 0.02).sum())} pairs > 0.02)   "
        f"arc-flipped max {np.nanmax(a_arc):.4f} "
        f"({int((a_arc > 0.02).sum())} pairs > 0.02)"
    )

    from hbsn_exp import nets

    hft = _infer_sel(nets.build_net(args.ckpt), ths, sel)

    n = len(COLS)
    nrow = 6
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gx0, gy0 = 0.75, 0.30
    gap_w, gap_h = 0.055, 0.085
    s = (FIG_W - gx0 - 0.12 - gap_w * (n - 1)) / n
    cw, rh = s + gap_w, s + gap_h
    grid_top = gy0 + nrow * s + (nrow - 1) * gap_h
    axs = np.empty((nrow, n), dtype=object)
    for r in range(nrow):
        for c in range(n):
            ax = fig.add_axes(
                [
                    (gx0 + c * cw) / FIG_W,
                    (gy0 + (nrow - 1 - r) * rh) / FIG_H,
                    s / FIG_W,
                    s / FIG_H,
                ]
            )
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            axs[r, c] = ax

    zzs = [_raw_slide(ps) for ps in used]
    gmax = max(np.abs(w).max() for w in zzs)
    for c_idx, ps in enumerate(used):
        k = sel[c_idx]
        xy = zzs[c_idx] / gmax * 0.85
        w = xy[:, 0] + 1j * xy[:, 1]
        ax0 = axs[0, c_idx]
        ax0.fill(w.real, w.imag, color="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\psi={ps:.2f}$", fontsize=FONTSIZE - 5, pad=7)
        cl = cf[k].astype(complex)
        fields = [
            np.zeros_like(mask) if np.all(np.isnan(cl)) else np.abs(cl) * mask,
            np.abs(cf_arc[k]) * mask if gvalid[k] else np.zeros_like(mask),
            np.abs(gt[k]) * mask if gvalid[k] else np.zeros_like(mask),
            np.abs(hf[k]) * mask,
            np.abs(hft[c_idx]) * mask,
        ]
        for r, field in enumerate(fields):
            axs[r + 1, c_idx].imshow(
                field,
                cmap="jet",
                extent=[-1.5, 1.5, -1.5, 1.5],
                vmin=0,
                vmax=0.8,
            )
            axs[r + 1, c_idx].axis("off")

    labels = [
        "input shape",
        "classical HBS",
        "classical + R180 per arc",
        "rotated HBS (gt)",
        "base HBSN",
        f"ft HBSN ({tag})",
    ]
    for r, label in enumerate(labels):
        row_y = gy0 + (nrow - 1 - r) * rh + s / 2
        fig.text(
            (gx0 - 0.10) / FIG_W,
            row_y / FIG_H,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=FONTSIZE - 4,
        )
    for f in FLIPS:
        i = min(range(n), key=lambda c: abs(COLS[c] - (f - 0.1)))
        xg = (gx0 + i * cw + s + gap_w / 2) / FIG_W
        fig.add_artist(
            plt.Line2D(
                [xg, xg],
                [gy0 / FIG_H, (grid_top + 0.26) / FIG_H],
                transform=fig.transFigure,
                color="k",
                ls="--",
                lw=1.4,
                alpha=0.9,
            )
        )
    fig.suptitle(
        "osc family: classical HBS vs the ideal continuous gt it is "
        "finetuned against",
        fontsize=FONTSIZE,
        y=0.985,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {png_path}")


if __name__ == "__main__":
    main()
