#!/usr/bin/env python3
"""Show the base <-> igtA16 180 deg branch difference as HBS field images.

base and the finetuned igtA16 differ by an EXACT 180 deg, alternating at
the four classical seams.  Measured over 102 probes: d180/d0 is either
~0.2 or ~4 with nothing in between (51 flipped / 51 same, zero
ambiguous), and the flip arcs are 40-140 and 220-319.

A curve would only restate that.  This figure makes it visible instead:
row 3 is the finetuned field, row 4 is the BASE field physically rotated
by 180 deg.  Where the two rows match, base and igtA16 are in opposite
branches; where row 3 instead matches row 2, they are in the same one.
The column title background is green for "same branch as base", red for
"opposite", so the arc-wise alternation reads off directly.

Columns are packed around the four seams (both sides of each) so the
switch points are bracketed rather than inferred.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_osc_standard as S
from e1_deform_osc_gt_compare import _nearest_valid, _raw_slide
from hbsn_exp import grid

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
FIG_W, FIG_H = 21.5, 6.6
FONTSIZE = 13
FLIPS = [40.38, 139.62, 220.89, 319.11]
CACHE = os.path.join(HERE, "../../runs/cache_osc")

# two probes inside each arc plus both sides of each seam; psi 270 is
# omitted because it sits in the 251.5-288.5 classical dead zone
COLS = [
    10.0,
    25.0,
    40.25,
    40.50,
    60.0,
    90.0,
    120.0,
    139.50,
    139.75,
    150.0,
    175.0,
    180.0,
    200.0,
    215.0,
    220.75,
    221.00,
    235.0,
    245.0,
    300.0,
    310.0,
    319.00,
    319.25,
    330.0,
    350.0,
]


def main() -> None:
    d = np.load(os.path.join(CACHE, "fields_osc.npz"))
    ths, cf, hf = d["ths"], d["cf"], d["hf"].astype(complex)
    ft = np.load(os.path.join(CACHE, "fields_osc_igtA16.npz"))["hf"].astype(
        complex
    )
    gt = np.load(os.path.join(CACHE, "ideal_gt_osc.npz"))["gt"].astype(complex)
    valid = np.array([not np.all(np.isnan(c)) for c in cf])
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)
    amp_b, amp_f, amp_g = (
        np.abs(hf).astype(np.float32),
        np.abs(ft).astype(np.float32),
        np.abs(gt).astype(np.float32),
    )

    sel = [_nearest_valid(ths, valid, ps) for ps in COLS]
    used = [float(ths[k]) for k in sel]
    # _rot_mod takes one 2-D field at a time, so rotate only the columns
    r180 = {k: S._rot_mod(hf[k], z, mask, [180.0])[0] for k in sel}

    rows = [
        ("input shape", None),
        ("classical HBS", np.abs(np.where(valid[:, None, None], cf, 0)) * mask),
        ("base HBSN", amp_b * mask),
        ("ft HBSN (igtA16)", amp_f * mask),
        ("R_180(base)", r180),
        ("gt (igtA16-framed)", amp_g * mask),
    ]
    n, nrow = len(COLS), len(rows)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gx0, gy0 = 0.80, 0.16
    gap_w, gap_h = 0.05, 0.085
    s = (FIG_W - gx0 - 0.12 - gap_w * (n - 1)) / n
    cw, rh = s + gap_w, s + gap_h
    grid_top = gy0 + nrow * s + (nrow - 1) * gap_h

    verdict = []
    for k in sel:
        d0 = S._wrms(amp_b[k], amp_f[k], w)
        d180 = S._wrms(r180[k], amp_f[k], w)
        verdict.append("same" if d0 < d180 else "flipped")

    zzs = [_raw_slide(ps) for ps in used]
    gmax = max(np.abs(v).max() for v in zzs)
    for c_idx, ps in enumerate(used):
        k = sel[c_idx]
        xy = zzs[c_idx] / gmax * 0.85
        ww = xy[:, 0] + 1j * xy[:, 1]
        ax0 = fig.add_axes(
            [
                (gx0 + c_idx * cw) / FIG_W,
                (gy0 + (nrow - 1) * rh) / FIG_H,
                s / FIG_W,
                s / FIG_H,
            ]
        )
        ax0.set_facecolor("black")
        ax0.fill(ww.real, ww.imag, color="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(
            rf"$\psi$={ps:.1f}",
            fontsize=FONTSIZE - 5,
            pad=7,
            backgroundcolor="#8fd18f"
            if verdict[c_idx] == "same"
            else "#e88a8a",
        )
        for r in range(1, nrow):
            ax = fig.add_axes(
                [
                    (gx0 + c_idx * cw) / FIG_W,
                    (gy0 + (nrow - 1 - r) * rh) / FIG_H,
                    s / FIG_W,
                    s / FIG_H,
                ]
            )
            ax.set_facecolor("black")
            ax.imshow(
                rows[r][1][k]
                if rows[r][1] is not None
                else np.zeros_like(mask),
                cmap="jet",
                extent=[-1.5, 1.5, -1.5, 1.5],
                vmin=0,
                vmax=0.8,
            )
            ax.axis("off")

    for r, (label, _) in enumerate(rows):
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
                [gy0 / FIG_H, (grid_top + 0.30) / FIG_H],
                transform=fig.transFigure,
                color="k",
                ls="--",
                lw=1.4,
                alpha=0.9,
            )
        )
    nflip = sum(v == "flipped" for v in verdict)
    fig.suptitle(
        "base vs ft (igtA16): an exact 180$^\\circ$ branch difference, "
        "alternating at the four classical seams\n"
        f"row 4 = base rotated 180$^\\circ$; title green = same branch as "
        f"base, red = opposite.  {nflip}/{n} columns opposite",
        fontsize=FONTSIZE,
        y=0.985,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "deformation_osc_frame_branch.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("columns used:", [round(u, 2) for u in used])
    print(
        "verdict:", list(zip([round(u, 1) for u in used], verdict, strict=True))
    )
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
