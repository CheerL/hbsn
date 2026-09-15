#!/usr/bin/env python3
"""Three-row figure: input shape / classical HBS / classical with the arcs flipped.

The classical HBS of the osc family breaks at four seams, but across a
seam the field jumps ~0.30 raw and only ~0.007 after R_180 -- the shape
is continuous and only the branch LABEL flips.  Row 3 is the classical
field with that label carried around: every sample inside the two arcs
between seams is physically rotated 180 deg.

The arcs are measured, not assumed (scripts/exp/classical_arcs.py), and
row 3 is indistinguishable from row 2 outside them -- the only visible
difference is at the four seams, where row 2 jumps and row 3 does not.
Measured: max adjacent-d 0.3031 -> 0.0136, pairs above 0.02 from 4 to 0,
against the base model's own 0.0519.

No checkpoint is needed: nothing here touches HBSN.
"""

import argparse
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
from e1_deform_osc_gt_compare import _adj_profile, _nearest_valid, _raw_slide
from finetune_osc import _images, _infer_all
from hbsn_exp import grid, nets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
CACHE = os.path.join(HERE, "../../runs/cache_osc")

FIG_W = 24.0
FONTSIZE = 13
FLIPS = [30.88, 149.12, 246.88, 293.12]
# both sides of each seam, plus interior probes of every arc; 270 is
# omitted because it sits in the 251.5-288.5 classical dead zone
COLS = [
    10.0,
    20.0,
    30.75,
    31.00,
    50.0,
    80.0,
    110.0,
    140.0,
    149.00,
    149.25,
    170.0,
    200.0,
    230.0,
    246.75,
    247.00,
    260.0,
    275.0,
    288.0,
    293.00,
    293.25,
    310.0,
    335.0,
    355.0,
]


def main() -> None:
    ap = argparse.ArgumentParser(description="3-row arc-flip figure")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="optional finetune; adds base-HBSN and ft-HBSN rows so the "
        "target and the two models can be read off the same columns",
    )
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    d = np.load(os.path.join(CACHE, f"fields_osc_{OF.FAMILY}.npz"))
    ths, cf = d["ths"], d["cf"]
    valid = np.array([not np.all(np.isnan(c)) for c in cf])
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)
    cfz = np.where(valid[:, None, None], cf, 0).astype(complex)

    states, seamed = CA.flip_states(cfz, valid, w, z, mask)
    cf_arc = CA.apply_flip(cfz, states, z, mask)
    a_raw = _adj_profile(np.abs(cfz), valid, w)
    a_arc = _adj_profile(np.abs(cf_arc), valid, w)

    print(
        f"{len(seamed)} measured seams at psi "
        f"{[round(float(ths[i]), 2) for i in seamed]}"
    )
    print(
        f"  raw classical       max {np.nanmax(a_raw):.4f} "
        f"({int((a_raw > 0.02).sum())} pairs > 0.02)"
    )
    print(
        f"  + R180 per arc      max {np.nanmax(a_arc):.4f} "
        f"({int((a_arc > 0.02).sum())} pairs > 0.02)"
    )
    i0, i1 = 0, len(ths) - 1
    print(
        f"  wrap 360->0         "
        f"{S._wrms(np.abs(cf_arc[i0]), np.abs(cf_arc[i1]), w):.4f}"
    )
    print("  flipped arcs: ", end="")
    i = 0
    while i < len(states):
        if states[i] == 1:
            j = i
            while j + 1 < len(states) and states[j + 1] == 1:
                j += 1
            print(f"{ths[i]:.2f}-{ths[j]:.2f}", end="  ")
            i = j
        i += 1
    print()

    sel = [_nearest_valid(ths, valid, ps) for ps in COLS]
    used = [float(ths[k]) for k in sel]
    # every row is stored PER COLUMN (indexed by c_idx), so the two optional
    # model rows -- which are only inferred at the display columns -- line up
    # with the two classical rows without a second indexing convention
    rows = [
        ("input shape", None),
        ("classical HBS", (np.abs(cfz) * mask)[sel]),
        ("classical + R180\nin arcs", (np.abs(cf_arc) * mask)[sel]),
    ]
    if args.ckpt:
        hf_base = np.load(os.path.join(CACHE, f"fields_osc_{OF.FAMILY}.npz"))[
            "hf"
        ].astype(complex)
        rows.append(("base HBSN", (np.abs(hf_base) * mask)[sel]))
        tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))
        print(
            f"[{tag}] inferring {len(sel)} fields from {args.ckpt}", flush=True
        )
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        hf_ft = _infer_all(nets.build_net(args.ckpt), _images(ths)[sel], dev)
        rows.append((f"ft HBSN ({tag})", np.abs(hf_ft) * mask))

    n, nrow = len(COLS), len(rows)
    gx0, gy0 = 0.95, 0.22
    gap_w, gap_h = 0.05, 0.085
    s = (FIG_W - gx0 - 0.12 - gap_w * (n - 1)) / n
    cw, rh = s + gap_w, s + gap_h
    grid_top = gy0 + nrow * s + (nrow - 1) * gap_h
    fig_h = grid_top + 0.95
    fig = plt.figure(figsize=(FIG_W, fig_h))

    zzs = [_raw_slide(ps) for ps in used]
    gmax = max(np.abs(v).max() for v in zzs)
    for c_idx, ps in enumerate(used):
        xy = zzs[c_idx] / gmax * 0.85
        ww = xy[:, 0] + 1j * xy[:, 1]
        ax0 = fig.add_axes(
            [
                (gx0 + c_idx * cw) / FIG_W,
                (gy0 + (nrow - 1) * rh) / fig_h,
                s / FIG_W,
                s / fig_h,
            ]
        )
        ax0.set_facecolor("black")
        ax0.fill(ww.real, ww.imag, color="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(rf"$\psi$={ps:.2f}", fontsize=FONTSIZE - 5, pad=7)
        for r in range(1, nrow):
            ax = fig.add_axes(
                [
                    (gx0 + c_idx * cw) / FIG_W,
                    (gy0 + (nrow - 1 - r) * rh) / fig_h,
                    s / FIG_W,
                    s / fig_h,
                ]
            )
            ax.set_facecolor("black")
            ax.imshow(
                rows[r][1][c_idx],
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
            row_y / fig_h,
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
                [gy0 / fig_h, (grid_top + 0.22) / fig_h],
                transform=fig.transFigure,
                color="k",
                ls="--",
                lw=1.4,
                alpha=0.9,
            )
        )
    fig.suptitle(
        "classical HBS, and the same field with the two mid-arcs rotated "
        "$180^\\circ$\n",
        fontsize=FONTSIZE,
        y=0.99,
    )
    fig.text(
        0.5,
        0.955,
        "dashed = the four classical seams    |    "
        f"max adjacent-$d$  {np.nanmax(a_raw):.4f} "
        f"$\\rightarrow$ {np.nanmax(a_arc):.4f}    |    "
        f"pairs above 0.02   {int((a_raw > 0.02).sum())} "
        f"$\\rightarrow$ {int((a_arc > 0.02).sum())}",
        ha="center",
        fontsize=FONTSIZE - 4,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "deformation_osc_arcs.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("columns used:", [round(u, 2) for u in used])
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
