#!/usr/bin/env python3
"""Final result figure: the finetuned HBSN's output across the osc family.

Two rows, the input shape and the model's own field.  No reference row and
no comparison: this is the deliverable, not a diagnostic.  Same columns,
rendering and scale as the diagnostics that preceded it (|field| * mask,
jet, 0-0.8), so the panels stay directly comparable in the paper even
though nothing is drawn beside them.

Columns bracket both sides of each classical seam and sample the interior
of every arc; a psi whose classical field is uncomputable is rendered from
the nearest computable one and the title reports the psi actually used.
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

import osc_family as OF
from e1_deform_osc_gt_compare import _nearest_valid, _raw_slide
from finetune_osc import _images, _infer_all
from hbsn_exp import grid, nets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
CACHE = os.path.join(HERE, "../../runs/cache_osc")
FLIP_PAIRS = [
    (30.75, 31.00),
    (149.00, 149.25),
    (246.75, 247.00),
    (293.00, 293.25),
]
FLIPS = [0.5 * (a + b) for a, b in FLIP_PAIRS]
FIG_W = 24.0
FONTSIZE = 13
FLIPS = [30.88, 149.12, 246.88, 293.12]
COLS = sorted(
    [0.0, 15.0, 90.0, 180.0, 215.0, 330.0, 350.0]
    + [v for pair in FLIP_PAIRS for v in pair]
)


def main() -> None:
    ap = argparse.ArgumentParser(description="final result figure")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))

    d = np.load(os.path.join(CACHE, f"fields_osc_{OF.FAMILY}.npz"))
    ths, cf = d["ths"], d["cf"]
    valid = np.array([not np.all(np.isnan(c)) for c in cf])
    _, mask = grid.get_ghbs_grid()

    sel = [_nearest_valid(ths, valid, ps) for ps in COLS]
    used = [float(ths[k]) for k in sel]
    print("columns used:", [round(u, 2) for u in used])

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] inferring {len(sel)} fields from {args.ckpt}", flush=True)
    hf = _infer_all(nets.build_net(args.ckpt), _images(ths)[sel], dev)
    fields = np.abs(hf) * mask

    n, nrow = len(COLS), 2
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
        ax = fig.add_axes(
            [
                (gx0 + c_idx * cw) / FIG_W,
                gy0 / fig_h,
                s / FIG_W,
                s / fig_h,
            ]
        )
        ax.set_facecolor("black")
        ax.imshow(
            fields[c_idx],
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        ax.axis("off")

    for r, label in enumerate(("input shape", f"HBSN ({tag})")):
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
        "HBSN output on the osc triangle family  "
        "(dashed = the four classical seams)",
        fontsize=FONTSIZE,
        y=0.99,
    )
    out = args.out or os.path.join(OUT_DIR, f"deformation_osc_final_{tag}.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
