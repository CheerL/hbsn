#!/usr/bin/env python3
"""4-row column comparison: input / classical / HBSN / HBSN finetuned.

Review figure for the osc-family finetune: the same 11 snapshot columns
as the preview, one row per source, identical rendering (|field|*mask,
jet, 0-0.8).  All field data come from the dense caches; no
recomputation.  Pinned hbs==1.0.1 (HBS_PYTHON_DIR) before importing.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import grid, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
FIG_W, FIG_H = 17.0, 7.4
H_SLIDE = 0.933
AMP = 0.55
FLIPS = [40.38, 139.62, 220.89, 319.11]
COLS = [40.3, 40.5, 90.0, 139.5, 139.7, 180.0, 220.8, 221.0, 289.0,
        319.0, 319.2]
FIELD_CACHE = "/tmp/h3scan/fields_osc.npz"
FT_CACHE = "/tmp/h3scan/fields_osc_ft.npz"
PNG_PATH = os.path.join(OUT_DIR, "deformation_osc_ft_compare.png")
ROWS = ["input shape", "classical HBS", "HBSN", "HBSN finetuned"]
FONTSIZE = 13


def _infer11(net, ths):
    out = []
    for ps in COLS:
        img = shapes.shape_to_image(
            shapes.triangle_slide(x=AMP * np.sin(np.deg2rad(ps)),
                                  h=H_SLIDE))[..., 0]
        out.append(nets_field(net, img))
    return out


def nets_field(net, img):
    from hbsn_exp import nets
    return nets.field_to_complex(nets.infer(net, img))


def _raw_slide(psi):
    """Common-scale input triangle (per-shape rescale removed)."""
    x = AMP * np.sin(np.deg2rad(psi))
    a = np.array([[-0.7, 0.0], [0.7, 0.0], [x, H_SLIDE]])
    edges = [a[i] + (a[(i + 1) % 3] - a[i])
             * np.linspace(0, 1, 101, endpoint=False)[:, None]
             for i in range(3)]
    bd = np.concatenate(edges)
    return bd - bd.mean(axis=0)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="4-row osc column comparison")
    ap.add_argument("--ft-cache", default=FT_CACHE,
                    help="npz with dense hf fields for the finetuned row")
    ap.add_argument("--ckpt", default=None,
                    help="or a ckpt: infer the finetuned row live")
    args = ap.parse_args()
    d = np.load(FIELD_CACHE)
    ths, cf, hf = d["ths"], d["cf"], d["hf"]
    if args.ckpt:
        from hbsn_exp import nets
        net = nets.build_net(args.ckpt)
        hft = np.array(_infer11(net, ths), dtype=np.complex64)
    else:
        hft = np.load(args.ft_cache)["hf"]
    _, mask = grid.get_ghbs_grid()
    n = len(COLS)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gx0, gy0 = 0.75, 0.30
    gap_w, gap_h = 0.055, 0.085
    s = (FIG_W - gx0 - 0.12 - gap_w * (n - 1)) / n
    cw, rh = s + gap_w, s + gap_h
    grid_top = gy0 + 4 * s + 3 * gap_h
    axs = []
    for r in range(4):
        row = []
        for c in range(n):
            ax = fig.add_axes([(gx0 + c * cw) / FIG_W,
                               (gy0 + (3 - r) * rh) / FIG_H,
                               s / FIG_W, s / FIG_H])
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            row.append(ax)
        axs.append(row)
    axs = np.array(axs)
    zzs = [_raw_slide(ps) for ps in COLS]
    gmax = max(np.abs(w).max() for w in zzs)
    for c_idx, ps in enumerate(COLS):
        k = int(np.argmin(np.abs(ths - ps)))
        xy = zzs[c_idx] / gmax * 0.85
        w = xy[:, 0] + 1j * xy[:, 1]
        ax0 = axs[0, c_idx]
        ax0.fill(w.real, w.imag, color="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\psi={ps:.1f}$", fontsize=FONTSIZE - 4, pad=7)
        cplx = cf[k].astype(complex)
        hftr = hft[c_idx] if args.ckpt else hft[k]
        fields = [np.zeros_like(mask) if np.all(np.isnan(cplx))
                  else np.abs(cplx) * mask,
                  np.abs(hf[k].astype(complex)) * mask,
                  np.abs(hftr) * mask]
        for r, field in enumerate(fields):
            axs[r + 1, c_idx].imshow(field, cmap="jet",
                                     extent=[-1.5, 1.5, -1.5, 1.5],
                                     vmin=0, vmax=0.8)
            axs[r + 1, c_idx].axis("off")
    for r, label in enumerate(ROWS):
        row_y = gy0 + (3 - r) * rh + s / 2
        fig.text((gx0 - 0.10) / FIG_W, row_y / FIG_H, label, rotation=90,
                 ha="right", va="center", fontsize=FONTSIZE - 4)
    for f in FLIPS:
        i = min(range(n), key=lambda c: abs(COLS[c] - (f - 0.1)))
        xg = (gx0 + i * cw + s + gap_w / 2) / FIG_W
        fig.add_artist(plt.Line2D([xg, xg],
                                  [gy0 / FIG_H, (grid_top + 0.26) / FIG_H],
                                  transform=fig.transFigure, color="k",
                                  ls="--", lw=1.4, alpha=0.9))
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
