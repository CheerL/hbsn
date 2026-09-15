#!/usr/bin/env python3
"""E1 - continuous deformation, OSCILLATING-SLIDE triangle family.

Same panel structure as e1_deform.py (semicircle sweep), retargeted to the
family the finetune was actually trained on (see osc_family):

    apex x = X0 + AMP sin(psi), h fixed   ->  4 classical flips per cycle

Window: psi* +- 0.72 deg (41 samples) about the seam at 246.88.  Of the
four seams this one and 293.12 are the only ones whose whole window is
computable -- 30.88 and 149.12 each have a classical dead zone overlapping
one side, leaving 29/41 fields and three spurious Im I2 crossings.  Here
the solver returns all 41 and Im I2 crosses exactly once.

(a) d(B_psi, B_psi*) vs psi: the classical curve (blue) is a step, the HBSN
    curve (red) is flat with zero R_pi flips.
(b) 3x4 snapshots at psi* +- eps, +-2eps: input shape / classical HBS /
    HBSN.

Pinned hbs==1.0.1 (HBS_PYTHON_DIR).  --ckpt selects the HBSN; rendering
|field|*mask, jet, 0-0.8.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import osc_family as OF
from hbsn_exp import classic, grid, nets, ops, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
AMP, H_SLIDE = OF.AMP, OF.H_SLIDE
PSI_STAR = 246.88
HALF_WIN, N_PSI = 0.72, 41
EPS = 0.0006  # snapshot offset in deg
FONTSIZE = 13
DEFAULT_CKPT = os.path.join(
    REPO, "runs/finetuned/osc/gtarc_x100a500_v1/best.pth"
)


def osc_bound(psi):
    return OF.bound(psi)


def main() -> None:
    ap = argparse.ArgumentParser(description="osc-family deformation figure")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(OUT_DIR, "f_deform.png")

    z, mask = grid.get_ghbs_grid()
    net = nets.build_net(args.ckpt)

    psis = np.linspace(PSI_STAR - HALF_WIN, PSI_STAR + HALF_WIN, N_PSI)
    classic_fields, hbsn_fields, i2s = [], [], []
    for ps in psis:
        bound = osc_bound(ps)
        try:
            cf = classic.classic_pixel(bound)
        except RuntimeError:
            cf = None
        img = shapes.shape_to_image(bound)[..., 0]
        hf = nets.field_to_complex(nets.infer(net, img))
        classic_fields.append(cf)
        hbsn_fields.append(hf)
        i2s.append(ops.i2(cf, z, mask) if cf is not None else np.nan + 0j)

    i2s = np.array(i2s)
    sign = np.sign(i2s.imag)
    valid = ~np.isnan(i2s.imag)
    crossings = np.where(valid[:-1] & valid[1:] & (np.diff(sign) != 0))[0]
    # Locate the flip by the R_pi test itself, not by the Im I2 zero.
    # Measured on this family the two do NOT coincide: Im I2 crosses at
    # 247.0075 while the field's 180 deg jump sits between the window
    # samples 246.9880 and 247.0240, because in x the flip is only about
    # 1.2e-4 wide.  Placing the snapshots at the zero crossing with the
    # usual eps = 6e-4 therefore put all four of them on ONE side and the
    # classical row showed no flip at all.  Anchor them on the endpoints of
    # the flipping pair instead -- that guarantees two before and two after.
    flip_pairs = [
        i
        for i in range(N_PSI - 1)
        if classic_fields[i] is not None
        and classic_fields[i + 1] is not None
        and ops.is_flip(classic_fields[i], classic_fields[i + 1], z, mask)
    ]
    if flip_pairs:
        i_star = int(flip_pairs[0])
        psi_lo, psi_hi = float(psis[i_star]), float(psis[i_star + 1])
        psi_star = 0.5 * (psi_lo + psi_hi)
    else:
        i_star = int(np.argmin(np.abs(i2s.imag)))
        psi_lo = psi_hi = psi_star = float(psis[i_star])

    d_classic = [
        np.nan
        if classic_fields[i] is None or classic_fields[i_star] is None
        else ops.d2(classic_fields[i], classic_fields[i_star], z, mask)
        for i in range(N_PSI)
    ]
    d_hbsn = [
        ops.d2(hbsn_fields[i], hbsn_fields[i_star], z, mask)
        for i in range(N_PSI)
    ]

    def count_flips(fields):
        n = 0
        for i in range(len(fields) - 1):
            if fields[i] is None or fields[i + 1] is None:
                continue
            if ops.is_flip(fields[i], fields[i + 1], z, mask):
                n += 1
        return n

    flip_classic = count_flips(classic_fields)
    flip_hbsn = count_flips(hbsn_fields)

    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.2)
    ax_a = fig.add_subplot(gs[0])
    gs_b = gs[1].subgridspec(3, 4, hspace=0.15, wspace=0.08)
    axs = np.array(
        [[fig.add_subplot(gs_b[r, c]) for c in range(4)] for r in range(3)]
    )

    # colour convention shared with the whole-period preview figure:
    # HBSN red, classical blue (the semicircle-era figure this replaces
    # had them the other way round)
    ax_a.plot(
        psis,
        d_classic,
        "b-o",
        lw=1.1,
        ms=3.2,
        label=f"classical (flips={flip_classic})",
    )
    ax_a.plot(
        psis,
        d_hbsn,
        "r-s",
        lw=1.6,
        ms=3,
        label=f"HBSN (flips={flip_hbsn})",
    )
    ax_a.axvline(psi_star, color="k", ls="--", lw=1)
    ax_a.text(
        psi_star + 0.01,
        np.nanmax(d_classic) * 1.18,
        rf"$\psi^*$={psi_star:.2f}$^\circ$",
        ha="left",
        va="center",
        fontsize=FONTSIZE,
    )
    ax_a.set_xlabel(
        rf"oscillation phase $\psi$ (deg), apex "
        rf"$x = {OF.X0:.2f} + {OF.AMP:.2f}\sin\psi$",
        fontsize=FONTSIZE,
    )
    ax_a.set_ylabel(r"$d(B_\psi, B_{\psi^*})$", fontsize=FONTSIZE)
    ax_a.tick_params(labelsize=FONTSIZE - 2)
    ax_a.set_xlim(psi_star - 0.35, psi_star + 0.35)
    ax_a.set_xticks([psi_star - 0.3, psi_star, psi_star + 0.3])
    ax_a.set_xticklabels(
        [f"{psi_star - 0.3:.1f}°", f"{psi_star:.1f}°", f"{psi_star + 0.3:.1f}°"]
    )
    ax_a.set_ylim(0, np.nanmax(d_classic) * 1.3)
    ax_a.legend(
        fontsize=FONTSIZE - 2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )

    # two before the flip, two after; +-EPS is far too small here (the
    # flip is ~1e-4 wide in x), so the pair endpoints carry the margin
    snap_psis = [psi_lo - 2 * EPS, psi_lo - EPS, psi_hi + EPS, psi_hi + 2 * EPS]
    row_labels = ["input shape", "classical HBS", "HBSN"]
    for c, pv in enumerate(snap_psis):
        # Fields at the TRUE snapshot phase, not at the nearest grid sample:
        # +-EPS is 6e-4 deg while the panel-(a) grid step is 3.6e-2, so
        # snapping to the grid would title a column 246.9868 and draw the
        # field of 246.9880, and would draw the same field twice in each
        # pair.  Four extra solver calls + inferences is cheap.
        bound = osc_bound(pv)
        try:
            f = classic.classic_pixel(bound)
        except RuntimeError:
            f = None
        hf = nets.field_to_complex(
            nets.infer(net, shapes.shape_to_image(bound)[..., 0])
        )

        zz = bound[:, 0] + 1j * bound[:, 1]
        zz = zz - zz.mean()
        zz = zz / np.abs(zz).max() * 0.85
        ax0 = axs[0, c]
        ax0.set_facecolor("black")
        ax0.fill(zz.real, zz.imag, color="white", edgecolor="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\psi={pv:.4f}^\circ$", fontsize=FONTSIZE + 1)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_visible(False)
        cx = float(OF.x_of(pv))
        ax0.text(
            zz[202].real,
            zz[202].imag + 0.12,
            f"apex x={cx:.5f}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 4,
            color="white",
            fontweight="bold",
            clip_on=False,
        )
        axs[1, c].imshow(
            np.abs(f) * mask if f is not None else np.zeros_like(mask),
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        axs[1, c].axis("off")
        axs[2, c].imshow(
            np.abs(hf) * mask,
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        axs[2, c].axis("off")

    for r, label in enumerate(row_labels):
        axs[r, 0].text(
            -0.08,
            0.5,
            label,
            transform=axs[r, 0].transAxes,
            va="center",
            ha="center",
            rotation=90,
            fontsize=FONTSIZE + 1,
        )

    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_b0 = axs[0, 0].get_position()
    pos_b3 = axs[0, 3].get_position()
    fig.text(
        (pos_a.x0 + pos_a.x1) / 2,
        0.965,
        r"(a) Distance from flip-point $d(B_\psi, B_{\psi^*})$",
        ha="center",
        fontsize=FONTSIZE + 1,
    )
    fig.text(
        (pos_b0.x0 + pos_b3.x1) / 2,
        0.965,
        rf"(b) HBS around flip at $\psi^*={psi_star:.2f}^\circ$",
        ha="center",
        fontsize=FONTSIZE + 1,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"ckpt: {args.ckpt}")
    print(
        f"psi* = {psi_star:.4f} deg  (Im I2 crossings in window: "
        f"{len(crossings)}; computable {int(valid.sum())}/{N_PSI})"
    )
    print(f"classical R_pi flips {flip_classic}, HBSN R_pi flips {flip_hbsn}")
    print(
        f"d(B,B*) classical max {np.nanmax(d_classic):.4f} -> "
        f"HBSN max {np.nanmax(d_hbsn):.4f}"
    )
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
