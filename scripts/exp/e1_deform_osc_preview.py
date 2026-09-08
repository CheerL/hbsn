#!/usr/bin/env python3
"""Oscillating-slide preview: apex x = 0.55 sin(psi) -> 4 flips/cycle.

The classical flip line is |x| = 0.358 (apex over the base endpoint).
Oscillating x = 0.55 sin(psi) crosses it four times per full cycle
(psi* = 40.6, 139.4, 220.6, 319.4 deg), so the classical representative
flips R_pi 4 times with visibly different shapes between (isosceles at
psi = 0/180, max tilt at 90/270).  HBSN morphs the same 180 deg net
reorientation continuously over ~1 deg windows (adjacent-d peak ~0.03 vs
classical 0.586).

Snapshots use a hybrid layout: wide context (isosceles psi = 0, max tilts
90/270) plus a tight +-0.1-deg pair around each flip - the classical row
flips 180 deg between visually identical input shapes, the HBSN row is
near-identical (d2 ~0.01).

Pinned hbs==1.0.1 (HBS_PYTHON_DIR); HBSN = nets.BEST_CKPT; rendering
|field|*mask, jet, 0-0.8.  Preview: deformation_osc_preview.png.
"""

import itertools
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, nets, ops, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
H_SLIDE = 0.933
AMP = 0.55  # osc amplitude; flips need |x| > 0.358
TH_LO, TH_HI = 0.0, 360.0
STEP = 0.25  # deg; spikes are instant, 0.25 resolves them
FLIPS = [40.38, 139.62, 220.89, 319.11]
# context (0/90/270) + tight +-0.1 deg pairs around the MEASURED flips
COLS = [0.0, 40.28, 40.48, 90.0, 139.52, 139.72, 220.79, 220.99, 270.0,
        319.01, 319.21]
FONTSIZE = 13
FIELD_CACHE = "/tmp/h3scan/fields_osc.npz"
PNG_PATH = os.path.join(OUT_DIR, "deformation_osc_preview.png")


def classic_or_none(bound):
    """classic_pixel; RuntimeError (1.0.1 zipper bug) -> None."""
    try:
        return classic.classic_pixel(bound, retries=8)
    except RuntimeError:
        return None


def osc_bound(psi):
    """Apex oscillates horizontally: x = AMP sin(psi), height h fixed."""
    return shapes.triangle_slide(x=AMP * np.sin(np.deg2rad(psi)), h=H_SLIDE)


def dense_scan(ths, net):
    """Per-psi: classic (None on fail) + HBSN field + Im I2."""
    z, mask = grid.get_ghbs_grid()
    cf_list, hf_list = [], []
    ims = np.full(ths.shape, np.nan)
    for k, ps in enumerate(ths):
        bound = osc_bound(ps)
        cf = classic_or_none(bound)
        img = shapes.shape_to_image(bound)[..., 0]
        hf = nets.field_to_complex(nets.infer(net, img))
        cf_list.append(cf)
        hf_list.append(hf)
        if cf is not None:
            ims[k] = ops.i2(cf, z, mask).imag
        if (k + 1) % 300 == 0:
            n_nan = sum(f is None for f in cf_list)
            print(f"scan {k + 1}/{len(ths)} psi={ps:.1f} nan={n_nan}", flush=True)
    return cf_list, hf_list, ims


def find_crossings(ths, ims):
    """Adjacent finite sign change, linear interp; merge 0.5-deg clusters."""
    valid = ~np.isnan(ims)
    sgn = np.sign(ims)
    idx = np.where(valid[:-1] & valid[1:] & (np.diff(sgn) != 0))[0]
    out = []
    for i in idx:
        a, b = ims[i], ims[i + 1]
        out.append(ths[i] + (ths[i + 1] - ths[i]) * abs(a) / (abs(a) + abs(b)))
    merged = []
    for c in out:
        if merged and c - merged[-1] <= 0.5:
            continue
        merged.append(c)
    return np.array(merged)


def verify_flip(psi, z, mask):
    """True if the representative flips R_pi across psi (+-0.15 deg ladder)."""
    lo = hi = None
    for off in [0.15, 0.3, 0.5, 0.8, 1.2]:
        if lo is None:
            lo = classic_or_none(osc_bound(psi - off))
        if hi is None:
            hi = classic_or_none(osc_bound(psi + off))
        if lo is not None and hi is not None:
            return bool(ops.is_flip(lo, hi, z, mask))
    return False


def register(fa, fb, z, mask) -> tuple[float, float]:
    """Best-fit rotation between fields: (phi_deg, registered residual d2)."""
    best = (0.0, np.inf)
    for lo, hi, st in [(0.0, 359.5, 5.0), (best[0] - 6, best[0] + 6, 0.25)]:
        for phi in np.arange(lo, hi + st / 2, st):
            d = ops.d2(ops.rotate_mu(fa, np.deg2rad(phi), z, mask), fb, z, mask)
            if d < best[1]:
                best = (float(phi % 360), float(d))
    return best


def curve_data(ths, fields, z, mask):
    """Adjacent-finite-sample d; x = pair midpoint psi."""
    fin = [i for i in range(len(fields)) if fields[i] is not None]
    xs, ds = [], []
    for i, j in itertools.pairwise(fin):
        xs.append(0.5 * (ths[i] + ths[j]))
        ds.append(ops.d2(fields[i], fields[j], z, mask))
    return np.array(xs), np.array(ds)


def plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask, flips) -> None:
    """Top: local d curves (classical spikes vs HBSN morph humps).
    Bottom: 3x11 snapshots (wide context + tight +-0.1-deg flip pairs).
    """
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.9], hspace=0.22)
    ax_a = fig.add_subplot(gs[0])
    gs_b = gs[1].subgridspec(3, 11, hspace=0.12, wspace=0.1)
    axs = np.array(
        [[fig.add_subplot(gs_b[r, c]) for c in range(11)] for r in range(3)]
    )
    ax_a.plot(xs_c, ds_c, color="r", lw=1.0, label="classical")
    ax_a.plot(xs_h, ds_h, color="b", lw=1.6, label="HBSN (morph)")
    for f in flips:
        ax_a.axvline(f, color="k", ls="--", lw=0.8)
    ax_a.set_xlim(TH_LO, TH_HI)
    ax_a.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax_a.set_xticklabels([f"{v}" for v in [0, 60, 120, 180, 240, 300, 360]])
    ax_a.set_ylabel(r"local field change $d(B_{\psi}, B_{\psi+0.25})$",
                    fontsize=FONTSIZE)
    ax_a.tick_params(labelsize=FONTSIZE - 2)
    ax_a.legend(fontsize=FONTSIZE - 2, loc="upper right")
    ax_a.set_xlabel(r"oscillation phase $\psi$ (deg), apex $x = 0.55\sin\psi$",
                    fontsize=FONTSIZE)
    for c_idx, ps in enumerate(cols):
        bound = osc_bound(ps)
        zz = bound[:, 0] + 1j * bound[:, 1]
        zz = zz - zz.mean()
        zz = zz / np.abs(zz).max() * 0.85
        ax0 = axs[0, c_idx]
        ax0.set_facecolor("black")
        ax0.fill(zz.real, zz.imag, color="white", edgecolor="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\psi={ps:.1f}$", fontsize=FONTSIZE - 4)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_visible(False)
        cf = classic_or_none(bound)
        img = shapes.shape_to_image(bound)[..., 0]
        hf = nets.field_to_complex(nets.infer(net, img))
        axs[1, c_idx].imshow(
            np.abs(cf) * mask if cf is not None else np.zeros_like(mask),
            cmap="jet", extent=[-1.5, 1.5, -1.5, 1.5], vmin=0, vmax=0.8)
        axs[1, c_idx].axis("off")
        axs[2, c_idx].imshow(
            np.abs(hf) * mask, cmap="jet", extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0, vmax=0.8)
        axs[2, c_idx].axis("off")
    for r, label in enumerate(["input shape", "classical HBS", "HBSN"]):
        axs[r, 0].text(-0.06, 0.5, label, transform=axs[r, 0].transAxes,
                       va="center", ha="center", rotation=90,
                       fontsize=FONTSIZE + 1)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_or_compute_fields(ths, net):
    """Field cache in /tmp; complex64 arrays; classic NaN rows kept."""
    if os.path.exists(FIELD_CACHE):
        d = np.load(FIELD_CACHE)
        if np.allclose(d["ths"], ths):
            cf = d["cf"].astype(complex)
            cf = [None if np.all(np.isnan(f)) else f for f in cf]
            return cf, list(d["hf"].astype(complex))
    cf_list, hf_list, _ = dense_scan(ths, net)
    cf_arr = np.array(
        [f if f is not None else np.full((128, 128), np.nan + 0j)
         for f in cf_list], dtype=np.complex64)
    hf_arr = np.array(hf_list, dtype=np.complex64)
    np.savez(FIELD_CACHE, ths=ths, cf=cf_arr, hf=hf_arr)
    cf_list = [None if np.all(np.isnan(f)) else f.astype(complex)
               for f in cf_arr]
    return cf_list, list(hf_arr.astype(complex))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="osc-slide preview figure")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore field cache")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    net = nets.build_net(nets.BEST_CKPT)
    z, mask = grid.get_ghbs_grid()
    ths = np.arange(TH_LO, TH_HI + STEP / 2, STEP)
    if args.recompute and os.path.exists(FIELD_CACHE):
        os.remove(FIELD_CACHE)
    cf_list, hf_list = load_or_compute_fields(ths, net)
    ims = np.array([np.nan if f is None else ops.i2(f, z, mask).imag
                    for f in cf_list])
    cros = find_crossings(ths, ims)
    flips = [float(c) for c in cros if verify_flip(c, z, mask)]
    print("crossings:", [f"{c:.2f}" for c in cros])
    print("verified flips:", [f"{c:.2f}" for c in flips])
    xs_c, ds_c = curve_data(ths, cf_list, z, mask)
    xs_h, ds_h = curve_data(ths, hf_list, z, mask)
    spike = ds_c > 0.3
    print("classical spikes (d>0.3): "
          + ", ".join(f"{x:.2f}" for x in xs_c[spike]))
    print(f"HBSN max step: {ds_h.max():.4f}")
    cols = COLS
    plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask, flips)
    print(f"figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
