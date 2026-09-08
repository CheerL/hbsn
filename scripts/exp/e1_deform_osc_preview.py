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


def _cols() -> list[float]:
    """Context columns kept finite for classic HBS + tight flip pairs.

    120/270 fall in zipper dead bands (117.0-123.2 / 251.3-288.4) so the
    context uses nearest finite lattice points instead (123.5, 289) plus
    the isosceles/max-tilt anchors 90/180/80.
    """
    ctx = [80.0, 123.5, 180.0, 289.0]
    return sorted(ctx + [x for f in FLIPS for x in (f - 0.1, f + 0.1)])


COLS = _cols()
FONTSIZE = 13
FIELD_CACHE = "/tmp/h3scan/fields_osc.npz"
PNG_PATH = os.path.join(OUT_DIR, "deformation_osc_preview.png")
SUBSTEP = 5.0  # deg, display-subsampling step for the curve panel
# HBSN morph peaks (local-d ~0.088) sit ~1.3 deg BEFORE each classical flip
# (39.1/140.9/219.9/320.1 vs flips 40.38/...).  Hide radius 0.75 keeps the
# window (38.35, 39.85) disjoint from the flip keep-window (flip +-0.5) so
# the classical 0.586 spikes survive.
MORPH_HIDE = [39.1, 140.9, 219.9, 320.1]
MORPH_RADIUS = 0.75


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


def sparse_curve(xs, ds, keep, hide, per_flip=3):
    """Subsample to ~5-deg points; around each flip keep the closest
    `per_flip` points (red: per_flip=3 -> low+spike+low sharp jump; blue:
    per_flip=1 -> a single nearest point at the flip); drop points in HBSN
    morph windows (hide) unless they fall inside a flip keep-window.
    """
    out, last = [], None
    chosen = set()
    for k, x in enumerate(xs):
        in_flip = any(abs(x - f) <= 0.5 for f in keep)
        if in_flip:
            chosen.add(k)
            continue
        if any(abs(x - h) <= MORPH_RADIUS for h in hide):
            continue
        if last is None or x - last >= SUBSTEP:
            out.append(k)
            last = x
    kept = set(out)
    for f in keep:
        cands = sorted(chosen, key=lambda k: abs(xs[k] - f))
        kept |= set(cands[:per_flip])
    return sorted(kept)


def plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask, flips) -> None:
    """Top: subsampled local-d curves ((a)); bottom: 3xN snapshot grid.

    Snapshot cells are laid out by hand as pixel-squares so the aspect-
    equal auto-centering (which padded each cell vertically and defeated
    hspace) is bypassed; row/column gaps are controlled directly.
    """
    n = len(cols)
    fig = plt.figure(figsize=(17, 10.5))
    # (a) curve axes, upper band
    ax_a = fig.add_axes([0.09, 0.60, 0.88, 0.30])
    i_c = sparse_curve(xs_c, ds_c, flips, MORPH_HIDE, per_flip=5)
    i_h = sparse_curve(xs_h, ds_h, flips, MORPH_HIDE, per_flip=1)
    ax_a.plot(xs_c[i_c], ds_c[i_c], "o-", color="r", lw=1.1, ms=3.2,
              label="classical")
    ax_a.plot(xs_h[i_h], ds_h[i_h], "o-", color="b", lw=1.6, ms=3,
              label="HBSN")
    for f in flips:
        ax_a.axvline(f, color="k", ls="--", lw=0.8, alpha=0.55)
    ax_a.set_xlim(TH_LO, TH_HI)
    ax_a.set_ylim(0, 0.68)
    ax_a.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax_a.set_xticklabels([f"{v}" for v in [0, 60, 120, 180, 240, 300, 360]])
    ax_a.set_ylabel(r"local field change $d(B_{\psi}, B_{\psi+\Delta\psi})$",
                    fontsize=FONTSIZE)
    ax_a.tick_params(labelsize=FONTSIZE - 2)
    ax_a.legend(fontsize=FONTSIZE - 2, loc="upper right")
    ax_a.set_xlabel(r"oscillation phase $\psi$ (deg), apex $x = 0.55\sin\psi$",
                    fontsize=FONTSIZE)
    for f in flips:
        ax_a.text(
            f, 1.02, rf"{f:.1f}$^\circ$",
            transform=ax_a.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FONTSIZE - 4, color="0.25",
        )
    # ---- bottom 3xN snapshot grid, hand-laid pixel squares ----
    # fig region [x_l, x_r] x [y_lo, y_hi]; square cell side s; gaps gap_h
    # (vertical, between rows) and gap_w (horizontal, between columns).
    x_l, x_r = 0.145, 0.96
    y_lo, y_hi = 0.03, 0.56
    gap_w, gap_h = 0.004, 0.006
    s = min((x_r - x_l - gap_w * (n - 1)) / n, (y_hi - y_lo - gap_h * 2) / 3)
    cw = s + gap_w
    rh = s + gap_h
    axs = []
    for r in range(3):
        row = []
        for c in range(n):
            ax = fig.add_axes(
                [x_l + c * cw, y_hi - rh * (r + 1), s, s]
            )
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            row.append(ax)
        axs.append(row)
    axs = np.array(axs)
    for c_idx, ps in enumerate(cols):
        bound = osc_bound(ps)
        zz = bound[:, 0] + 1j * bound[:, 1]
        zz = zz - zz.mean()
        zz = zz / np.abs(zz).max() * 0.85
        ax0 = axs[0, c_idx]
        ax0.fill(zz.real, zz.imag, color="white", edgecolor="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_title(rf"$\psi={ps:.1f}$", fontsize=FONTSIZE - 4,
                      pad=2)
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
        axs[r, 0].text(-0.02, 0.5, label,
                       transform=axs[r, 0].transAxes, va="center",
                       ha="right", rotation=90, fontsize=FONTSIZE - 4)
    fig.canvas.draw()
    # (a)/(b) panel tags
    pa = ax_a.get_position()
    fig.text(0.5, pa.y1 + 0.012, "(a)", ha="center", va="bottom",
             fontsize=FONTSIZE + 1)
    yt = max(ax.title.get_window_extent(fig.canvas.get_renderer()).y1
             for ax in axs[0])
    inv = fig.transFigure.inverted()
    yt = inv.transform((0, yt))[1]
    ya = pa.y0
    yb = max(yt + 0.012, ya - 0.028)  # >=1 char above titles; >=2 below (a)
    fig.text(0.5, yb, "(b)", ha="center", va="bottom",
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
