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

    120/270 fall in zipper dead bands (117.0-123.2 / 251.3-288.4), so the
    context uses the max-tilt anchor 90, the isosceles anchor 180, and the
    nearest finite point past the second dead band (289).
    """
    ctx = [90.0, 180.0, 289.0]
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


def sparse_curve(xs, hide, spikes=()):
    """Uniform ~5-deg subsample (HBSN morph windows dropped); the extra
    nearest sample is added only for `spikes` (classical flip peaks)."""
    out, last = [], None
    for k, x in enumerate(xs):
        if any(abs(x - h) <= MORPH_RADIUS for h in hide):
            continue
        if last is None or x - last >= SUBSTEP:
            out.append(k)
            last = x
    kept = set(out)
    for f in spikes:
        kept.add(int(min(range(len(xs)), key=lambda k: abs(xs[k] - f))))
    return sorted(kept)


def _raw_slide(psi):
    """Raw (unscaled) slide-triangle boundary, mean-centered like
    triangle_slide; input row draws all columns at one COMMON scale
    (triangle_slide's per-shape rescale magnifies the isosceles ~14%)."""
    x = AMP * np.sin(np.deg2rad(psi))
    a = np.array([[-0.7, 0.0], [0.7, 0.0], [x, H_SLIDE]])
    edges = []
    for i in range(3):
        p0, p1 = a[i], a[(i + 1) % 3]
        seg = np.linspace(0, 1, 101, endpoint=False)
        edges.append(p0 + (p1 - p0) * seg[:, None])
    bd = np.concatenate(edges)
    return bd - bd.mean(axis=0)


def plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask, flips) -> None:
    """Top (a): subsampled local-d curves; bottom (b): 3xN snapshot grid.

    Layout is computed in inches: each snapshot cell is a true physical
    square (fraction width s/17 vs height s/9.8), so the input row (fill)
    matches the jet rows (imshow) exactly, and the band between the two
    panels is sized to hold titles + panel tags without overlap.
    """
    n = len(cols)
    fig = plt.figure(figsize=(17.0, 9.8))
    # ---- geometry (inches) ----
    gx0, gy0 = 0.42, 0.30          # grid left edge / bottom margin
    gap_w, gap_h = 0.055, 0.085    # col gap; row gap = 1.5 x col gap
    s = (17.0 - gx0 - 0.12 - gap_w * (n - 1)) / n  # true square side
    cw, rh = s + gap_w, s + gap_h
    grid_top = gy0 + 3 * s + 2 * gap_h
    # (a) curve axes, upper band
    a_x0, a_x1 = 1.05, gx0 + n * cw - gap_w
    a_y0, a_y1 = 5.85, 9.25
    ax_a = fig.add_axes([a_x0 / 17.0, a_y0 / 9.8, (a_x1 - a_x0) / 17.0,
                         (a_y1 - a_y0) / 9.8])
    i_c = sparse_curve(xs_c, MORPH_HIDE, spikes=flips)
    i_h = sparse_curve(xs_h, MORPH_HIDE)
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
    # ---- bottom 3xN snapshot grid: true square cells (inches) ----
    axs = []
    for r in range(3):
        row = []
        for c in range(n):
            ax = fig.add_axes([(gx0 + c * cw) / 17.0,
                               (gy0 + (2 - r) * rh) / 9.8,
                               s / 17.0, s / 9.8])
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            row.append(ax)
        axs.append(row)
    axs = np.array(axs)
    zzs = [_raw_slide(ps) for ps in cols]
    gmax = max(np.abs(zz).max() for zz in zzs)
    for c_idx, ps in enumerate(cols):
        bound = osc_bound(ps)
        xy = zzs[c_idx] / gmax * 0.85
        zz = xy[:, 0] + 1j * xy[:, 1]
        ax0 = axs[0, c_idx]
        ax0.fill(zz.real, zz.imag, color="white", edgecolor="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\psi={ps:.1f}$", fontsize=FONTSIZE - 4,
                      pad=7)
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
    # dashed lines between each classical flip pair, spanning (b)
    for f in flips:
        i = min(range(n), key=lambda c: abs(cols[c] - (f - 0.1)))
        xg = (gx0 + i * cw + s + gap_w / 2) / 17.0
        fig.add_artist(plt.Line2D([xg, xg], [gy0 / 9.8,
                                  (grid_top + 0.26) / 9.8],
                                  transform=fig.transFigure, color="k",
                                  ls="--", lw=1.4, alpha=0.9))
    for r, label in enumerate(["input shape", "classical HBS", "HBSN"]):
        row_y = gy0 + (2 - r) * rh + s / 2
        fig.text((gx0 - 0.10) / 17.0, row_y / 9.8, label,
                 rotation=90, ha="right", va="center",
                 fontsize=FONTSIZE - 4)
    # (a)/(b) panel tags on fixed inch bands
    fig.text(0.5, (a_y1 + 0.24) / 9.8, "(a)", ha="center", va="bottom",
             fontsize=FONTSIZE + 1)
    fig.text(0.5, (grid_top + 0.34) / 9.8, "(b)", ha="center", va="bottom",
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
    parser.add_argument("--ckpt", default=None,
                        help="override BEST_CKPT for HBSN inference")
    parser.add_argument("--out", default=None,
                        help="output png path (default: preview[_ft].png)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    global PNG_PATH
    net = nets.build_net(args.ckpt or nets.BEST_CKPT)
    if args.out:
        PNG_PATH = args.out
    elif args.ckpt:
        PNG_PATH = os.path.join(OUT_DIR, "deformation_osc_preview_ft.png")
    z, mask = grid.get_ghbs_grid()
    ths = np.arange(TH_LO, TH_HI + STEP / 2, STEP)
    if args.recompute and os.path.exists(FIELD_CACHE):
        os.remove(FIELD_CACHE)
    if args.ckpt:
        # finetuned net: recompute HBSN fields (own cache), reuse classic
        cache_ft = FIELD_CACHE.replace(".npz", "_ft.npz")
        if os.path.exists(cache_ft) and not args.recompute:
            hf_list = list(np.load(cache_ft)["hf"].astype(complex))
        else:
            hf_list = []
            for ps in ths:
                img = shapes.shape_to_image(osc_bound(ps))[..., 0]
                hf_list.append(nets.field_to_complex(nets.infer(net, img)))
            np.savez(cache_ft, ths=ths,
                     hf=np.array(hf_list, dtype=np.complex64))
        cf_list, _ = load_or_compute_fields(ths, net)
    else:
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
