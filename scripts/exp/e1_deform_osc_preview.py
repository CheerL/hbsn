#!/usr/bin/env python3
"""Oscillating-slide preview: apex x = X0 + AMP sin(psi) -> 4 flips/cycle.

The apex slides horizontally at constant chirality.  With X0 = 0 the apex
would cross the base midpoint twice per cycle, making the triangle
isoceles (mirror-symmetric) at those two phases and flipping the family's
chirality; the offset X0 > 0 removes that, so every member is scalene and
the whole cycle is symmetry-free (see osc_family).

The classical representative flips R_pi at the four phases where the apex
crosses the flip line, measured at |x| = 0.35565 on the + side (NOT 0.358;
see the note on FLIP_PAIRS).  The finetuned net goes through the same
180 deg reorientation with no local increase at all -- the run prints the
measured steps; on gtarc_x100a500_v1 the HBSN max step is 0.0132 against a
classical peak of 0.586.  (That peak used to be ~0.088 with a ~1 deg morph
window around each flip, back when the fit target was the raw,
discontinuous classical field.  The target is now built continuous by
scripts/exp/build_gt_arcs.py, so the window is gone and what is plotted is
the plain "does it hold still" measurement.)

Panel (a) plots the local jump between adjacent psi: red = HBSN, blue =
classical, on a LINEAR 0-0.68 axis.  The linear axis is deliberate -- a log
axis shows both curves clearly but squashes the ~44x gap into ~1.6
decades, which visually argues the opposite of the result.  Dashed
verticals mark the four flips.

Panel (b) is a 3xN snapshot grid: input shape / classical HBS / HBSN.
Columns are context anchors clear of this family's six classical dead
zones, plus the two measured grid samples bracketing each flip: the
classical row flips 180 deg between those two columns, the HBSN row does
not.

Pinned hbs==1.0.1 (HBS_PYTHON_DIR); rendering |field|*mask, jet, 0-0.8.
HBSN comes from --ckpt.  Output: --out, else deformation_osc_preview[_ft].png.
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

import classical_arcs as CA
import osc_family as OF
from hbsn_exp import classic, grid, nets, ops, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
H_SLIDE, AMP = OF.H_SLIDE, OF.AMP
TH_LO, TH_HI = 0.0, 360.0
STEP = 0.25  # deg; spikes are instant, 0.25 resolves them
# The four flips are NOT at a fixed |x|: the classical half-plane selection
# switches where Im I2 changes sign, measured at x* = 0.35565 on the +side,
# which in THIS family lands between psi 30.75/31.00, 149.24/149.25,
# 246.75/247.00 and 292.99/293.00... i.e. the flip is extremely sharp in x
# and a column pair picked as (flip +- 0.13 deg) can easily land BOTH on the
# same side (that is exactly what happened at 149).  So the pairs below are
# the measured grid samples that straddle each flip, not offsets from it.
FLIP_PAIRS = [
    (30.75, 31.00),
    (149.00, 149.25),
    (246.75, 247.00),
    (293.00, 293.25),
]
FLIPS = [0.5 * (a + b) for a, b in FLIP_PAIRS]


def _cols() -> list[float]:
    """Context columns kept finite for classic HBS + tight flip pairs.

    Every entry is checked against this family's six classical dead zones
    (28.75-30.50, 46.50-51.00, 129.00-133.50, 149.50-151.25, 233.75-234.75,
    305.25-306.25); the context anchors sit clear of all of them, and each
    flip is bracketed by the two grid samples that straddle it.
    """
    ctx = [0.0, 15.0, 90.0, 180.0, 215.0, 330.0, 350.0]
    return sorted(ctx + [v for pair in FLIP_PAIRS for v in pair])


COLS = _cols()
FONTSIZE = 13
FIELD_CACHE = (
    os.path.join(HERE, "../../runs/cache_osc", f"fields_osc_{OF.FAMILY}.npz")
)
PNG_PATH = os.path.join(OUT_DIR, "deformation_osc_preview.png")
SUBSTEP = 5.0  # deg, display-subsampling step for the curve panel
# Empty: the four windows this used to hide were the OLD model's morph
# peaks, defined in the OLD family's psi coordinates.  Both changed, so
# the windows are meaningless here and every curve point is now drawn.
MORPH_HIDE = []
MORPH_RADIUS = 0.75
A_YLIM_LO, A_YLIM_HI = 0.0, 0.68  # panel (a) linear y-range (see plot_figure)


def classic_or_none(bound):
    """classic_pixel; RuntimeError (1.0.1 zipper bug) -> None."""
    try:
        return classic.classic_pixel(bound, retries=8)
    except RuntimeError:
        return None


def osc_bound(psi):
    """Apex slides at constant chirality: x = X0 + AMP sin(psi)."""
    return OF.bound(psi)


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
            print(
                f"scan {k + 1}/{len(ths)} psi={ps:.1f} nan={n_nan}", flush=True
            )
    return cf_list, hf_list, ims


def find_crossings(ths, ims):
    """Sign change of Im I2 over any two computable phases; linear interp.

    Gap-straddling pairs count, and they are not an edge case: the
    classical solver's failure band sits right against the flip
    line (measured: NaN for apex offsets in roughly [0.342, 0.354], valid
    again from 0.356).  At 0.25 deg the two samples bracketing a flip can
    therefore land one inside the band and one already past the crossing,
    so an adjacent-VALID-pair-only scan misses the flip completely -- it
    is visible only in the pair spanning the gap.  classical_arcs.links
    returns both kinds.
    """
    valid = ~np.isnan(ims)
    out = []
    for i, j in CA.links(valid):
        if np.sign(ims[i]) != np.sign(ims[j]):
            a, b = abs(ims[i]), abs(ims[j])
            out.append(ths[i] + (ths[j] - ths[i]) * a / (a + b))
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
    (triangle_slide's per-shape rescale magnifies the near-isosceles ~14%)."""
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
    gx0, gy0 = 0.42, 0.75  # grid left edge / bottom margin; gy0 sets the
    # gap to panel (a) -- 0.75 puts (b) back to about half the clearance
    # it had at 0.10 (the empty band below is cropped by bbox_inches)
    gap_w, gap_h = 0.055, 0.085  # col gap; row gap = 1.5 x col gap
    s = (17.0 - gx0 - 0.12 - gap_w * (n - 1)) / n  # true square side
    cw, rh = s + gap_w, s + gap_h
    grid_top = gy0 + 3 * s + 2 * gap_h
    # (a) curve axes, upper band
    a_x0, a_x1 = 1.05, gx0 + n * cw - gap_w
    a_y0, a_y1 = 5.85, 9.25
    ax_a = fig.add_axes(
        [a_x0 / 17.0, a_y0 / 9.8, (a_x1 - a_x0) / 17.0, (a_y1 - a_y0) / 9.8]
    )
    i_c = sparse_curve(xs_c, MORPH_HIDE, spikes=flips)
    i_h = sparse_curve(xs_h, MORPH_HIDE)
    ax_a.plot(
        xs_c[i_c],
        ds_c[i_c],
        "o-",
        color="b",
        lw=1.1,
        ms=3.2,
        label="classical",
    )
    ax_a.plot(
        xs_h[i_h],
        ds_h[i_h],
        "o-",
        color="r",
        lw=1.6,
        ms=3,
        label="HBSN",
    )
    for f in flips:
        ax_a.axvline(f, color="k", ls="--", lw=0.8, alpha=0.55)
    ax_a.set_xlim(TH_LO, TH_HI)
    # Linear y, 0-based.  Log was tried to lift the HBSN curve off the
    # spine; it does, but it also squashes the 32x classical/HBSN gap into
    # ~1.5 decades, which visually argues the opposite of the result.  On
    # the linear axis the message is immediate: classical (blue) spikes to
    # 0.586 at the four seams, HBSN (red) stays flat near 0.
    ax_a.set_ylim(A_YLIM_LO, A_YLIM_HI)
    ax_a.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax_a.set_xticklabels([f"{v}" for v in [0, 60, 120, 180, 240, 300, 360]])
    ax_a.set_ylabel(
        r"local field change $d(B_{\psi}, B_{\psi+\Delta\psi})$",
        fontsize=FONTSIZE,
    )
    ax_a.tick_params(labelsize=FONTSIZE - 2)
    ax_a.legend(fontsize=FONTSIZE - 2, loc="upper right")
    ax_a.set_xlabel(
        rf"oscillation phase $\psi$ (deg), apex "
        rf"$x = {OF.X0:.2f} + {OF.AMP:.2f}\sin\psi$",
        fontsize=FONTSIZE,
    )
    for f in flips:
        ax_a.text(
            f,
            1.02,
            rf"{f:.1f}$^\circ$",
            transform=ax_a.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 4,
            color="0.25",
        )
    # ---- bottom 3xN snapshot grid: true square cells (inches) ----
    axs = []
    for r in range(3):
        row = []
        for c in range(n):
            ax = fig.add_axes(
                [
                    (gx0 + c * cw) / 17.0,
                    (gy0 + (2 - r) * rh) / 9.8,
                    s / 17.0,
                    s / 9.8,
                ]
            )
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
        ax0.set_title(rf"$\psi={ps:.1f}$", fontsize=FONTSIZE - 4, pad=7)
        cf = classic_or_none(bound)
        img = shapes.shape_to_image(bound)[..., 0]
        hf = nets.field_to_complex(nets.infer(net, img))
        axs[1, c_idx].imshow(
            np.abs(cf) * mask if cf is not None else np.zeros_like(mask),
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        axs[1, c_idx].axis("off")
        axs[2, c_idx].imshow(
            np.abs(hf) * mask,
            cmap="jet",
            extent=[-1.5, 1.5, -1.5, 1.5],
            vmin=0,
            vmax=0.8,
        )
        axs[2, c_idx].axis("off")
    # dashed lines between each classical flip pair, spanning (b)
    for f in flips:
        i = min(range(n), key=lambda c: abs(cols[c] - (f - 0.1)))
        xg = (gx0 + i * cw + s + gap_w / 2) / 17.0
        fig.add_artist(
            plt.Line2D(
                [xg, xg],
                [gy0 / 9.8, (grid_top + 0.26) / 9.8],
                transform=fig.transFigure,
                color="k",
                ls="--",
                lw=1.4,
                alpha=0.9,
            )
        )
    for r, label in enumerate(["input shape", "classical HBS", "HBSN"]):
        row_y = gy0 + (2 - r) * rh + s / 2
        fig.text(
            (gx0 - 0.10) / 17.0,
            row_y / 9.8,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=FONTSIZE - 4,
        )
    # (a)/(b) panel tags on fixed inch bands
    fig.text(
        0.5,
        (a_y1 + 0.24) / 9.8,
        "(a)",
        ha="center",
        va="bottom",
        fontsize=FONTSIZE + 1,
    )
    fig.text(
        0.5,
        (grid_top + 0.34) / 9.8,
        "(b)",
        ha="center",
        va="bottom",
        fontsize=FONTSIZE + 1,
    )
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
        [
            f if f is not None else np.full((128, 128), np.nan + 0j)
            for f in cf_list
        ],
        dtype=np.complex64,
    )
    hf_arr = np.array(hf_list, dtype=np.complex64)
    np.savez(FIELD_CACHE, ths=ths, cf=cf_arr, hf=hf_arr)
    cf_list = [
        None if np.all(np.isnan(f)) else f.astype(complex) for f in cf_arr
    ]
    return cf_list, list(hf_arr.astype(complex))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="osc-slide preview figure")
    parser.add_argument(
        "--recompute", action="store_true", help="ignore field cache"
    )
    parser.add_argument(
        "--ckpt", default=None, help="override BEST_CKPT for HBSN inference"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output png path (default: preview[_ft].png)",
    )
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
        # finetuned net: recompute HBSN fields (own cache), reuse classic.
        # Key the cache by the checkpoint directory: a fixed `_ft.npz` is
        # silently reused for a DIFFERENT --ckpt, which plots one model's
        # fields under another model's name.
        tag = os.path.basename(os.path.dirname(args.ckpt))
        cache_ft = FIELD_CACHE.replace(".npz", f"_ft_{tag}.npz")
        if os.path.exists(cache_ft) and not args.recompute:
            hf_list = list(np.load(cache_ft)["hf"].astype(complex))
        else:
            hf_list = []
            for ps in ths:
                img = shapes.shape_to_image(osc_bound(ps))[..., 0]
                hf_list.append(nets.field_to_complex(nets.infer(net, img)))
            np.savez(
                cache_ft, ths=ths, hf=np.array(hf_list, dtype=np.complex64)
            )
        cf_list, _ = load_or_compute_fields(ths, net)
    else:
        cf_list, hf_list = load_or_compute_fields(ths, net)
    ims = np.array(
        [np.nan if f is None else ops.i2(f, z, mask).imag for f in cf_list]
    )
    cros = find_crossings(ths, ims)
    flips = [float(c) for c in cros if verify_flip(c, z, mask)]
    print("crossings:", [f"{c:.2f}" for c in cros])
    print("verified flips:", [f"{c:.2f}" for c in flips])
    xs_c, ds_c = curve_data(ths, cf_list, z, mask)
    xs_h, ds_h = curve_data(ths, hf_list, z, mask)
    spike = ds_c > 0.3
    print(
        "classical spikes (d>0.3): "
        + ", ".join(f"{x:.2f}" for x in xs_c[spike])
    )
    print(f"HBSN max step: {ds_h.max():.4f}")
    print(
        f"curve range: classical {ds_c.min():.3g}..{ds_c.max():.3g}"
        f"   HBSN {ds_h.min():.3g}..{ds_h.max():.3g}"
    )
    cols = COLS
    plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask, flips)
    print(f"figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
