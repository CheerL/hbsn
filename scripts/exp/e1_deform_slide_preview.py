#!/usr/bin/env python3
"""Slide-family deformation preview (user review).

triangle_slide family (base fixed, apex slides at h=0.933): crossings
53.70/57.57/122.03/126.44, verified R_pi flips at 57.57 and 122.03
(0.05-deg full-range scan).  Same layout/conventions as e1_deform_wide:
adjacent-sample d curve (classical spikes vs HBSN continuous) + 3x11
snapshots on a uniform 8-deg grid, columns nudged off flips, HBSN
bump cores (auto-detected), and classic-failure points.  Pinned
hbs==1.0.1; HBSN = nets.BEST_CKPT; rendering |field|*mask jet 0-0.8.
Preview figure: figures/hbsn/candidates/deformation_slide_preview.png
"""

import argparse
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
H_SLIDE = 0.933  # apex height aligned to the semicircle flip height
TH_LO, TH_HI = 50.0, 130.0  # slide-family preview interval
STEP = 0.05  # deg between samples
CROSSINGS = [53.70, 57.57, 122.03, 126.44]  # Im I2 zeros (0.05-deg scan)
FLIPS = [57.57, 122.03]  # verified R_pi flips (+-0.12 post-hoc verify)
FONTSIZE = 13
DEG_PER_T = 180.0
FIELD_CACHE = "/tmp/h3scan/fields_slide.npz"
PNG_PATH = os.path.join(OUT_DIR, "deformation_slide_preview.png")

SNAP_OFFSETS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]  # auto-widen ladder


def classic_or_none(bound):
    """classic_pixel; RuntimeError (1.0.1 zipper bug) -> None."""
    try:
        return classic.classic_pixel(bound, retries=8)
    except RuntimeError:
        return None


def slide_bound(th):
    """Slide family: base fixed, apex (x, 0.933) slides, x = 2*(t-0.5)."""
    return shapes.triangle_slide(x=2.0 * (th / DEG_PER_T - 0.5), h=H_SLIDE)


def hbsn_bump_cores(ths, ds) -> list[tuple[float, float]]:
    """HBSN transition-bump cores from the smoothed d curve.

    0.55-deg moving average kills sample-level noise ripples; cores are
    clustered peaks above 3x the smoothed baseline (p25).
    """
    sm = np.convolve(ds, np.ones(11) / 11, mode="same")
    base = max(float(np.percentile(sm, 25)), 1e-6)
    cores = []
    for i in range(2, len(sm) - 2):
        is_peak = sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]
        if not (is_peak and sm[i] > 3.0 * base):
            continue
        if cores and ths[i] - cores[-1][0] <= 2.0:
            if sm[i] > cores[-1][1]:
                cores[-1] = (ths[i], sm[i])
        else:
            cores.append((ths[i], sm[i]))
    return [(t - 0.35, t + 0.35) for t, _ in cores]


def compute_fields(ths: np.ndarray, net):
    """Classic (None on failure) + HBSN complex fields per theta sample."""
    cf_list, hf_list = [], []
    for k, th in enumerate(ths):
        bound = slide_bound(th)
        cf = classic_or_none(bound)
        img = shapes.shape_to_image(bound)[..., 0]
        hf = nets.field_to_complex(nets.infer(net, img))
        cf_list.append(cf)
        hf_list.append(hf)
        if (k + 1) % 100 == 0:
            n_nan = sum(f is None for f in cf_list)
            print(
                f"fields {k + 1}/{len(ths)} theta={th:.2f} nan={n_nan}",
                flush=True,
            )
    return cf_list, hf_list


def snapshot_thetas(z, mask, bumps) -> list[float]:
    """Uniform 8-deg grid over the interval; nudge if needed.

    A column is nudged +-0.5 deg if it falls inside an HBSN transition
    bump core +-0.4 deg (auto-detected from the HBSN d curve), within
    0.1 deg of a classical flip, or if the classic field fails there.
    """
    ths_u = [50.0 + 8.0 * k for k in range(11)]
    cols = []
    for th in ths_u:
        for cand in [th, th - 0.5, th + 0.5, th - 1.0, th + 1.0]:
            ok = all(not (lo - 0.4 <= cand <= hi + 0.4) for lo, hi in bumps)
            ok = ok and all(abs(cand - f) > 0.1 for f in FLIPS)
            if ok:
                cf = classic_or_none(slide_bound(cand))
                ok = cf is not None
            if ok:
                cols.append(cand)
                break
        else:
            raise RuntimeError(f"no valid snapshot column near {th}")
    return cols


def plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask) -> None:
    """Top: adjacent-sample d curves; bottom: 3x8 snapshot grid."""
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.9], hspace=0.22)
    ax_a = fig.add_subplot(gs[0])
    gs_b = gs[1].subgridspec(3, 11, hspace=0.12, wspace=0.08)
    axs = np.array(
        [[fig.add_subplot(gs_b[r, c]) for c in range(11)] for r in range(3)]
    )
    ax_a.plot(
        xs_c,
        ds_c,
        color="r",
        lw=1.3,
        label="classical (spikes: 180$^\\circ$ flips)",
    )
    ax_a.plot(
        xs_h,
        ds_h,
        color="b",
        lw=1.5,
        label="HBSN (continuous)",
    )
    for f in FLIPS:
        ax_a.axvline(f, color="k", ls="--", lw=0.8)
        ax_a.text(
            f,
            0.92,
            rf"$\theta^*$={f:.2f}$^\circ$",
            transform=ax_a.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=FONTSIZE - 2,
            color="k",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    ax_a.set_xlim(TH_LO, TH_HI)
    ax_a.set_xticks([50, 60, 70, 80, 90, 100, 110, 120, 130])
    ax_a.set_xticklabels(
        [f"{v}°" for v in [50, 60, 70, 80, 90, 100, 110, 120, 130]]
    )
    ax_a.set_ylabel(
        r"local field change $d(B_\theta, B_{\theta+\Delta\theta})$",
        fontsize=FONTSIZE,
    )
    ax_a.tick_params(labelsize=FONTSIZE - 2)
    ax_a.legend(
        fontsize=FONTSIZE - 2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    ax_a.set_xlabel(
        r"apex offset $x=2(\theta/180^{\circ}-0.5)$ of $C=(x,\,0.933)$",
        fontsize=FONTSIZE,
    )
    # (b) snapshots: input / classical / HBSN rows over 8 columns
    for c_idx, th in enumerate(cols):
        bound = slide_bound(th)
        zz = bound[:, 0] + 1j * bound[:, 1]
        zz = zz - zz.mean()
        zz = zz / np.abs(zz).max() * 0.85
        ax0 = axs[0, c_idx]
        ax0.set_facecolor("black")
        ax0.fill(zz.real, zz.imag, color="white", edgecolor="white", lw=0)
        ax0.set_xlim(-1.5, 1.5)
        ax0.set_ylim(-1.5, 1.5)
        ax0.set_aspect("equal")
        ax0.set_title(rf"$\theta={th:.2f}^\circ$", fontsize=FONTSIZE - 3)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_visible(False)
        cx, cy = np.cos(np.pi * th / DEG_PER_T), np.sin(np.pi * th / DEG_PER_T)
        ax0.text(
            zz[202].real,
            zz[202].imag + 0.12,
            f"C({cx:.4f},{cy:.4f})",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 5,
            color="white",
            fontweight="bold",
            clip_on=False,
        )
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
    for r, label in enumerate(["input shape", "classical HBS", "HBSN"]):
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
    pos_l = axs[0, 0].get_position()
    pos_r = axs[0, 10].get_position()
    fig.text(
        (pos_a.x0 + pos_a.x1) / 2,
        0.955,
        "(a) Local field change between adjacent samples "
        r"($\Delta\theta$=0.05$^\circ$)",
        ha="center",
        fontsize=FONTSIZE + 1,
    )
    fig.text(
        (pos_l.x0 + pos_r.x1) / 2,
        0.955,
        "(b) Snapshots, uniform 8-degree sampling",
        ha="center",
        fontsize=FONTSIZE + 1,
    )
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_or_compute_fields(ths, net):
    """Field cache in /tmp (not synced); complex64 to halve memory."""
    if os.path.exists(FIELD_CACHE):
        d = np.load(FIELD_CACHE)
        if np.allclose(d["ths"], ths):
            cf = d["cf"].astype(complex)
            cf = [None if np.all(np.isnan(f)) else f for f in cf]
            return cf, list(d["hf"].astype(complex))
    cf_list, hf_list = compute_fields(ths, net)
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
    parser = argparse.ArgumentParser(
        description="H3 Step 2: adjacent-d metric figure"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="ignore field cache"
    )
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    net = nets.build_net(nets.BEST_CKPT)
    z, mask = grid.get_ghbs_grid()
    ths = np.arange(TH_LO, TH_HI + STEP / 2, STEP)
    if args.recompute and os.path.exists(FIELD_CACHE):
        os.remove(FIELD_CACHE)
    cf_list, hf_list = load_or_compute_fields(ths, net)

    def adjacent_d(fields):
        """Adjacent-finite-sample d; x = pair midpoint theta."""
        fin = [i for i in range(len(fields)) if fields[i] is not None]
        xs, ds = [], []
        for i, j in itertools.pairwise(fin):
            xs.append(0.5 * (ths[i] + ths[j]))
            ds.append(ops.d2(fields[i], fields[j], z, mask))
        return np.array(xs), np.array(ds)

    xs_c, ds_c = adjacent_d(cf_list)
    xs_h, ds_h = adjacent_d(hf_list)
    spike = ds_c > 0.3
    print(
        f"classical spikes (d>0.3): {int(spike.sum())} -> "
        + ", ".join(f"{x:.2f}" for x in xs_c[spike])
    )
    print(f"HBSN max step: {ds_h.max():.4f}")
    bumps = hbsn_bump_cores(ths, ds_h)
    print(
        "HBSN bump cores: "
        + ", ".join(f"{0.5 * (a + b):.2f}" for a, b in bumps)
    )
    cols = snapshot_thetas(z, mask, bumps)
    plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask)
    print(f"figure: {PNG_PATH}")
    print(f"snapshot thetas: {[f'{c:.2f}' for c in cols]}")


if __name__ == "__main__":
    main()
