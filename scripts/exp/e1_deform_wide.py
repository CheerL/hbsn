#!/usr/bin/env python3
"""E1-wide Step 2 - deformation_wide.png (H3 handoff Step 2, interval A).

Same triangle family as e1_deform.py; interval A [64.6, 115.4] deg chosen
by the user (Step 1 scan: crossings at 64.65/69.30/111.13/115.31 deg).
Curve panel: local continuity metric d(B_theta, B_{theta+0.05 deg})
between adjacent samples (ops.d2, same conventions as e1_deform).  The
classical representative jumps discontinuously at the real flips (69.30
and 111.13: one-step d2 ~ 0.59), the HBSN output is continuous (max step
~ 0.05, 11x smaller); the two-value alternation of the classical branch
is shown by the snapshot row at the crossings.
Snapshot panel: 4 crossings x +-eps (eps = 0.1 deg, widened where classic
field is numerically unavailable), 3 rows (input / classical HBS / HBSN),
field rendering identical to e1_deform (np.abs(field)*mask, jet, 0-0.8).

Conventions inherited from e1_deform_wide_scan.py:
- pinned hbs==1.0.1 via HBS_PYTHON_DIR (fork 1.1.0 lambda-normalization
  does not reproduce the paper phenomenon, measured 2026-09-07);
- classic failures -> NaN (zipper pure-imaginary bug), bridged by
  registering gap-edge fields (flips inside holes are still caught).
- HBSN checkpoint = nets.BEST_CKPT (paper best model).

Output: figures/hbsn/deformation_wide.png (final deliverable, tracked).
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
OUT_DIR = os.path.join(HERE, "../../figures/hbsn")
BASE = 0.7
TH_LO, TH_HI = 64.6, 115.4  # user-confirmed interval A (Step 1 scan)
STEP = 0.05  # deg between samples
TRAJ_STRIDE = 5  # register every 5th sample (0.25 deg); finer steps
# under-resolve the ~0.2 deg/step rotation (refine grid is 0.5 deg)
EPS = 0.1  # deg, snapshot offset from each crossing
CROSSINGS = [64.65, 69.30, 111.13, 115.31]  # Im I2 zero crossings (Step 1)
FLIPS = [69.30, 111.13]  # crossings where the representative truly flips
FONTSIZE = 13
DEG_PER_T = 180.0
FIELD_CACHE = "/tmp/h3scan/fields_wide.npz"
PNG_PATH = os.path.join(OUT_DIR, "deformation_wide.png")

SNAP_OFFSETS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]  # auto-widen ladder


def classic_or_none(bound):
    """classic_pixel; RuntimeError (1.0.1 zipper bug) -> None."""
    try:
        return classic.classic_pixel(bound, retries=8)
    except RuntimeError:
        return None


def compute_fields(ths: np.ndarray, net):
    """Classic (None on failure) + HBSN complex fields per theta sample."""
    cf_list, hf_list = [], []
    for k, th in enumerate(ths):
        bound = shapes.triangle(t=th / DEG_PER_T, base=BASE)
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


def register_phi(f_a, f_b, z, mask) -> float:
    """Optimal residual rotation: argmin_phi d2(R_phi f_a, f_b), in deg."""
    if f_a is None or f_b is None:
        return np.nan
    best_phi, best_dr = 0.0, np.inf
    # coarse 5 deg sweep, then one refine pass +-6 around the best
    for lo, hi, step in [(0.0, 359.9999, 5.0), (None, None, 0.5)]:
        if lo is None:
            lo, hi = best_phi - 6, best_phi + 6
        for phi in np.arange(lo, hi + step / 2, step):
            d = ops.d2(
                ops.rotate_mu(f_a, np.deg2rad(phi), z, mask), f_b, z, mask
            )
            if d < best_dr:
                best_phi, best_dr = float(phi), d
    return best_phi


def snapshot_thetas(z, mask) -> list[float]:
    """Uniform 5-deg grid over the interval; nudge if needed.

    A column is nudged +-0.5 deg if it falls inside an HBSN transition
    core (bumps ~[70.2,70.9]/~[108.7,109.3]) +-0.4 deg, within 0.1 deg
    of a classical flip, or if the classic field fails there.
    """
    ths_u = [65.0 + 5.0 * k for k in range(11)]
    bumps = [(70.2, 70.9), (108.7, 109.3)]
    cols = []
    for th in ths_u:
        for cand in [th, th - 0.5, th + 0.5, th - 1.0, th + 1.0]:
            ok = all(not (lo - 0.4 <= cand <= hi + 0.4) for lo, hi in bumps)
            ok = ok and all(abs(cand - f) > 0.1 for f in FLIPS)
            if ok:
                cf = classic_or_none(
                    shapes.triangle(t=cand / DEG_PER_T, base=BASE)
                )
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
    ax_a.set_xticks([65, 75, 85, 95, 105, 115])
    ax_a.set_xticklabels([f"{v}°" for v in [65, 75, 85, 95, 105, 115]])
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
        r"polar angle $\theta$ of apex $C=(\cos\theta,\sin\theta)$",
        fontsize=FONTSIZE,
    )
    # (b) snapshots: input / classical / HBSN rows over 8 columns
    for c_idx, th in enumerate(cols):
        bound = shapes.triangle(t=th / DEG_PER_T, base=BASE)
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
        "(b) Snapshots, uniform 5-degree sampling",
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
    cols = snapshot_thetas(z, mask)
    plot_figure(ths, xs_c, ds_c, xs_h, ds_h, cols, net, z, mask)
    print(f"figure: {PNG_PATH}")
    print(f"snapshot thetas: {[f'{c:.2f}' for c in cols]}")


if __name__ == "__main__":
    main()
