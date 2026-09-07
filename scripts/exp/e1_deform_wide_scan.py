#!/usr/bin/env python3
"""E1-wide Step 1 - Im I2(theta) wide sweep: find >=3 zero-crossing interval.

Triangle family of e1_deform.py: A=(-0.7,0), B=(0.7,0), apex C=(cos t, sin t)
sweeps the upper semicircle, theta = 180*t (deg).  Scan Im I2(theta) over
theta in [30, 150] deg at 0.02 deg step; zero crossings by adjacent sign
change + linear interpolation (2 decimals); classic_pixel RuntimeError ->
NaN, excluded from crossing detection (same convention as e1_deform.py).

Key convention (measured 2026-09-07):
- MUST run on pinned hbs==1.0.1 (pyproject pin, installed in venv).
  scripts/exp hbsn_exp prefers the ~/projects/hbs_python fork (1.1.0
  lambda-normalization arg I2->0: continuous representative, no R_pi jumps)
  which does NOT reproduce the paper phenomenon (re-run e1_deform on
  2026-09-07: 0 flips, d-max ~ 0.04).  With pinned hbs==1.0.1 the narrow
  window shows Im I2 sign change at theta* ~ 111.10 deg, matching paper.tex
  sec 7.  We therefore point HBS_PYTHON_DIR to a non-existent path so that
  hbsn_exp falls back to the pinned hbs==1.0.1.  Rendering/field pipeline
  identical to e1_deform.py.
- The fork's lambda-normalization (2026-08-17 robustness session) conflicts
  with this experiment's convention - reported as conflict point (H3 sec 6).

Deliverable (H3 Step 1, intermediate, awaits user confirmation):
scan figure + zero-crossing list (2 decimals) + candidate intervals.
"""

import argparse
import itertools
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pinned hbs 1.0.1: must be set before importing hbsn_exp
os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hbsn_exp import classic, grid, ops, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "../../figures/hbsn/candidates")
BASE = 0.7
TH_LO, TH_HI = 30.0, 150.0  # avoid degenerate 0/180 deg (apex near base)
STEP = 0.02  # deg; handoff allows 0.02-0.05
FONTSIZE = 13
DEG_PER_T = 180.0  # theta(deg) = 180*t

NPZ_PATH = os.path.join(OUT_DIR, "deformation_wide_scan.npz")
PNG_PATH = os.path.join(OUT_DIR, "deformation_wide_scan.png")
GAP_MAX = 45.0  # candidate interval grouping: neighbor gap <= 45 deg
# (crossing structure found: pairs {64.65,69.30} and {111.13,115.31}
# ~4-5 deg apart, 42 deg between pairs -> GAP_MAX must bridge the pairs)
GAP_MAX_PROBE = 5.0  # bridge probing: only NaN runs <= 5 deg wide


def scan(ths: np.ndarray) -> np.ndarray:
    """Im I2 per theta via classic_pixel -> ops.i2; failure -> NaN."""
    z, mask = grid.get_ghbs_grid()
    ims = np.full(ths.shape, np.nan)
    for k, th in enumerate(ths):
        try:
            cf = classic.classic_pixel(
                shapes.triangle(t=th / DEG_PER_T, base=BASE), retries=8
            )
            ims[k] = ops.i2(cf, z, mask).imag
        except RuntimeError:
            pass
        if (k + 1) % 250 == 0:
            print(f"scan {k + 1}/{len(ths)} theta={th:.2f}", flush=True)
    return ims


def find_crossings(ths: np.ndarray, ims: np.ndarray) -> np.ndarray:
    """Crossings: adjacent finite sign change -> linear interp theta*."""
    valid = ~np.isnan(ims)
    sgn = np.sign(ims)
    idx = np.where(valid[:-1] & valid[1:] & (np.diff(sgn) != 0))[0]
    out = []
    for i in idx:
        a, b = ims[i], ims[i + 1]
        frac = abs(a) / (abs(a) + abs(b))
        out.append(ths[i] + (ths[i + 1] - ths[i]) * frac)
    return np.array(out)


def gap_edge_crossings(ths, ims) -> tuple[np.ndarray, np.ndarray]:
    """Crossings across remaining NaN runs: opposite-sign edges -> midpoint.

    Returns (crossings, uncertain_flags); midpoint theta* is approximate
    (the gap may still hide the exact location).
    """
    out, unc = [], []
    nan = np.isnan(ims)
    if not nan.any():
        return np.array([]), np.array([], dtype=bool)
    idx = np.where(nan)[0]
    runs, start = [], idx[0]
    for a, b in itertools.pairwise(idx):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    for s, e in runs:
        if s == 0 or e == len(ims) - 1:
            continue
        a, b = ims[s - 1], ims[e + 1]
        if np.isfinite(a) and np.isfinite(b) and np.sign(a) != np.sign(b):
            out.append(0.5 * (ths[s] + ths[e]))
            unc.append(True)
    return np.array(out), np.array(unc, dtype=bool)


def merge_crossings(crossings: np.ndarray) -> np.ndarray:
    """Merge jitter re-crossings: cluster (gap < 0.5 deg) keeps first."""
    if len(crossings) == 0:
        return crossings
    out = [crossings[0]]
    for c in crossings[1:]:
        if c - out[-1] > 0.5:
            out.append(c)
    return np.array(out)


def group_runs(crossings: np.ndarray) -> list[tuple[float, float, int]]:
    """Group crossings with neighbor gap <= GAP_MAX; keep groups >= 3."""
    if len(crossings) < 3:
        return []
    groups = [[crossings[0]]]
    for c in crossings[1:]:
        if c - groups[-1][-1] <= GAP_MAX:
            groups[-1].append(c)
        else:
            groups.append([c])
    return [(g[0], g[-1], len(g)) for g in groups if len(g) >= 3]


def bridge_gaps(ths: np.ndarray, ims: np.ndarray) -> np.ndarray:
    """Probe NaN runs that hide a sign flip (crossing detection is blind

    across NaN gaps).  For each NaN run with opposite finite signs at its
    edges and width <= GAP_MAX_PROBE deg, probe the interior (deterministic
    per-theta retries; a succeeded probe is a real classic_pixel value).
    """
    out = ims.copy()
    z, mask = grid.get_ghbs_grid()
    for _pass in range(2):
        nan = np.isnan(out)
        if not nan.any():
            break
        idx = np.where(nan)[0]
        # split into consecutive runs
        runs, start = [], idx[0]
        for a, b in itertools.pairwise(idx):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        for s, e in runs:
            if s == 0 or e == len(out) - 1:
                continue
            a, b = out[s - 1], out[e + 1]
            if np.sign(a) == 0 or np.sign(b) == 0:
                continue
            if np.sign(a) == np.sign(b):
                continue
            width = ths[e] - ths[s]
            if width > GAP_MAX_PROBE:
                continue
            for k in np.linspace(s, e, 9)[1:-1].round().astype(int):
                try:
                    cf = classic.classic_pixel(
                        shapes.triangle(t=ths[k] / DEG_PER_T, base=BASE),
                        retries=8,
                    )
                    out[k] = ops.i2(cf, z, mask).imag
                except RuntimeError:
                    pass
    return out


def plot_scan(ths, ims, crossings, runs, png_path: str) -> None:
    """Scan curve with crossings annotated + candidate intervals shaded."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ths, ims, lw=1.2, color="0.35")
    ax.axhline(0, color="k", ls="--", lw=0.8)

    for k, c in enumerate(crossings):
        ax.plot(c, 0, "ro", ms=5, zorder=3)
        off = 7 if k % 2 == 0 else -13
        va = "bottom" if k % 2 == 0 else "top"
        ax.annotate(
            f"{c:.2f}°",
            (c, 0),
            xytext=(0, off),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            color="r",
        )
    for k, (a, b, cnt) in enumerate(runs):
        ax.axvspan(a, b, color="tab:blue", alpha=0.08, zorder=0)
        ax.text(
            (a + b) / 2,
            0.94,
            f"candidate {k + 1} ({cnt} crossings)",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color="tab:blue",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    ax.set_xlabel(
        r"polar angle $\theta$ of apex $C=(\cos\theta,\sin\theta)$",
        fontsize=FONTSIZE,
    )
    ax.set_ylabel(r"$\operatorname{Im} I_2(B_\theta)$", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 2)
    ax.set_title(
        "Wide sweep of Im I2(theta) - classical HBS (pinned hbs 1.0.1)",
        fontsize=FONTSIZE,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H3 Step 1 wide Im I2 scan for zero crossings"
    )
    parser.add_argument(
        "--rescan", action="store_true", help="ignore npz cache, rescan"
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ths = np.arange(TH_LO, TH_HI + STEP / 2, STEP)
    if os.path.exists(NPZ_PATH) and not args.rescan:
        d = np.load(NPZ_PATH)
        ths, ims = d["theta"], d["im"]
        print(f"loaded cache: {NPZ_PATH} ({len(ths)} pts)")
    else:
        ims = scan(ths)
    ims = bridge_gaps(ths, ims)
    np.savez(NPZ_PATH, theta=ths, im=ims)

    c_adj = merge_crossings(find_crossings(ths, ims))
    c_gap, _ = gap_edge_crossings(ths, ims)
    crossings = np.sort(np.concatenate([c_adj, c_gap]))
    runs = group_runs(crossings)
    print(
        f"scan step {STEP} deg, finite {np.isfinite(ims).sum()}/{len(ims)},"
        f" crossings {len(crossings)}"
    )
    print("zero crossings (deg):")
    print("  " + ", ".join(f"{c:.2f}" for c in c_adj))
    if len(c_gap):
        print(
            "  gap-bridged (approx, inside NaN runs): "
            + ", ".join(f"{c:.2f}" for c in c_gap)
        )
    print(f"candidate intervals (>=3 crossings, gap<={GAP_MAX:g} deg):")
    if not runs:
        print("  none")
    for k, (a, b, cnt) in enumerate(runs):
        inside = [c for c in crossings if a <= c <= b]
        print(f"  candidate {k + 1}: [{a:.2f}, {b:.2f}] ({cnt} crossings)")
        inside_s = ", ".join(f"{c:.2f}" for c in inside)
        print(f"    crossings: {inside_s}")

    plot_scan(ths, ims, crossings, runs, PNG_PATH)
    print(f"figure: {PNG_PATH}")
    print(f"data: {NPZ_PATH}")


if __name__ == "__main__":
    main()
