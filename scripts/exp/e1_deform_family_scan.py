#!/usr/bin/env python3
"""H3 follow-up: hunt for flip-dense triangle-family variants.

User asked for intervals with MORE frequent flips ("back-and-forth" very
visible).  The semicircle-sweep family (base=0.7) has only 2 real flips
in [30,150] deg (69.30, 111.13; measured 2026-09-07/08 on pinned
hbs==1.0.1), so this script scans VARIANT families for flip density:
- triangle(base=b), b in {0.5, 0.6, 0.8, 0.9}
- triangle_slide(x, h=0.933) (apex sliding horizontally; h aligned to
  the semicircle family's flip-height apex)

Per variant: 0.1-deg coarse scan of Im I2 over [30,150] -> NaN-gap
bridging probes -> crossings (sign change, linear interp, 0.5-deg
jitter-cluster merge) -> flip verification per crossing (is_flip at
+-0.15 deg, widened on failure).  Output per variant: crossing list
(flip-verified vs graze), scan figure under figures/hbsn/candidates/.
Pinned hbs==1.0.1 via HBS_PYTHON_DIR (fork 1.1.0 does not reproduce the
paper phenomenon).  Rendering/field pipeline identical to e1_deform.
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

from e1_deform_wide_scan import find_crossings, merge_crossings
from hbsn_exp import classic, grid, ops, shapes

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../figures/hbsn/candidates",
)
TH_LO, TH_HI = 30.0, 150.0
STEP = 0.1  # deg, coarse scan step
DEG_PER_T = 180.0
H_SLIDE = 0.933  # slide-family apex height (flip-height alignment)

VARIANTS = [
    ("base05", lambda t: shapes.triangle(t=t, base=0.5)),
    ("base06", lambda t: shapes.triangle(t=t, base=0.6)),
    ("base08", lambda t: shapes.triangle(t=t, base=0.8)),
    ("base09", lambda t: shapes.triangle(t=t, base=0.9)),
    (
        "slide",
        lambda t: shapes.triangle_slide(x=2.0 * (t - 0.5), h=H_SLIDE),
    ),
]


def variant_scan(bound_fn, ths: np.ndarray) -> np.ndarray:
    """Im I2 per theta for a variant family; classic failure -> NaN."""
    z, mask = grid.get_ghbs_grid()
    ims = np.full(ths.shape, np.nan)
    for k, th in enumerate(ths):
        try:
            cf = classic.classic_pixel(bound_fn(th / DEG_PER_T), retries=8)
            ims[k] = ops.i2(cf, z, mask).imag
        except RuntimeError:
            pass
        if (k + 1) % 200 == 0:
            print(
                f"  scan {k + 1}/{len(ths)} theta={th:.1f} nan="
                f"{int(np.isnan(ims[: k + 1]).sum())}",
                flush=True,
            )
    return ims


def bridge_small_gaps(bound_fn, ths, ims, gap_max: float = 0.5):
    """Probe narrow NaN runs (<= gap_max deg) with sign-flipping edges."""
    out = ims.copy()
    z, mask = grid.get_ghbs_grid()
    for _ in range(2):
        nan = np.isnan(out)
        if not nan.any():
            break
        idx = np.where(nan)[0]
        runs, start = [], idx[0]
        for a, b in itertools.pairwise(idx):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        for s, e in runs:
            if s == 0 or e == len(out) - 1:
                continue
            if ths[e] - ths[s] > gap_max:
                continue
            if np.sign(out[s - 1]) == np.sign(out[e + 1]):
                continue
            for k in np.linspace(s, e, 9)[1:-1].round().astype(int):
                try:
                    cf = classic.classic_pixel(
                        bound_fn(ths[k] / DEG_PER_T), retries=8
                    )
                    out[k] = ops.i2(cf, z, mask).imag
                except RuntimeError:
                    pass
    return out


def verify_flips(bound_fn, ths, crossings, z, mask) -> list[dict]:
    """Per crossing: is_flip across +-0.15 deg (widen on classic failure)."""
    out = []
    for c in crossings:
        lo, hi = None, None
        for off in [0.15, 0.3, 0.5, 0.8, 1.2]:
            if lo is None:
                lo = try_classic(bound_fn, c - off)
            if hi is None:
                hi = try_classic(bound_fn, c + off)
            if lo is not None and hi is not None:
                break
        if lo is None or hi is None:
            out.append({"th": c, "flip": None, "d2": np.nan})
            continue
        d2v = ops.d2(lo, hi, z, mask)
        out.append(
            {
                "th": c,
                "flip": bool(ops.is_flip(lo, hi, z, mask)),
                "d2": d2v,
            }
        )
    return out


def try_classic(bound_fn, th):
    """classic_pixel or None (1.0.1 zipper failures)."""
    try:
        return classic.classic_pixel(bound_fn(th / DEG_PER_T), retries=8)
    except RuntimeError:
        return None


def plot_variant(tag, ths, ims, cros, flips, path) -> None:
    """Im I2 curve with flip-verified (red) vs graze (orange) crossings."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(ths, ims, lw=1.1, color="0.35")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    for c in cros:
        rec = next((r for r in flips if np.isclose(r["th"], c)), None)
        is_f = rec is not None and rec["flip"]
        ax.plot(c, 0, "o", ms=6, color="r" if is_f else "orange", zorder=3)
    ax.set_xlabel(r"polar angle $\theta$ (deg)", fontsize=12)
    ax.set_ylabel(r"$\operatorname{Im} I_2$", fontsize=12)
    ax.set_title(
        f"variant {tag}: {len(cros)} crossings, "
        f"{sum(1 for r in flips if r['flip'])} verified flips"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    z, mask = grid.get_ghbs_grid()
    ths = np.arange(TH_LO, TH_HI + STEP / 2, STEP)
    for tag, bound_fn in VARIANTS:
        print(f"=== variant {tag} ===", flush=True)
        ims = variant_scan(bound_fn, ths)
        ims = bridge_small_gaps(bound_fn, ths, ims)
        cros = merge_crossings(find_crossings(ths, ims))
        flips = verify_flips(bound_fn, ths, cros, z, mask)
        n_flip = sum(1 for r in flips if r["flip"])
        print(f"crossings: {[f'{c:.2f}' for c in cros]}")
        for r in flips:
            print(f"  {r['th']:.2f}: flip={r['flip']} d2={r['d2']:.3f}")
        print(f"=> {len(cros)} crossings, {n_flip} verified flips")
        plot_variant(
            tag,
            ths,
            ims,
            cros,
            flips,
            os.path.join(OUT_DIR, f"family_scan_{tag}.png"),
        )
    print("done")


if __name__ == "__main__":
    main()
