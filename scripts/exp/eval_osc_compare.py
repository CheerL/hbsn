#!/usr/bin/env python3
"""Cross-candidate summary of the osc evaluation standard.

Reads runs/cache_osc/eval_standard_<tag>.npz (written by eval_osc_standard.py)
and prints one row per candidate:

  D_free    mean RMS_disk(|h|,|R_theta* cf|) -- similarity up to any rotation
  D_branch  the same but with rotation restricted to the allowed seam {0,180}
            (the honest number; D_free - D_branch is the gain from drift)
  lo / hi   D on the blue core (<0.25 max) / red rim (>0.55 max)
  drift     p95 angular distance of theta* to {0,180}
  plan      p95 deviation of theta*(psi) from base's plan, mod the seam branch
  flip      worst adjacent-d in the +-5 deg windows around the classical flips
  jumps     count of adjacent-d > 0.20 over the whole cycle

Usage: eval_osc_compare.py [tag ...]   (default: every cached candidate)
"""

import glob
import os
import sys

import numpy as np

OUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "runs",
        "cache_osc",
    )
)
FLIPS = [40.38, 139.62, 220.89, 319.11]


def _circ(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main() -> None:
    tags = sys.argv[1:] or [
        os.path.basename(p)[len("eval_standard_") : -len(".npz")]
        for p in sorted(glob.glob(os.path.join(OUT_DIR, "eval_standard_*.npz")))
    ]
    data = {
        t: np.load(os.path.join(OUT_DIR, f"eval_standard_{t}.npz"))
        for t in tags
    }
    ref = data["base"]["rows"] if "base" in data else None
    hdr = (
        f"{'tag':>10} | {'D_free':>7} {'D_branch':>8} {'lo':>6} {'hi':>6}"
        f" | {'drift95':>7} {'plan95':>6} | {'flip':>6} {'jumps':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tag, d in data.items():
        r = d["rows"]
        plan = float("nan")
        if ref is not None and len(ref) == len(r):
            dev = [
                min(_circ(a, b), 180 - _circ(a, b))
                for a, b in zip(r[:, 4], ref[:, 4], strict=True)
            ]
            plan = float(np.percentile(dev, 95))
        fj = d["d_adj_cand"]
        mid = 0.5 * (d["ths"][:-1] + d["ths"][1:])
        flips = [fj[np.abs(mid - f) <= 5.0].max() for f in FLIPS]
        print(
            f"{tag:>10} | {r[:, 2].mean():7.4f} {r[:, 11].mean():8.4f} "
            f"{r[:, 7].mean():6.4f} {r[:, 9].mean():6.4f} | "
            f"{np.percentile(r[:, 5], 95):7.1f} {plan:6.1f} | "
            f"{max(flips):6.4f} {int((fj > 0.20).sum()):5d}"
        )


if __name__ == "__main__":
    main()
