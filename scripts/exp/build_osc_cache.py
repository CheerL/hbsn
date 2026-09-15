#!/usr/bin/env python3
"""Build the field + image cache for one osc family.

Classical HBS and the pretrained HBSN over the whole 0..360 deg cycle at
0.25 deg, plus the rendered input images.  Everything downstream (the
training target, the audits, the figures) reads this, so the family id is
part of the filename: a cache built for one family must never be silently
reused for another.

The classical solver fails on a narrow band of apex offsets that sits
right against the +0.358 flip line (measured: NaN for |x| in about
[0.342, 0.354], valid again from 0.356).  That band is intrinsic to the
solver's zipper routine, not to the family, so `retries` is raised and the
resulting NaN runs are reported rather than hidden -- the flip inside such
a run is only recoverable from the pair STRADDLING it, which is what
classical_arcs.links() is for.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HBS_PYTHON_DIR"] = "/nonexistent"

import osc_family as F
from hbsn_exp import classic, nets, shapes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(REPO, "runs/cache_osc")
STEP = 0.25
RETRIES = 8


def classic_or_none(bound):
    for _ in range(RETRIES):
        try:
            return classic.classic_pixel(bound)
        except RuntimeError:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="build the osc family cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ths = np.arange(0.0, 360.0 + STEP / 2, STEP)
    fpath = os.path.join(CACHE, f"fields_osc_{F.FAMILY}.npz")
    ipath = os.path.join(CACHE, f"imgs_osc_{F.FAMILY}.npz")
    if os.path.exists(fpath) and not args.force:
        print(f"exists (use --force to rebuild): {fpath}")
        return

    net = nets.build_net(nets.BEST_CKPT)

    F.selfcheck()
    n = len(ths)
    cf = np.full((n, 128, 128), np.nan + 0j, dtype=np.complex64)
    hf = np.zeros((n, 128, 128), dtype=np.complex64)
    imgs = np.zeros((n, 256, 256), dtype=np.uint8)
    t0, nnan = time.time(), 0
    for k, ps in enumerate(ths):
        b = F.bound(ps)
        c = classic_or_none(b)
        if c is None:
            nnan += 1
        else:
            cf[k] = c
        img = shapes.shape_to_image(b)[..., 0]
        imgs[k] = img
        hf[k] = nets.field_to_complex(nets.infer(net, img))
        if (k + 1) % 100 == 0 or k + 1 == n:
            el = time.time() - t0
            eta = el / (k + 1) * (n - k - 1)
            print(
                f"  {k + 1}/{n} psi={ps:.2f} nan={nnan} "
                f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m",
                flush=True,
            )

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(fpath, ths=ths, cf=cf, hf=hf, family=F.FAMILY)
    np.savez_compressed(ipath, ths=ths, imgs=imgs, family=F.FAMILY)
    print(f"saved: {fpath}\nsaved: {ipath}", flush=True)
    # per-psi validity: cf is (n,128,128), so reduce over the image axes
    ok = ~np.all(np.isnan(cf.real), axis=(1, 2))
    x = F.x_of(ths)
    print(f"classical NaN {int((~ok).sum())}/{n}")
    runs, s_ = [], None
    for i in range(n):
        if not ok[i] and s_ is None:
            s_ = i
        if ok[i] and s_ is not None:
            runs.append((s_, i - 1))
            s_ = None
    if s_ is not None:
        runs.append((s_, n - 1))
    for a, b in runs:
        print(
            f"  NaN run psi {ths[a]:.2f}..{ths[b]:.2f} (x {x[a]:+.4f}..{x[b]:+.4f})"
        )
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
