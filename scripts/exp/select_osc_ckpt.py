#!/usr/bin/env python3
"""Select the best osc-family checkpoint under the F+C standard.

Why this exists: the saved `best.pth` is chosen by `_quick_eval` (mean F
over the 11 display columns) at training time.  That is NOT the standard:
the standard averages 616 psi and adds the continuity gates.  Measured on
v17, the two disagree badly -- the saved ep2 has flip peak 1.16x base
while discarded ep3 sits at 0.91x, i.e. a PASSING checkpoint was thrown
away in favour of a failing one.

Usage:
    select_osc_ckpt.py --glob 'runs/finetuned/osc/v17/ep*.pth'
    select_osc_ckpt.py --ckpts a.pth b.pth c.pth
    select_osc_ckpt.py --root runs/finetuned/osc --tags v17 v19

Prints a table scored exactly like eval_osc_standard, then the winner.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_osc_standard as S
from finetune_osc import FLIPS, _images
from hbsn_exp import grid, nets


def score(ckpt, ths, cf, imgs, hf_base, w, z, mask, valid, stride=2):
    import torch

    net = nets.build_net(ckpt)
    hf = S._infer_all(net, imgs, "cuda" if torch.cuda.is_available() else "cpu")
    sel = list(np.where(valid)[0][::stride])
    amp_b = np.abs(hf_base).astype(np.float32)
    amp_c = np.abs(hf).astype(np.float32)
    fb, fc = [], []
    for i in sel:
        fb.append(S._best_theta(cf[i], amp_b[i], w, z, mask)[1])
        fc.append(S._best_theta(cf[i], amp_c[i], w, z, mask)[1])
    fb, fc = np.array(fb), np.array(fc)
    worse = int((fc > fb + S.TOL).sum())
    win = 1.0 - worse / len(sel)
    adj_b = S._adjacent_d(amp_b, w, 0, len(ths) - 1)
    adj_c = S._adjacent_d(amp_c, w, 0, len(ths) - 1)
    mid = 0.5 * (ths[:-1] + ths[1:])
    pk = [max(float(adj_c[np.abs(mid - f) <= S.FLIP_WIN].max()) for f in FLIPS)]
    pk_b = max(float(adj_b[np.abs(mid - f) <= S.FLIP_WIN].max()) for f in FLIPS)
    return {
        "ckpt": ckpt,
        "F": float(fc.mean()),
        "F_down": float(1 - fc.mean() / fb.mean()),
        "win": float(win),
        "peak": pk[0],
        "peak_ratio": pk[0] / pk_b,
        "jump": float(adj_c.max()),
        "gates": {
            "F improves": bool(fc.mean() <= fb.mean()),
            "F no per-psi regression": bool(win >= S.MIN_WIN),
            "C no discontinuity": bool(adj_c.max() <= S.JUMP),
            "C flip peak <= base": bool(pk[0] <= pk_b * S.PEAK_TOL),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="pick the F+C-best ckpt")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--ckpts", nargs="*", default=None)
    ap.add_argument("--root", default="runs/finetuned/osc")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    if args.glob:
        ckpts = sorted(glob.glob(args.glob))
    elif args.ckpts:
        ckpts = args.ckpts
    else:
        tags = args.tags or []
        ckpts = []
        for t in tags:
            ckpts += sorted(glob.glob(os.path.join(args.root, t, "ep*.pth")))
            ckpts += sorted(glob.glob(os.path.join(args.root, t, "best.pth")))
    if not ckpts:
        raise SystemExit("no checkpoints matched")

    d = np.load(S.FIELD_CACHE)
    ths, cf, hf_base = d["ths"], d["cf"], d["hf"].astype(complex)
    imgs = _images(ths)
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)
    valid = np.array([not np.all(np.isnan(c)) for c in cf])
    cf = np.where(valid[:, None, None], cf, 0).astype(complex)

    rows = [
        score(c, ths, cf, imgs, hf_base, w, z, mask, valid, args.stride)
        for c in ckpts
    ]
    print(
        f"{'checkpoint':<34} {'F':>8} {'down':>7} {'win':>6} "
        f"{'peak':>7} {'xbase':>6}  gates"
    )
    for r in rows:
        g = "".join("Y" if v else "." for v in r["gates"].values())
        print(
            f"{os.path.basename(os.path.dirname(r['ckpt'])) + '/' + os.path.basename(r['ckpt']):<34} "
            f"{r['F']:8.4f} {r['F_down'] * 100:6.1f}% {r['win'] * 100:5.1f}% "
            f"{r['peak']:7.4f} {r['peak_ratio']:6.2f}  {g}"
        )
    ok = [r for r in rows if all(r["gates"].values())]
    print(f"\ngate order: {' '.join(next(iter(rows))['gates'])}")
    if not ok:
        # nothing passes: rank by how many gates pass, then F
        rows.sort(key=lambda r: (-sum(r["gates"].values()), r["F"]))
        r = rows[0]
        print(f"NO checkpoint passes.  Fewest-failing: {r['ckpt']}")
        print(f"  fails: {[k for k, v in r['gates'].items() if not v]}")
    else:
        ok.sort(key=lambda r: r["F"])
        print(f"BEST (passes all gates, lowest F): {ok[0]['ckpt']}")
        print(
            f"  F {ok[0]['F']:.4f} ({ok[0]['F_down'] * 100:.1f}% down), "
            f"peak {ok[0]['peak_ratio']:.2f}x base"
        )


if __name__ == "__main__":
    main()
