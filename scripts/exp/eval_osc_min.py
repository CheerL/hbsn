#!/usr/bin/env python3
"""Minimal osc audit: similarity (F) + continuity (C). Nothing else.

The previous standard carried a frame gate (O) on top of these two.  O
existed because the old target was built by re-rotating the classical
field per psi (--align full), which made the FIT rotation-invariant and
so let a 180 deg mis-framed candidate pass both F and C.  The current
target is one fixed continuous field and the loss is a direct pixel MSE
against it, so the frame is pinned by the training objective itself and O
has nothing left to guard.  It is dropped here by request.

F -- similarity to the target, rotation removed.
    F(psi) = wRMS_w(|model(psi)|, |R_theta gt(psi)|) at the best theta.
    Removing the single best rotation leaves a pure SHAPE distance, so a
    model is neither penalised nor rewarded for its orientation.  Reported
    against the same radial weight the trainer uses.

C -- continuity along psi.
    d(psi) = wRMS_w(|model(psi)|, |model(psi+dpsi)|), NO rotation removed:
    continuity is a property of the raw sequence, and removing a rotation
    would hide exactly the drift we are looking for.

psi the target excludes (classical NaN, and anything outside the training
range recorded in the gt npz) are skipped in both.
"""

import argparse
import os
import sys

import numpy as np

os.environ["HBS_PYTHON_DIR"] = "/nonexistent"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_osc_standard as S
import osc_family as OF
from finetune_osc import _images, _infer_all
from hbsn_exp import grid, nets

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
CACHE = os.path.join(REPO, "runs/cache_osc")
FIELD_CACHE = os.path.join(CACHE, f"fields_osc_{OF.FAMILY}.npz")
FT_TOL = 0.005  # per-psi F regression tolerance
PEAK_TOL = 1.02  # C peak vs base, matching the trainer's own noise floor


def _infer(net, imgs):
    import torch

    return _infer_all(net, imgs, "cuda" if torch.cuda.is_available() else "cpu")


def audit(hf, ref, ok, ths, w, z, mask, sel):
    """(F per psi, C per psi pair) for one field set."""
    amp = np.abs(hf).astype(np.float32)
    f = np.array([S._best_theta(ref[i], amp[i], w, z, mask)[1] for i in sel])
    d = np.array(
        [
            S._wrms(amp[i], amp[i + 1], w) if (ok[i] and ok[i + 1]) else np.nan
            for i in range(len(ths) - 1)
        ]
    )
    return f, d


def main() -> None:
    ap = argparse.ArgumentParser(description="F + C only")
    ap.add_argument(
        "--gt-cache", default=os.path.join(CACHE, f"gt_arcs_{OF.FAMILY}.npz")
    )
    ap.add_argument("--ckpt", action="append", default=[], help="repeatable")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    d = np.load(FIELD_CACHE)
    ths, hf_base = d["ths"], d["hf"].astype(complex)
    g = np.load(args.gt_cache)
    ref, ok = g["gt"].astype(complex), g["valid"]
    z, mask = grid.get_ghbs_grid()
    w = S._radial_weight(mask, z)
    sel = [int(i) for i in np.where(ok)[0][:: args.stride]]
    mid = 0.5 * (ths[:-1] + ths[1:])

    print(
        f"reference: {os.path.basename(args.gt_cache)}  "
        f"valid psi {int(ok.sum())}/{len(ths)}  "
        f"F sampled {len(sel)} (stride {args.stride})"
    )

    models = [("base", hf_base)]
    if args.ckpt:
        imgs = _images(ths)
        for ck in args.ckpt:
            tag = args.tag or os.path.basename(os.path.dirname(ck))
            print(f"[{tag}] inferring {len(ths)} fields from {ck}", flush=True)
            models.append((tag, _infer(nets.build_net(ck), imgs)))

    res = {}
    for name, hf in models:
        res[name] = audit(hf, ref, ok, ths, w, z, mask, sel)

    print("\n== F: similarity to the target, best rotation removed ==")
    print(
        f"{'model':>14} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8} {'vs base':>9}"
    )
    b_f = res["base"][0]
    for name, (f, _) in res.items():
        print(
            f"{name:>14} {f.mean():8.4f} {np.median(f):8.4f} "
            f"{np.percentile(f, 95):8.4f} {f.max():8.4f} "
            f"{100 * (1 - f.mean() / b_f.mean()):8.1f}%"
        )

    print("\n== C: adjacent-d along psi, no rotation removed ==")
    print(
        f"{'model':>14} {'max':>8} {'p99':>8} {'mean':>8} "
        f"{'>0.02':>7} {'vs base':>9}"
    )
    b_d = res["base"][1]
    for name, (_, dd) in res.items():
        v = dd[~np.isnan(dd)]
        print(
            f"{name:>14} {v.max():8.4f} {np.percentile(v, 99):8.4f} "
            f"{v.mean():8.4f} {int((v > 0.02).sum()):7d} "
            f"{100 * (1 - v.max() / b_d[~np.isnan(b_d)].max()):8.1f}%"
        )
    print("\n  largest C pairs after the flip:")
    for name in res:
        if name == "base":
            continue
        dd = res[name][1]
        v = np.nan_to_num(dd, nan=-1)
        top = np.argsort(v)[::-1][:5]
        print(
            f"    {name:>12}: "
            + "  ".join(f"{mid[i]:.2f}({dd[i]:.4f})" for i in top)
        )

    gates = {}
    for name, (f, dd) in res.items():
        if name == "base":
            continue
        worse = int((f > b_f + FT_TOL).sum())
        cmax = np.nanmax(dd)
        gates[name] = {
            "F improves": bool(f.mean() <= b_f.mean()),
            "F no per-psi regression": bool(worse / len(f) <= 0.12),
            "C max <= base": bool(cmax <= np.nanmax(b_d) * PEAK_TOL),
        }
        print(
            f"\n[{name}] regressed psi {worse}/{len(f)}  C max {cmax:.4f} "
            f"(base {np.nanmax(b_d):.4f})"
        )
        print(f"  gates: {gates[name]}")
        print(f"  VERDICT: {'PASS' if all(gates[name].values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
