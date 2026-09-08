#!/usr/bin/env python3
"""E5 noise candidate pool (Round A): per-σ 5×5 sheets for user selection.

Widens the (b)-panel candidate pool: per σ∈{0,0.1,0.2,0.3,0.5} up to 5 columns
(5 rows = Noisy / GT / w-o HBSN / w-o FT / w HBSN+FT), column header = image ID
+ three model IoU (2 dp). All 25 candidates are distinct images (global dedup);
noise/seed/VARIANT/561-pool follow vis_m1m2m3w.py unchanged.

Two passes:
  1. scan the whole 561-image pool once, recording per-(image,σ) IoU triples +
     M3' coherence/size-ratio (no images kept); tier gates applied on the full
     table, so gate-lowering is done honestly;
  2. replay (fresh dataloader, same seed → identical crops and noise) and run
     inference only for the chosen pairs; sheet IoU is recomputed from the very
     predictions shown, guaranteeing header == figure.

Tiers (recorded per candidate — gate-lowering is explicit, not hidden):
  σ≤0.3: A = M3'≥0.93 and ≥max(M1,M2)+0.10; B = ≥0.85 and ≥max+0.05;
         C = M3' strictly best (no IoU floor).
  σ=0.5: A = ≥max+0.10 and coherence≥0.7 and size_ratio∈[0.3,2.5];
         B = ≥max+0.05 and coherence≥0.5; C = M3' strictly best.

Usage: uv run python scripts/exp/vis_noise_candidates.py [max_samples]
Output: figures/hbsn/candidates/noise_candidates_sigma{X}.png ×5 +
        noise_candidates.csv / noise_candidates.json
"""

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_coco import _build_configs
from eval_noise import COMMON_SETS, load_compare_nets_no_hbsn
from vis_m1m2m3w import (
    ROW_LABELS,
    SIGMAS,
    VARIANT,
    _coherence,
    _iou,
    _size_ratio,
)

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

OUT_DIR = "figures/hbsn/candidates"
N_PICK = 5  # candidates per σ
TIER_DESC = {
    "≤0.3": {
        "A": "M3' IoU≥0.93 and ≥max(M1,M2)+0.10",
        "B": "M3' IoU≥0.85 and ≥max(M1,M2)+0.05",
        "C": "M3' strictly best (no IoU floor)",
    },
    "0.5": {
        "A": "M3'≥max(M1,M2)+0.10, coherence≥0.7, size_ratio∈[0.3,2.5]",
        "B": "M3'≥max(M1,M2)+0.05, coherence≥0.5",
        "C": "M3' strictly best (no IoU floor)",
    },
}


def _tier(iou3, coh, sr, sigma):
    """Gate tier for one (image, σ): 'A' preferred, 'B' lowered, 'C' M3'-best."""
    m1, m2, m3 = iou3
    best = max(m1, m2)
    if sigma > 0.3:
        if m3 >= best + 0.10 and coh >= 0.7 and 0.3 <= sr <= 2.5:
            return "A"
        if m3 >= best + 0.05 and coh >= 0.5:
            return "B"
    else:
        if m3 >= 0.93 and m3 >= m1 + 0.10 and m3 >= m2 + 0.10:
            return "A"
        if m3 >= 0.85 and m3 >= best + 0.05:
            return "B"
    return "C" if m3 > best else None


def _pick(cands_by_sigma):
    """Global greedy, hardest σ first: ≤5 distinct images per σ, lower tier
    fills the remainder (recorded per candidate)."""
    picked = {}
    used = set()
    for s in sorted(cands_by_sigma, reverse=True):  # rare stories first
        picked[s] = []
        for c in cands_by_sigma[s]:  # sorted by (tier, -margin)
            if len(picked[s]) >= N_PICK:
                break
            if c["idx"] in used:
                continue
            used.add(c["idx"])
            picked[s].append(c)
    return picked


def _classify(stats):
    """Tier-classify every (image, σ) record; return per-σ sorted lists+counts."""
    lists, counts = {}, {}
    for s in SIGMAS:
        entries = []
        counts[s] = {"A": 0, "B": 0, "C": 0}
        for rec in stats[s]:
            t = _tier(
                (rec["iou1"], rec["iou2"], rec["iou3"]),
                rec["coh"],
                rec["sr"],
                s,
            )
            if t is None:
                continue
            counts[s][t] += 1
            entries.append(
                {
                    **rec,
                    "tier": t,
                    "margin": rec["iou3"] - max(rec["iou1"], rec["iou2"]),
                }
            )
        entries.sort(key=lambda e: (e["tier"], -e["margin"]))
        lists[s] = entries
    return lists, counts


def _scan(n_max, ds, dl, nets, dtype, dev):
    """Pass 1: IoU triples + M3' coh/sr for every (image, σ). No images kept."""
    stats = {s: [] for s in SIGMAS}
    torch.manual_seed(0)
    n = 0
    for i, (img, mask) in enumerate(dl):
        if n_max and i >= n_max:
            break
        n += 1
        img_t = img.to(dev, dtype=dtype)
        mask_t = mask.to(dev, dtype=dtype)
        mask_hw = mask[0, 0].float().cpu().numpy()
        noisies = [
            (img_t + s * torch.randn_like(img_t)).clamp(0, 1) for s in SIGMAS
        ]
        for s, noisy in zip(SIGMAS, noisies, strict=True):
            ious, pred_m3 = [], None
            for mi, (_, net) in enumerate(nets):
                hard = net.get_hard_mask(net(noisy)[0])
                _, iou = net.get_metrics(hard, mask_t)
                ious.append(iou.mean().item())
                if mi == 2:
                    pred_m3 = hard[0, 0].float().detach().cpu().numpy()
            stats[s].append(
                {
                    "idx": i,
                    "img_id": ds.img_ids[i],
                    "iou1": ious[0],
                    "iou2": ious[1],
                    "iou3": ious[2],
                    "coh": _coherence(pred_m3),
                    "sr": _size_ratio(pred_m3, mask_hw),
                }
            )
        if n % 50 == 0:
            print(f"  scanned {n}", flush=True)
    return stats, n


def _replay(ds, nets, dtype, dev, pairs):
    """Replay seed-0 crops/noise; 3-model inference only for (idx, σ) pairs.

    Per image the RNG consumes all 5 σ noisies in σ order (identical stream to
    the scan pass), so noise/crops reproduce the sheets exactly. Returns
    {(idx, σ): entry with noisy/mask_hw/preds/iou(pass-2)}. Stops once all
    pairs are covered.
    """
    _, dl = ds.get_dataloader(batch_size=1, split_rate=0, drop_last=True)
    torch.manual_seed(0)
    want = {}
    for idx, s in pairs:
        want.setdefault(idx, set()).add(s)
    out = {}
    for i, (img, mask) in enumerate(dl):
        img_t = img.to(dev, dtype=dtype)
        noisies = [
            (img_t + s * torch.randn_like(img_t)).clamp(0, 1) for s in SIGMAS
        ]
        if i not in want:
            continue
        mask_hw = mask[0, 0].float().cpu().numpy()
        for s, noisy in zip(SIGMAS, noisies, strict=True):
            if s not in want[i]:
                continue
            preds, ious = [], []
            for _, net in nets:
                hard = net.get_hard_mask(net(noisy)[0])
                p = hard[0, 0].float().detach().cpu().numpy()
                preds.append(p)
                ious.append(_iou(p, mask_hw))
            out[(i, s)] = {
                "idx": i,
                "noisy": noisy[0].float().detach().cpu().numpy(),
                "mask_hw": mask_hw,
                "preds": preds,
                "iou": ious,
            }
        if len(out) == len(pairs):
            break
    return out


def _replay_and_render(ds, nets, dtype, dev, picked):
    """Pass 2: replay chosen (idx, σ) pairs, drift-check vs pass 1, group by σ
    in picked (tier/margin) order."""
    pairs = [(c["idx"], s) for s in SIGMAS for c in picked[s]]
    out = _replay(ds, nets, dtype, dev, pairs)
    got = {s: [] for s in SIGMAS}
    for s in SIGMAS:
        for c in picked[s]:
            e = out[(c["idx"], s)]
            c["iou"] = e[
                "iou"
            ]  # pass-2 IoU（与图一致）；pass-1 值保留供漂移检查
            for k in range(3):
                if abs(e["iou"][k] - c[f"iou{k + 1}"]) > 0.01:
                    print(
                        f"  [warn] σ={s} img#{c['idx']} M{k + 1} IoU drift "
                        f"pass1={c[f'iou{k + 1}']:.3f} pass2={e['iou'][k]:.3f}"
                    )
            got[s].append(
                {
                    **c,
                    "noisy": e["noisy"],
                    "mask_hw": e["mask_hw"],
                    "preds": e["preds"],
                }
            )
    return got


def _render_sheet(s, entries, out):
    """5 rows × len(entries) columns; header = image ID + tier + IoU ×3 (2dp)."""
    ncols = len(entries)
    nrows = len(ROW_LABELS)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(2.7 * ncols + 1.0, 2.7 * nrows + 1.0),
        squeeze=False,
    )
    left, right, top, bottom = 0.07, 0.99, 0.91, 0.03
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=0.04, hspace=0.06
    )
    for ci, e in enumerate(entries):
        iou1, iou2, iou3 = e["iou"]
        axs[0, ci].set_title(
            f"#{e['img_id']} ({e['tier']})\n{iou1:.2f}/{iou2:.2f}/{iou3:.2f}",
            fontsize=11,
        )
        axs[0, ci].imshow(np.clip(e["noisy"].transpose(1, 2, 0), 0, 1))
        axs[0, ci].axis("off")
        axs[1, ci].imshow(e["mask_hw"], cmap="gray", vmin=0, vmax=1)
        axs[1, ci].axis("off")
        for mi, pred in enumerate(e["preds"]):
            axs[2 + mi, ci].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axs[2 + mi, ci].axis("off")
    for ri, lab in enumerate(ROW_LABELS):
        y = top - (ri + 0.5) / nrows * (top - bottom)
        fig.text(
            left / 2, y, lab, ha="center", va="center", rotation=90, fontsize=13
        )
    fig.text(
        left,
        0.005,
        f"σ={s:g}  header IoU order: w/o HBSN / w-o FT / w HBSN+FT (2 dp)",
        fontsize=9,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _write_summary(picked, n_pool):
    """CSV + JSON summary of all candidates (pass-2 IoU) with honest meta."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for s in sorted(picked):
        for c in picked[s]:
            iou1, iou2, iou3 = c["iou"]
            path = f"{OUT_DIR}/noise_candidates_sigma{s:g}.png"
            rows.append(
                {
                    "sigma": s,
                    "image_id": c["img_id"],
                    "iou_m1": round(iou1, 4),
                    "iou_m2": round(iou2, 4),
                    "iou_m3": round(iou3, 4),
                    "tier": c["tier"],
                    "margin": round(iou3 - max(iou1, iou2), 4),
                    "path": path,
                }
            )
    if not rows:
        print("[warn] no candidates picked; nothing to write")
        return []
    with open(f"{OUT_DIR}/noise_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tier_counts = dict(_TIER_COUNTS)
    meta = {
        "seed": 0,
        "sigmas": SIGMAS,
        "pool_size": n_pool,
        "tier_definitions": TIER_DESC,
        "tier_counts_over_pool": tier_counts,
        "models": [lab for lab, _ in VARIANT],
        "note": "IoU from pass-2 predictions (identical to the sheets); "
        "tier judged in pass 1 on the same deterministic inference.",
    }
    with open(f"{OUT_DIR}/noise_candidates.json", "w") as f:
        json.dump({"meta": meta, "candidates": rows}, f, indent=2)
    return rows


_TIER_COUNTS = {}


def _setup():
    """Dataset + 3 nets exactly as vis_m1m2m3w.py (ckpt config + COMMON_SETS)."""
    spec = get_spec("unetpp")
    ckpt = torch.load(VARIANT[0][1], map_location="cpu", weights_only=False)
    _, dataset_cfg = _build_configs(
        spec,
        ckpt.get("config", {}),
        {
            "data_dir": "coco/val2017",
            "annotation_path": "coco/annotations/instances_val2017.json",
        },
        COMMON_SETS,
    )
    ds = spec.dataset(dataset_cfg)
    _, dl = ds.get_dataloader(batch_size=1, split_rate=0, drop_last=True)
    nets = load_compare_nets_no_hbsn(
        [("unetpp", p, lab) for lab, p in VARIANT],
        COMMON_SETS,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    dtype = torch_dtype(nets[0][1].config)
    dev = nets[0][1].config.device
    return ds, dl, nets, dtype, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()

    ds, dl, nets, dtype, dev = _setup()

    print(f"pass 1: scanning ≤{args.max_samples} images × {len(SIGMAS)} σ × 3")
    stats, n = _scan(args.max_samples, ds, dl, nets, dtype, dev)
    lists, counts = _classify(stats)
    for s in SIGMAS:
        print(
            f"  σ={s:g}: tier counts A={counts[s]['A']} "
            f"B={counts[s]['B']} C={counts[s]['C']} (pool {n})"
        )
    _TIER_COUNTS.update({f"{s:g}": counts[s] for s in SIGMAS})

    picked = _pick(lists)
    for s in sorted(picked, reverse=True):
        comp = "".join(c["tier"] for c in picked[s])
        print(
            f"  σ={s:g}: picked {len(picked[s])} tiers=[{comp}] "
            f"imgs={[c['img_id'] for c in picked[s]]}"
        )

    print("pass 2: replay + render")
    got = _replay_and_render(ds, nets, dtype, dev, picked)
    for s in SIGMAS:
        if got[s]:
            _render_sheet(
                s, got[s], f"{OUT_DIR}/noise_candidates_sigma{s:g}.png"
            )
            print(f"  σ={s:g}: sheet with {len(got[s])} candidates")
    rows = _write_summary(picked, n)
    assert len({r["image_id"] for r in rows}) == len(rows), "duplicate image"
    print(f"\noutputs: {OUT_DIR}/noise_candidates_sigma*.png (5 sheets),")
    print(f"  {OUT_DIR}/noise_candidates.csv, {OUT_DIR}/noise_candidates.json")
    print(f"scanned {n} images; picked {len(rows)} candidates (all distinct)")


if __name__ == "__main__":
    main()
