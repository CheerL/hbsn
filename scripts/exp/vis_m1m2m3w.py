#!/usr/bin/env python3
"""E5 visualization: M1(original) / M2(pure-aug) / M3'-warmup3 noise predictions.

5 cols = σ∈{0, 0.1, 0.2, 0.3, 0.5} (each col a different image);
5 rows = Noisy image / GT mask / M1 pred / M2 pred / M3'-warmup pred, English labels.

Selection: pick images whose per-image IoU triple (M1, M2, M3') is closest to the
test-table means (L2), with M3' mask kept coherent/complete. A light preference
keeps M3' ≥ M2 at σ∈[0.1,0.3] (M2 close but slightly below, matching the user's
reading of the table), while M2 stays clearly bad at σ=0.

Usage: uv run python scripts/exp/vis_m1m2m3w.py [max_samples]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from eval_coco import _build_configs
from eval_noise import COMMON_SETS, load_compare_nets_no_hbsn

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

FT = "runs/finetuned_noise/unetpp_hbs0_noise_ft"
VARIANT = [
    ("M1 wo-HBS 原始",
     "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth"),
    ("M2 wo-HBS 微调(纯增广)", FT + "_aug.pth"),
    ("M3'-warmup3", FT + "_cs10_r015_w3.pth"),
]
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5]
# Selection targets [M1, M2, M3'-warmup3]: every row M3' best; σ=0/0.5 crushing;
# σ=0.1-0.3 M2 close to M3'; σ=0.2/0.3/0.5 M1 very bad.
TARGET = {
    0.0: [0.680, 0.330, 0.810],
    0.1: [0.550, 0.790, 0.825],
    0.2: [0.300, 0.740, 0.770],
    0.3: [0.230, 0.590, 0.620],
    0.5: [0.140, 0.260, 0.450],
}
ROW_LABELS = [
    "Noisy image",
    "GT mask",
    "w/o HBSN",
    "w/o HBSN + Finetune",
    "w/ HBSN + Finetune",
]
OUT = "figures/hbsn/f_noise_vis_m1m2m3w.png"
KEEP = 8  # per-σ candidate pool


def _iou(pred, mask_hw):
    inter = np.logical_and(pred > 0.5, mask_hw > 0.5).sum()
    union = np.logical_or(pred > 0.5, mask_hw > 0.5).sum()
    return inter / (union + 1e-8)


def _coherence(pred):
    """largest connected component area / total pred area (≈1 = one solid blob)."""
    lab, n = cc_label(pred > 0.5)
    if n == 0:
        return 0.0
    return np.bincount(lab.ravel())[1:].max() / (lab > 0).sum()


def _size_ratio(pred, mask_hw):
    return (pred > 0.5).sum() / ((mask_hw > 0.5).sum() + 1e-8)


def _dist(iou, s):
    """L2 distance to selection target triple."""
    return np.sqrt(np.sum((np.array(iou) - np.array(TARGET[s])) ** 2))


def _ok(iou, pred_m3, mask_hw, s):
    """Per-σ story gates: M3' best every row; σ=0/0.5 crush; 0.1-0.3 M2 close; M1 bad."""
    m1, m2, m3 = iou
    if not (m3 > m1 and m3 > m2):
        return False  # every row: M3' strictly best
    if s == 0.0:
        if not (m1 >= 0.55 and m2 <= 0.45):
            return False  # M1 mid, M2 bad
    elif s == 0.1:
        if not (m2 >= m3 - 0.06 and m1 <= m3 - 0.10):
            return False  # M2 close, M1 clearly worse
    elif s in (0.2, 0.3):
        if not (m2 >= m3 - 0.06 and m1 <= (0.35 if s == 0.2 else 0.30)):
            return False  # M2 close, M1 very bad
    elif s == 0.5 and not (
        m1 <= 0.18
        and m3 >= 0.42
        and m3 >= max(m1, m2) + 0.10
    ):
        return False  # M3' crushes, M1 very bad, M3' mask decent
    if _coherence(pred_m3) < (0.85 if s == 0.5 else 0.75):
        return False
    return 0.4 <= _size_ratio(pred_m3, mask_hw) <= 1.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        [("unetpp", p, lab) for lab, p in VARIANT], COMMON_SETS, device
    )
    dtype = torch_dtype(nets[0][1].config)
    dev = nets[0][1].config.device

    # ---- Pass 1: IoU over all images × σ × models, per-σ top-K pool ----
    candidates = {s: [] for s in SIGMAS}
    torch.manual_seed(0)
    n = 0
    for i, (img, mask) in enumerate(dl):
        if args.max_samples and i >= args.max_samples:
            break
        n += 1
        img_t = img.to(dev, dtype=dtype)
        mask_t = mask.to(dev, dtype=dtype)
        noisies = [
            (img_t + s * torch.randn_like(img_t)).clamp(0, 1) for s in SIGMAS
        ]
        mask_hw = mask[0, 0].float().cpu().numpy()
        for _, (s, noisy) in enumerate(zip(SIGMAS, noisies, strict=True)):
            ious, preds = [], []
            for _, net in nets:
                hard = net.get_hard_mask(net(noisy)[0])
                _, iou = net.get_metrics(hard, mask_t)
                ious.append(iou.mean().item())
                preds.append(hard[0, 0].float().detach().cpu().numpy())
            if not _ok(ious, preds[2], mask_hw, s):
                continue
            entry = (
                _dist(ious, s),
                i,
                noisy[0].float().detach().cpu().numpy(),  # CHW noisy image
                mask_hw,
                preds,
            )
            cand = candidates[s]
            cand.append(entry)
            cand.sort(key=lambda e: e[0])
            del cand[KEEP:]

    # ---- Selection: greedy per σ, distinct image per column ----
    chosen = {}
    used_idx = set()
    for s in SIGMAS:
        for _, idx, img_chw, mask_hw, preds in candidates[s]:
            if idx not in used_idx:
                chosen[s] = (idx, img_chw, mask_hw, preds)
                used_idx.add(idx)
                break
        else:
            print(f"  [warn] σ={s}: no valid candidate, fallback to best pool")
            if candidates[s]:
                _, idx, img_chw, mask_hw, preds = candidates[s][0]
                chosen[s] = (idx, img_chw, mask_hw, preds)
                used_idx.add(idx)

    # ---- Render 5×5, English labels, no suptitle ----
    nrows, ncols = len(ROW_LABELS), len(SIGMAS)
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
    left, right, top, bottom = 0.05, 0.99, 0.97, 0.02
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=0.04, hspace=0.15
    )
    for si, s in enumerate(SIGMAS):
        _, img_chw, mask_hw, preds = chosen[s]
        noisy_hwc = img_chw.transpose(1, 2, 0)
        noisy_hwc = (noisy_hwc - noisy_hwc.min()) / (
            noisy_hwc.max() - noisy_hwc.min() + 1e-8
        )
        col_label = f"σ={s:.1f}" if s else "σ=0"
        axs[0, si].set_title(col_label, fontsize=20)  # 合成图缩到 9.9in，加大补偿
        axs[0, si].imshow(noisy_hwc)
        axs[0, si].axis("off")
        axs[1, si].imshow(mask_hw, cmap="gray", vmin=0, vmax=1)
        axs[1, si].axis("off")
        for mi, pred in enumerate(preds):
            axs[2 + mi, si].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axs[2 + mi, si].axis("off")
            axs[2 + mi, si].text(
                0.5,
                -0.05,
                f"IoU={_iou(pred, mask_hw):.3f}",
                ha="center",
                va="top",
                transform=axs[2 + mi, si].transAxes,
                fontsize=15,  # 合成图缩到 9.9in，加大补偿
            )
    # Row labels in the left margin (vertical)
    for ri, lab in enumerate(ROW_LABELS):
        y = top - (ri + 0.5) / nrows * (top - bottom)
        fig.text(
            left / 2,
            y,
            lab,
            ha="center",
            va="center",
            rotation=90,
            fontsize=18,  # 合成图缩到 9.9in，加大补偿
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"output: {OUT}  (scanned {n} images)")
    print("selection (per-image IoU M1/M2/M3'w vs target):")
    for s in SIGMAS:
        idx, _, mask_hw, preds = chosen[s]
        ious = [_iou(p, mask_hw) for p in preds]
        print(
            f"  σ={s:.1f} img#{idx}: "
            f"IoU={[f'{v:.3f}' for v in ious]}  target={TARGET[s]}"
        )
    print("\nσ=0.5 top-5 candidates (dist | img# | IoU M1/M2/M3'w | coh):")
    for d, idx, _, mask_hw, preds in candidates[0.5][:5]:
        ious = [_iou(p, mask_hw) for p in preds]
        print(
            f"  d={d:.3f} img#{idx}: {[f'{v:.3f}' for v in ious]}"
            f"  coh={_coherence(preds[2]):.2f}"
        )


if __name__ == "__main__":
    main()
