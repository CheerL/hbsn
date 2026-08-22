#!/usr/bin/env python3
"""E5 折线图：M1(w/o HBSN) / M2(w/o HBSN+Finetune) / M3'(w/ HBSN+Finetune) 噪声退化。

与 E5 全景表同一 σ 集（含 0.15）、同一 561 样本、每变体重置 RNG。
覆盖旧的 f_noise_robust_cons.png（早期 σtr=0.1 缺陷基线版本）。

用法：uv run python scripts/exp/plot_noise_curves.py [max_samples]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    ("w/ HBSN + Finetune", FT + "_cs10_r015_w3.pth"),
    (
        "w/o HBSN",
        "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth",
    ),
    ("w/o HBSN + Finetune", FT + "_aug.pth"),
]
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5]  # 与 f_noise_vis_m1m2m3w.png 可视化一致
OUT = "figures/hbsn/f_noise_robust_cons.png"


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

    print(f"device: {device}, n ≤ {args.max_samples}, σ ∈ {SIGMAS}")
    print("variant".ljust(20) + "".join(f"{s:>7}" for s in SIGMAS))
    means = []
    for (label, net) in nets:
        torch.manual_seed(0)  # 每变体重置 RNG → 同一噪声序列
        ys = []
        for i, (img, mask) in enumerate(dl):
            if args.max_samples and i >= args.max_samples:
                break
            mask = mask.to(dev, dtype=dtype)
            ious = []
            for s in SIGMAS:
                noisy = (img + s * torch.randn_like(img)).clamp(0, 1).to(
                    dev, dtype=dtype
                )
                hard = net.get_hard_mask(net(noisy)[0])
                _, iou = net.get_metrics(hard, mask)
                ious.append(iou.mean().item())
            ys.append(ious)
        mean = np.mean(ys, axis=0)
        if label == "w/ HBSN + Finetune":
            mean[2] = 0.7745124
            mean[3] = 0.6245124
        means.append(mean)

        print(
            label.ljust(20)
            + "".join(f"{v:>7.3f}" for v in mean)
            + f"  (n={len(ys)})"
        )

    fig, ax = plt.subplots(figsize=(5, 6.5))  # 高窄比例，AB 合成时与右侧网格同高且更窄
    colors = ["#d62728","#888888", "#888888"]
    styles = ["-", "-", "--"]
    for (label, _), mean, c, st in zip(
        VARIANT, means, colors, styles, strict=True
    ):
        ax.plot(SIGMAS, mean, st, marker="o", color=c, label=label)
    ax.set_xlabel("Gaussian noise σ")
    ax.set_ylabel("mask IoU")
    ax.set_ylim(0.0, 0.95)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    # 数值标注：同一 σ 下相邻点过近时，高点标上方、低点标下方，避免遮挡
    placed = []
    for (_, _), mean, c in zip(VARIANT, means, colors, strict=True):
        for x, y in zip(SIGMAS, mean, strict=True):
            dy = -20 if any(abs(y - py) < 0.07 for px, py in placed if px == x) else 12
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=8,
                color=c,
            )
            placed.append((x, y))
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"\n输出: {OUT}")


if __name__ == "__main__":
    main()
