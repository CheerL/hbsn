#!/usr/bin/env python3
"""E5 全景表：M1/M2/M2'/M3/M3'/M3'-warmup 同一 σ 集（含 0.15）。

unetpp hbs0 起点，n=561。每变体加载 ckpt、重置 RNG、逐 σ 评估 mask IoU。
输出单表 + Δ vs M2（纯增广微调）。

用法：uv run python scripts/exp/eval_all_table.py [max_samples]
"""

import argparse
import os
import sys

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
VARIANTS = [
    ("M1 wo-HBS 原始",
     "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth"),
    ("M2 wo-HBS 微调(纯增广)", FT + "_aug.pth"),
    ("M2' clean-sup(公平基线)", FT + "_cs10aug.pth"),
    ("M3 w-HBS 微调(cond0.15)", FT + "_cg015.pth"),
    ("M3' w-HBS+clean-sup", FT + "_cs10_r015.pth"),
    ("M3' warmup3", FT + "_cs10_r015_w3.pth"),
]
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = get_spec("unetpp")
    ckpt = torch.load(VARIANTS[0][1], map_location="cpu", weights_only=False)
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

    print(f"device: {device}, n ≤ {args.max_samples}, σ ∈ {SIGMAS}")
    print("\n===== E5 全景表 (unetpp) =====")
    print("variant".ljust(24) + "".join(f"{s:>7}" for s in SIGMAS))
    means = {}
    for label, path in VARIANTS:
        nets = load_compare_nets_no_hbsn(
            [("unetpp", path, label)], COMMON_SETS, device
        )
        net = nets[0][1]
        dtype = torch_dtype(net.config)
        dev = net.config.device
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
        means[label] = np.mean(ys, axis=0)
        print(
            label.ljust(24)
            + "".join(f"{v:>7.3f}" for v in means[label])
            + f"  (n={len(ys)})"
        )

    base = "M2 wo-HBS 微调(纯增广)"
    print("\nΔ vs M2（纯增广微调）:")
    for label in [lab for lab, _ in VARIANTS if lab != base]:
        print(
            label.ljust(24)
            + "".join(f"{d:>+7.3f}" for d in means[label] - means[base])
        )
    print("\nΔ vs M2'（clean-sup 公平基线）:")
    base2 = "M2' clean-sup(公平基线)"
    for label in [lab for lab, _ in VARIANTS if lab not in (base, base2)]:
        print(
            label.ljust(24)
            + "".join(f"{d:>+7.3f}" for d in means[label] - means[base2])
        )


if __name__ == "__main__":
    main()
