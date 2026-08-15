#!/usr/bin/env python3
"""E5 公平对比：GT-锚定一致性（方案 A）是否全面比公平基线 M2' 强。

对比同一 unetpp hbs0 起点的 5 个变体：
  M2  = 纯增广（wo-HBS 旧基线，无 clean-sup）
  M2' = clean-sup 1.0、无一致性（wo-HBS 公平基线，同协议）
  M3' = clean-sup 1.0 + 一致性 0.15
  cs15r015 = clean-sup 1.5 + 一致性 0.15
  cs10r02  = clean-sup 1.0 + 一致性 0.2

输出对比表（行=变体，列=σ）与 Δ vs M2'。

用法：uv run python scripts/exp/eval_fair_cons.py [max_samples]
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

BASE = "runs/finetuned_noise/unetpp_hbs0_noise_ft"
VARIANTS = [
    ("M2 纯增广", "_aug"),
    ("M2' cs10aug", "_cs10aug"),
    ("M3' cs10r015", "_cs10_r015"),
    ("warmup2", "_cs10_r015_w2"),
    ("warmup3", "_cs10_r015_w3"),
]
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = get_spec("unetpp")
    ckpt = torch.load(BASE + ".pth", map_location="cpu", weights_only=False)
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
    print("\n===== 公平对比网格 =====")
    print("variant".ljust(16) + "".join(f"{s:>7}" for s in SIGMAS))
    means = {}
    for label, suffix in VARIANTS:
        path = BASE + suffix + ".pth"
        nets = load_compare_nets_no_hbsn(
            [("unetpp", path, label)], COMMON_SETS, device
        )
        net = nets[0][1]
        dtype = torch_dtype(net.config)
        dev = net.config.device
        torch.manual_seed(0)  # 每个变体重置 RNG → 同一噪声
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
            label.ljust(16)
            + "".join(f"{v:>7.4f}" for v in means[label])
            + f"  (n={len(ys)})"
        )

    m2p = means["M2' cs10aug"]
    print("\nΔ vs M2' cs10aug:")
    for label in [lab for lab, _ in VARIANTS if lab != "M2' cs10aug"]:
        print(
            label.ljust(16)
            + "".join(f"{d:>+7.4f}" for d in means[label] - m2p)
        )
    print("\nΔ vs M2 纯增广:")
    m2 = means["M2 纯增广"]
    for label in [lab for lab, _ in VARIANTS if lab != "M2 纯增广"]:
        print(
            label.ljust(16)
            + "".join(f"{d:>+7.4f}" for d in means[label] - m2)
        )


if __name__ == "__main__":
    main()
