#!/usr/bin/env python3
"""E5 重新对比：M1（原始 hbs0，未微调）vs M2（纯增广微调）。

用户要求重新对比"不那么公平"的 M1 和 M2：M1 是原始模型（未做噪声微调），
M2 是同一 hbs0 起点在带噪 connected 子集上纯增广微调。输出单表 + Δ(M2-M1)。

用法：uv run python scripts/exp/eval_m1_m2.py [max_samples]
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

M1 = "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth"
M2 = "runs/finetuned_noise/unetpp_hbs0_noise_ft_aug.pth"
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = get_spec("unetpp")
    ckpt = torch.load(M2, map_location="cpu", weights_only=False)
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
    print("\n===== M1(原始) vs M2(纯增广微调) =====")
    print("variant".ljust(20) + "".join(f"{s:>7}" for s in SIGMAS))
    means = {}
    for label, path in [("M1 原始", M1), ("M2 纯增广微调", M2)]:
        nets = load_compare_nets_no_hbsn(
            [("unetpp", path, label)], COMMON_SETS, device
        )
        net = nets[0][1]
        dtype = torch_dtype(net.config)
        dev = net.config.device
        torch.manual_seed(0)  # 同一噪声序列
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
            label.ljust(20)
            + "".join(f"{v:>7.4f}" for v in means[label])
            + f"  (n={len(ys)})"
        )

    d = means["M2 纯增广微调"] - means["M1 原始"]
    print("\nΔ (M2 - M1):")
    print("".join(f"{v:>+7.4f}" for v in d))


if __name__ == "__main__":
    main()
