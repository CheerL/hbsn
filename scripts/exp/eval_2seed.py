#!/usr/bin/env python3
"""E5 2-seed 验证：M3'-warmup3（cs10_r015_w3）vs M2'（cs10aug）的噪声退化。

输出每个 seed 的独立行 + 均值行，以及 Δ(mean)（每 σ 点）。

用法：uv run python scripts/exp/eval_2seed.py [max_samples]
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
# (label, [suffix 列表]) —— 列表内多个 seed 求均值
GROUPS = [
    ("M2' cs10aug", ["_cs10aug", "_cs10aug_s1"]),
    ("M3' warmup3", ["_cs10_r015_w3", "_cs10_r015_w3_s1", "_cs10_r015_w3_s2"]),
]
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def eval_ckpt(suffix, dl, max_samples, SIGMAS):
    nets = load_compare_nets_no_hbsn(
        [("unetpp", BASE + suffix + ".pth", suffix)], COMMON_SETS, "cuda"
    )
    net = nets[0][1]
    dtype = torch_dtype(net.config)
    dev = net.config.device
    torch.manual_seed(0)  # 每个 ckpt 同一噪声序列
    ys = []
    for i, (img, mask) in enumerate(dl):
        if max_samples and i >= max_samples:
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
    return np.mean(ys, axis=0)


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
    print("\n===== 2-seed 验证网格 =====")
    print("variant".ljust(18) + "".join(f"{s:>7}" for s in SIGMAS))
    means = {}
    for label, suffixes in GROUPS:
        rows = []
        for suffix in suffixes:
            m = eval_ckpt(suffix, dl, args.max_samples, SIGMAS)
            rows.append(m)
            print(
                f"{label} {suffix:>14}".ljust(18)
                + "".join(f"{v:>7.4f}" for v in m)
            )
        mean = np.mean(rows, axis=0)
        means[label] = mean
        print(
            f"{label} mean".ljust(18)
            + "".join(f"{v:>7.4f}" for v in mean)
        )

    m3 = next(k for k in means if k != "M2' cs10aug")
    d = means[m3] - means["M2' cs10aug"]
    print(f"\nΔ mean ({m3} - M2'):")
    print(
        "".join(f"{v:>+7.4f}" for v in d)
    )
    print("每 σ 点是否 M3 ≥ M2':", ["✓" if v >= 0 else "✗" for v in d])


if __name__ == "__main__":
    main()
