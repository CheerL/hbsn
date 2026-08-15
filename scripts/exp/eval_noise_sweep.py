#!/usr/bin/env python3
"""E5 终局统一评估：所有 E5 变体的噪声退化对比。

对 (model × start) 各组加载全部变体 ckpt（基线 + P1 权重扫掠 r05/r10 + P2 解冻
ufz + P3 σtr=0.2 s02），在 connected 单实例子集多 σ 下评估 mask IoU，
输出对比表（行=变体，列=σ）与曲线图 f_noise_variants.png。

用法：uv run python scripts/exp/eval_noise_sweep.py [max_samples] [--sigmas ...]
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
)  # scripts/（eval_coco 所在）

from eval_coco import _build_configs
from eval_noise import COMMON_SETS, load_compare_nets_no_hbsn

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

OUT = "figures/hbsn"
# 终局计划：(model, start, [(label, ckpt 后缀), ...])。后缀 "" = 基线（方案 A）。
PLAN = [
    (
        "unetpp",
        "hbs0",
        [
            ("rate 0 (基线)", ""),
            ("rate 0.5", "_r05"),
            ("rate 1.0", "_r10"),
            ("unfreeze", "_ufz"),
            ("σtr=0.2", "_s02"),
            ("cons 2.0", "_c2"),
            ("cons 8.0", "_c8"),
        ],
    ),
    (
        "unetpp",
        "hbs0.05",
        [
            ("rate 0.05 (基线)", ""),
            ("rate 0.5", "_r05"),
            ("rate 1.0", "_r10"),
            ("unfreeze", "_ufz"),
            ("σtr=0.2", "_s02"),
            ("cons 2.0", "_c2"),
            ("cons 8.0", "_c8"),
        ],
    ),
    (
        "deeplab",
        "hbs0",
        [
            ("rate 0 (基线)", ""),
            ("rate 0.5", "_r05"),
            ("rate 1.0", "_r10"),
            ("unfreeze", "_ufz"),
            ("σtr=0.2", "_s02"),
            ("cons 2.0", "_c2"),
            ("cons 8.0", "_c8"),
        ],
    ),
    (
        "deeplab",
        "hbs0.05",
        [
            ("rate 0.05 (基线)", ""),
            ("rate 0.5", "_r05"),
            ("rate 1.0", "_r10"),
            ("unfreeze", "_ufz"),
            ("σtr=0.2", "_s02"),
            ("cons 2.0", "_c2"),
            ("cons 8.0", "_c8"),
        ],
    ),
]


def ckpt_path(model: str, start: str, suffix: str) -> str:
    return f"runs/finetuned_noise/{model}_{start}_noise_ft{suffix}.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    ap.add_argument("--sigmas", default="0,0.05,0.1,0.2,0.3,0.5")
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device: {device}, max_samples: {args.max_samples}, sigmas: {sigmas}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (model, start, variants) in zip(
        axes.ravel(), PLAN, strict=True
    ):
        spec = get_spec(model)
        ckpt = torch.load(
            ckpt_path(model, start, variants[0][1]),
            map_location="cpu",
            weights_only=False,
        )
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

        print(f"\n===== {model} / start={start} =====")
        print("variant".ljust(20) + "".join(f"{s:>8}" for s in sigmas))
        for label, suffix in variants:
            path = ckpt_path(model, start, suffix)
            if not os.path.exists(path):
                print(f"{label.ljust(20)}  (缺 ckpt: {path})")
                continue
            entries = [(model, path, label)]
            nets = load_compare_nets_no_hbsn(entries, COMMON_SETS, device)
            dtype = torch_dtype(nets[0][1].config)
            dev = nets[0][1].config.device
            torch.manual_seed(0)  # 每个变体重置 RNG → 所有变体看到同一噪声
            ys = []
            for i, (img, mask) in enumerate(dl):
                if args.max_samples and i >= args.max_samples:
                    break
                mask = mask.to(dev, dtype=dtype)
                ious = []
                for s in sigmas:
                    noisy = (img + s * torch.randn_like(img)).clamp(0, 1).to(
                        dev, dtype=dtype
                    )
                    hard = nets[0][1].get_hard_mask(nets[0][1](noisy)[0])
                    _, iou = nets[0][1].get_metrics(hard, mask)
                    ious.append(iou.mean().item())
                ys.append(ious)
            means = np.mean(ys, axis=0)
            print(
                label.ljust(20)
                + "".join(f"{v:>8.4f}" for v in means)
                + f"  (n={len(ys)})"
            )
            ax.plot(sigmas, means, marker="o", label=label)

        ax.set_xlabel("Gaussian noise σ")
        ax.set_ylabel("mask IoU")
        ax.set_title(f"{model} / start={start}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f_noise_variants.png"), dpi=150)
    plt.close(fig)
    print(f"\n输出: {OUT}/f_noise_variants.png")


if __name__ == "__main__":
    main()
