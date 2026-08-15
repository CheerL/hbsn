#!/usr/bin/env python3
"""E5 噪声鲁棒性：原版 vs 带 HBSN 分割在带噪图上的退化对比。

对 connected（单连通）coco val 子集，加多强度高斯噪声 σ∈{0,0.05,0.1,0.2}，
评估 unet/deeplab 的 hbs0（wo HBSN）vs hbs0.05（w HBSN）mask IoU，
生成退化曲线（IoU vs σ），unet/deeplab 分开子图。

用法：uv run python scripts/exp/eval_noise.py [max_samples]
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

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

SIGMAS = [0.0, 0.05, 0.1, 0.2]
OUT = "figures/hbsn"
COMMON_SETS = ["connected=true", "single_instance=true"]

FAMILIES = {
    "unetpp": [
        (
            "unetpp",
            "runs/finetuned_noise/unetpp_hbs0_noise_ft.pth",
            "hbs0-ft (wo HBSN)",
        ),
        (
            "unetpp",
            "runs/finetuned_noise/unetpp_hbs0.05_noise_ft.pth",
            "hbs0.05-ft (w HBSN)",
        ),
    ],
    "deeplab": [
        (
            "deeplab",
            "runs/finetuned_noise/deeplab_hbs0_noise_ft.pth",
            "hbs0-ft (wo HBSN)",
        ),
        (
            "deeplab",
            "runs/finetuned_noise/deeplab_hbs0.05_noise_ft.pth",
            "hbs0.05-ft (w HBSN)",
        ),
    ],
}


def load_compare_nets_no_hbsn(entries, sets, device):
    """加载对比模型，过滤 hbsn.* 键（finetuned ckpt 的 hbsn 是 bce 架构，
    与默认 hbsn 子网 shape 不匹配；mask 指标只看 predict_mask，hbsn 不参与）。"""
    nets = []
    for model, ckpt_path, label in entries:
        spec = get_spec(model)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net_cfg, _ = _build_configs(
            spec, ckpt.get("config", {}), {"device": device}, sets
        )
        net = spec.net.factory(net_cfg)
        sd = {
            k: v
            for k, v in ckpt["state_dict"].items()
            if not k.startswith("hbsn.")
        }
        net.load_state_dict(sd, strict=False)
        net.eval()
        nets.append((label, net))
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=200)
    ap.add_argument("--sigmas", default="0,0.05,0.1,0.2")
    ap.add_argument("--out", default="f_noise_robust.png")
    args = ap.parse_args()
    max_samples = args.max_samples
    SIGMAS = [float(s) for s in args.sigmas.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)  # 固定噪声，可复现
    print(f"device: {device}, max_samples: {max_samples}, sigmas: {SIGMAS}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (fam, entries) in zip(
        axes, FAMILIES.items(), strict=True
    ):
        spec = get_spec(entries[0][0])
        ckpt = torch.load(
            entries[0][1], map_location="cpu", weights_only=False
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
        dataset = spec.dataset(dataset_cfg)
        _, dl = dataset.get_dataloader(
            batch_size=1, split_rate=0, drop_last=True
        )
        nets = load_compare_nets_no_hbsn(entries, COMMON_SETS, device)
        dtype = torch_dtype(nets[0][1].config)
        labels = [lab for lab, _ in nets]
        dev = nets[0][1].config.device

        results = {lab: {s: [] for s in SIGMAS} for lab, _ in nets}
        for i, (img, mask) in enumerate(dl):
            if max_samples and i >= max_samples:
                break
            mask = mask.to(dev, dtype=dtype)
            for s in SIGMAS:
                noisy = (img + s * torch.randn_like(img)).clamp(0, 1).to(
                    dev, dtype=dtype
                )
                for lab, net in nets:
                    hard = net.get_hard_mask(net(noisy)[0])
                    _, iou = net.get_metrics(hard, mask)
                    results[lab][s].append(iou.mean().item())
        print(f"\n===== {fam}（{len(next(iter(results.values()))[SIGMAS[0]])} 样本）=====")
        for lab in labels:
            ys = [np.mean(results[lab][s]) for s in SIGMAS]
            print(
                f"  {lab:16s} "
                + " ".join(f"σ={s}:{y:.4f}" for s, y in zip(SIGMAS, ys, strict=True))
            )
            ax.plot(SIGMAS, ys, marker="o", label=lab)
        ax.set_xlabel("Gaussian noise σ")
        ax.set_ylabel("mask IoU")
        ax.set_title(fam)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, args.out), dpi=150)
    plt.close(fig)
    print(f"\n输出: {OUT}/{args.out}")


if __name__ == "__main__":
    main()
