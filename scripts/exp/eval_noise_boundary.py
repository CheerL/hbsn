#!/usr/bin/env python3
"""E5 边界/形状鲁棒性评估：像素 IoU 之外，测边界 IoU + 多噪声类型。

对 (model × start × 协议) 评估 mask IoU 与边界 IoU（HBS 是边界形状描述子，
像素 IoU 可能漏检形状先验的真实收益），噪声类型：高斯（σ）与 salt-and-pepper（p）。
输出 Δ = hbs0.05 − hbs0 表，直接回答「带 HBSN 是否在边界上更鲁棒」。

用法：uv run python scripts/exp/eval_noise_boundary.py [max_samples]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from eval_coco import _build_configs
from eval_noise import COMMON_SETS, load_compare_nets_no_hbsn

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

# (model, start, [(协议, ckpt 后缀), ...]) — 后缀 "" = 冻结基线
PROTOCOLS = ["", "_ufz", "_s02"]
GROUPS = [
    ("unetpp", "hbs0", [("冻结", ""), ("解冻", "_ufz"), ("σtr=0.2", "_s02")]),
    ("unetpp", "hbs0.05", [("冻结", ""), ("解冻", "_ufz"), ("σtr=0.2", "_s02")]),
    ("deeplab", "hbs0", [("冻结", ""), ("解冻", "_ufz"), ("σtr=0.2", "_s02")]),
    ("deeplab", "hbs0.05", [("冻结", ""), ("解冻", "_ufz"), ("σtr=0.2", "_s02")]),
]
GAUSS_SIGMAS = [0.0, 0.1, 0.2, 0.3]
SP_PROBS = [0.0, 0.1, 0.2, 0.3]


def ckpt_path(model, start, suffix):
    return f"runs/finetuned_noise/{model}_{start}_noise_ft{suffix}.pth"


def boundary_iou(pred_bin, gt_bin, k=5):
    """边界 IoU：dilate 两 mask 后取边界环（dilate∧¬orig）的 IoU。k=5 → 2px 边界带。"""
    pd = F.max_pool2d(pred_bin.float(), k, stride=1, padding=k // 2)
    gd = F.max_pool2d(gt_bin.float(), k, stride=1, padding=k // 2)
    pb = pd * (1 - pred_bin)
    gb = gd * (1 - gt_bin)
    inter = (pb * gb).sum(dim=(1, 2, 3))
    union = pb.sum(dim=(1, 2, 3)) + gb.sum(dim=(1, 2, 3)) - inter
    return inter / union.clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    for model, start, protocols in GROUPS:
        spec = get_spec(model)
        ckpt = torch.load(
            ckpt_path(model, start, protocols[0][1]),
            map_location="cpu",
            weights_only=False,
        )
        _, dcfg = _build_configs(
            spec,
            ckpt.get("config", {}),
            {
                "data_dir": "coco/val2017",
                "annotation_path": "coco/annotations/instances_val2017.json",
            },
            COMMON_SETS,
        )
        ds = spec.dataset(dcfg)
        _, dl = ds.get_dataloader(batch_size=1, split_rate=0, drop_last=True)

        print(f"\n===== {model} / {start} =====")
        for pname, suffix in protocols:
            nets = load_compare_nets_no_hbsn(
                [(model, ckpt_path(model, start, suffix), pname)],
                COMMON_SETS,
                device,
            )
            net = nets[0][1]
            dtype = torch_dtype(net.config)
            dev = net.config.device
            # 每个噪声类型收集 IoU 与边界 IoU
            acc = {"g": {s: [[], []] for s in GAUSS_SIGMAS}, "sp": {p: [[], []] for p in SP_PROBS}}
            torch.manual_seed(0)
            for i, (img, mask) in enumerate(dl):
                if args.max_samples and i >= args.max_samples:
                    break
                mask = mask.to(dev, dtype=dtype)
                gt_bin = (mask > 0.5).float()
                for s in GAUSS_SIGMAS:
                    noisy = (img + s * torch.randn_like(img)).clamp(0, 1).to(dev, dtype=dtype)
                    hard = net.get_hard_mask(net(noisy)[0]).float()
                    _, iou = net.get_metrics(hard, mask)
                    acc["g"][s][0].append(iou.mean().item())
                    acc["g"][s][1].append(boundary_iou(hard, gt_bin).mean().item())
                for p in SP_PROBS:
                    if p == 0.0:
                        noisy = img.to(dev, dtype=dtype)
                    else:
                        sp = torch.rand_like(img)
                        noisy = img.clone().to(dev, dtype=dtype)
                        noisy[sp < p / 2] = 0.0
                        noisy[sp > 1 - p / 2] = 1.0
                    hard = net.get_hard_mask(net(noisy)[0]).float()
                    _, iou = net.get_metrics(hard, mask)
                    acc["sp"][p][0].append(iou.mean().item())
                    acc["sp"][p][1].append(boundary_iou(hard, gt_bin).mean().item())

            print(f"\n-- {model}/{start} / {pname} --")
            for nt, levels in (("gauss", GAUSS_SIGMAS), ("saltpepper", SP_PROBS)):
                lbls = levels
                ious = [sum(acc["g" if nt == "gauss" else "sp"][l][0]) / len(acc["g" if nt == "gauss" else "sp"][l][0]) for l in levels]
                bis = [sum(acc["g" if nt == "gauss" else "sp"][l][1]) / len(acc["g" if nt == "gauss" else "sp"][l][1]) for l in levels]
                print(f"  {nt:11s} IoU:   " + " ".join(f"{l}:{v:.3f}" for l, v in zip(lbls, ious, strict=True)))
                print(f"  {nt:11s} bIoU:  " + " ".join(f"{l}:{v:.3f}" for l, v in zip(lbls, bis, strict=True)))


if __name__ == "__main__":
    main()
