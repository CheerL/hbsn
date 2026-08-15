#!/usr/bin/env python3
"""E5 HBS 场稳定性：噪声下预测 mask 的 HBS 是否更贴近 GT HBS。

hbs_loss 训练目标是让预测 mask 的 HBS 场贴近 GT mask 的 HBS 场。直接测这个量
（hbsn(预测mask) 与 hbsn(GTmask) 的 MSE）在噪声下的表现——若 hbs0.05 模型
在噪声下 HBS 场误差更小，则「HBSN 稳定形状签名」是可辩护的正结果。

hbsn 用 bce 旧架构预训练权重（与 finetune 相同），冻结；两模型共用同一 HBS 函数。

用法：uv run python scripts/exp/eval_noise_hbsfield.py [max_samples]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from eval_coco import _build_configs
from eval_noise import COMMON_SETS, load_compare_nets_no_hbsn
from finetune_noise import HBSN_CFG, build_bce_hbsn

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

GAUSS_SIGMAS = [0.0, 0.1, 0.2, 0.3]
# (model, [(start, ckpt 后缀), ...])
GROUPS = [
    ("unetpp", [("hbs0", ""), ("hbs0.05", ""), ("hbs0.05", "_ufz")]),
    ("deeplab", [("hbs0", ""), ("hbs0.05", ""), ("hbs0.05", "_ufz")]),
]


def ckpt_path(model, start, suffix):
    return f"runs/finetuned_noise/{model}_{start}_noise_ft{suffix}.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_samples", nargs="?", type=int, default=561)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    for model, starts in GROUPS:
        spec = get_spec(model)
        ckpt = torch.load(
            ckpt_path(model, starts[0][0], starts[0][1]),
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
        # 共用同一 bce hbsn（冻结）计算 HBS 场
        hcfg = HBSN_CFG[model]
        hbsn = build_bce_hbsn(hcfg["ckpt"], hcfg["cd"], hcfg["cu"], device)
        hbsn.eval()

        print(f"\n===== {model} =====")
        for start, suffix in starts:
            nets = load_compare_nets_no_hbsn(
                [(model, ckpt_path(model, start, suffix), start)],
                COMMON_SETS,
                device,
            )
            net = nets[0][1]
            dtype = torch_dtype(net.config)
            dev = net.config.device
            errs = {s: [] for s in GAUSS_SIGMAS}
            torch.manual_seed(0)
            with torch.no_grad():
                for i, (img, mask) in enumerate(dl):
                    if args.max_samples and i >= args.max_samples:
                        break
                    mask = mask.to(dev, dtype=dtype)
                    gt_hbs = hbsn(mask)
                    for s in GAUSS_SIGMAS:
                        noisy = (
                            img + s * torch.randn_like(img)
                        ).clamp(0, 1).to(dev, dtype=dtype)
                        hard = net.get_hard_mask(net(noisy)[0]).float()
                        pred_hbs = hbsn(hard)
                        errs[s].append(
                            (pred_hbs - gt_hbs).pow(2).mean().item()
                        )
            line = " ".join(f"σ={s}:{sum(v)/len(v):.5f}" for s, v in errs.items())
            print(f"  {start:10s} HBS场MSE: {line}")


if __name__ == "__main__":
    main()
