#!/usr/bin/env python3
"""E5 阶段 B：带 HBSN 分割模型在带噪 connected 子集上 finetune 几轮。

复现远程训练配置（方案 A）：hbsn 子网用 bce 旧架构（soft_all2/loog3，
加载原始 hbsn_checkpoint 权重并冻结），lr=5e-5、batch=16、hbs_loss_rate 保留
config 值（hbs0=0 无先验 vs hbs0.05=0.05 有先验），从 hbs0 加载续训。

用法：uv run python scripts/exp/finetune_noise.py [epochs]
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # scripts/（eval_coco 所在）

from eval_coco import _build_configs

from hbsn._legacy import pickle_shim
from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

pickle_shim.install()  # 旧 ckpt 的 pickle Config 反序列化桩

OUT_DIR = "runs/finetuned_noise"
COMMON_SETS = ["connected=true", "single_instance=true"]
NOISE_SIGMA = 0.1
LR = 5e-5  # 远程训练值（此前误用 1e-4）
BATCH_SIZE = 16  # 远程训练值（此前误用 8）

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# 原始 hbsn_checkpoint（bce 旧架构，远程训练配对的 HBSN 子网权重）
HBSN_CFG = {
    "deeplab": {
        "ckpt": os.path.join(
            _REPO, "runs/remote_hbsn/hbsn_stn3_soft_all2_best.pth"
        ),
        "cd": [8, 8, 16, 32, 64, 128],
        "cu": [8, 16, 32, 64, 128],
    },
    "unetpp": {
        "ckpt": os.path.join(
            _REPO, "runs/remote_hbsn/hbsn_stn3_loog3_best1481.pth"
        ),
        "cd": [8, 16, 32, 64, 128, 256],
        "cu": [8, 16, 32, 64, 128, 256],
    },
}

MODELS = {
    "unetpp": {
        "hbs0": "runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth",
        "hbs0.05": "runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/best.pth",
    },
    "deeplab": {
        "hbs0": "runs/migrated/deeplab/May15_11-02-15_hbs0_all_c/checkpoints/best.pth",
        "hbs0.05": "runs/migrated/deeplab/May24_12-58-07_hbs0.05_all_newhbsn/checkpoints/best.pth",
    },
}


def build_bce_hbsn(ckpt_path, channels_down, channels_up, device):
    """构建 bce 旧架构 HBSNet 并 strict 加载原始权重（形状先验，冻结）。

    soft_all2/loog3 的 bce 主干 + pre/post_stn；bce→backbone 改名 + fc_loc 冗余复制。
    """
    spec = get_spec("hbsn")
    cfg = OmegaConf.merge(
        OmegaConf.structured(spec.net_schema),
        OmegaConf.create(
            {
                "channels_down": channels_down,
                "channels_up": channels_up,
                "stn_mode": 3,
                "device": device,
            }
        ),
    )
    net = spec.net.factory(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mapped = {}
    for k, v in ck["state_dict"].items():
        nk = k.replace("bce.", "backbone.") if k.startswith("bce.") else k
        mapped[nk] = v
    for k in list(mapped):
        if ".fc_loc.0." in k:
            mapped[k.replace(".fc_loc.0.", ".fc_loc1.")] = mapped[k]
        if ".fc_loc.2." in k:
            mapped[k.replace(".fc_loc.2.", ".fc_loc2.")] = mapped[k]
    net.load_state_dict(mapped, strict=True)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", nargs="?", type=int, default=5)
    ap.add_argument(
        "--hbs-loss-rate",
        type=float,
        default=None,
        help="覆盖 finetune 期 hbs_loss_rate（默认用 ckpt config 原值）",
    )
    ap.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA)
    ap.add_argument(
        "--noise-sigma-range",
        default=None,
        help='"lo,hi" 每 batch 均匀采样噪声强度（覆盖多个 σ 训练）',
    )
    ap.add_argument(
        "--cons-min-sigma",
        type=float,
        default=None,
        help="仅当采样噪声 σ ≥ 该阈值时施加一致性正则（轻噪声跟随像素，重噪声用形状锚定）",
    )
    ap.add_argument(
        "--cons-noise-sigma-range",
        default=None,
        help='"lo,hi" 一致性专用重噪声前向（与 mask 噪声解耦）；设置后 mask 保持原噪声、'
        "一致性在 lo..hi 独立采样",
    )
    ap.add_argument(
        "--clean-sup-rate",
        type=float,
        default=0.0,
        help="GT-锚定一致性：>0 时 pred_clean 带梯度前向并对其 GT 加 mask loss（w_clean），"
        "使一致性目标锚定到正确形状",
    )
    ap.add_argument("--unfreeze-hbsn", action="store_true")
    ap.add_argument(
        "--hbs-consistency-rate",
        type=float,
        default=0.0,
        help="HBS 形状-噪声不变性正则权重：MSE(hbs(pred_noisy), hbs(pred_clean).detach())",
    )
    ap.add_argument(
        "--cons-warmup-epochs",
        type=int,
        default=0,
        help="一致性权重线性 ramp 的 epoch 数（0=全程满权重）；先学好噪声增广再叠形状先验，"
        "降低高噪声下 mask/consistency 梯度冲突导致的训练方差",
    )
    ap.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="默认 16；一致性正则建议 8"
    )
    ap.add_argument(
        "--models",
        default="unetpp,deeplab",
        help="逗号分隔的模型族（默认全部）；聚焦单网络时传 unetpp 或 deeplab",
    )
    ap.add_argument(
        "--starts",
        default="hbs0,hbs0.05",
        help="逗号分隔的起点（默认全部）；纯一致性法传 hbs0",
    )
    ap.add_argument(
        "--out-suffix", default="", help="输出 ckpt 文件名后缀（默认空=覆盖方案 A 名称）"
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="torch 手动种子（多 seed 可复现性验证）"
    )
    args = ap.parse_args()
    epochs = args.epochs
    noise_sigma = args.noise_sigma
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(
        f"device: {device}, epochs: {epochs}, σ={noise_sigma}, lr={LR}, "
        f"hbs_loss_rate={args.hbs_loss_rate}, unfreeze_hbsn={args.unfreeze_hbsn}, "
        f"hbs_consistency_rate={args.hbs_consistency_rate}"
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    models = args.models.split(",")
    starts = args.starts.split(",")
    for model in models:
        variants = MODELS[model]
        for name in list(variants):
            if name not in starts:
                del variants[name]
        for name, src in variants.items():
            spec = get_spec(model)
            ckpt = torch.load(src, map_location="cpu", weights_only=False)
            net_cfg, dataset_cfg = _build_configs(
                spec,
                ckpt.get("config", {}),
                {
                    "device": device,
                    "data_dir": "coco/val2017",
                    "annotation_path": "coco/annotations/instances_val2017.json",
                },
                COMMON_SETS,
            )
            if args.hbs_loss_rate is not None:
                net_cfg.hbs_loss_rate = args.hbs_loss_rate
            net = spec.net.factory(net_cfg)
            net.load_state_dict(ckpt["state_dict"], strict=False)
            # 替换 hbsn 子网为 bce 旧架构（加载原始预训练权重）；默认冻结（形状先验固定）
            hcfg = HBSN_CFG[model]
            net.hbsn = build_bce_hbsn(
                hcfg["ckpt"], hcfg["cd"], hcfg["cu"], device
            )
            net.hbsn.requires_grad_(args.unfreeze_hbsn)
            net.train().to(device)
            dtype = torch_dtype(net_cfg)

            dataset = spec.dataset(dataset_cfg)
            train_dl, _ = dataset.get_dataloader(
                batch_size=args.batch_size, split_rate=1, drop_last=True
            )
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)

            print(f"\n===== finetune {model} ({name}) =====")
            for epoch in range(epochs):
                losses = []
                hbs_losses = []
                cons_losses = []
                sigma_lo, sigma_hi = (
                    (float(x) for x in args.noise_sigma_range.split(","))
                    if args.noise_sigma_range
                    else (noise_sigma, noise_sigma)
                )
                cons_lo = cons_hi = None
                if args.cons_noise_sigma_range:
                    cons_lo, cons_hi = (
                        float(x)
                        for x in args.cons_noise_sigma_range.split(",")
                    )
                cons_rate = args.hbs_consistency_rate
                if args.cons_warmup_epochs > 0:
                    cons_rate *= min(
                        1.0, (epoch + 1) / args.cons_warmup_epochs
                    )
                for img, mask in train_dl:
                    mask = mask.to(device, dtype=dtype)
                    clean = img.to(device, dtype=dtype)
                    batch_sigma = float(
                        np.random.uniform(sigma_lo, sigma_hi)
                    )
                    noisy = (
                        img + batch_sigma * torch.randn_like(img)
                    ).clamp(0, 1).to(device, dtype=dtype)
                    pred_noisy = net(noisy)
                    loss_dict, _ = net.loss(pred_noisy, mask)
                    loss = loss_dict["loss"]
                    if args.clean_sup_rate > 0:
                        # GT-锚定：pred_clean 带梯度前向并监督到 GT，其 HBS 成为正确形状目标
                        # （独立于一致性，rate=0 时也生效 → 公平的 wo-HBS 基线）
                        pred_clean = net(clean)
                        loss_clean_dict, _ = net.loss(pred_clean, mask)
                        loss = (
                            loss
                            + args.clean_sup_rate
                            * loss_clean_dict["loss"]
                        )
                    if args.hbs_consistency_rate > 0:
                        if args.clean_sup_rate == 0:
                            # 原实现：clean 前向只作 detached 形状目标，no_grad 省一半图显存
                            with torch.no_grad():
                                pred_clean = net(clean)
                        if cons_lo is not None:
                            # 双噪声：一致性用独立重噪声前向，mask 噪声保持原值
                            cons_sigma = float(
                                np.random.uniform(cons_lo, cons_hi)
                            )
                            noisy_heavy = (
                                img + cons_sigma * torch.randn_like(img)
                            ).clamp(0, 1).to(device, dtype=dtype)
                            pred_heavy = net(noisy_heavy)
                            cons = F.mse_loss(
                                pred_heavy[1], pred_clean[1].detach()
                            )
                            loss = loss + cons_rate * cons
                            cons_losses.append(cons.item())
                        elif (
                            args.cons_min_sigma is None
                            or batch_sigma >= args.cons_min_sigma
                        ):
                            cons = F.mse_loss(
                                pred_noisy[1], pred_clean[1].detach()
                            )
                            loss = loss + cons_rate * cons
                            cons_losses.append(cons.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    hbs_losses.append(loss_dict["hbs_loss"].item())
                cons_txt = (
                    f" cons={np.mean(cons_losses):.4f}"
                    if cons_losses
                    else ""
                )
                print(
                    f"  epoch {epoch + 1}/{epochs}: "
                    f"loss={np.mean(losses):.4f} "
                    f"hbs_loss={np.mean(hbs_losses):.4f}{cons_txt}"
                )

            # 保存（迁移格式：state_dict + config dict，供 eval 加载）
            out_path = os.path.join(
                OUT_DIR, f"{model}_{name}_noise_ft{args.out_suffix}.pth"
            )
            torch.save(
                {
                    "state_dict": net.cpu().state_dict(),
                    "config": {
                        "net": dict(net_cfg),
                        "dataset": dict(dataset_cfg),
                    },
                },
                out_path,
            )
        print(f"  保存: {out_path}")


if __name__ == "__main__":
    main()
