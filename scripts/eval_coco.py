#!/usr/bin/env python3
"""COCO 分割模型评估：加载 checkpoint → val2017 上输出 F1/IoU 均值。

用法（仓库根目录）：
    python scripts/eval_coco.py --model unetpp \
        --checkpoint runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth

net/dataset 配置优先取 checkpoint 内存档（含 hbsn_checkpoint 路径），
可用 --<key> 覆盖，如 --data-dir coco/val2017。
"""
import argparse
import glob
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec


def evaluate(model: str, checkpoint_path: str, sets: list[str] | None = None, **overrides) -> tuple[float, float]:
    spec = get_spec(model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {})

    net_cfg = OmegaConf.merge(
        OmegaConf.structured(spec.net_schema), OmegaConf.create(ckpt_config.get("net", {}))
    )
    dataset_cfg = OmegaConf.merge(
        OmegaConf.structured(spec.dataset_schema),
        OmegaConf.create(ckpt_config.get("dataset", {})),
    )
    for key, value in overrides.items():
        if value is not None and key in net_cfg:
            net_cfg[key] = value
        elif value is not None and key in dataset_cfg:
            dataset_cfg[key] = value
        elif value is not None:
            print(f"warning: 未知覆盖 key {key}，忽略", file=sys.stderr)
    for kv in sets or []:
        key, _, value = kv.partition("=")
        value = value.strip().lower()
        if value in ("true", "false"):
            value = value == "true"
        elif value.isdigit():
            value = int(value)
        if key in net_cfg:
            net_cfg[key] = value
        elif key in dataset_cfg:
            dataset_cfg[key] = value
        else:
            print(f"warning: 未知覆盖 key {key}，忽略", file=sys.stderr)

    # hbsn 子网权重不在指标路径上（原 test_full 亦如此：hbsn_checkpoint 注释、
    # 随机 hbsn），且迁移后的 checkpoint 路径可能指向本机不存在的旧文件 → 清空
    if "hbsn_checkpoint" in net_cfg:
        net_cfg.hbsn_checkpoint = ""

    net = spec.net.factory(net_cfg)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()

    dataset = spec.dataset(dataset_cfg)
    _, dataloader = dataset.get_dataloader(batch_size=1, split_rate=0, drop_last=True)

    results = []
    with torch.no_grad():
        for img, mask in dataloader:
            img = img.to(net.config.device, dtype=torch_dtype(net.config))
            mask = mask.to(net.config.device, dtype=torch_dtype(net.config))

            predict_mask, _ = net(img)
            hard = net.get_hard_mask(predict_mask)
            f1, iou = net.get_metrics(hard, mask)
            results.append(torch.stack([f1, iou], dim=1).cpu().numpy())

    mean = np.concatenate(results, axis=0).mean(axis=0)
    return float(mean[0]), float(mean[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["deeplab", "unetpp", "tpsn", "maskrcnn", "hbsn"])
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径或 glob（含 runs/migrated/...）")
    parser.add_argument("--data-dir", help="覆盖 dataset.data_dir")
    parser.add_argument("--annotation-path", help="覆盖 dataset.annotation_path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="任意配置覆盖，可重复（如 --set connected=true --set cat_ids=[16]）")
    args = parser.parse_args()

    for path in sorted(glob.glob(args.checkpoint)):
        f1, iou = evaluate(
            args.model, path, sets=args.set,
            data_dir=args.data_dir, annotation_path=args.annotation_path,
        )
        print(f"{path} [F1 {f1:.6f} | IoU {iou:.6f}]")


if __name__ == "__main__":
    main()
