#!/usr/bin/env python3
"""COCO 分割模型评估：单模型 F1/IoU 均值，或多模型逐样本对比 + 改进量分析。

单模型评估（用法见 --help 之外的示例）：
    python scripts/eval_coco.py --model unetpp \
        --checkpoint runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth

多模型逐样本对比（共享同一 dataloader，保证样本对齐；相邻两两一组 base→improved）：
    python scripts/eval_coco.py \
        --compare unetpp:runs/migrated/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth:unet \
        --compare unetpp:runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth:unet+hbsn \
        --compare deeplab:runs/migrated/deeplab/May15_11-02-15_hbs0_all_c/checkpoints/best.pth:deeplab \
        --compare deeplab:runs/migrated/deeplab/May23_21-02-42_hbs0.05_all/checkpoints/epoch_195.pth:deeplab+hbsn \
        --data-dir coco/val2017 --annotation-path coco/annotations/instances_val2017.json \
        --set single_instance=true --set connected=true \
        --top-n 10 --out-dir outputs/analysis

net/dataset 配置优先取 checkpoint 内存档，可用 --<key> 覆盖（如 --data-dir）。
对比模式即旧 test.ipynb 逐样本分析功能的固化：每模型逐样本 IoU、成对改进量
统计（mean/median/正改进比例）、top-N 改进样本可视化。
"""
import argparse
import glob
import os
import sys
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from hbsn.config.schemas import HBSNetSchema
from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec


def filter_known(schema_cls, cfg_dict: dict) -> dict:
    """迁移 ckpt 的 config 可能带旧字段（如 hbsn_version）——按 schema 字段过滤。"""
    known = {f.name for f in fields(schema_cls)}
    filtered = {}
    for key, value in cfg_dict.items():
        if key not in known:
            continue
        if key == "hbsn_config" and isinstance(value, dict):
            value = filter_known(HBSNetSchema, value)
        filtered[key] = value
    return filtered


def _build_configs(spec, ckpt_config: dict, overrides: dict, sets: list[str] | None):
    """从 checkpoint 存档 config + 命令行覆盖构建 (net_cfg, dataset_cfg)。

    overrides 为 {字段名: 值}（None 跳过），sets 为 "--set KEY=VALUE" 列表。
    迁移 ckpt 的 hbsn_checkpoint 路径指向本机可能不存在的旧文件，且 hbsn 子网
    权重不在指标路径上（指标只看 predict_mask）——统一清空，避免加载失败。
    """
    net_cfg = OmegaConf.merge(
        OmegaConf.structured(spec.net_schema),
        OmegaConf.create(filter_known(spec.net_schema, ckpt_config.get("net", {}))),
    )
    dataset_cfg = OmegaConf.merge(
        OmegaConf.structured(spec.dataset_schema),
        OmegaConf.create(filter_known(spec.dataset_schema, ckpt_config.get("dataset", {}))),
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

    if "hbsn_checkpoint" in net_cfg:
        net_cfg.hbsn_checkpoint = ""
    return net_cfg, dataset_cfg


def evaluate(model: str, checkpoint_path: str, sets: list[str] | None = None, **overrides) -> tuple[float, float]:
    spec = get_spec(model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {})

    net_cfg, dataset_cfg = _build_configs(spec, ckpt_config, overrides, sets)
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

            # 模型返回 (predict_mask, hbs[, others])——tpsn 等返回 3 元组
            predict_mask = net(img)[0]
            hard = net.get_hard_mask(predict_mask)
            f1, iou = net.get_metrics(hard, mask)
            results.append(torch.stack([f1, iou], dim=1).cpu().numpy())

    mean = np.concatenate(results, axis=0).mean(axis=0)
    return float(mean[0]), float(mean[1])


def _load_compare_nets(entries, sets, device: str):
    """加载对比模型列表，返回 [(label, net), ...]。device 强制覆盖存档值（旧 ckpt 可能存 cuda:2）。"""
    nets = []
    for model, ckpt_path, label in entries:
        spec = get_spec(model)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net_cfg, _ = _build_configs(spec, ckpt.get("config", {}), {"device": device}, sets)
        net = spec.net.factory(net_cfg)
        net.load_state_dict(ckpt["state_dict"], strict=False)
        net.eval()
        nets.append((label, net))
    return nets


def _save_improvement_scatter(improvements, out_dir):
    """改进量排序散点图（旧 test.ipynb cell 10 的固化）。"""
    order = np.argsort(-improvements)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(range(len(improvements)), improvements[order], alpha=0.7, s=8)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample index (sorted by improvement)")
    ax.set_ylabel("Improvement (sum over base→improved pairs)")
    ax.set_title("Per-sample IoU improvement of HBSN variant")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "improvement_scatter.png"), dpi=120)
    plt.close(fig)  # 防 matplotlib 全局 registry 累积


def _save_top_n_figure(top, labels, out_dir):
    """top-N 改进样本 6 宫格图：原图 + GT + 各模型预测（旧 test.ipynb cell 11 的固化）。"""
    n_rows = len(top)
    n_cols = 2 + len(labels)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = np.atleast_2d(axs)
    for r, (imp, idx, img, gt, preds) in enumerate(top):
        img = img.transpose(1, 2, 0)  # CHW → HWC
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axs[r, 0].imshow(img)
        axs[r, 0].set_title(f"#{idx} img")
        axs[r, 0].axis("off")
        axs[r, 1].imshow(gt, cmap="gray")
        axs[r, 1].set_title("GT")
        axs[r, 1].axis("off")
        for c, (label, pred) in enumerate(zip(labels, preds, strict=True)):
            axs[r, 2 + c].imshow(pred, cmap="gray")
            axs[r, 2 + c].set_title(f"{label}\nIoU+{imp:.3f}" if c % 2 == 1 else label)
            axs[r, 2 + c].axis("off")
    fig.suptitle("Top-N samples with largest HBSN improvement")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(out_dir, "top_n_improved.png"), dpi=100)
    plt.close(fig)


def compare_evaluate(
    entries: list[tuple[str, str, str]],
    top_n: int = 10,
    out_dir: str | None = None,
    data_dir: str | None = None,
    annotation_path: str | None = None,
    sets: list[str] | None = None,
    device: str = "cpu",
    max_samples: int | None = None,
) -> None:
    """多模型共享同一 dataloader 逐样本评估（旧 test.ipynb cell 5/8/10/11 的固化）。

    entries 相邻两两一组 (base, improved)，改进量 = improved - base 逐样本求和。
    内存有界：逐样本 IoU 全保留，预测图只在线维护 top-N 候选。
    device 默认 cpu：旧 ckpt 存档可能是 cuda:2 之类已失效的设备。
    """
    first_spec = get_spec(entries[0][0])
    first_ckpt = torch.load(entries[0][1], map_location="cpu", weights_only=False)
    _, dataset_cfg = _build_configs(
        first_spec,
        first_ckpt.get("config", {}),
        {"data_dir": data_dir, "annotation_path": annotation_path},
        sets,
    )
    dataset = first_spec.dataset(dataset_cfg)
    _, dataloader = dataset.get_dataloader(batch_size=1, split_rate=0, drop_last=True)
    num_samples = len(dataloader)
    if max_samples is not None:
        num_samples = min(max_samples, num_samples)  # 冒烟/快速验证用

    nets = _load_compare_nets(entries, sets, device)
    n_models = len(nets)
    device = nets[0][1].config.device
    dtype = torch_dtype(nets[0][1].config)
    labels = [label for label, _ in nets]

    all_ious = np.zeros((n_models, num_samples))
    improvements = np.zeros(num_samples)
    top: list = []  # (improvement, idx, img, gt, preds)，在线维护 top-N

    with torch.no_grad():
        for i, (img, mask) in enumerate(dataloader):
            if i >= num_samples:
                break
            img = img.to(device, dtype=dtype)
            mask = mask.to(device, dtype=dtype)
            sample_ious = np.zeros(n_models)
            sample_preds = []
            for j, (_, net) in enumerate(nets):
                predict_mask = net(img)[0]
                hard = net.get_hard_mask(predict_mask)
                _, iou = net.get_metrics(hard, mask)
                sample_ious[j] = iou.mean().item()
                sample_preds.append(hard[0, 0].float().cpu().numpy())
            all_ious[:, i] = sample_ious
            imp = sum(
                sample_ious[2 * k + 1] - sample_ious[2 * k]
                for k in range(n_models // 2)
            )
            improvements[i] = imp
            if len(top) < top_n or imp > top[-1][0]:
                top.append((imp, i, img[0].float().cpu().numpy(), mask[0, 0].float().cpu().numpy(), sample_preds))
                top.sort(key=lambda t: t[0], reverse=True)
                top = top[:top_n]

    print("=== 逐模型 IoU（mean / median）===")
    for j, label in enumerate(labels):
        print(f"  {label:24s} mean {all_ious[j].mean():.4f}  median {np.median(all_ious[j]):.4f}")
    if n_models >= 2:
        print(
            f"\n改进量（成对 base→improved 求和）: mean {improvements.mean():.4f}, "
            f"median {np.median(improvements):.4f}, "
            f"正改进比例 {(improvements > 0).mean() * 100:.1f}%"
        )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _save_improvement_scatter(improvements, out_dir)
        _save_top_n_figure(top, labels, out_dir)
        print(f"\n可视化已保存到 {out_dir}/")


def _parse_compare_entries(compare_args: list[str]) -> list[tuple[str, str, str]]:
    """解析 --compare MODEL:CKPT[:LABEL]，支持 checkpoint glob 展开。"""
    entries = []
    for item in compare_args:
        parts = item.split(":")
        model, ckpt = parts[0], parts[1]
        base_label = parts[2] if len(parts) > 2 else model
        paths = sorted(glob.glob(ckpt))
        for path in paths:
            label = base_label if len(paths) == 1 else f"{base_label}:{os.path.basename(path)}"
            entries.append((model, path, label))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="模型名（单模型评估模式）")
    parser.add_argument("--checkpoint", help="checkpoint 路径或 glob（单模型评估模式）")
    parser.add_argument("--data-dir", help="覆盖 dataset.data_dir")
    parser.add_argument("--annotation-path", help="覆盖 dataset.annotation_path")
    parser.add_argument("--device", help="覆盖 net.device（旧 ckpt 可能无 device 或指向不存在的 GPU）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="任意配置覆盖，可重复（如 --set connected=true --set cat_ids=[16]）")
    parser.add_argument("--compare", action="append", default=[], metavar="MODEL:CKPT[:LABEL]",
                        help="多模型对比模式：可重复，相邻两两一组 (base, improved)；支持 glob")
    parser.add_argument("--top-n", type=int, default=10, help="对比模式 top-N 改进样本数")
    parser.add_argument("--out-dir", default=None, help="对比模式可视化输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="对比模式仅评估前 N 个样本（冒烟/快速验证）")
    args = parser.parse_args()

    if args.compare:
        entries = _parse_compare_entries(args.compare)
        if not entries:
            print("未找到匹配的 checkpoint", file=sys.stderr)
            return 1
        if len(entries) % 2 != 0:
            print("--compare 必须成对提供（base, improved）", file=sys.stderr)
            return 1
        compare_evaluate(
            entries,
            top_n=args.top_n,
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            annotation_path=args.annotation_path,
            sets=args.set,
            device=args.device or "cpu",
            max_samples=args.max_samples,
        )
        return 0

    if not args.model or not args.checkpoint:
        parser.error("单模型模式需要 --model 与 --checkpoint；多模型对比用 --compare")

    for path in sorted(glob.glob(args.checkpoint)):
        f1, iou = evaluate(
            args.model, path, sets=args.set,
            data_dir=args.data_dir, annotation_path=args.annotation_path,
            device=args.device,
        )
        print(f"{path} [F1 {f1:.6f} | IoU {iou:.6f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
