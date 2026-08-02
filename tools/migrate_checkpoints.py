#!/usr/bin/env python3
"""旧 checkpoint（pre-rewrite，pickle 旧 Config）→ 新格式迁移。

新格式：config 键为纯 dict（net/dataset/recorder/run 四节），无类路径依赖；
optimizer 状态丢弃（重写后参数顺序变化，续训语义不可保）。

键映射规则（按类型）：
- hbsn 独立 ckpt：pre_stn.*/post_stn.*/backbone.* 全部保留（新 HBSNet 同名属性）
- seg 类 ckpt（deeplab/unetpp/tpsn）：hbsn.* 键丢弃（旧运行时 _handle_checkpoint
  即丢弃，hbsn 子网来自 net.hbsn_checkpoint 指向的独立 ckpt）；其余 model.*/
  mask_conv.*/weight_layer.*/lap_loss.weight 保留（smp/torchvision 已钉版）

原文件只读不动；输出到 <--out-dir>/<type>/<原目录名>/<原文件名>。

用法：python tools/migrate_checkpoints.py "runs/*/*/checkpoints/*.pth"
"""
import argparse
import glob
import os
import sys

import torch

from hbsn._legacy import pickle_shim


def detect_type(state_dict: dict, config) -> str:
    """按 state_dict 键集 + config 属性推断旧 checkpoint 类型。

    config 可能是完整 Config（属性在 net_config 上）或裸 net config。
    """
    net_cfg = getattr(config, "net_config", None) or config
    if any(k.startswith("hbsn.") for k in state_dict):
        # seg 类：config 区分 tpsn/maskrcnn，键集区分 deeplab/unetpp
        if hasattr(net_cfg, "qc_loss_rate"):
            return "tpsn"
        if hasattr(net_cfg, "select_num"):
            return "maskrcnn"
        if any(k.startswith("model.encoder.") for k in state_dict):
            return "unetpp"
        if any(k.startswith("model.backbone.") for k in state_dict):
            return "deeplab"
        raise ValueError("无法识别 seg 类型（无 model.encoder./model.backbone. 键）")
    if any(k.startswith(("pre_stn.", "post_stn.", "backbone.")) for k in state_dict):
        return "hbsn"
    raise ValueError("无法识别 checkpoint 类型（无 hbsn./backbone. 等特征键）")


def map_state_dict(state_dict: dict, type_: str) -> tuple[dict, list[str]]:
    kept, dropped = {}, []
    for key, value in state_dict.items():
        if type_ != "hbsn" and key.startswith("hbsn."):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, dropped


def obj_to_dict(obj):
    """旧 Config 对象 → 纯 dict（torch dtype → 名称；嵌套对象递归展开）。"""
    if isinstance(obj, torch.dtype):
        return str(obj).split(".")[-1]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [obj_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {
            k: obj_to_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


def rewrite_hbsn_checkpoint_path(config_dict: dict, out_dir: str) -> dict:
    """hbsn_checkpoint 指向迁移后的独立 hbsn ckpt（runs/hbsn/... → runs/migrated/hbsn/...）。"""
    net = config_dict.get("net", {})
    path = net.get("hbsn_checkpoint", "")
    if path and "runs/hbsn/" in path:
        net["hbsn_checkpoint"] = path.replace("runs/hbsn/", os.path.join(out_dir, "hbsn") + "/")
    return config_dict


# 更早布局（May17 时代）的扁平 config：net+dataset+run 字段混在一个类里。
# dict 值 = 新结构里的字段名（改名），恒等映射也显式写出。
FLAT_DATASET_KEYS = {
    "coco_annotation": "annotation_path",
    "coco_root": "data_dir",
    "augment_rotation": "augment_rotation",
    "augment_scale": "augment_scale",
    "augment_translate": "augment_translate",
    "cat_ids": "cat_ids",
    "img_ids": "img_ids",
    "resize_rate": "resize_rate",
    "min_area": "min_area",
    "connected": "connected",
    "single_instance": "single_instance",
    "is_augment": "is_augment",
}
FLAT_RUN_KEYS = {
    "batch_size", "lr", "lr_decay_rate", "lr_decay_steps", "moments",
    "weight_norm", "total_epoches", "version", "checkpoint_path",
    "checkpoint_dir", "load",
}
FLAT_RECORDER_KEYS = {"log_dir", "log_base_dir", "comment"}


def flat_config_to_sections(obj) -> dict:
    """扁平旧 config → 新四段结构（net/dataset/recorder/run）；未识别字段归 net。"""
    sections = {"net": {}, "dataset": {}, "recorder": {}, "run": {}}
    for key, value in vars(obj).items():
        if key.startswith("_"):
            continue
        value = obj_to_dict(value)
        if key in FLAT_DATASET_KEYS:
            target, key = "dataset", FLAT_DATASET_KEYS[key]
        elif key in FLAT_RUN_KEYS:
            target = "run"
        elif key in FLAT_RECORDER_KEYS:
            target = "recorder"
        else:
            target = "net"
        if key == "data_dir" and isinstance(value, str):
            value = value.rstrip("/")
        sections[target][key] = value
    return sections


def migrate_one(src_path: str, out_dir: str, dry_run: bool = False) -> str | None:
    pickle_shim.install()
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    old_config = ckpt.get("config")

    if isinstance(old_config, dict) and "net" in old_config:
        print(f"跳过（已是新格式）: {src_path}")
        return None

    type_ = detect_type(ckpt["state_dict"], old_config)
    state_dict, dropped = map_state_dict(ckpt["state_dict"], type_)

    # 输出目录：<out_dir>/<type>/<运行目录名>/checkpoints/<原文件名>（镜像原结构，
    # hbsn_checkpoint 路径重写依赖此层级）
    rel_dir = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
    out_path = os.path.join(out_dir, type_, rel_dir, "checkpoints", os.path.basename(src_path))

    if dry_run:
        print(f"[dry-run] {src_path} -> {out_path} ({type_}, "
              f"{len(state_dict)} 键保留, {len(dropped)} 键丢弃)")
        return out_path

    if old_config and hasattr(old_config, "net_config"):
        new_config = {
            "net": obj_to_dict(old_config.net_config),
            "dataset": obj_to_dict(old_config.dataset_config),
            "recorder": obj_to_dict(old_config.recorder_config),
            "run": obj_to_dict(old_config.run_config),
        }
    elif old_config and hasattr(old_config, "__dict__"):
        new_config = flat_config_to_sections(old_config)  # 扁平旧 config（May17 时代）
    else:  # 无 config 存档的旧 ckpt（save(config={}) 默认值）
        new_config = {"net": {}, "dataset": {}, "recorder": {}, "run": {}}
    new_config["legacy"] = {"type": type_, "source": src_path}
    new_config = rewrite_hbsn_checkpoint_path(new_config, out_dir)

    new_ckpt = {
        "state_dict": state_dict,
        "epoch": ckpt.get("epoch", -1),
        "best_epoch": ckpt.get("best_epoch", -1),
        "best_loss": ckpt.get("best_loss", float("inf")),
        "config": new_config,
        "optimizer": None,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(new_ckpt, out_path)
    dropped_note = f", 丢弃 {len(dropped)} 个 hbsn.* 键" if dropped else ""
    print(f"{src_path} -> {out_path} ({type_}{dropped_note})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", help="旧 checkpoint 路径或 glob")
    parser.add_argument("--out-dir", default="runs/migrated")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.checkpoints))
    if not paths:
        print(f"未找到匹配的 checkpoint: {args.checkpoints}")
        return 1
    print(f"发现 {len(paths)} 个 checkpoint，输出到 {args.out_dir}")

    if not args.dry_run:
        # 原文件只读不动；请自行确保已有备份（如 tar 打包）——本工具不复制
        print("提示：迁移不修改原文件；请确认已备份（tar 打包）")

    for path in paths:
        migrate_one(path, args.out_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
