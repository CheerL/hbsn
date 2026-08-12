#!/usr/bin/env python3
"""本地全部 checkpoint 加载/推理冒烟测试 + migrated 前后 state_dict 一致性。

对每个本地 ckpt：
- hbsn：nets.build_net（或 legacy.build_old 若 bce 旧架构），dummy 256×256 推理
- 分割（deeplab/unetpp/tpsn）：eval_coco._build_configs + factory + dummy 推理
报告加载缺失键数、推理输出 shape、输出 hash；migrated 前后（原文件 vs migrated）
对比 state_dict 权重一致性（迁移工具应保权重 → 推理必然一致）。

用法：uv run python scripts/exp/test_local_ckpts.py
"""

import glob
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # scripts/（eval_coco 所在）

from eval_coco import _build_configs
from hbsn_exp import legacy

from hbsn._legacy import pickle_shim
from hbsn.registry import get_spec

pickle_shim.install()  # 注册旧 Config 反序列化桩（原文件可加载）


def obj_to_dict(obj):
    """旧 Config 占位对象 → 纯 dict（与 migrate_checkpoints.obj_to_dict 同逻辑）。"""
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


def config_to_dict(cfg):
    """ckpt config → 新格式 dict（旧对象按 migrate 四段展开，扁平归 net）。"""
    if isinstance(cfg, dict):
        return cfg
    if cfg is None:
        return {}
    if hasattr(cfg, "net_config"):
        return {
            "net": obj_to_dict(cfg.net_config),
            "dataset": obj_to_dict(cfg.dataset_config),
            "recorder": obj_to_dict(cfg.recorder_config),
            "run": obj_to_dict(cfg.run_config),
        }
    return obj_to_dict(cfg)


def out_hash(t):
    return hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()[:10]


def sd_hash(sd):
    h = hashlib.md5()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def try_load_infer(path, device="cpu"):
    """加载并推理一个 ckpt。返回 (type, stn_mode, missing, out_shape, hash)。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    top = {k.split(".")[0] for k in sd}

    if "bce" in top:
        net, mode = legacy.build_old(path, device=device)
        x = torch.randn(1, 1, 256, 256).to(device)
    else:
        if "backbone" in top:
            name = "hbsn"
        elif "model" in top or "mask_conv" in top or "lap_loss" in top:
            # 分割：lap_loss=tpsn 特有键；encoder=unetpp；mask_conv/backbone=deeplab
            if "lap_loss" in top:
                name = "tpsn"
            elif any("encoder" in k for k in sd):
                name = "unetpp"
            elif "mask_conv" in top or any("backbone" in k for k in sd):
                name = "deeplab"
            else:
                return None, None, None, None, None
        else:
            return None, None, None, None, None
        spec = get_spec(name)
        net_cfg, _ = _build_configs(
            spec, config_to_dict(ck.get("config", {})), {"device": device}, None
        )
        net = spec.net.factory(net_cfg)
        missing, _unexpected = net.load_state_dict(sd, strict=False)
        net.eval().to(device)
        h, w = net_cfg.height, net_cfg.width
        x = torch.randn(1, net_cfg.input_channels, h, w).to(device)
        mode = getattr(net_cfg, "stn_mode", None)
    with torch.no_grad():
        out = net(x)
    if isinstance(out, (tuple, list)):  # 分割网络 forward 返回 (mask, ...)
        out = out[0]
    return name, mode, len(missing), tuple(out.shape), out_hash(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    ckpts = sorted(glob.glob("runs/hbsn/**/best.pth", recursive=True))
    ckpts += sorted(glob.glob("runs/migrated/hbsn/**/best.pth", recursive=True))
    for sub in ("deeplab", "unetpp", "tpsn"):
        ckpts += sorted(glob.glob(f"runs/{sub}/**/best.pth", recursive=True))
        ckpts += sorted(
            glob.glob(f"runs/migrated/{sub}/**/best.pth", recursive=True)
        )

    print(f"{'path':<74}{'type':<9}{'stn':>4}{'miss':>5}  {'out':<18}{'hash'}")
    results = {}
    for p in ckpts:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            results[p] = sd_hash(ck["state_dict"])
        except Exception as e:
            print(f"{p:<74}LOAD-FAIL: {type(e).__name__}")
            continue
        try:
            name, mode, miss, shape, h = try_load_infer(p, device)
            if name is None:
                print(f"{p:<74}{'?':<9}{'':>4}{'':>5}  SKIP")
                continue
            print(f"{p:<74}{name:<9}{mode!s:>4}{miss:>5}  {shape!s:<18}{h}")
        except Exception as e:
            print(f"{p:<74}FAIL: {type(e).__name__}: {str(e)[:50]}")

    # ---- migrated 前后一致性：state_dict 权重对比（按实验目录配对） ----
    print("\n===== migrated 前后 state_dict 一致性 =====")
    pairs = {}
    for p, h in results.items():
        rel = p.replace("runs/migrated/", "").replace("runs/", "")
        exp = "/".join(rel.split("/")[:2])  # <type>/<exp_name>
        pairs.setdefault(exp, {})[p] = h
    matched = 0
    for exp, variants in sorted(pairs.items()):
        if len(variants) >= 2:
            hashes = set(variants.values())
            same = "✅ 一致" if len(hashes) == 1 else f"❌ 不同 {variants}"
            print(f"  {exp:<56} {same}")
            matched += 1
    if matched == 0:
        print(
            "  （无成对原文件+migrated——本地原文件多为 UNMIGRATED，需先迁移）"
        )


if __name__ == "__main__":
    main()
