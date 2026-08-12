"""HBSN 模型加载与推理（复用 hbsn 包，零修改主干）。

输入：256×256 灰度图（黑底白形状，[0,255]）
输出：(2,128,128) HBS 场（Beltrami 系数实/虚），定义在 GHBS 像素网格。
"""

import os
from dataclasses import fields

import torch
from omegaconf import OmegaConf

from hbsn.config.schemas import HBSNetSchema
from hbsn.nets.base import torch_dtype
from hbsn.registry import get_spec

# 论文最优模型（runs/migrated 下迁移后的 dict 格式 config，best_loss≈0.0051）
# 绝对路径：脚本可能在任意 cwd 下运行（E1-E4 各自独立）。
_REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
BEST_CKPT = os.path.join(
    _REPO, "runs/migrated/hbsn/Jun11_13-01-53_big_ns/checkpoints/best.pth"
)
BEST_CKPT_STN3 = BEST_CKPT  # 带 PoseAligner + RotationNormalizer（stn_mode=3）


def _filter_known(cfg_dict: dict) -> dict:
    """迁移 ckpt 的 config 带旧字段（如 output_height）——按 schema 字段过滤。"""
    known = {f.name for f in fields(HBSNetSchema)}
    return {k: v for k, v in cfg_dict.items() if k in known}


def build_net(checkpoint_path, device="cpu"):
    """从 checkpoint 构建 HBSNet（config 取存档 net 节，过滤旧字段，device 强制覆盖）。"""
    spec = get_spec("hbsn")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net_cfg = OmegaConf.merge(
        OmegaConf.structured(spec.net_schema),
        OmegaConf.create(_filter_known(ck.get("config", {}).get("net", {}))),
    )
    net_cfg.device = device
    net = spec.net.factory(net_cfg)
    net.load_state_dict(ck["state_dict"], strict=False)
    net.eval()
    return net


def infer(net, gray_img, device="cpu"):
    """gray_img: (256,256) uint8/float [0,255] → (2,128,128) HBS 场。"""
    if gray_img.dtype != torch.float32:
        gray_img = torch.as_tensor(gray_img, dtype=torch.float32)
    x = (gray_img / 255.0).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
    x = x.to(device, dtype=torch_dtype(net.config))
    with torch.no_grad():
        predict = net(x)[0]  # (1,2,128,128) → (2,128,128)
    return predict.cpu().numpy()  # (2,128,128)


def field_to_complex(hbs_np):
    """(2,128,128) HBS 场 → (128,128) 复数场。"""
    return hbs_np[0] + 1j * hbs_np[1]


def no_rn(net):
    """E3 无 RotationNormalizer 变体：推理时禁用 post_stn（不重训）。"""
    net.post_stn = None
    return net
