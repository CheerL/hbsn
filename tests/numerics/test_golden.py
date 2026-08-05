"""黄金数值测试：固定 seed 的模型前向/loss 输出与提交的 golden 逐元素一致。

用途：**等价重构的回归闸门**——重构后跑本文件，任何输出漂移都说明重构改变了
数值行为（即非等价）。golden 由 `_generate_golden.py` 一次性生成
（tests/numerics/golden/*.npy + *_loss.json），提交入库。

若重构**有意**改变数值：重跑生成器覆盖 golden，并人工 diff 新旧值确认合理。

覆盖确定性核心路径：HBSNet（stn_mode 0/3）、STN 三模式、UNet、UNet2D、QC/LAP 损失。
分割模型（deeplab/unetpp/maskrcnn）依赖预训练权重，由 test_nets 冒烟覆盖，不入 golden。
"""
import json
import os

import numpy as np
import pytest
import torch

from hbsn.config.schemas import HBSNetSchema
from hbsn.nets.hbsn import HBSNet
from hbsn.nets.stn import STN
from hbsn.nets.tpsn import BCLossFunc, LAPLossFunc
from hbsn.nets.unet import UNet, UNet2D

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
# 与 _generate_golden.py 完全同序：manual_seed(0) → build → x=rand → forward → gt=rand
SEED = 0


def _builders() -> dict:
    return {
        "hbsn_stn3": (
            lambda: HBSNet.factory(HBSNetSchema(stn_mode=3)),
            (2, 1, 256, 256),
            (2, 2, 128, 128),
        ),
        "hbsn_stn0": (
            lambda: HBSNet.factory(HBSNetSchema(stn_mode=0)),
            (2, 1, 256, 256),
            None,
        ),
        "stn_mode0": (
            lambda: STN(input_channels=1, height=32, width=32, stn_mode=0),
            (2, 1, 32, 32),
            None,
        ),
        "stn_mode1": (
            lambda: STN(input_channels=1, height=32, width=32, stn_mode=1),
            (2, 1, 32, 32),
            None,
        ),
        "stn_mode2": (
            lambda: STN(input_channels=1, height=32, width=32, stn_mode=2),
            (2, 1, 32, 32),
            None,
        ),
        "unet": (
            lambda: UNet(
                n_channels=1, n_classes=2,
                channels_down=[4, 8, 16], channels_up=[8, 16],
                is_bilinear=True, is_skip=True,
            ),
            (2, 1, 64, 64),
            None,
        ),
        "unet2d": (
            lambda: UNet2D(n_input=3, n_output=2, n_feature=4, depth_down=2, depth_hidden=1),
            (2, 3, 64, 64),
            None,
        ),
    }


def _loss_module_cases() -> dict:
    """QC/LAP 损失：平滑非共形映射（各向异性缩放），与生成器一致。"""
    h, w = 32, 32
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    mapping = torch.stack([0.9 * x, 0.8 * y], dim=0).unsqueeze(0)
    return {"bcloss": (BCLossFunc([h, w]), mapping), "laploss": (LAPLossFunc([h, w]), mapping)}


def _load_golden(name, suffix):
    path = os.path.join(GOLDEN_DIR, f"{name}_{suffix}")
    assert os.path.exists(path), f"golden 缺失，请先跑 tests/numerics/_generate_golden.py: {path}"
    return path


@pytest.mark.parametrize("name", sorted(_builders()))
def test_golden_forward(name):
    build, input_shape, _ = _builders()[name]
    golden = np.load(_load_golden(name, "output.npy"))

    torch.manual_seed(SEED)
    net = build()
    net.eval()
    x = torch.rand(*input_shape)
    with torch.no_grad():
        out = net(x)
        if isinstance(out, tuple):
            out = out[0]  # STN 返回 (x, theta)
    actual = out.detach().float().cpu()
    assert torch.allclose(
        actual, torch.from_numpy(golden), atol=1e-5, rtol=1e-5
    ), f"{name} 前向输出漂移——重构改变了数值"


def test_golden_loss():
    """HBSNet(stn3) 全损失项逐项与 golden 比对（唯一带 loss 的确定性网络案例）。"""
    name = "hbsn_stn3"
    build, input_shape, gt_shape = _builders()[name]
    with open(_load_golden(name, "loss.json")) as f:
        golden_losses = json.load(f)

    torch.manual_seed(SEED)
    net = build()
    net.eval()
    x = torch.rand(*input_shape)
    gt = torch.rand(*gt_shape)
    with torch.no_grad():
        out = net(x)
        loss_dict, _ = net.loss(out, gt)
    for key, expected in golden_losses.items():
        assert loss_dict[key] == pytest.approx(expected, rel=1e-5), (
            f"{name} loss[{key}] 漂移：{loss_dict[key]:.6f} vs golden {expected:.6f}"
        )


@pytest.mark.parametrize("name", sorted(_loss_module_cases()))
def test_golden_loss_module(name):
    module, mapping = _loss_module_cases()[name]
    with open(_load_golden(name, "loss.json")) as f:
        golden = json.load(f)[name]
    with torch.no_grad():
        value = module(mapping).item()
    assert value == pytest.approx(golden, rel=1e-5), (
        f"{name} 漂移：{value:.6g} vs golden {golden:.6g}"
    )
