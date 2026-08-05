"""生成黄金数值基准（确定性模型前向/loss 输出），运行一次后提交 golden/*.npy。

用法（仓库根）：
    uv run --extra dev python tests/numerics/_generate_golden.py

生成物：
    golden/<case>_output.npy   前向输出张量（CPU float32）
    golden/<case>_loss.json    前向的标量损失（若有）

等价重构的验收标准：重构前后跑 test_golden.py 全部通过（输出逐元素一致）。
若重构是有意改变数值，需人工重跑本脚本并确认新旧 golden diff 合理。
"""
import json
import os

import numpy as np
import torch

from hbsn.config.schemas import HBSNetSchema
from hbsn.nets.hbsn import HBSNet
from hbsn.nets.stn import STN
from hbsn.nets.tpsn import BCLossFunc, LAPLossFunc
from hbsn.nets.unet import UNet, UNet2D

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _run_case(name, build, input_shape, loss=False, gt_shape=None):
    torch.manual_seed(0)  # 权重初始化 + 输入生成共用同一种子（numpy 未参与，不需 np.seed）
    net = build()
    net.eval()  # BN 用 running stats，保证确定性
    x = torch.rand(*input_shape)
    with torch.no_grad():
        out = net(x)
        if isinstance(out, tuple):
            out = out[0]  # STN.forward 返回 (x, theta)
        out = out.detach().float().cpu().numpy()
    np.save(os.path.join(GOLDEN_DIR, f"{name}_output.npy"), out)

    if loss:
        gt = torch.rand(*gt_shape)
        with torch.no_grad():
            loss_dict, _ = net.loss(torch.from_numpy(out), gt)
        losses = {k: float(v.item()) for k, v in loss_dict.items()}
        with open(os.path.join(GOLDEN_DIR, f"{name}_loss.json"), "w") as f:
            json.dump(losses, f, indent=2, sort_keys=True)
        print(f"{name}: 输出 {out.shape} + loss {losses}")
    else:
        print(f"{name}: 输出 {out.shape}")


def _run_loss_module(name, module, mapping):
    with torch.no_grad():
        value = float(module(mapping).item())
    with open(os.path.join(GOLDEN_DIR, f"{name}_loss.json"), "w") as f:
        json.dump({name: value}, f, indent=2, sort_keys=True)
    print(f"{name}: {value}")


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    # 确定性核心路径：HBSNet（pre/post STN 组合）、STN 三模式、UNet/UNet2D、QC/LAP 损失
    _run_case(
        "hbsn_stn3", lambda: HBSNet.factory(HBSNetSchema(stn_mode=3)),
        (2, 1, 256, 256), loss=True, gt_shape=(2, 2, 128, 128),
    )
    _run_case(
        "hbsn_stn0", lambda: HBSNet.factory(HBSNetSchema(stn_mode=0)),
        (2, 1, 256, 256),
    )
    for mode in (0, 1, 2):
        _run_case(
            f"stn_mode{mode}",
            lambda m=mode: STN(input_channels=1, height=32, width=32, stn_mode=m),
            (2, 1, 32, 32),
        )
    _run_case(
        "unet",
        lambda: UNet(
            n_channels=1, n_classes=2,
            channels_down=[4, 8, 16], channels_up=[8, 16],
            is_bilinear=True, is_skip=True,
        ),
        (2, 1, 64, 64),
    )
    _run_case(
        "unet2d",
        lambda: UNet2D(n_input=3, n_output=2, n_feature=4, depth_down=2, depth_hidden=1),
        (2, 3, 64, 64),
    )

    # QC/LAP 损失：平滑非共形映射（各向异性缩放），不能用纯随机——QC 对近奇异雅可比会爆
    h, w = 32, 32
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    mapping = torch.stack([0.9 * x, 0.8 * y], dim=0).unsqueeze(0)
    _run_loss_module("bcloss", BCLossFunc([h, w]), mapping)
    _run_loss_module("laploss", LAPLossFunc([h, w]), mapping)

    print(f"\ngolden 已写入 {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
