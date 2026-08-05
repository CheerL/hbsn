"""5 种模型 CPU 前向 + loss 冒烟：形状正确、loss 有限。"""
import pytest
import torch

from hbsn.config.schemas import (
    HBSNetSchema,
    MaskRcnnNetSchema,
    SegNetSchema,
    TpsnNetSchema,
)
from hbsn.nets.hbsn import HBSNet
from hbsn.nets.stn import STN
from hbsn.nets.tpsn import BCLossFunc, LAPLossFunc
from hbsn.registry import MODEL_REGISTRY

CASES = [
    ("hbsn", (2, 1, 256, 256), (2, 2, 128, 128), HBSNetSchema()),
    ("unetpp", (2, 3, 256, 256), (2, 1, 256, 256), SegNetSchema()),
    ("deeplab", (2, 3, 256, 256), (2, 1, 256, 256), SegNetSchema()),
    ("tpsn", (2, 3, 256, 256), (2, 1, 256, 256), TpsnNetSchema()),
    ("maskrcnn", (2, 3, 256, 256), (2, 1, 256, 256), MaskRcnnNetSchema()),
]


@pytest.mark.parametrize("model,img_shape,gt_shape,schema", CASES)
def test_forward_and_loss(model, img_shape, gt_shape, schema):
    torch.manual_seed(0)
    spec = MODEL_REGISTRY[model]
    net = spec.net.factory(schema)
    net.eval()

    img = torch.rand(img_shape)
    gt = torch.rand(gt_shape)
    with torch.no_grad():
        predict = net(img)
        loss_dict, output_data = net.loss(predict, gt)

    assert torch.isfinite(loss_dict["loss"]), f"{model}: non-finite loss"
    assert len(output_data) >= 2, f"{model}: output_data 结构异常"


# ------------------------------------------------------------- STN 模式分支


@pytest.mark.parametrize("stn_mode", [0, 1, 2])
def test_stn_forward_modes(stn_mode):
    torch.manual_seed(0)
    stn = STN(input_channels=1, height=32, width=32, stn_mode=stn_mode)
    x = torch.rand(2, 1, 32, 32)
    out, theta = stn(x)
    assert out.shape == x.shape
    # mode 0 → 全仿射 (B,2,3)；mode 1 → (B,1)；mode 2 → 旋转 (B,)
    expected_theta = (2, 2, 3) if stn_mode == 0 else (2, 1) if stn_mode == 1 else (2,)
    assert theta.shape == expected_theta


# ------------------------------------------------------------- HBSNet factory / loss


@pytest.mark.parametrize(
    "stn_mode,pre,post",
    [(0, False, False), (1, True, False), (2, False, True), (3, True, True)],
)
def test_hbsnet_factory_stn_modes(stn_mode, pre, post):
    cfg = HBSNetSchema(stn_mode=stn_mode)
    net = HBSNet.factory(cfg)
    assert (net.pre_stn is not None) == pre
    assert (net.post_stn is not None) == post


def test_hbsnet_loss_is_mask_false():
    torch.manual_seed(0)
    cfg = HBSNetSchema(stn_mode=0)  # 无 STN → 纯 hbs+grad loss
    net = HBSNet.factory(cfg)
    predict = torch.rand(2, 2, 128, 128)
    ground_truth = torch.rand(2, 2, 128, 128)
    loss_dict, output_data = net.loss(predict, ground_truth, is_mask=False)
    assert torch.isfinite(loss_dict["loss"])
    assert set(loss_dict) == {"loss", "hbs_loss", "stn_loss", "grad_loss"}
    assert len(output_data) == 2


# ------------------------------------------------------------- TPSN 损失组件


def test_tpsn_bc_lap_loss_finite():
    """平滑非共形映射（各向异性缩放）：QC 有限且 > 0。

    注意不能用纯随机映射——QC 损失对近奇异雅可比会爆炸（inf），
    这是该损失的真实属性，TPSN.loss 里有 inf/NaN 守卫。
    """
    h, w = 32, 32
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    u = 0.9 * x
    v = 0.8 * y  # 各向异性缩放 → 非共形 → Beltrami 系数 > 0 但有限
    mapping = torch.stack([u, v], dim=0).unsqueeze(0)
    qc = BCLossFunc([h, w])(mapping)
    lap = LAPLossFunc([h, w])(mapping)
    assert torch.isfinite(qc)
    assert torch.isfinite(lap)
    assert qc > 0
