"""5 种模型 CPU 前向 + loss 冒烟：形状正确、loss 有限。"""
import pytest
import torch

from hbsn.config.schemas import (
    HBSNetSchema,
    MaskRcnnNetSchema,
    SegNetSchema,
    TpsnNetSchema,
)
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
