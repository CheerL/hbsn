"""BaseNet 通用逻辑：torch_dtype/output_size 辅助 + save/load/initialize/参数分组。"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hbsn.nets.base import BaseNet, output_size, torch_dtype


class DummyNet(BaseNet):
    """最小 BaseNet 子类：一个普通 conv + 一个 fixable conv。"""

    def __init__(self, config):
        super().__init__(config)
        self.conv = nn.Conv2d(1, 1, kernel_size=1)
        self.fix = nn.Conv2d(1, 1, kernel_size=1)

    @property
    def fixable_layers(self):
        return self.fix

    @property
    def uninitializable_layers(self):
        return nn.Module()

    def forward(self, img):
        return img

    def loss(self, predict, ground_truth):
        return {"loss": torch.tensor(0.0)}, (predict,)


def make_cfg(is_freeze=False):
    return SimpleNamespace(
        device="cpu",
        dtype="float32",
        height=256,
        width=256,
        input_channels=1,
        output_channels=2,
        load_strict=True,
        is_freeze=is_freeze,
        finetune_rate=0.1,
        channels_up=[8, 16],
        channels_down=[8, 16, 32],
    )


def test_torch_dtype():
    assert torch_dtype(SimpleNamespace(dtype="float32")) is torch.float32
    assert torch_dtype(SimpleNamespace(dtype="float64")) is torch.float64


def test_output_size():
    # channels_up 比 channels_down 少 1 层 → 降采样 1/2 → 256→128
    assert output_size(make_cfg()) == (128, 128)


def test_save_load_roundtrip(tmp_path):
    net = DummyNet(make_cfg())
    path = tmp_path / "ckpt.pth"
    net.save(
        str(path),
        epoch=1,
        best_epoch=0,
        best_loss=0.5,
        config={"x": 1},
        optimizer=None,
    )

    net2 = DummyNet(make_cfg())
    epoch, best_epoch, best_loss, optimizer = net2.load(str(path))
    assert epoch == 1
    assert best_epoch == 0
    assert best_loss == 0.5
    assert optimizer is None
    # 权重往返一致
    assert torch.equal(net.conv.weight, net2.conv.weight)


def test_load_model_static(tmp_path):
    net = DummyNet(make_cfg())
    path = tmp_path / "ckpt.pth"
    net.save(str(path), epoch=2, best_epoch=1, best_loss=0.3)
    ckpt, _, epoch, best_epoch, best_loss = BaseNet.load_model(str(path))
    assert ckpt["epoch"] == 2
    assert epoch == 2 and best_epoch == 1 and best_loss == 0.3


def test_get_param_dict_no_freeze():
    net = DummyNet(make_cfg(is_freeze=False))
    param_dict = net.get_param_dict(1e-3)
    assert len(param_dict) == 2  # main + fixable(finetune_rate)
    assert param_dict[0]["initial_lr"] == 1e-3
    assert param_dict[1]["initial_lr"] == 1e-4  # 1e-3 * finetune_rate(0.1)
    assert net.fix.weight.requires_grad  # 未冻结


def test_get_param_dict_freeze():
    net = DummyNet(make_cfg(is_freeze=True))
    param_dict = net.get_param_dict(1e-3)
    assert len(param_dict) == 1  # fixable 被冻结，只有 main
    assert not net.fix.weight.requires_grad  # 冻结生效


def test_freeze_fixable_layers():
    net = DummyNet(make_cfg(is_freeze=False))
    net.freeze_fixable_layers()
    assert not net.fix.weight.requires_grad
    assert net.conv.weight.requires_grad


def test_initialize_xavier():
    net = DummyNet(make_cfg())
    net.conv.weight.data.fill_(0.0)  # 先清零，再初始化应被覆盖
    net.initialize()
    assert net.conv.weight.abs().sum() > 0  # xavier 非零
    # uninitializable 为空 → conv 全部被初始化
    assert not torch.allclose(
        net.conv.weight, torch.zeros_like(net.conv.weight)
    )


class DummyNetUninit(DummyNet):
    """uninitializable_layers 非空：initialize 应跳过该层。"""

    @property
    def uninitializable_layers(self):
        return self.fix


def test_initialize_skips_uninitializable():
    net = DummyNetUninit(make_cfg())
    net.conv.weight.data.fill_(0.0)
    net.fix.weight.data.fill_(0.123)  # 预设值，initialize 后应保持
    net.initialize()
    # fix 被跳过：权重不变
    assert torch.allclose(
        net.fix.weight, torch.full_like(net.fix.weight, 0.123)
    )
    # conv 仍被 xavier 初始化
    assert net.conv.weight.abs().sum() > 0


class DummyNetNoFix(DummyNet):
    """fixable_layers 为空 + is_freeze=True：get_param_dict 不应崩。"""

    @property
    def fixable_layers(self):
        return nn.Module()


def test_get_param_dict_freeze_empty_fixable():
    cfg = make_cfg(is_freeze=True)
    cfg.finetune_rate = 0.1
    net = DummyNetNoFix(cfg)
    param_dict = net.get_param_dict(1e-3)
    assert len(param_dict) == 1  # 无 fixable → 只有 main 分组
    assert param_dict[0]["initial_lr"] == 1e-3


def test_base_defaults_and_abstracts():
    """BaseNet 基类默认属性（空 Module）+ forward/loss 抽象抛错。"""
    net = BaseNet(make_cfg())
    assert isinstance(net.fixable_layers, nn.Module)
    assert isinstance(net.uninitializable_layers, nn.Module)
    with pytest.raises(NotImplementedError):
        net.forward(torch.rand(1, 1, 8, 8))
    with pytest.raises(NotImplementedError):
        net.loss(None, None)


def test_get_input_shape():
    net = BaseNet(make_cfg())
    assert net.get_input_shape() == (1, 1, 256, 256)
    assert net.get_input_shape(batch_size=4) == (4, 1, 256, 256)
