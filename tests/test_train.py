"""train 模块：run/epoch_run/initialization/save/load（mock net + recorder，不跑真实训练）。"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

from hbsn import train


class FakeNet:
    """最小可训练对象：带真实权重供 Adam/保存用。"""

    def __init__(self):
        self.config = SimpleNamespace(device="cpu", dtype="float32")
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        self._saved = []

    def forward(self, img):
        return img

    def loss(self, predict, ground_truth):
        # loss 依赖 self.w（requires_grad），backward 才有效
        return {"loss": self.w * 0.5}, (predict, ground_truth)

    def initialize(self):
        pass

    def get_param_dict(self, lr):
        return [{"params": [self.w], "initial_lr": lr}]

    def save(self, path, *args, **kwargs):
        self._saved.append(path)
        torch.save({"epoch": args[0], "state_dict": {}}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("epoch", 0), 0, 1e10, None

    def __call__(self, img):
        return self.forward(img)

    def train(self):
        pass

    def eval(self):
        pass


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return torch.rand(1, 8, 8), torch.rand(1, 8, 8)


def test_run():
    net = FakeNet()
    img = torch.rand(1, 1, 8, 8)
    gt = torch.rand(1, 1, 8, 8)
    loss_dict, output_data = train.run(net, (img, gt))
    assert "loss" in loss_dict
    assert len(output_data) == 2


def test_epoch_run_train_and_test():
    net = FakeNet()
    dl = torch.utils.data.DataLoader(TinyDataset(), batch_size=1)
    opt = torch.optim.Adam([net.w])
    recorder = MagicMock()
    train.epoch_run(net, dl, opt, recorder, 0, is_train=True)
    recorder.add_loss.assert_called()
    recorder.add_epoch_loss.assert_called()

    recorder.reset_mock()
    train.epoch_run(net, dl, opt, recorder, 0, is_train=False)
    recorder.add_loss.assert_called()


def test_initialization():
    net = FakeNet()
    recorder = MagicMock()
    # initialization 里 config_to_dict 用 OmegaConf.to_container → 传 OmegaConf 配置
    run_cfg = OmegaConf.create(
        {"lr": 1e-3, "weight_norm": 1e-5, "moments": 0.9, "lr_decay_steps": [2], "lr_decay_rate": 0.5}
    )
    optimizer, scheduler = train.initialization(net, recorder, run_cfg)
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
    recorder.init_recorder.assert_called_once()


def test_save_checkpoint_best(tmp_path):
    net = FakeNet()
    recorder = MagicMock()
    recorder.checkpoint_dir = str(tmp_path)
    recorder.best_epoch = 0
    recorder.best_loss = 0.1
    recorder.update_best.return_value = True
    train.save_checkpoint(net, recorder, SimpleNamespace(), {"net": {}}, None, 0)
    assert any(os.path.basename(p) == "best.pth" for p in net._saved)
    recorder.update_best.assert_called_once()


def test_save_checkpoint_interval(tmp_path):
    net = FakeNet()
    recorder = MagicMock()
    recorder.checkpoint_dir = str(tmp_path)
    recorder.best_epoch = 0
    recorder.best_loss = 0.1
    recorder.update_best.return_value = False
    train.save_checkpoint(net, recorder, SimpleNamespace(), {"net": {}}, None, 5)
    assert any(os.path.basename(p) == "epoch_5.pth" for p in net._saved)


def test_load_checkpoint_missing(tmp_path):
    net = FakeNet()
    recorder = MagicMock()
    run_cfg = SimpleNamespace(checkpoint_path=str(tmp_path / "nope.pth"))
    init_epoch = train.load_checkpoint(net, recorder, None, None, run_cfg)
    assert init_epoch == -1


def test_load_checkpoint_present(tmp_path):
    path = tmp_path / "ckpt.pth"
    torch.save({"epoch": 3}, path)
    net = FakeNet()
    recorder = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    run_cfg = SimpleNamespace(checkpoint_path=str(path))
    init_epoch = train.load_checkpoint(net, recorder, optimizer, scheduler, run_cfg)
    assert init_epoch == 3
    assert recorder.best_epoch == 0
    scheduler.last_epoch = init_epoch


def test_config_to_dict():
    cfg = OmegaConf.create({"a": 1, "b": {"c": 2}})
    assert train.config_to_dict(cfg) == {"a": 1, "b": {"c": 2}}
