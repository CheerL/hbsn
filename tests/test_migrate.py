"""迁移工具全流程测试：合成旧格式 ckpt → 桩反序列化 → 键映射 → 新格式可加载。"""
import os
import sys
import types

import torch

from tools.migrate_checkpoints import detect_type, map_state_dict, migrate_one


class HBSNetConfig:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.stn_mode = 3
        self.radius = 50
        self.input_channels = 1
        self.output_channels = 2
        self.height = 256
        self.width = 256
        self.channels_down = [8, 8, 16, 32, 64, 128]
        self.channels_up = [8, 16, 32, 64, 128]
        self.is_skip = True


class SegHBSNNetConfig:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.hbs_loss_rate = 1.0
        self.hbsn_checkpoint = "runs/hbsn/Jun11_13-01-53_big_ns/checkpoints/best.pth"
        self.input_channels = 3


class HBSNDatasetConfig:
    def __init__(self):
        self.data_dir = "img/generated"
        self.masked_size = 64


class RecorderConfig:
    def __init__(self):
        self.log_base_dir = "runs"


class RunConfig:
    def __init__(self):
        self.total_epoches = 1000
        self.batch_size = 64


class Config:
    def __init__(self, net_config, dataset_config, recorder_config, run_config):
        self.net_config = net_config
        self.dataset_config = dataset_config
        self.recorder_config = recorder_config
        self.run_config = run_config


# 类路径挂在旧模块路径上，pickle 时按路径序列化（与真实旧 checkpoint 一致）
HBSNetConfig.__module__ = "config.net"
SegHBSNNetConfig.__module__ = "config.net"
HBSNDatasetConfig.__module__ = "config.dataset"
RecorderConfig.__module__ = "config.recoder"
RunConfig.__module__ = "config.run"
Config.__module__ = "config.config"


def _make_legacy_config():
    """构造旧格式 Config 对象。"""
    top_mod = types.ModuleType("config")
    net_mod = types.ModuleType("config.net")
    ds_mod = types.ModuleType("config.dataset")
    rec_mod = types.ModuleType("config.recoder")
    run_mod = types.ModuleType("config.run")
    config_mod = types.ModuleType("config.config")

    net_mod.HBSNetConfig = HBSNetConfig
    net_mod.SegHBSNNetConfig = SegHBSNNetConfig
    ds_mod.HBSNDatasetConfig = HBSNDatasetConfig
    rec_mod.RecorderConfig = RecorderConfig
    run_mod.RunConfig = RunConfig
    config_mod.Config = Config
    # pickle 反序列化类路径时需 import "config.config"：顶层包也要挂桩
    sys.modules["config"] = top_mod
    sys.modules["config.net"] = net_mod
    sys.modules["config.dataset"] = ds_mod
    sys.modules["config.recoder"] = rec_mod
    sys.modules["config.run"] = run_mod
    sys.modules["config.config"] = config_mod

    return Config(SegHBSNNetConfig(), HBSNDatasetConfig(), RecorderConfig(), RunConfig())


def test_migrate_seg_checkpoint(tmp_path):
    # 合成旧格式 unetpp 类 ckpt：hbsn.* + model.encoder.* 键
    cfg = _make_legacy_config()
    old_ckpt = {
        "state_dict": {
            "hbsn.backbone.inc.double_conv.0.weight": torch.zeros(8, 1, 3, 3),
            "hbsn.pre_stn.localization.0.weight": torch.ones(8, 1, 7, 7),
            "model.encoder.conv1.weight": torch.ones(64, 3, 7, 7),
            "model.decoder.blocks.block1.0.weight": torch.ones(32, 32, 3, 3),
        },
        "epoch": 350,
        "best_epoch": 349,
        "best_loss": 0.1,
        "config": cfg,
        "optimizer": None,
    }
    src = tmp_path / "epoch_350.pth"
    torch.save(old_ckpt, src)

    from hbsn._legacy import pickle_shim
    pickle_shim.install()

    # 键映射
    assert detect_type(old_ckpt["state_dict"], cfg) == "unetpp"
    kept, dropped = map_state_dict(old_ckpt["state_dict"], "unetpp")
    assert len(dropped) == 2 and all(k.startswith("hbsn.") for k in dropped)
    assert all(k.startswith("model.") for k in kept)

    # 全流程迁移
    out_dir = tmp_path / "migrated"
    out_path = migrate_one(str(src), str(out_dir))
    new_ckpt = torch.load(out_path, map_location="cpu", weights_only=False)

    # 新格式断言
    assert new_ckpt["optimizer"] is None
    assert isinstance(new_ckpt["config"], dict)
    assert "hbsn." not in new_ckpt["state_dict"]
    # hbsn_checkpoint 路径重写：runs/hbsn/... → <out_dir>/hbsn/...
    assert os.path.join(out_dir, "hbsn") in new_ckpt["config"]["net"]["hbsn_checkpoint"]
    # dtype 转名称
    assert new_ckpt["config"]["net"]["dtype"] == "float32"
    # 重叠张量一致
    for k in kept:
        assert torch.equal(new_ckpt["state_dict"][k], old_ckpt["state_dict"][k])


def test_migrate_hbsn_checkpoint(tmp_path):
    # 独立 hbsn ckpt 的 config 也是完整 Config（net_config 为 HBSNetConfig）
    cfg = _make_legacy_config()
    cfg.net_config = HBSNetConfig()
    old_ckpt = {
        "state_dict": {
            "pre_stn.localization.0.weight": torch.zeros(8, 1, 7, 7),
            "backbone.inc.double_conv.0.weight": torch.zeros(8, 1, 3, 3),
        },
        "epoch": 100,
        "best_epoch": 90,
        "best_loss": 0.01,
        "config": cfg,
        "optimizer": None,
    }
    src = tmp_path / "best.pth"
    torch.save(old_ckpt, src)

    from hbsn._legacy import pickle_shim
    pickle_shim.install()

    assert detect_type(old_ckpt["state_dict"], cfg) == "hbsn"
    out_dir = tmp_path / "migrated"
    out_path = migrate_one(str(src), str(out_dir))
    new_ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
    assert set(new_ckpt["state_dict"]) == set(old_ckpt["state_dict"])  # hbsn 键全保留
