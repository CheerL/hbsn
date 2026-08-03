"""配置校验：未知 key 报错、跨字段约束。"""
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from hbsn.config.schemas import HbsnDatasetSchema, HBSNetSchema
from hbsn.config.validate import validate_config


def test_unknown_key_raises():
    net_cfg = OmegaConf.create({"unknown_key": 1})
    with pytest.raises(ConfigKeyError):
        OmegaConf.merge(OmegaConf.structured(HBSNetSchema), net_cfg)


def test_valid_merge():
    cfg = OmegaConf.create({"net": {"stn_mode": 3, "device": "cpu"}, "dataset": {"masked_size": 64}})
    merged_net, merged_dataset = validate_config(cfg, HBSNetSchema, HbsnDatasetSchema)
    assert merged_net.stn_mode == 3
    assert merged_dataset.masked_size == 64
    # 输出降采样：2^(len(up)-len(down)) = 1/2 → 128
    rate = 2 ** (len(merged_net.channels_up) - len(merged_net.channels_down))
    assert int(merged_net.height * rate) == 128


def test_channel_mismatch_raises():
    cfg = OmegaConf.create({"net": {"channels_up": [8]}, "dataset": {}})
    with pytest.raises(AssertionError):
        validate_config(cfg, HBSNetSchema, HbsnDatasetSchema)


def test_stn_mode_raises():
    cfg = OmegaConf.create({"net": {"stn_mode": 9}, "dataset": {}})
    with pytest.raises(AssertionError):
        validate_config(cfg, HBSNetSchema, HbsnDatasetSchema)
