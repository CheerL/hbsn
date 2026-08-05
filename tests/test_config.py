"""配置校验：未知 key 报错、跨字段约束。"""
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from hbsn.config.schemas import CocoDatasetSchema, HbsnDatasetSchema, HBSNetSchema
from hbsn.config.validate import validate_config


def test_unknown_key_raises():
    net_cfg = OmegaConf.create({"unknown_key": 1})
    with pytest.raises(ConfigKeyError):
        OmegaConf.merge(OmegaConf.structured(HBSNetSchema), net_cfg)


def test_valid_merge(hbsn_data_dir):
    # validate_config 对 HbsnDatasetSchema 断言 data_dir 存在——用合成目录（hermetic）
    train_dir, test_dir = hbsn_data_dir
    cfg = OmegaConf.create({
        "net": {"stn_mode": 3, "device": "cpu"},
        "dataset": {"masked_size": 64, "data_dir": train_dir, "test_data_dir": test_dir},
    })
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


# ------------------------------------------------------------- 数据目录/标注文件存在性


def test_validate_hbsn_dir_missing():
    cfg = OmegaConf.create({"net": {}, "dataset": {"data_dir": "/nonexistent"}})
    with pytest.raises(AssertionError, match="data_dir 不存在"):
        validate_config(cfg, HBSNetSchema, HbsnDatasetSchema)


def test_validate_coco_annotation_missing():
    cfg = OmegaConf.create(
        {"net": {}, "dataset": {"annotation_path": "/nonexistent.json"}}
    )
    with pytest.raises(AssertionError, match="annotation_path 不存在"):
        validate_config(cfg, HBSNetSchema, CocoDatasetSchema)


def test_validate_coco_valid(tmp_path):
    """Coco 分支：annotation_path 存在即通过（只查文件存在性，不解析内容）。"""
    ann = tmp_path / "instances.json"
    ann.write_text("{}")
    cfg = OmegaConf.create({"net": {}, "dataset": {"annotation_path": str(ann)}})
    merged_net, merged_dataset = validate_config(cfg, HBSNetSchema, CocoDatasetSchema)
    assert merged_dataset.annotation_path == str(ann)
    assert merged_net.stn_mode == 3  # HBSNetSchema 默认值保留
