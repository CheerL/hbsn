"""eval_coco 纯逻辑测试：filter_known / _build_configs / _parse_compare_entries（无需数据）。"""
from hbsn.config.schemas import HBSNetSchema, SegNetSchema
from hbsn.registry import get_spec
from scripts.eval_coco import (
    _build_configs,
    _parse_compare_entries,
    filter_known,
)


def test_filter_known_drops_unknown():
    cfg = {"device": "cpu", "stn_mode": 3, "old_field": 1}
    filtered = filter_known(HBSNetSchema, cfg)
    assert "old_field" not in filtered
    assert filtered["stn_mode"] == 3


def test_filter_known_recurses_hbsn_config():
    cfg = {"hbsn_config": {"radius": 40, "junk": 1}, "input_channels": 3}
    filtered = filter_known(SegNetSchema, cfg)
    assert filtered["hbsn_config"] == {"radius": 40}


def test_build_configs_clears_hbsn_checkpoint():
    """deeplab 的 SegNetSchema 有 hbsn_checkpoint 字段 → 清空防路径失效。"""
    spec = get_spec("deeplab")
    net_cfg, _ = _build_configs(
        spec, {"net": {"hbsn_checkpoint": "runs/old.pth"}}, {"device": "cpu"}, None
    )
    assert net_cfg.hbsn_checkpoint == ""


def test_build_configs_hbsn_schema_defaults():
    """hbsn 模型 schema 无 hbsn_checkpoint 字段 → 不崩，默认值保留。"""
    spec = get_spec("hbsn")
    net_cfg, dataset_cfg = _build_configs(spec, {}, {"device": "cpu"}, None)
    assert net_cfg.stn_mode == 3
    assert dataset_cfg.data_dir == "img/generated"


def test_build_configs_set_bool_int_parsing():
    spec = get_spec("hbsn")
    net_cfg, _ = _build_configs(spec, {}, {}, ["is_freeze=true", "stn_mode=2"])
    assert net_cfg.is_freeze is True  # "true" 解析为 bool
    assert net_cfg.stn_mode == 2


def test_build_configs_unknown_override_warns(capsys):
    spec = get_spec("hbsn")
    _build_configs(spec, {}, {"no_such_field": 1}, None)
    assert "未知覆盖" in capsys.readouterr().err


def test_parse_compare_entries(tmp_path):
    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"x")
    entries = _parse_compare_entries([f"unetpp:{ckpt}:unet"])
    assert entries == [("unetpp", str(ckpt), "unet")]
