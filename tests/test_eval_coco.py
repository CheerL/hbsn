"""eval_coco 纯逻辑测试：filter_known / _build_configs / _parse_compare_entries / _prepare_input（无需数据）。"""
from types import SimpleNamespace

import torch

from hbsn.config.schemas import HBSNetSchema, SegNetSchema
from hbsn.registry import get_spec
from scripts.eval_coco import (
    _build_configs,
    _parse_compare_entries,
    _prepare_input,
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


def test_build_configs_set_list_parsing():
    """--set cat_ids=[16] 列表字面量 → json.loads 解析（review 发现：docstring 声称可用但原实现崩溃）。"""
    spec = get_spec("deeplab")
    _, dataset_cfg = _build_configs(spec, {}, {}, ["cat_ids=[16]", "connected=true"])
    assert dataset_cfg.cat_ids == [16]
    assert dataset_cfg.connected is True


def test_build_configs_unknown_override_warns(capsys):
    spec = get_spec("hbsn")
    _build_configs(spec, {}, {"no_such_field": 1}, None)
    assert "未知覆盖" in capsys.readouterr().err


def test_parse_compare_entries(tmp_path):
    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"x")
    entries = _parse_compare_entries([f"unetpp:{ckpt}:unet"])
    assert entries == [("unetpp", str(ckpt), "unet")]


# ------------------------------------------------------------- _prepare_input


def _cfg(in_ch=3, h=256, w=256):
    return SimpleNamespace(input_channels=in_ch, height=h, width=w)


def test_prepare_input_grayscale_to_rgb():
    """灰度 (1,64,64) uint8 → 模型要 RGB → (1,3,256,256) float32 [0,1]。"""
    img = torch.randint(0, 256, (1, 64, 64), dtype=torch.uint8)
    out = _prepare_input(img, _cfg(in_ch=3))
    assert out.shape == (1, 3, 256, 256)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.equal(out[:, 0], out[:, 1])  # 灰度复制三通道


def test_prepare_input_rgb_to_grayscale():
    """RGB (3,64,64) uint8 → 模型要 1 通道 → (1,1,256,256)。"""
    img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    out = _prepare_input(img, _cfg(in_ch=1))
    assert out.shape == (1, 1, 256, 256)
    assert out.dtype == torch.float32


def test_prepare_input_resizes_and_normalizes():
    """resize 到网络输入尺寸 + 归一化 [0,1]。"""
    img = torch.full((3, 64, 64), 128, dtype=torch.uint8)
    out = _prepare_input(img, _cfg(in_ch=3, h=32, w=32))
    assert out.shape == (1, 3, 32, 32)
    assert torch.allclose(out, torch.full_like(out, 128 / 255.0), atol=1e-6)


def test_prepare_input_rgba_drops_alpha():
    """RGBA (4,64,64) → 丢弃 alpha → 适配 3 通道，避免 repeat 后 12 通道。"""
    img = torch.randint(0, 256, (4, 64, 64), dtype=torch.uint8)
    out = _prepare_input(img, _cfg(in_ch=3))
    assert out.shape == (1, 3, 256, 256)
