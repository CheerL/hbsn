"""模型注册表：resolve_model_name 双输入形态 + get_spec 已知/未知名。"""
import pytest

from hbsn.registry import MODEL_REGISTRY, get_spec, resolve_model_name


def test_resolve_model_name_str():
    assert resolve_model_name("hbsn") == "hbsn"


def test_resolve_model_name_dict():
    # hydra choice group 的 cfg.model 是单键 dict {选择名: 内容}
    assert resolve_model_name({"tpsn": {}}) == "tpsn"


def test_get_spec_known():
    spec = get_spec("hbsn")
    assert spec.name == "hbsn"
    assert spec.net is MODEL_REGISTRY["hbsn"].net


def test_get_spec_unknown():
    with pytest.raises(AssertionError):
        get_spec("not_a_model")
