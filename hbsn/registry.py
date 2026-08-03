"""模型注册表：net/schema/dataset 三元组（取代旧 factory.py 的四合一 TYPE_DICT）。"""
from dataclasses import dataclass

from hbsn.config.schemas import (
    CocoDatasetSchema,
    HbsnDatasetSchema,
    HBSNetSchema,
    MaskRcnnNetSchema,
    SegNetSchema,
    TpsnNetSchema,
)
from hbsn.data.coco import CocoDataset
from hbsn.data.hbsn import HBSNDataset
from hbsn.nets.deeplab import DeepLab
from hbsn.nets.hbsn import HBSNet
from hbsn.nets.maskrcnn import MaskRCNN
from hbsn.nets.tpsn import TPSN
from hbsn.nets.unetpp import UnetPP


@dataclass(frozen=True)
class ModelSpec:
    name: str
    net: type
    net_schema: type
    dataset: type
    dataset_schema: type


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("hbsn", HBSNet, HBSNetSchema, HBSNDataset, HbsnDatasetSchema),
        ModelSpec("deeplab", DeepLab, SegNetSchema, CocoDataset, CocoDatasetSchema),
        ModelSpec("unetpp", UnetPP, SegNetSchema, CocoDataset, CocoDatasetSchema),
        ModelSpec("maskrcnn", MaskRCNN, MaskRcnnNetSchema, CocoDataset, CocoDatasetSchema),
        ModelSpec("tpsn", TPSN, TpsnNetSchema, CocoDataset, CocoDatasetSchema),
    ]
}


def resolve_model_name(model) -> str:
    """hydra choice group 的 cfg.model 是单键 dict {'选择名': 内容}；直接传字符串也兼容。"""
    if isinstance(model, str):
        return model
    return next(iter(model))


def get_spec(model: str) -> ModelSpec:
    assert model in MODEL_REGISTRY, (
        f"Unknown model '{model}', expected one of {sorted(MODEL_REGISTRY)}"
    )
    return MODEL_REGISTRY[model]
