"""数据集冒烟：hbsn 读 .npy（真实数据 1 样本）、coco 读掩码。"""
import numpy as np
import pytest

from hbsn.config.schemas import CocoDatasetSchema, HbsnDatasetSchema
from hbsn.data.coco import CocoDataset
from hbsn.data.hbsn import HBSNDataset


def test_hbsn_dataset_shapes():
    cfg = HbsnDatasetSchema(data_dir="img/generated", test_data_dir="img/gen2")
    dataset = HBSNDataset(cfg, is_test=True)  # gen2 只有 1.5k 样本，加载快
    assert len(dataset) > 0
    image, hbs = dataset[0]
    image, hbs = dataset.transform((image, hbs))
    assert image.shape == (1, 256, 256)
    assert hbs.shape == (2, 128, 128)  # masked_size=64 裁剪后


def test_hbsn_dataset_npy_values():
    """npy 数据健全性：float32、裁剪后形状、数值有限（.mat 已删除，无源对照）。"""
    cfg = HbsnDatasetSchema(data_dir="img/simple", test_data_dir="")
    dataset = HBSNDataset(cfg)
    _, hbs = dataset[0]
    assert hbs.dtype == np.float32
    assert hbs.shape == (2, 128, 128)  # masked_size=64 裁剪后
    assert np.isfinite(hbs).all(), "npy 含 NaN/Inf"
    assert hbs.std() > 0, "npy 全等值，疑似损坏"


@pytest.mark.skipif(not __import__("os").path.exists("coco/annotations/instances_val2017.json"), reason="coco 数据缺失")
def test_coco_dataset_shapes():
    cfg = CocoDatasetSchema(
        data_dir="coco/val2017",
        annotation_path="coco/annotations/instances_val2017.json",
        connected=True,
        single_instance=True,
    )
    dataset = CocoDataset(cfg)
    assert len(dataset) > 0
    img, mask = dataset[0]
    assert img.shape[0] == 3
    assert mask.ndim == 3 and mask.shape[0] == 1
