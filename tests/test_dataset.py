"""数据集冒烟：hbsn 读 .npy（真实数据 1 样本）、coco 读掩码、get_dataloader 分支。"""

import json

import numpy as np
import pytest
from torch.utils.data import Subset

from hbsn.config.schemas import CocoDatasetSchema, HbsnDatasetSchema
from hbsn.data.coco import CocoDataset
from hbsn.data.dataset import TransformSubset
from hbsn.data.hbsn import HBSNDataset


@pytest.fixture
def simple_dataset(hbsn_data_dir):
    """合成数据小数据集，num_workers=0 避免多进程开销（hermetic，不依赖真实 img/）。"""
    train_dir, _ = hbsn_data_dir
    cfg = HbsnDatasetSchema(data_dir=train_dir, test_data_dir="", num_workers=0)
    return HBSNDataset(cfg)


def test_hbsn_dataset_shapes(hbsn_data_dir):
    """合成数据：train/test 双目录加载，形状与 masked_size 裁剪后一致。"""
    train_dir, test_dir = hbsn_data_dir
    cfg = HbsnDatasetSchema(data_dir=train_dir, test_data_dir=test_dir)
    dataset = HBSNDataset(cfg, is_test=True)  # is_test=True → 读 test_data_dir
    assert len(dataset) > 0
    image, hbs = dataset[0]
    image, hbs = dataset.transform((image, hbs))
    assert image.shape == (1, 256, 256)
    assert hbs.shape == (2, 128, 128)  # masked_size=64 裁剪后


def test_hbsn_dataset_npy_values(hbsn_data_dir):
    """npy 数据健全性：float32、裁剪后形状、数值有限（.mat 已删除，无源对照）。"""
    train_dir, _ = hbsn_data_dir
    cfg = HbsnDatasetSchema(data_dir=train_dir, test_data_dir="")
    dataset = HBSNDataset(cfg)
    _, hbs = dataset[0]
    assert hbs.dtype == np.float32
    assert hbs.shape == (2, 128, 128)  # masked_size=64 裁剪后
    assert np.isfinite(hbs).all(), "npy 含 NaN/Inf"
    assert hbs.std() > 0, "npy 全等值，疑似损坏"


@pytest.mark.skipif(
    not __import__("os").path.exists("coco/annotations/instances_val2017.json"),
    reason="coco 数据缺失",
)
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


# ------------------------------------------------------------- get_dataloader 分支


def test_get_dataloader_split_rate_1(simple_dataset):
    train_dl, test_dl = simple_dataset.get_dataloader(
        batch_size=2, split_rate=1
    )
    assert train_dl is not None and test_dl is None
    batch = next(iter(train_dl))
    assert batch[0].shape[0] <= 2


def test_get_dataloader_split_rate_0(simple_dataset):
    train_dl, test_dl = simple_dataset.get_dataloader(
        batch_size=2, split_rate=0
    )
    assert train_dl is None and test_dl is not None


def test_get_dataloader_split(simple_dataset):
    train_dl, test_dl = simple_dataset.get_dataloader(
        batch_size=2, split_rate=0.8
    )
    assert train_dl is not None and test_dl is not None
    assert len(train_dl.dataset) + len(test_dl.dataset) == len(simple_dataset)


def test_get_dataloader_augment_uses_augment_transform(simple_dataset):
    # 非 None → shuffle=True 分支的 loader 用 augment_transform
    marker = object()
    simple_dataset.augment_transform = marker  # type: ignore[assignment]
    train_dl, _ = simple_dataset.get_dataloader(batch_size=2, split_rate=1)
    assert train_dl.dataset.transform is marker
    # test loader（split_rate=1 无 test）不涉及；单独验证非 shuffle 分支
    simple_dataset.augment_transform = None
    _, test_dl = simple_dataset.get_dataloader(batch_size=2, split_rate=0)
    assert test_dl.dataset.transform is simple_dataset.transform


# ------------------------------------------------------------- TransformSubset


def test_transform_subset_from_dataset(simple_dataset):
    ts = TransformSubset.from_dataset(simple_dataset, transform=lambda x: x)
    assert isinstance(ts, Subset)
    assert len(ts) == len(simple_dataset)
    data = ts[0]
    assert len(data) == 2  # (image, hbs)


def test_transform_subset_from_subset(simple_dataset):
    inner = Subset(simple_dataset, [0, 1])
    ts = TransformSubset.from_dataset(inner, transform=None)
    assert len(ts) == 2
    assert len(ts[0]) == 2


def test_transform_subset_wraps():
    class NotADataset:
        pass

    with pytest.raises(TypeError):
        TransformSubset.from_dataset(NotADataset(), transform=None)  # type: ignore[arg-type]


def test_transform_subset_batch(simple_dataset):
    ts = TransformSubset.from_dataset(simple_dataset, transform=lambda x: x)
    batch = ts.__getitems__([0, 1])  # DataLoader 的批量 API 路径
    assert len(batch) == 2


# ------------------------------------------------------------- HBSNDataset 分支


def test_hbsn_dataset_is_test_no_test_dir_raises(hbsn_data_dir):
    """is_test=True 但未配 test_data_dir → FileNotFoundError（hbsn.py 21）。"""
    train_dir, _ = hbsn_data_dir
    cfg = HbsnDatasetSchema(data_dir=train_dir, test_data_dir="")
    with pytest.raises(FileNotFoundError):
        HBSNDataset(cfg, is_test=True)


def test_hbsn_dataset_augment(hbsn_data_dir):
    """is_augment=True → 构建 augment_transform（hbsn.py 51-60）。"""
    train_dir, _ = hbsn_data_dir
    cfg = HbsnDatasetSchema(
        data_dir=train_dir, test_data_dir="", is_augment=True
    )
    ds = HBSNDataset(cfg)
    assert ds.augment_transform is not None
    assert len(ds) == 3


# ------------------------------------------------------------- CocoDataset 分支


def _write_minimal_coco(tmp_path) -> str:
    """最小合法 COCO 标注 json（无图无标注，仅一个类别），供 CocoDataset 构造。"""
    ann = tmp_path / "instances.json"
    ann.write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "obj", "supercategory": "obj"}
                ],
            }
        )
    )
    return str(ann)


def test_coco_dataset_is_test_branches(tmp_path):
    """is_test=True 用 test_data_dir/test_annotation_path（29-31）；无 test 字段则抛错（33）。"""
    ann = _write_minimal_coco(tmp_path)
    cfg = CocoDatasetSchema(
        data_dir=str(tmp_path),
        annotation_path=ann,
        test_data_dir=str(tmp_path),
        test_annotation_path=ann,
    )
    ds = CocoDataset(cfg, is_test=True)
    assert ds.annotation_path == ann
    assert len(ds) == 0  # json 无图

    cfg2 = CocoDatasetSchema(
        data_dir=str(tmp_path),
        annotation_path=ann,
        test_data_dir="",
        test_annotation_path="",
    )
    with pytest.raises(FileNotFoundError):
        CocoDataset(cfg2, is_test=True)


def test_coco_dataset_img_ids_and_augment(tmp_path):
    """img_ids 直接指定（53）+ is_augment 建 augment_transform（88）。"""
    ann = tmp_path / "instances.json"
    ann.write_text(
        json.dumps(
            {
                # 带一张 id=1 的图（loadImgs 对不存在 id 抛 KeyError）
                "images": [
                    {"id": 1, "file_name": "a.jpg", "width": 16, "height": 16}
                ],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "obj", "supercategory": "obj"}
                ],
            }
        )
    )
    cfg = CocoDatasetSchema(
        data_dir=str(tmp_path),
        annotation_path=str(ann),
        img_ids=[1],
        is_augment=True,
    )
    ds = CocoDataset(cfg)
    assert len(ds) == 1
    assert ds.augment_transform is not None
