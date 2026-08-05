"""数据集冒烟：hbsn 读 .npy（真实数据 1 样本）、coco 读掩码、get_dataloader 分支。"""
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


# ------------------------------------------------------------- get_dataloader 分支


def test_get_dataloader_split_rate_1(simple_dataset):
    train_dl, test_dl = simple_dataset.get_dataloader(batch_size=2, split_rate=1)
    assert train_dl is not None and test_dl is None
    batch = next(iter(train_dl))
    assert batch[0].shape[0] <= 2


def test_get_dataloader_split_rate_0(simple_dataset):
    train_dl, test_dl = simple_dataset.get_dataloader(batch_size=2, split_rate=0)
    assert train_dl is None and test_dl is not None


def test_get_dataloader_split(simple_dataset):
    train_dl, test_dl = simple_dataset.get_dataloader(batch_size=2, split_rate=0.8)
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
