"""CocoDataset 单测：合成 COCO 注解 JSON + 合成图片，hermetic 覆盖数据管线核心逻辑。

真实 COCO 数据（coco/val2017）不随 git 走且体积大，无法作为单元测试依赖。
本文件在 tmp_path 生成最小合成 COCO 数据（3 张 64×64 图 + annotations JSON），
覆盖：基础加载/__getitem__ 掩码、single_instance 过滤、connected 过滤、_anns_to_mask。
"""

import json

import numpy as np
import torch
from PIL import Image

from hbsn.config.schemas import CocoDatasetSchema
from hbsn.data.coco import CocoDataset


def _write_synthetic_coco(tmp_path):
    """生成合成 COCO 数据目录，返回 (data_dir, annotation_path)。

    图 1：1 个 ann（单 polygon，area 500）→ connected+min_area 保留
    图 2：2 个 ann → single_instance 过滤
    图 3：1 个 ann（area 100）→ connected+min_area=300 过滤
    """
    data_dir = tmp_path / "coco_val"
    ann_dir = tmp_path / "annotations"
    data_dir.mkdir()
    ann_dir.mkdir()
    for i in (1, 2, 3):
        arr = np.full((64, 64, 3), i * 40, dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(data_dir / f"{i}.png")

    annotations = [
        {
            "id": 11,
            "image_id": 1,
            "category_id": 16,
            "bbox": [10, 10, 20, 20],
            "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            "area": 500,
            "iscrowd": 0,
        },
        {
            "id": 21,
            "image_id": 2,
            "category_id": 16,
            "bbox": [5, 5, 15, 15],
            "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]],
            "area": 200,
            "iscrowd": 0,
        },
        {
            "id": 22,
            "image_id": 2,
            "category_id": 16,
            "bbox": [30, 30, 15, 15],
            "segmentation": [[30, 30, 45, 30, 45, 45, 30, 45]],
            "area": 200,
            "iscrowd": 0,
        },
        {
            "id": 31,
            "image_id": 3,
            "category_id": 16,
            "bbox": [10, 10, 10, 10],
            "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
            "area": 100,
            "iscrowd": 0,
        },
    ]
    payload = {
        "images": [
            {"id": 1, "file_name": "1.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "2.png", "width": 64, "height": 64},
            {"id": 3, "file_name": "3.png", "width": 64, "height": 64},
        ],
        "annotations": annotations,
        "categories": [{"id": 16, "name": "cat", "supercategory": "animal"}],
    }
    ann_path = ann_dir / "instances.json"
    ann_path.write_text(json.dumps(payload))
    return str(data_dir), str(ann_path)


def _make_cfg(data_dir, ann_path, **kw):
    defaults = {
        "data_dir": data_dir,
        "annotation_path": ann_path,
        "cat_ids": [16],
        "height": 32,
        "width": 32,
        "resize_rate": 1.0,
        "num_workers": 0,
    }
    defaults.update(kw)
    return CocoDatasetSchema(**defaults)


def test_coco_basic_load_and_getitem(tmp_path):
    """基础加载：3 张图全保留；__getitem__ 返回 (3,H,W) img + (1,H,W) 0/1 mask。"""
    data_dir, ann_path = _write_synthetic_coco(tmp_path)
    ds = CocoDataset(_make_cfg(data_dir, ann_path))
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape[0] == 3  # io.read_image 返回 (C,H,W)
    assert mask.ndim == 3 and mask.shape[0] == 1
    assert set(torch.unique(mask).tolist()) <= {0, 1}  # 掩码是 0/1
    assert mask.sum() > 0  # 有前景


def test_coco_single_instance_filters(tmp_path):
    """single_instance=True：图 2（2 个 ann）被过滤，剩图 1、图 3。"""
    data_dir, ann_path = _write_synthetic_coco(tmp_path)
    ds = CocoDataset(_make_cfg(data_dir, ann_path, single_instance=True))
    assert len(ds) == 2
    assert ds.img_ids == [1, 3]


def test_coco_connected_filters_by_area(tmp_path):
    """connected=True + min_area=300：图 3（area 100）被过滤，剩图 1（area 500）。"""
    data_dir, ann_path = _write_synthetic_coco(tmp_path)
    ds = CocoDataset(
        _make_cfg(data_dir, ann_path, connected=True, min_area=300)
    )
    assert len(ds) == 1
    assert ds.img_ids == [1]


def test_anns_to_mask_static(tmp_path):
    """_anns_to_mask：给定 ann id 列表 → LongTensor (H,W)，前景区域与 polygon 一致。"""
    data_dir, ann_path = _write_synthetic_coco(tmp_path)
    ds = CocoDataset(_make_cfg(data_dir, ann_path))
    mask = CocoDataset._anns_to_mask(ds.coco, (11,))
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.int64
    assert mask.sum() > 0
