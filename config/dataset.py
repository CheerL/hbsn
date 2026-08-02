from typing import List, Optional

from config.base import BaseConfig


class BaseDatasetConfig(BaseConfig):
    data_dir = ""
    test_data_dir = ""
    is_augment = False
    augment_rotation = 180
    augment_scale = [0.8, 1.2]
    augment_translate = [0.1, 0.1]
    is_soft_label = True
    # pin_memory=True 在 WSL2 + torch 2.13 下 CachingHostAllocator 不复用，
    # 主进程 RSS 以 ~36MB/迭代 无界增长（本机 15GB 内存会 OOM）。
    # 实测 pin_memory=False 时内存有界封顶。可显式改回 True 以换取传输速度。
    pin_memory = False
    num_workers = 4

    @property
    def augment(self):
        return (
            (
                self.augment_rotation,
                self.augment_scale,
                self.augment_translate,
            )
            if self.is_augment
            else False
        )

    @property
    def _except_keys(self):
        return super()._except_keys + [
            "is_augment",
            "augment_rotation",
            "augment_scale",
            "augment_translate",
        ]


class HBSNDatasetConfig(BaseDatasetConfig):
    data_dir = "img/generated"
    test_data_dir = "img/gen2"
    augment_rotation = 180
    augment_scale = [0.5, 2]
    augment_translate = [0.5, 0.5]
    masked_size = 64


class CocoDatasetConfig(BaseDatasetConfig):
    data_dir: str = "coco/train2017"
    test_data_dir: str = ""
    annotation_path: str = "coco/annotations/instances_train2017.json"
    test_annotation_path: str = ""
    height: int = 256
    width: int = 256
    img_ids: Optional[List[int]] = []
    cat_ids: Optional[List[int]] = []  # [16]
    connected: bool = False
    single_instance: bool = False
    resize_rate: float = 1.5
    min_area: float = 500
    augment_rotation: float = 30
    augment_scale: List[float] = [0.8, 1.2]
    augment_translate: List[float] = [0.1, 0.1]
