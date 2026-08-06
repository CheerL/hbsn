"""CocoDataset：COCO 分割数据（图片 + 实例掩码）。"""

import functools
import os

import numpy as np
import torch
from loguru import logger
from pycocotools.coco import COCO
from torchvision import io
from torchvision.transforms import Compose

from hbsn.data.dataset import BaseDataset
from hbsn.data.transforms import (
    BoundedRandomCrop,
    RandomFlip,
    RandomRotation,
    ResizeMax,
    ToTensor,
)


class CocoDataset(BaseDataset):
    def __init__(self, config, is_test: bool = False):
        self.config = config

        if not is_test:
            self.annotation_path = self.config.annotation_path
            self.data_dir = self.config.data_dir
        elif self.config.test_data_dir and self.config.test_annotation_path:
            self.annotation_path = self.config.test_annotation_path
            self.data_dir = self.config.test_data_dir
        else:
            raise FileNotFoundError(
                "Test data directory or annotation path not found"
            )

        self.coco: COCO = COCO(self.annotation_path)
        logger.info(f"Loading {self.annotation_path}...")

        # cat ids
        self.cat_ids = (
            self.coco.getCatIds()
            if not self.config.cat_ids
            else self.config.cat_ids
        )

        # img ids（多类别合并去重：一张图含多个目标类只出现一次）
        if not self.config.img_ids:
            self.img_ids = list(
                dict.fromkeys(
                    img_id
                    for cat_id in self.cat_ids
                    for img_id in self.coco.getImgIds(catIds=cat_id)
                )
            )
        else:
            self.img_ids = self.config.img_ids

        # single instance
        if self.config.single_instance:
            self.img_ids = [
                img_id
                for img_id in self.img_ids
                if len(self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids))
                == 1
            ]

        img_data = self.coco.loadImgs(self.img_ids)
        self.anns = [
            self.coco.loadAnns(
                self.coco.getAnnIds(
                    imgIds=img["id"], catIds=self.cat_ids, iscrowd=None
                )
            )
            for img in img_data
        ]
        self.files = [
            os.path.join(self.data_dir, img["file_name"]) for img in img_data
        ]

        # connected
        if self.config.connected:
            self.anns = [
                [ann for ann in anns if len(ann["segmentation"]) == 1]
                for anns in self.anns
            ]
            filter_list = [
                len(anns) > 0 and anns[0]["area"] > self.config.min_area
                for anns in self.anns
            ]
            self.anns = [
                a
                for a, keep in zip(self.anns, filter_list, strict=True)
                if keep
            ]
            self.img_ids = [
                i
                for i, keep in zip(self.img_ids, filter_list, strict=True)
                if keep
            ]
            self.files = [
                f
                for f, keep in zip(self.files, filter_list, strict=True)
                if keep
            ]

        self.transform = Compose(self._transform_list())
        if self.config.is_augment and not is_test:
            self.augment_transform = Compose(
                [
                    ToTensor(),
                    RandomFlip(),
                    RandomRotation(self.config.augment_rotation, expand=True),
                    *self._transform_list()[1:],
                ]
            )

        logger.info(f"Dataset contains {len(self)} images")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # annToMask 结果按 img_id 缓存（有界，掩码构建与 anns 内容一一对应）
        ann_ids = tuple(sorted(ann["id"] for ann in self.anns[idx]))
        mask = self._anns_to_mask(self.coco, ann_ids).unsqueeze(0)

        img = io.read_image(self.files[idx])
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, mask

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def _anns_to_mask(coco: COCO, ann_ids: tuple[int, ...]) -> torch.Tensor:
        anns = [coco.loadAnns([ann_id])[0] for ann_id in ann_ids]
        return torch.LongTensor(
            np.max(np.stack([coco.annToMask(ann) for ann in anns]), axis=0)
        )

    def _transform_list(self):
        return [
            ToTensor(),
            ResizeMax(
                int(
                    self.config.resize_rate
                    * max(self.config.height, self.config.width)
                )
            ),
            BoundedRandomCrop(
                (self.config.height, self.config.width), pad_if_needed=True
            ),
        ]
