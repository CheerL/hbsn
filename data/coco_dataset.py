import itertools
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision import io
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset, BaseDatasetConfig
from data.custom_transform import (
    BoundedRandomCrop,
    RandomFlip,
    RandomRotation,
    ResizeMax,
    ToTensor,
)


class CocoDatasetConfig(BaseDatasetConfig):
    data_dir: str='coco/train2017'
    test_data_dir: str='coco/val2017'
    annotation_path: str='coco/annotations/instances_train2017.json'
    test_annotation_path: str='coco/annotations/instances_val2017.json'
    height: int=256
    width: int=256
    img_ids: Optional[List[int]]=[]
    cat_ids: Optional[List[int]]=[] # [16]
    connected: bool=False
    single_instance: bool=False
    resize_rate: float=1.5
    min_area: float=500
    augment_rotation: float=30
    augment_scale: List[float]=[0.8, 1.2]
    augment_translate: List[float]=[0.1, 0.1]


class CocoDataset(BaseDataset):
    def __init__(self, config: CocoDatasetConfig, is_test: bool=False):
        self.config = config
        
        if not is_test:
            self.annotation_path = self.config.annotation_path
            self.data_dir = self.config.data_dir
        elif self.config.test_data_dir and self.config.test_annotation_path:
            self.annotation_path = self.config.test_annotation_path
            self.data_dir = self.config.test_data_dir
        else:
            raise FileNotFoundError("Test data directory or annotation path not found")
        
        self.coco: COCO = COCO(self.annotation_path)
        
        # cat ids
        if not self.config.cat_ids:
            self.cat_ids = self.coco.getCatIds()
        else:
            self.cat_ids = self.config.cat_ids

        # img ids
        if not self.config.img_ids:
            self.img_ids = list(itertools.chain(*[
                self.coco.getImgIds(catIds=cat_id)
                for cat_id in self.cat_ids
            ]))
        else:
            self.img_ids = self.config.img_ids
        
        # single instance
        if self.config.single_instance:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(
                    imgIds=img_id, catIds=self.cat_ids
                )) == 1
            ]
        
        # load imgs
        img_data = self.coco.loadImgs(self.img_ids)
        self.anns = [
            self.coco.loadAnns(
                self.coco.getAnnIds(
                    imgIds=img['id'], 
                    catIds=self.cat_ids, 
                    iscrowd=None
                )
            )
            for img in img_data
        ]
        self.files = [
            os.path.join(self.data_dir, img['file_name']) 
            for img in img_data
        ]
        
        # connected
        if self.config.connected:
            def filter_func(x_list, filter_list):
                return [
                    x for x, is_filter in zip(x_list, filter_list)
                    if is_filter
                ]
            
            self.anns = [
                [
                    ann for ann in anns
                    if len(ann['segmentation']) == 1
                ]
                for anns in self.anns
            ]
            filter_list = [
                len(anns) > 0
                and anns[0]['area'] > self.config.min_area
                for anns in self.anns
            ]
            # for i, is_filter in enumerate(filter_list):
            #     if not is_filter:
            #         print(f"Image {self.files[i]}({i}) has no connected component.")

            self.anns = filter_func(self.anns, filter_list)
            self.img_ids = filter_func(self.img_ids, filter_list)
            self.files = filter_func(self.files, filter_list)

        self.transform = transforms.Compose([
            ToTensor(),
            ResizeMax(int(self.config.resize_rate * max(self.config.height, self.config.width))),
            BoundedRandomCrop((self.config.height, self.config.width), pad_if_needed=True),
        ])
        
        # augment
        if self.config.is_augment and not is_test:
            self.augment_transform = transforms.Compose([
                ToTensor(),
                RandomFlip(),
                RandomRotation(self.config.augment_rotation, expand=True),
                ResizeMax(int(self.config.resize_rate * max(self.config.height, self.config.width))),
                BoundedRandomCrop((self.config.height, self.config.width), pad_if_needed=True),
            ])

        print(f'Dataset contains {len(self)} images')
        
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anns = self.anns[idx]

        mask = torch.LongTensor(np.max(np.stack([
            self.coco.annToMask(ann)
            for ann in anns
        ]), axis=0)).unsqueeze(0)

        img = io.read_image(self.files[idx])
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, mask