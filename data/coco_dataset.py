import itertools
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision import io
from torchvision.datapoints import Mask
from torchvision.transforms import v2 as transforms

from data.augment_dataset import AugmentDataset
from data.custom_transform import BoundedRandomCrop, PadSquare, ResizeMax


class CocoDataset(AugmentDataset):
    def __init__(
        self, 
        root_path: str, 
        annotation_path: str,
        img_ids: Optional[List[int]]=[],
        cat_ids: Optional[List[int]]=[],
        # transform: Optional[Callable]=None,
        connected: bool=False,
        single_instance: bool=False,
        is_augment: bool=False,
        augment_rotation: float=30,
        augment_scale: List[float]=[0.8, 1.2],
        augment_translate: List[float]=[0.1, 0.1]
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.annotation_path = annotation_path
        self.coco: COCO = COCO(annotation_path)
        self.connected = connected
        self.single_instance = single_instance

        # cat ids
        if not cat_ids:
            self.cat_ids = self.coco.getCatIds()
        else:
            self.cat_ids = cat_ids
        
        # img ids
        if not img_ids:
            self.img_ids = list(itertools.chain(*[
                self.coco.getImgIds(catIds=cat_id)
                for cat_id in self.cat_ids
            ]))
        else:
            self.img_ids = img_ids
        
        # single instance
        if self.single_instance:
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
            os.path.join(self.root_path, img['file_name']) 
            for img in img_data
        ]
        
        # connected
        if self.connected:
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
                for anns in self.anns
            ]
            # for i, is_filter in enumerate(filter_list):
            #     if not is_filter:
            #         print(f"Image {self.files[i]}({i}) has no connected component.")

            self.anns = filter_func(self.anns, filter_list)
            self.img_ids = filter_func(self.img_ids, filter_list)
            self.files = filter_func(self.files, filter_list)

        
        self.transform = transforms.Compose([
            ResizeMax(512),
            BoundedRandomCrop(256, pad_if_needed=True),
        ])
        
        # augment
        if is_augment:
            self.augment_rotation = augment_rotation
            self.augment_scale = augment_scale
            self.augment_translate = augment_translate
            
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(augment_rotation, expand=True),
                # transforms.RandomAffine(augment_rotation, scale=augment_scale, translate=augment_translate),
                self.transform,
                # transforms.RandomRotation(augment_rotation),
                # transforms.RandomAffine(augment_rotation, scale=augment_scale, translate=augment_translate),
            ])

        
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        anns = self.anns[idx]

        mask = torch.LongTensor(np.max(np.stack([
            self.coco.annToMask(ann) * ann["category_id"]
            for ann in anns
        ]), axis=0)).unsqueeze(0)
        mask = Mask(mask)
        
        img = io.read_image(self.files[idx])
        if img.shape[0] == 1:
            img = torch.cat([img]*3)

        return img, mask