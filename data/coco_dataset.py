from typing import List, Tuple, Optional, Callable
from pathlib import Path
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import io
import torch
import numpy as np
import itertools
import os

class ImageData(Dataset):
    def __init__(
        self, 
        root_path: Path, 
        annotation_path: Path,
        img_ids: Optional[List[int]]=[],
        cat_ids: Optional[List[int]]=[],
        transform: Optional[Callable]=None,
        connected: bool=True
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.annotation_path = annotation_path
        self.coco: COCO = COCO(annotation_path)
        self.connected = connected

        if not cat_ids:
            self.cat_ids = self.coco.getCatIds()
        else:
            self.cat_ids = cat_ids
        
        if not img_ids:
            self.img_ids = list(itertools.chain(*[
                self.coco.getImgIds(catIds=cat_id)
                for cat_id in self.cat_ids
            ]))
        else:
            self.img_ids = img_ids
        
        self.img_data = self.coco.loadImgs(self.img_ids)
        self.files = [
            os.path.join(self.root_path, img['file_name']) 
            for img in self.img_data
        ]
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.coco.getAnnIds(
            imgIds=self.img_data[idx]['id'], 
            catIds=self.cat_ids, 
            iscrowd=None
        )
        anns = self.coco.loadAnns(ann_ids)
        if self.connected:
            anns = [
                ann for ann in anns
                if len(ann['segmentation']) == 1
            ]

        mask = torch.LongTensor(np.max(np.stack([
            self.coco.annToMask(ann) * ann["category_id"]
            for ann in anns
        ]), axis=0)).unsqueeze(0)
        
        img = io.read_image(self.files[idx])
        if img.shape[0] == 1:
            img = torch.cat([img]*3)
        
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask