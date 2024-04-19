import os

import scipy.io as sio
import torch
from genericpath import exists
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import functional as F

DEFAULT_DATA_DIR = 'img/generated'

def load_data(file_path):
    image_path = file_path
    hbs_path = file_path.replace('.png', '.mat')
    
    if not exists(hbs_path) or not exists(image_path):
        raise FileNotFoundError("Image or HBS file not found")
    
    image = Image.open(image_path)
    hbs = sio.loadmat(hbs_path)['hbs']
    return image, hbs

class CustomRandomAffine(transforms.RandomAffine):
    def forward(self, img):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        x, y = torch.where(img[0]>0.5)
        dis = ((x-width/2).pow(2) + (y-height/2).pow(2)).sqrt()
        max_dis = dis.max()
        
        
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        
        if self.scale is not None:
            max_scale = min(self.scale[1], min(width,height) / (max_dis * 2))
            min_scale = min(self.scale[0], max_scale)
            scale_ranges = [min_scale, max_scale]
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        if self.translate is not None:
            scaled_max_dis = scale * max_dis
            translate = self.translate
            max_dx = min(float(translate[0] * width), width/2 - scaled_max_dis)
            max_dy = min(float(translate[1] * height), height/2 - scaled_max_dis)
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        shear_x = shear_y = 0.0
        if self.shear is not None:
            shears = self.shear
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())
        shear = (shear_x, shear_y)
        
        ret = [angle, translations, scale, shear]
        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)

class TrainSubset(Subset):
    def __init__(self, dataset, indices, transforms) -> None:
        super().__init__(dataset, indices)
        self.transforms = transforms
        
    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

class HBSNDataset(Dataset):
    def __init__(self, root=DEFAULT_DATA_DIR, is_augment=False, 
                 augment_rotation=180, augment_scale=[0.8,1.2], augment_translate=[0.1,0.1]
                 ):
        self.root = root
        self.is_augment = is_augment
        self.augment_rotation = augment_rotation
        self.augment_scale = augment_scale
        self.augment_translate = augment_translate
        
        
        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        hbs_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.transform = transforms.Lambda(
            lambda x: (
                image_transform(x[0]), 
                hbs_transform(x[1])
            )
        )
        
        augment_image_transform = transforms.Compose([
            CustomRandomAffine(augment_rotation, scale=augment_scale, translate=augment_translate),
        ])
        self.augment_transform = transforms.Lambda(
            lambda x: (
                augment_image_transform(x[0]), 
                x[1]
            )
        )

        self.data_list = sorted([
            f"{root}/{file}" for file in os.listdir(root) 
            if file.endswith('.png')
            and exists(f"{root}/{file.replace('.png', '.mat')}")
        ])
        self.data = {}
        self.num_sample = len(self.data_list)
        
        if self.num_sample > 0:
            image, hbs = self[0]
            C_image, H_image, W_image = image.shape
            C_hbs, H_hbs, W_hbs = hbs.shape
            assert H_image == H_hbs and W_image == W_hbs, "Image and HBS size mismatch"
            assert C_image == 1, "Image channel should be 1"
            assert C_hbs == 2, "HBS channel should be 2"
            
            self.H = H_image
            self.W = W_image
            self.C_image = C_image
            self.C_hbs = C_hbs
    
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self, index):
        if isinstance(index, list):
            return [self[i] for i in index]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            file_name = self.data_list[index]
            if file_name not in self.data:
                image, hbs = self.transform(load_data(file_name))
                self.data[file_name] = (image, hbs)
            else:
                image, hbs = self.data[file_name]
            return image, hbs
    
    def get_size(self):
        return self.H, self.W, self.C_image, self.C_hbs
    
    def get_dataloader(self, batch_size=32, split_rate=0.8, drop_last=True):
        # Split dataset into train and test sets
        # train_size = int(split_rate * len(self))
        # test_size = len(self) - train_size
        if split_rate == 1:
            if self.is_augment:
                train_dataset = TrainSubset(
                    self, 
                    range(len(self)),
                    self.augment_transform
                    )
            else:
                train_dataset = self

            test_dataset = None
        elif split_rate == 0:
            train_dataset = None
            test_dataset = self
        else:
            train_dataset, test_dataset = random_split(self, [split_rate, 1-split_rate])
            
            if self.is_augment:
                train_dataset = TrainSubset(
                train_dataset.dataset,
                train_dataset.indices,
                self.augment_transform
                )

        # train_dataset.transform = self.augment_transform if self.is_augment else self.transform
        # test_dataset.transform = self.transform

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=True, num_workers=4, drop_last=drop_last, persistent_workers=True
            ) if train_dataset else None
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            pin_memory=True, num_workers=4, drop_last=drop_last, persistent_workers=True
            ) if test_dataset else None
        return train_dataloader, test_dataloader
