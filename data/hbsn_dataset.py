import os

import scipy.io as sio
from genericpath import exists
from PIL import Image
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from data.custom_transform import BoundedRandomAffine

DEFAULT_DATA_DIR = 'img/generated'

def load_data(file_path):
    image_path = file_path
    hbs_path = file_path.replace('.png', '.mat')
    
    if not exists(hbs_path) or not exists(image_path):
        raise FileNotFoundError("Image or HBS file not found")
    
    image = Image.open(image_path)
    hbs = sio.loadmat(hbs_path)['hbs']
    return image, hbs


class HBSNDataset(BaseDataset):
    def __init__(
        self, root=DEFAULT_DATA_DIR, is_augment=False, 
        augment_rotation=180, augment_scale=[0.8,1.2], augment_translate=[0.1,0.1]
    ):
        self.root = root
        self.data_list = sorted([
            f"{root}/{file}" for file in os.listdir(root) 
            if file.endswith('.png')
            and exists(f"{root}/{file.replace('.png', '.mat')}")
        ])
        self.data = {}
        self.num_sample = len(self.data_list)
        
        
        # Base Transforms
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
        
        # Augmentation
        if is_augment:
            self.augment_rotation = augment_rotation
            self.augment_scale = augment_scale
            self.augment_translate = augment_translate
            
            augment_image_transform = transforms.Compose([
                BoundedRandomAffine(augment_rotation, scale=augment_scale, translate=augment_translate),
            ])
            self.augment_transform = transforms.Compose([
                self.transform,
                transforms.Lambda(
                    lambda x: (
                        augment_image_transform(x[0]), 
                        x[1]
                    )
                )
            ])
            
        if self.num_sample > 0:
            image, hbs = self[0]
            image, hbs = self.transform((image, hbs))
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
                image, hbs = load_data(file_name)
                self.data[file_name] = (image, hbs)
            else:
                image, hbs = self.data[file_name]
            return image, hbs
    
    def get_size(self):
        return self.H, self.W, self.C_image, self.C_hbs
