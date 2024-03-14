import os

import scipy.io as sio
from genericpath import exists
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms

DEFAULT_DATA_DIR = 'img/generated'

def load_data(file_path):
    image_path = file_path
    hbs_path = file_path.replace('.png', '.mat')
    
    if not exists(hbs_path) or not exists(image_path):
        raise FileNotFoundError("Image or HBS file not found")
    
    image = Image.open(image_path)
    hbs = sio.loadmat(hbs_path)['hbs']
    return image, hbs

class TrainSubset(Subset):
    def __init__(self, dataset, indices, transforms) -> None:
        super().__init__(dataset, indices)
        self.transforms = transforms
        
    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

class HBSNDataset(Dataset):
    def __init__(self, root=DEFAULT_DATA_DIR, is_augment=False):
        self.root = root
        self.is_augment = is_augment
        
        # self.augment_transform = transforms.Compose([
        #     transforms.RandomAffine(180, scale=(0.5, 1.2), translate=(0.2, 0.2))
        # ])
        
        augment_image_transform = transforms.Compose([
            transforms.RandomAffine(180, scale=(0.5, 1.2), translate=(0.2, 0.2)),
        ])
        image_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        hbs_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.augment_transform = transforms.Lambda(
            lambda x: (
                augment_image_transform(x[0]), 
                x[1]
            )
        )
        self.transform = transforms.Lambda(
            lambda x: (
                image_transform(x[0]), 
                hbs_transform(x[1])
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
        file_name = self.data_list[index]
        if file_name not in self.data:
            image, hbs = self.transform(load_data(file_name))
            self.data[file_name] = (image, hbs)
        else:
            image, hbs = self.data[file_name]
        return image, hbs
    
    def get_size(self):
        return self.H, self.W, self.C_image, self.C_hbs
    
    def get_dataloader(self, batch_size=32, split_rate=0.8):
        # Split dataset into train and test sets
        # train_size = int(split_rate * len(self))
        # test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [split_rate, 1-split_rate])
        train_dataset = TrainSubset(train_dataset.dataset, train_dataset.indices, self.augment_transform)
        # train_dataset.transform = self.augment_transform if self.is_augment else self.transform
        # test_dataset.transform = self.transform

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=True, num_workers=4, drop_last=True
            )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            pin_memory=True, num_workers=4, drop_last=True
            )
        return train_dataloader, test_dataloader
