import os

import scipy.io as sio
from genericpath import exists
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

DEFAULT_DATA_DIR = 'img/generated'

def load_data(file_path):
    image_path = file_path
    hbs_path = file_path.replace('.png', '.mat')
    image = Image.open(image_path)
    hbs = load_hbs(hbs_path)
    return image, hbs

def load_hbs(hbs_path):
    if not exists(hbs_path):
        return None

    hbs = sio.loadmat(hbs_path)['hbs']
    # Process the label data as needed
    return hbs

class HBSNDataset(Dataset):
    def __init__(self, root=DEFAULT_DATA_DIR):
        self.root = root
        
        self.image_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Grayscale(),
            # transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        self.hbs_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transform = transforms.Lambda(
            lambda x: (
                self.image_transform(x[0]), 
                self.hbs_transform(x[1])
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
        train_size = int(split_rate * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])

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
