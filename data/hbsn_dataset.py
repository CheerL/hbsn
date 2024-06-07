import os
from genericpath import exists

import scipy.io as sio
from PIL import Image
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset, BaseDatasetConfig
from data.custom_transform import BoundedRandomAffine, SoftLabel


def load_data(file_path):
    image_path = file_path
    hbs_path = file_path.replace(".png", ".mat")

    if not exists(hbs_path) or not exists(image_path):
        raise FileNotFoundError("Image or HBS file not found")

    image = Image.open(image_path)
    hbs = sio.loadmat(hbs_path)["hbs"]
    return image, hbs


class HBSNDatasetConfig(BaseDatasetConfig):
    data_dir = "img/generated"
    test_data_dir = "img/gen2"
    augment_rotation = 180
    augment_scale = [0.5, 2]
    augment_translate = [0.5, 0.5]
    output_size = 64


class HBSNDataset(BaseDataset):
    def __init__(self, config: HBSNDatasetConfig, is_test: bool = False):
        self.config = config
        
        if not is_test:
            self.data_dir = self.config.data_dir
        elif self.config.test_data_dir:
            self.data_dir = self.config.test_data_dir
        else:
            raise FileNotFoundError("Test data directory not found")

        self.data_list = sorted(
            [
                f"{self.data_dir}/{file}"
                for file in os.listdir(self.data_dir)
                if file.endswith(".png")
                and exists(f"{self.data_dir}/{file.replace('.png', '.mat')}")
            ]
        )
        self.data = {}
        self.num_sample = len(self.data_list)

        # Base Transforms
        image_transform = (
            transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    SoftLabel(kernel_size=5),
                ]
            )
            if self.config.is_soft_label
            else transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        )
        hbs_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Lambda(
            lambda x: (image_transform(x[0]), hbs_transform(x[1]))
        )

        # Augmentation
        if self.config.is_augment and not is_test:
            augment_image_transform = transforms.Compose(
                [
                    BoundedRandomAffine(
                        self.config.augment_rotation,
                        scale=self.config.augment_scale,
                        translate=self.config.augment_translate,
                    ),
                ]
            )
            self.augment_transform = transforms.Compose(
                [
                    self.transform,
                    transforms.Lambda(lambda x: (augment_image_transform(x[0]), x[1])),
                ]
            )

        if self.num_sample > 0:
            data = self[0]
            if data is not None:
                image, hbs = data
                image, hbs = self.transform((image, hbs))
                C_image, H_image, W_image = image.shape
                C_hbs, H_hbs, W_hbs = hbs.shape
                # print(image.shape, hbs.shape)
                # assert H_image == H_hbs and W_image == W_hbs, "Image and HBS size mismatch"
                assert C_image == 1, "Image channel should be 1"
                assert C_hbs == 2, "HBS channel should be 2"

                self.height = H_image
                self.width = W_image
                self.input_channels = C_image
                self.output_channels = C_hbs

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        file_name = self.data_list[index]
        if file_name not in self.data:
            image, hbs = load_data(file_name)
            self.data[file_name] = (image, hbs)
        else:
            image, hbs = self.data[file_name]
            
        if self.config.output_size > 0:
            hbs = hbs[
                self.config.output_size : -self.config.output_size,
                self.config.output_size : -self.config.output_size,
                :,
            ]
        return image, hbs

    def get_size(self):
        return self.height, self.width, self.input_channels, self.output_channels
