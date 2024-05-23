import os

import scipy.io as sio
from genericpath import exists
from PIL import Image
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from data.custom_transform import BoundedRandomAffine, SoftLabel
from typing import Optional
from config import HBSNetConfig

DEFAULT_DATA_DIR = "img/generated"


def load_data(file_path):
    image_path = file_path
    hbs_path = file_path.replace(".png", ".mat")

    if not exists(hbs_path) or not exists(image_path):
        raise FileNotFoundError("Image or HBS file not found")

    image = Image.open(image_path)
    hbs = sio.loadmat(hbs_path)["hbs"]
    return image, hbs


class HBSNDataset(BaseDataset):
    def __init__(
        self,
        root=DEFAULT_DATA_DIR,
        is_augment=False,
        augment_rotation=180,
        augment_scale=[0.8, 1.2],
        augment_translate=[0.1, 0.1],
        is_soft_label=True,
        config: Optional[HBSNetConfig] = None,
    ):
        if config:
            self.root = config.data_dir
            self.is_augment = config.is_augment
            self.augment_rotation = config.augment_rotation
            self.augment_scale = config.augment_scale
            self.augment_translate = config.augment_translate
            self.is_soft_label = config.is_soft_label
        else:
            self.root = root
            self.is_augment = is_augment
            self.augment_rotation = augment_rotation
            self.augment_scale = augment_scale
            self.augment_translate = augment_translate
            self.is_soft_label = is_soft_label

        self.data_list = sorted(
            [
                f"{root}/{file}"
                for file in os.listdir(root)
                if file.endswith(".png")
                and exists(f"{root}/{file.replace('.png', '.mat')}")
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
            if self.is_soft_label
            else transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        )
        hbs_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Lambda(
            lambda x: (image_transform(x[0]), hbs_transform(x[1]))
        )

        # Augmentation
        if self.is_augment:
            augment_image_transform = transforms.Compose(
                [
                    BoundedRandomAffine(
                        self.augment_rotation,
                        scale=self.augment_scale,
                        translate=self.augment_translate,
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
        return self.height, self.width, self.input_channels, self.output_channels


class HBSNDataset_V2(HBSNDataset):
    output_size = 64

    def __getitem__(self, index):
        image, hbs = super().__getitem__(index)
        # print(hbs.shape)
        hbs = hbs[
            self.output_size : -self.output_size,
            self.output_size : -self.output_size,
            :,
        ]
        return image, hbs
