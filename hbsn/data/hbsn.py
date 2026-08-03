"""HBSNDataset：读 img/**/*.npy（float32 CHW (2,256,256)，由 tools/convert_mat_to_npy.py 生成）。"""
import os

import numpy as np
from PIL import Image
from torchvision.transforms import transforms

from hbsn.data.dataset import BaseDataset
from hbsn.data.transforms import BoundedRandomAffine, SoftLabel


class HBSNDataset(BaseDataset):
    def __init__(self, config, is_test: bool = False):
        self.config = config

        if not is_test:
            self.data_dir = self.config.data_dir
        elif self.config.test_data_dir:
            self.data_dir = self.config.test_data_dir
        else:
            raise FileNotFoundError("Test data directory not found")

        self.data_list = sorted(
            [
                os.path.join(self.data_dir, file)
                for file in os.listdir(self.data_dir)
                if file.endswith(".png")
                and os.path.exists(
                    os.path.join(self.data_dir, file.replace(".png", ".npy"))
                )
            ]
        )
        self.num_sample = len(self.data_list)

        # Base transforms（SoftLabel 语义上应作用于标签，但旧行为作用于输入图像；
        # is_soft_label 保留旧行为以保证数值等价，见 README）
        image_transform = (
            transforms.Compose(
                [transforms.Grayscale(), transforms.ToTensor(), SoftLabel(kernel_size=5)]
            )
            if self.config.is_soft_label
            else transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        )
        # npy 已是 (C,H,W) float32 张量，无需 ToTensor
        self.transform = transforms.Lambda(
            lambda x: (image_transform(x[0]), x[1])
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
                    transforms.Lambda(
                        lambda x: (augment_image_transform(x[0]), x[1])
                    ),
                ]
            )

        if self.num_sample > 0:
            image, hbs = self[0]
            image, hbs = self.transform((image, hbs))
            C_image, H_image, W_image = image.shape
            C_hbs, _, _ = hbs.shape
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
        image = Image.open(file_name)
        hbs = np.load(file_name.replace(".png", ".npy"))
        if self.config.masked_size > 0:
            m = self.config.masked_size
            hbs = hbs[:, m:-m, m:-m]  # npy 为 CHW，裁剪 H/W 两维
        return image, hbs

    def get_size(self):
        return (
            self.height,
            self.width,
            self.input_channels,
            self.output_channels,
        )
