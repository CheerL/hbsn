from torch.utils.data import DataLoader, Dataset, random_split

from config import BaseConfig
from data.transform_subset import TransformSubset


class BaseDatasetConfig(BaseConfig):
    data_dir = ""
    test_data_dir = ""
    is_augment = False
    augment_rotation = 180
    augment_scale = [0.8, 1.2]
    augment_translate = [0.1, 0.1]
    is_soft_label = True

    @property
    def augment(self):
        return (
            (
                self.augment_rotation,
                self.augment_scale,
                self.augment_translate,
            )
            if self.is_augment
            else False
        )

    @property
    def _except_keys(self):
        return super()._except_keys + [
            "is_augment",
            "augment_rotation",
            "augment_scale",
            "augment_translate",
        ]


class BaseDataset(Dataset):
    augment_transform = None
    transform = None

    def __len__(self):
        raise NotImplementedError

    def get_dataloader(self, batch_size=32, split_rate=0.8, drop_last=True):
        if split_rate == 1:
            train_dataset = self
            test_dataset = None
        elif split_rate == 0:
            train_dataset = None
            test_dataset = self
        else:
            train_num = int(len(self) * split_rate)
            train_dataset, test_dataset = random_split(
                self, [train_num, len(self) - train_num]
            )

        if train_dataset:
            train_dataset = TransformSubset.from_dataset(
                train_dataset,
                self.augment_transform
                if self.augment_transform is not None
                else self.transform,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=1,
                drop_last=drop_last,
                persistent_workers=True,
            )
        else:
            train_dataloader = None

        if test_dataset:
            test_dataset = TransformSubset.from_dataset(
                test_dataset, self.transform
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=1,
                drop_last=drop_last,
                persistent_workers=True,
            )
        else:
            test_dataloader = None

        return train_dataloader, test_dataloader
