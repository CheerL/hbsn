"""BaseDataset：dataloader 构造（num_workers/pin_memory 从配置透传）+ TransformSubset。"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

RANDOM_SEED = 960717  # train.py 从这里 import，保证 split/worker/全局种子一致


def worker_init_fn(worker_id: int) -> None:
    # numpy 与 python 的 RNG 都是全局的：按 worker 播种，
    # 增强序列在不同 worker 间不同、跨运行可复现
    np.random.seed(RANDOM_SEED + worker_id)
    random.seed(RANDOM_SEED + worker_id)


class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None) -> None:
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            return self.transform(data)
        return data

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    @classmethod
    def from_dataset(cls, dataset: Dataset | Subset, transform):
        if isinstance(dataset, Subset):
            return cls(dataset.dataset, dataset.indices, transform)
        elif isinstance(dataset, Dataset):
            return cls(dataset, range(len(dataset)), transform)
        else:
            raise TypeError(f"Expected Dataset or Subset, but got {type(dataset)}")


class BaseDataset(Dataset):
    augment_transform = None
    transform = None

    def __len__(self):
        raise NotImplementedError

    def get_dataloader(
        self,
        batch_size=32,
        split_rate=0.8,
        drop_last=True,
    ):
        num_workers = getattr(self.config, "num_workers", 1)
        pin_memory = getattr(self.config, "pin_memory", False)

        if split_rate == 1:
            train_dataset = self
            test_dataset = None
        elif split_rate == 0:
            train_dataset = None
            test_dataset = self
        else:
            train_num = int(len(self) * split_rate)
            train_dataset, test_dataset = random_split(
                self,
                [train_num, len(self) - train_num],
                generator=torch.Generator().manual_seed(RANDOM_SEED),
            )

        def make_loader(dataset, shuffle):
            return DataLoader(
                TransformSubset.from_dataset(
                    dataset,
                    self.augment_transform
                    if shuffle and self.augment_transform is not None
                    else self.transform,
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=drop_last,
                persistent_workers=num_workers > 0,
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
            )

        train_dataloader = make_loader(train_dataset, True) if train_dataset else None
        test_dataloader = make_loader(test_dataset, False) if test_dataset else None
        return train_dataloader, test_dataloader
