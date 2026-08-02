import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from data.transform_subset import TransformSubset

RANDOM_SEED = 960717  # must match train.RANDOM_SEED for reproducible splits


def worker_init_fn(worker_id: int) -> None:
    # numpy RNG is global per worker; seed so augmentation differs across
    # workers but is reproducible across runs
    np.random.seed(RANDOM_SEED + worker_id)


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
        pin_memory=True,
        num_workers=None,
    ):
        if num_workers is None:
            num_workers = getattr(self.config, "num_workers", 1)

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
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=drop_last,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
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
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=drop_last,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )
        else:
            test_dataloader = None

        return train_dataloader, test_dataloader
