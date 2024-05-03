from torch.utils.data import Dataset, Subset


class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None) -> None:
        super().__init__(dataset, indices)
        self.transform = transform
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            return self.transform(data)
        return data
    
    @classmethod
    def from_dataset(cls, dataset: Dataset | Subset, transform):
        if isinstance(dataset, Subset):
            return cls(dataset.dataset, dataset.indices, transform)
        elif isinstance(dataset, Dataset):
            return cls(dataset, range(len(dataset)), transform)
        else:
            raise TypeError(f"Expected Dataset or Subset, but got {type(dataset)}")