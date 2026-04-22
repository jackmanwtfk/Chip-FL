import numpy as np

from torch.utils.data import DataLoader, Subset




class DataManager:
    def __init__(self, name, batch_size, resize=None):
        self.name = name
        self.batch_size = batch_size
        self.resize = resize

    def get_datasets(self):
        raise NotImplementedError
    
    def get_dataloaders(self):
        train_set, test_set = self.get_datasets()

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return train_loader, test_loader


def split_dataset(dataset, num_splits=1):
    " split dataset into multiple subsets "
    total_size = len(dataset)
    split_size = total_size // num_splits
    indices = list(range(total_size))
    np.random.shuffle(indices)

    subsets = []
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else total_size
        subset_indices = indices[start_idx:end_idx]
        subsets.append(Subset(dataset, subset_indices))

    return subsets
