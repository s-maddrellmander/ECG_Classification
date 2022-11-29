import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from functools import partial

class ECG_Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data).to(torch.float32)
        self.targets = torch.IntTensor(targets)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    """Collate Function to process each batch of the dataset
    Then any processing on the dataset that needs to be applied is done here
    """
    # import ipdb; ipdb.set_trace()
    # batch = [(torch.Tensor(t[0]), t[1]) for t in batch]
    return tuple(zip(*batch))


def preprocess_data(X, y, batch_size=1024):
    logging.info(f"Building Dataset")
    # Convert y list of values into numpy array
    y = np.array(y.tolist())
    
    dataset = ECG_Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=None)
    logging.info(f"Loaded data into dataloader")
    train_features, train_labels = next(iter(dataloader))
    # import ipdb; ipdb.set_trace()
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    return dataloader