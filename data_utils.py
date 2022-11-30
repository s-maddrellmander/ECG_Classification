import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from functools import partial

class ECG_Dataset(Dataset):
    def __init__(self, data, targets):
        # self.data = torch.tensor(data).to(torch.float32)
        # self.targets = torch.LongTensor(targets)
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    """Collate Function to process each batch of the dataset
    Then any processing on the dataset that needs to be applied is done here
    From the PyTorch Docs:
    `After fetching a list of samples using the indices from sampler,
    the function passed as the collate_fn
    argument is used to collate lists of samples into batches.`
    """
    data = torch.Tensor(np.array([b[0] for b in batch])).to(dtype=torch.float32)
    target = torch.LongTensor(np.array([b[1] for b in batch]))
    return data, target


def preprocess_data(X, y, batch_size=1024):
    logging.info(f"Building Dataset")
    # Convert y list of values into numpy array
    y = np.array(y.tolist())
    
    dataset = ECG_Dataset(X, y)
    try:
        import poptorch
        options = poptorch.Options()
        options.outputMode(poptorch.OutputMode.All)
        options.deviceIterations(10)
        dataloader = poptorch.DataLoader(poptorch.Options(),
                                    dataset=dataset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=True,
                                    num_workers=16)
        IS_IPU = True
    except ImportError:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn)
        IS_IPU = False
    logging.info(f"Loaded data into dataloader")
    train_features, train_labels = next(iter(dataloader))
    logging.info(f"Feature batch shape: {train_features.size()}")
    logging.info(f"Labels batch shape: {train_labels.size()}")
    return dataloader