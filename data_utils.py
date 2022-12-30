import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECG_Dataset(Dataset):

    def __init__(self, data, targets):
        self.data = torch.tensor(data).to(torch.float32)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


def preprocess_data(X, y):
    logging.info(f"Building Dataset")
    # Convert y list of values into numpy array
    # Transpose the order of the dimmensions here
    X = np.transpose(X, (0, 2, 1))
    y = np.array(y.tolist())

    dataset = ECG_Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    logging.info(f"Loaded data into dataloader")
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    return dataloader
