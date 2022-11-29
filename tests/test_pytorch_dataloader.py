import logging
import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils import preprocess_data, ECG_Dataset

@pytest.mark.parametrize("batch_size", ([5, 12, 16, 1024]))
def test_preprocess_data(batch_size):
    X = np.random.random((batch_size*10, 100, 12))
    y = np.random.randint(low=0, high=4, size=(batch_size*10))
    
    dataloader = preprocess_data(X, y, batch_size=batch_size)
    assert type(dataloader) == torch.utils.data.dataloader.DataLoader
    assert dataloader.batch_size == batch_size
    for data in dataloader:
        assert data[0].shape == (batch_size, 100, 12)
        assert data[1].shape == (batch_size,)


@pytest.mark.parametrize("batch_size", ([5]))
@pytest.mark.parametrize("d_type", ([np.float32, np.float64]))
def test_ECG_Dataset(batch_size, d_type):
    X = np.random.random((batch_size*10, 100, 12)).astype(d_type)
    y = np.random.randint(low=0, high=4, size=(batch_size*10))
    
    dataset = ECG_Dataset(X, y)
    assert dataset.data.shape == (batch_size * 10, 100, 12)
    assert dataset.data.dtype == torch.float32
    assert dataset.targets.dtype == torch.int32
    assert dataset.targets.shape == (batch_size * 10, )
    # import ipdb; ipdb.set_trace()
    assert dataset[0][0].shape == (100, 12)
    assert dataset[0][0].dtype == torch.float32
    assert dataset[0][1].dtype == torch.int32
    assert dataset[0][1].shape == ()