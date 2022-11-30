import logging
import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils import preprocess_data, ECG_Dataset, collate_fn

@pytest.mark.parametrize("batch_size", ([5, 12, 16, 1024]))
def test_preprocess_data(batch_size):
    X = np.random.random((batch_size*10, 100, 12))
    y = np.random.randint(low=0, high=4, size=(batch_size*10))
    
    dataloader = preprocess_data(X, y, batch_size=batch_size)
    # assert type(dataloader) == torch.utils.data.dataloader.DataLoader
    assert dataloader.batch_size == batch_size
    for data in dataloader:
        # assert type(data) == list
        assert type(data[0]) == torch.Tensor
        assert type(data[1]) == torch.Tensor
        assert data[0].shape == (batch_size, 100, 12)
        assert data[1].shape == (batch_size,)


@pytest.mark.parametrize("batch_size", ([5]))
def test_ECG_Dataset(batch_size):
    X = np.random.random((batch_size*10, 100, 12))
    y = np.random.randint(low=0, high=4, size=(batch_size*10))
    
    dataset = ECG_Dataset(X, y)
    assert dataset.data.shape == (batch_size * 10, 100, 12)
    assert dataset.targets.shape == (batch_size * 10, )
    assert dataset[0][0].shape == (100, 12)

    assert dataset[0][1].shape == ()
    
@pytest.mark.parametrize("d_type", ([np.float32, np.float64]))
def test_collate_fn(d_type):
    # Inputs: batch - what is the batch object
    # Input is a list of length batch size, containing tuples of tensors for the 
    # dataset. In this case X, and y being (bs x seq_len x 12 channels)
    # and the target a tensor of a single value
    # Returns: The batch once processed - what format is that?
    batch_size = 10
    data = [(np.random.random((1, 100, 12)).astype(d_type), np.random.randint((1, ))) for _ in range(batch_size)]
    outputs = collate_fn(data)
    assert type(outputs) == tuple
    assert outputs[0].dtype == torch.float32
    assert outputs[1].dtype == torch.int64
    