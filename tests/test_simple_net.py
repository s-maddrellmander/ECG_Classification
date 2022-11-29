import logging

from models.simple_net import NeuralNetwork
import torch

def test_NeuralNetwork():
    model = NeuralNetwork()
    X = torch.rand(1, 12000)
    probs = model(X)
    assert probs.shape == (1, 5)
    