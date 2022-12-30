import pytest
import torch

from models import conv_net


@pytest.mark.parametrize("out_channels,kernel_size,stride,padding", [
    (3, 1, 1, 0),
    (3, 1, 3, 0),
    (3, 2, 1, 4),
    (3, 3, 1, 1),
    (3, 3, 1, 2),
    (3, 2, 1, 0),
])
def test_conv1d(out_channels, kernel_size, stride, padding):
    X = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                     dtype=torch.float32)
    m = torch.nn.Conv1d(3,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding)
    out = m(X)
    l_out = (X.shape[1] + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    assert out.shape == (out_channels, l_out)


def test_conv_net():
    X = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]],
                     dtype=torch.float32)
    y = torch.tensor([1])
    # seq_len, in_channels, out_channels, kernel_size, stride, padding, num_classes
    model = conv_net.ConvNet(X.shape[2], 3, 5, 3, 1, 1, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossFn = torch.nn.CrossEntropyLoss()

    pred = model(X)
    assert pred.size(1) == (2)
    loss = lossFn(pred, y)
    # TODO: Add some extra assertations here
