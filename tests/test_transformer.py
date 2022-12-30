import logging
import torch
import torch.nn as nn
import pytest
from torchsummary import summary

from models.transformer import TransformerModel, BaseEncoder


@pytest.mark.parametrize("in_x, n_emb, dim_emb, out_y", [(4, 10, 3, 3),
                                                         (128, 10, 32, 32)])
def test_embedding(in_x, n_emb, dim_emb, out_y, bs=2):
    embedding = nn.Embedding(n_emb, dim_emb)
    rand_inp = torch.randint(10, (bs, in_x))
    input = torch.LongTensor(rand_inp)
    output = embedding(input)
    assert output.shape == (bs, in_x, out_y)  # shape = (bs, seq_len, emb_dim)


class TestTransformerModel:
    """Build a main test class for the transformer model."""

    def setup_method(self, ):
        self.model = TransformerModel(
            ntoken=256,  # small embedding table
            emb_dim=128,
            nhead=4,
            nhid=128,
            nlayers=2,
            dropout=0.1,
            nclasses=5)
        # configure self.attribute
        self.datum = torch.rand((1, 1000, 12))
        assert self.datum.shape == (1, 1000, 12)

    def teardown_method(self, ):
        # tear down self.attribute
        pass

    def test_encoder(self, ):
        assert self.model.float_encoder(self.datum).shape == (1, 1000, 128)

    def test_forward(self):
        output = self.model(self.datum)
        # import ipdb; ipdb.set_trace()
        assert output.shape == torch.Size([1, 5])
        pass

    def test_classification_head(self):
        # Make sure the output is the correct size
        input = torch.rand(torch.Size([1, 12000,
                                       128]))  # shape = (bs, seq_len, emb_dim)
        # input = nn.Flatten()(input)
        input = self.model.pooler(input)
        out = self.model.decoder(input)
        assert out.shape == torch.Size([1, 5])

    # @pytest.mark.skip("Sizes varrying too fast atm.")
    def test_model_summary(self):
        summary_out = summary(self.model,
                              input_size=(1000, 12),
                              batch_size=-1,
                              verbose=1)
        # assert summary_out.total_params == 249738
        # assert summary_out.trainable_params == 249738


class TestBaseEncoder:

    def setup_method(self, ):
        self.encoder = BaseEncoder(n_inputs=12, n_hidden=1024)

    @pytest.mark.parametrize('batch_size', [1, 2, 3, 4, 128])
    def test_forward(self, batch_size):
        datum = torch.rand((batch_size, 256, 12))
        out = self.encoder(datum)
        assert out.shape == (batch_size, 256, 1024)

    def test_forward_simple(self, ):
        datum = torch.rand((1, 12))
        out = self.encoder(datum)
        assert out.shape == (1, 1024)
