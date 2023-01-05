import logging
import sys
import torch
import torch.nn as nn
import pytest
from torchsummary import summary

from models.transformer import TransformerModel, BaseEncoder
from models.transformer import (TransformerForSequenceClassification,
                                TransformerEncoder, Embeddings,
                                TransformerEncoderLayer, FeedForward,
                                MultiHeadAttention, AttentionHead,
                                scaled_dot_product_attention)

from utils import parse_cmd_args


@pytest.mark.parametrize("in_x, n_emb, dim_emb, out_y", [(4, 10, 3, 3),
                                                         (128, 10, 32, 32)])
def test_embedding(in_x, n_emb, dim_emb, out_y, bs=2):
    embedding = nn.Embedding(n_emb, dim_emb)
    rand_inp = torch.randint(10, (bs, in_x))
    input = torch.LongTensor(rand_inp)
    output = embedding(input)
    assert output.shape == (bs, in_x, out_y)  # shape = (bs, seq_len, emb_dim)


class TestTransformerSections:

    def setup_method(self, ):
        # Set and get default command line args
        sys.argv[1:] = []
        config = parse_cmd_args()
        config.model.vocab_size = 10000
        config.model.embed_dim = 128
        config.model.num_heads = 4
        config.model.hidden_size = 128
        config.model.num_hidden_layers = 2
        config.model.num_labels = 5
        self.seq_cls = TransformerForSequenceClassification(config.model)
        self.encoder = TransformerEncoder(config.model)
        self.emb = Embeddings(config.model)
        self.layer = TransformerEncoderLayer(config.model)
        self.ff = FeedForward(config.model)
        self.mhsa = MultiHeadAttention(config.model)
        self.layer_norm = nn.LayerNorm(config.model.hidden_size)
        self.attn_out_size = 384
        self.attn = AttentionHead(embed_dim=config.model.embed_dim,
                                  head_dim=self.attn_out_size)
        self.base = BaseEncoder(config.model.num_channels,
                                config.model.embed_dim)

        self.config = config

    def test_AttentionHead(self, ):
        x = torch.rand((1, 1000, self.config.model.embed_dim))
        out = self.attn(x)
        assert out.size() == (1, 1000, self.attn_out_size)

    def test_MultiHeadAttention(self):
        x = torch.rand((1, 1000, self.config.model.embed_dim))
        out = self.mhsa(x)
        assert out.size() == (1, 1000, self.config.model.embed_dim)

    def test_FeedForward(self):
        x = torch.randn((1, 1000, self.config.model.embed_dim))
        x = self.layer_norm(x)
        assert x.size() == (1, 1000, self.config.model.hidden_size)
        out = self.ff(x)
        assert out.size() == (1, 1000, self.config.model.embed_dim)

    def test_TransformerForSequenceClassification(self):
        x = torch.rand((1, 1000, 12))
        out = self.seq_cls(x)
        assert out.size() == (1, 5)

    def test_TransformerEncoder(self):
        x = torch.rand((1, 1000, 12))
        out = self.encoder(x)
        assert out.size() == (1, 1000, self.config.model.embed_dim)

    def test_TransformerEncoderLayer(self):
        x = torch.rand((1, 1000, self.config.model.embed_dim))
        out = self.layer(x)
        assert out.size() == x.size()

    def test_BaseEncoder(self):
        x = torch.rand((1, 1000, 12))
        out = self.base(x)
        assert out.size() == (1, 1000, self.config.model.embed_dim)

    def test_Embeddings(self):
        # Start with the absolute raw inputs
        x = torch.rand((1, 1000, 12))
        emb = self.emb(x)
        assert emb.size() == (1, 1000, self.config.model.embed_dim)


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

        # Set and get default command line args
        sys.argv[1:] = []
        config = parse_cmd_args()
        config.model.vocab_size = 256
        config.model.embed_dim = 128
        config.model.num_heads = 4
        config.model.hidden_size = 128
        config.model.num_hidden_layers = 2
        config.model.num_labels = 5
        self.hf_transformer = TransformerForSequenceClassification(
            config.model)

    def teardown_method(self, ):
        # tear down self.attribute
        pass

    def test_compare_model_parameters(self):
        summary_base = summary(self.model,
                               input_size=(1000, 12),
                               batch_size=-1,
                               verbose=1)
        summary_hf = summary(self.hf_transformer,
                             input_size=(1000, 12),
                             batch_size=-1,
                             verbose=1)
        assert summary_base.summary_list[
            4].num_params == summary_hf.summary_list[9].num_params

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
