import torch
from torch import optim
from torch import nn
from tqdm import tqdm
import sys

from torchsummary import summary

# Transformer model + utility functions

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------
# Using the HuggingFace textbook as a guide here


def scaled_dot_product_attention(query, key, value):
    # Core attention function - this is worth sketching out the dimmensions for
    dim_k = query.size(-1)  # Get the last dimmension
    scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(
        torch.tensor(dim_k))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):

    def __init__(self, embed_dim, head_dim) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state),
                                                    self.k(hidden_state),
                                                    self.v(hidden_state))
        return attn_outputs


class MultiHeadAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([head(hidden_state) for head in self.heads], dim=-1)
        x = self.output_layer(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class Embeddings(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        # self.token_embeddings = nn.Embedding(config.vocab_size,
        #                                      config.hidden_size)
        # assert config.embed_dim == config.hidden_size
        self.token_embeddings = BaseEncoder(config.num_channels,
                                            config.embed_dim)
        self.positional_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.positional_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForSequenceClassification(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(
            x
        )[:,
          0, :]  # This is to select the [CLS] token state TODO: Need to add this to datastructre
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def test_dummy_transformer(config):
    model = TransformerForSequenceClassification(config.model)
    summary(model, input_size=(1000, 12), batch_size=-1)


# ----------------------


class Pooler(torch.nn.Module):

    def __init__(self, n_hidden):
        super().__init__()
        self.fc = torch.nn.Linear(n_hidden, n_hidden, bias=True)
        torch.nn.init.normal_(self.fc.weight, std=0.02)  # scale with size?
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x[..., 0, :]  # put output at CLS token
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class BaseEncoder(torch.nn.Module):

    def __init__(self, n_inputs, n_hidden=128) -> None:
        super().__init__()
        self.layer_0 = nn.Linear(n_inputs, n_hidden)
        # self.layers = [nn.Linear]
        self.out = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.out(x)
        return x


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 ntoken,
                 emb_dim,
                 nhead,
                 nhid,
                 nlayers,
                 nclasses,
                 dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError(
                'TransformerEncoder module does not exist in PyTorch 1.1 or '
                'lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(
            ntoken, emb_dim)  # out_shape = (bs, seq_len, emb_dim)
        self.ninp = emb_dim
        self.decoder = nn.Linear(emb_dim, nclasses)
        self.pooler = Pooler(nhid)
        self.classification_head = nn.Linear(emb_dim, nclasses)

        self.float_encoder = BaseEncoder(n_inputs=12, n_hidden=emb_dim)

        # self.init_weights()
        # self.batch_size = batch_size
        # self.seq_len = seq_len
        self.emb_dim = emb_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.bias)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        self.src_mask = None
        # TODO: Why i the batchsize coming in as 2?
        # src = src[0]
        # src = torch.reshape(src, [12, 1000])
        # assert src.dtype
        # src = self.encoder(src) * math.sqrt(self.ninp)  # out_shape = (bs, seq_len, emb_dim)
        src = self.float_encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        # return output
        # output = nn.Flatten()(output)
        output = self.pooler(output)
        output = self.classification_head(output)
        # TODO: If we want to add the self-supervision, could do this by just adding another head
        # Return output (target) plus masked in-fill
        # Use self.decoder(output), where the dimension is set to the number of channels
        return output
        # output = self.decoder(output)

        # return F.log_softmax(output, dim=-1)


def train_transformer(trainloader, testloader, opts):
    # seq_len = trainloader.dataset.data[0].shape[1]
    # in_channels = trainloader.dataset.data[0].shape[0]
    # num_classes = trainloader.dataset.targets.max() + 1

    # model = TransformerModel(ntoken=opts.model.n_token,
    #                          emb_dim=opts.model.emb_dim,
    #                          nhead=opts.model.n_head,
    #                          nhid=opts.model.n_hid,
    #                          nlayers=opts.model.n_layers,
    #                          dropout=opts.model.dropout,
    #                          nclasses=5)
    model = TransformerForSequenceClassification(opts.model)
    summary(model, input_size=(1000, 12), batch_size=-1)
    total_epochs = opts.model.epochs
    criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # TODO: Tidy up the TQDM sections here
    with tqdm(range(total_epochs), colour="#FF6F79"
              ) as all_epochs:  # loop over the dataset multiple times
        for epoch in all_epochs:
            all_epochs.set_description(
                f"Training for {epoch} / {total_epochs} epochs")
            running_loss = 0.0
            running_acc = 0.0
            with tqdm(trainloader, unit="batch", colour="#B5E4EB") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    # TODO: This is hackey really, need a longer term soltuion here
                    inputs = torch.permute(inputs, (0, 2, 1))

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    running_acc += 100 * (
                        (predicted == labels).sum().item() / labels.size(0))
                    # print statistics
                    running_loss += loss.item()
                    if i % 1 == 0:
                        tepoch.set_postfix(
                            loss=f"{running_loss / (i + 1):.5f}",
                            acc=f"{running_acc / (i + 1):.1f}%")

            # TODO: Add the validation to get an accuracy as well / ROC for comparisson
            # print('Finished Training')
            correct = 0
            total = 0
            val_loss = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for i, data in enumerate(testloader, 1):
                    inputs, labels = data
                    # calculate outputs by running images through the network
                    # TODO: This is hackey really, need a longer term soltuion here
                    inputs = torch.permute(inputs, (0, 2, 1))
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            all_epochs.set_postfix(
                val_acc=f'{100 * correct / total:.1f}%',
                val_loss=f"{val_loss / len(testloader):.4f}")
