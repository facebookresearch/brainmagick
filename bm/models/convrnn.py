# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from bm.utils import capture_init
from .common import ScaledEmbedding, SubjectLayers, ConvSequence


class LSTM(nn.Module):
    """A wrapper for nn.LSTM that outputs the same amount
    of features if bidirectional or not bidirectional.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x):
        x, h = self.lstm(x)
        if self.linear:
            x = self.linear(x)
        return x, h


class Attention(nn.Module):
    def __init__(self, channels: int, radius: int = 50, heads: int = 4):
        super().__init__()
        assert channels % heads == 0
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.embedding = nn.Embedding(radius * 2 + 1, channels // heads)
        # Let's make this embedding a bit smoother
        weight = self.embedding.weight.data
        weight[:] = weight.cumsum(0) / torch.arange(1, len(weight) + 1).float().view(-1, 1).sqrt()
        self.heads = heads
        self.radius = radius

        self.bn = nn.BatchNorm1d(channels)
        self.fc = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full([channels], 0.1))

    def forward(self, x):
        def _split(y):
            return y.view(y.shape[0], self.heads, -1, y.shape[2])

        content = _split(self.content(x))
        query = _split(self.query(x))
        key = _split(self.key(x))

        batch_size, _, dim, length = content.shape

        # first index `t` is query, second index `s` is key.
        dots = torch.einsum("bhct,bhcs->bhts", query, key)

        steps = torch.arange(length, device=x.device)
        relative = (steps[:, None] - steps[None, :])
        embs = self.embedding.weight.gather(
            0, self.radius + relative.clamp_(-self.radius, self.radius).view(-1, 1).expand(-1, dim))
        embs = embs.view(length, length, -1)
        dots += 0.3 * torch.einsum("bhct,tsc->bhts", query, embs)

        # we kill any reference outside of the radius
        dots = torch.where(
            relative.abs() <= self.radius, dots, torch.tensor(-float('inf')).to(embs))

        weights = torch.softmax(dots, dim=-1)
        out = torch.einsum("bhts,bhcs->bhct", weights, content)
        out += 0.3 * torch.einsum("bhts,tsc->bhct", weights, embs)
        out = out.reshape(batch_size, -1, length)
        out = F.relu(self.bn(self.fc(out))) * self.scale.view(1, -1, 1)
        return out


class ConvRNN(nn.Module):
    @capture_init
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 # Overall structure
                 depth: int = 2,
                 linear_out: bool = False,
                 complex_out: bool = False,
                 concatenate: bool = False,  # concatenate the inputs
                 # Conv structure
                 kernel_size: int = 4,
                 stride: int = 2,
                 growth: float = 1.,
                 # LSTM
                 lstm: int = 2,
                 flip_lstm: bool = False,
                 bidirectional_lstm: bool = False,
                 # Attention
                 attention: int = 0,
                 heads: int = 4,
                 # Dropout, BN, activations
                 conv_dropout: float = 0.0,
                 lstm_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 batch_norm: bool = False,
                 relu_leakiness: float = 0.0,
                 # Subject embeddings,
                 n_subjects: int = 200,
                 subject_dim: int = 64,
                 embedding_location: tp.List[str] = ["lstm"],  # can be lstm or input
                 embedding_scale: float = 1.0,
                 subject_layers: bool = False,
                 subject_layers_dim: str = "input",  # or hidden
                 ):
        super().__init__()
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")

        self._concatenate = concatenate
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.embedding_location = embedding_location

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects)
            in_channels["meg"] = dim
        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(n_subjects, subject_dim, embedding_scale)
            if "input" in embedding_location:
                in_channels["meg"] += subject_dim

        # concatenate inputs if need be
        if concatenate:
            in_channels = {"concat": sum(in_channels.values())}
            hidden = {"concat": sum(hidden.values())}

        # compute the sequences of channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]

        lstm_hidden = sum(sizes[n][-1] for n in in_channels)
        lstm_input = lstm_hidden
        if "lstm" in embedding_location:
            lstm_input += subject_dim

        # encoders and decoder
        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=stride,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm)
        self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                       for name, channels in sizes.items()})

        # lstm
        self.lstm = None
        self.linear = None
        if lstm:
            self.lstm = LSTM(
                input_size=lstm_input,
                hidden_size=lstm_hidden,
                dropout=lstm_dropout,
                num_layers=lstm,
                bidirectional=bidirectional_lstm)
            self._flip_lstm = flip_lstm

        self.attentions = nn.ModuleList()
        for _ in range(attention):
            self.attentions.append(Attention(lstm_hidden, heads=heads))

        # decoder
        decoder_sizes = [int(round(lstm_hidden / growth ** k)) for k in range(depth + 1)]
        self.final = None
        if linear_out:
            assert not complex_out
            self.final = nn.Conv1d(decoder_sizes[-1], out_channels, 1)
        elif complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(decoder_sizes[-1], 2 * decoder_sizes[-1], 1),
                nn.ReLU(),
                nn.Conv1d(2 * decoder_sizes[-1], out_channels, 1))
        else:
            params['activation_on_last'] = False
            decoder_sizes[-1] = out_channels
            assert depth > 0, "if no linear out, depth must be > 0"
        self.decoder = ConvSequence(decoder_sizes, decode=True, **params)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, (size of the input - kernel_size) % stride = 0.

        If the input has a valid length, the output
        will have exactly the same length.
        """
        for idx in range(self.depth):
            length = math.ceil(length / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride
        return int(length)

    def pad(self, x):
        length = x.size(-1)
        valid_length = self.valid_length(length)
        delta = valid_length - length
        return F.pad(x, (0, delta))

    def forward(self, inputs, batch):
        subjects = batch.subject_index
        length = next(iter(inputs.values())).shape[-1]  # length of any of the inputs

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)
        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            if "input" in self.embedding_location:
                inputs["meg"] = torch.cat([inputs["meg"], emb.expand(-1, -1, length)], dim=1)

        if self._concatenate:
            input_list = [input_ for _, input_ in sorted(inputs.items())]
            inputs = {"concat": torch.cat(input_list, dim=1)}

        inputs = {name: self.pad(input_) for name, input_ in inputs.items()}
        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](self.pad(x))

        inputs = [x[1] for x in sorted(encoded.items())]
        if "lstm" in self.embedding_location and self.subject_embedding is not None:
            inputs.append(emb.expand(-1, -1, inputs[0].shape[-1]))

        x = torch.cat(inputs, dim=1)
        if self.lstm is not None:
            x = x.permute(2, 0, 1)
            if self._flip_lstm:
                x = x.flip([0])
            x, _ = self.lstm(x)
            if self._flip_lstm:
                x = x.flip([0])
            x = x.permute(1, 2, 0)

        for attention in self.attentions:
            x = x + attention(x)

        x = self.decoder(x)

        if self.final is not None:
            x = self.final(x)

        out = x[:, :, :length]
        return out
