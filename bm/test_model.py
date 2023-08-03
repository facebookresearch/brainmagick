# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest
from . import models as _models
from .dataset import SegmentBatch


def fake_batch():
    meg = torch.randn(2, 12, 128)
    features = torch.randn(2, 8, 128)
    subject_index = torch.tensor([1, 0], dtype=torch.int64)
    batch = SegmentBatch(
        meg=meg,
        features=features,
        features_mask=torch.ones(2, 1, 128),
        subject_index=subject_index,
        recording_index=subject_index)
    return batch


@pytest.mark.parametrize("concat", (True, False))
def test_model(concat: bool) -> None:
    model = _models.ConvRNN(in_channels=dict(meg=12),
                            out_channels=23,
                            hidden=dict(meg=200),
                            conv_dropout=0.1, growth=1.2,
                            concatenate=concat)
    batch = fake_batch()
    out = model(dict(meg=batch.meg), batch)
    assert out.shape == (2, 23, 128)


@pytest.mark.parametrize("concat", (True, False))
@pytest.mark.parametrize("depth", (0, 1, 3))
def test_model_2_inputs(concat: bool, depth: int) -> None:
    model = _models.ConvRNN(in_channels=dict(meg=12, features=8),
                            out_channels=12,
                            hidden=dict(meg=200, features=20),
                            depth=depth,
                            linear_out=True,
                            conv_dropout=0.1, growth=1.2,
                            concatenate=concat)
    batch = fake_batch()
    out = model(dict(meg=batch.meg, features=batch.features), batch)
    assert out.shape == (2, 12, 128)


def test_deep_mel():
    features = fake_batch().features

    n_in_channels = features.shape[1]
    n_hidden_channels = 3
    n_hidden_layers = 5
    n_out_channels = 2

    model = _models.DeepMel(
        n_in_channels, n_hidden_channels, n_hidden_layers, n_out_channels, kernel=3, stride=1,
        dilation_growth=2, dilation_period=5, batch_norm=True, activation_on_last=False, skip=True,
        glu_context=1, glu=2)
    out = model(features)

    assert len(model.sequence) == n_hidden_layers
    assert out.shape == (features.shape[0], n_out_channels, features.shape[2])
