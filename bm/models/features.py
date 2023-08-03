# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Models to be applied on the features before applying the contrastive loss.
"""

from .common import ConvSequence
import logging

logger = logging.getLogger(__name__)


class DeepMel(ConvSequence):
    """DeepMel model that extracts features from the Mel spectrogram.

    Parameters
    ----------
    n_in_channels :
        Number of input channels.
    n_hidden_channels :
        Number of channels in hidden layers.
    n_hidden_layers :
        Number of hidden layers.
    n_out_channels :
        Number of output channels.
    kwargs:
        Additional keyword arguments to pass to ConvSequence.
    """
    def __init__(self, n_in_channels: int, n_hidden_channels: int, n_hidden_layers: int,
                 n_out_channels: int, **kwargs):
        channels = \
            [n_in_channels] + [n_hidden_channels] * (n_hidden_layers - 1) + [n_out_channels]
        super().__init__(channels, **kwargs)
