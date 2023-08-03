# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# flake8: noqa
"""
All the models used, ConvRNN is taken from https://arxiv.org/abs/2103.02339,
and SimpleConv is the one used in the NMI paper, https://arxiv.org/abs/2208.12266.
"""

from .convrnn import ConvRNN
from .simpleconv import SimpleConv
from .features import DeepMel
