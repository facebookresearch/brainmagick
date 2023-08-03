# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Features describe how to transform the input sparse annotation, e.g.
words or wav file names, into actual dense features for training neural network,
or to be used as targets for the contrastive loss.
"""
from .base import FeaturesBuilder, Feature  # noqa
from . import basic  # noqa
from . import audio  # noqa
from . import embeddings  # noqa
