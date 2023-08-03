# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Ways to make the model stronger."""
import random
import torch


# We need a shared RNG to make sure all the distributed worker will skip the penalty together,
# as otherwise we wouldn't get any speed up.
penalty_rng = random.Random(1234)


def svd_penalty(model, min_size=1., dim=16, niters=2, proba=1, exact=False):
    """
    Penalty on the largest singular value for a layer.
    Args:
        - model: model to penalize
        - min_size: minimum size in kB of a layer to penalize.
        - dim: projection dimension for the svd_lowrank. Higher is better but slower.
        - niters: number of iterations in the algorithm used by svd_lowrank.
        - powm: use power method instead of lowrank SVD, my own experience
            is that it is both slower and less stable.
        - proba: probability to apply the penalty.
        - exact: use exact SVD (slow but useful at validation).
    """
    total = 0
    if penalty_rng.random() > proba:
        return 0.

    for mn, m in model.named_modules():
        if not isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
            continue
        p = m.weight
        if p.numel() / 2**8 < min_size:
            continue
        p = p.view(p.shape[0], -1)
        if exact:
            estimate = torch.svd(p, compute_uv=False)[1].pow(2).max()
        else:
            estimate = torch.svd_lowrank(p, dim, niters)[1][0].pow(2)
        total += estimate
    return total / proba
