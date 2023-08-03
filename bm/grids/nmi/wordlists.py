# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Training models on word lists, with shorter context."""

from itertools import product  # noqa
from .._explorers import ClipExplorer


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200,
        partition="devlab,learnlab",
    )
    launcher.bind_({'model': 'clip_conv', 'optim.batch_size': 128})

    seeds = [2036, 2037, 2038]
    # specific grid for word list evaluations, e.g. Table A.1
    launcher.bind_({'dset.force_uid_assignement': True})
    with launcher.job_array():
        for seed in seeds:
            sub = launcher.bind({'dset.selections': ['audio_mous_wl']}, seed=seed)
            sub.bind_({'dset.tmin': -0.3, 'dset.tmax': 0.5})
            sub()
