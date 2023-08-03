# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Testing different learning rates and batch size."""
from itertools import product  # noqa
from .._explorers import ClipExplorer


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200,
        partition="learnlab",
    )
    # See conf/model/clip_conv.yaml for the configuration used.
    launcher.bind_({'model': 'clip_conv'})

    seeds = [2036, 2037, 2038]
    audio_sets = [
        'gwilliams2022',
    ]
    lrs = [1e-4, 3e-4, 6e-4, 1e-3]
    batch_sizes = [32, 64, 128, 256]
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset]}, seed=seed)
            if dset == 'broderick2019':
                sub.bind_({'test.wer_recordings': 100})
            for lr, batch_size in product(lrs, batch_sizes):
                sub({'optim.lr': lr, 'optim.batch_size': batch_size})
            for offset in [0, 50, 100, 150, 200, 250, 300]:
                sub({'task.offset_meg_ms': offset})
            # comparing with autoreject
            for n_rec in [16]:
                sub.bind_({'dset.n_recordings': n_rec})
                sub()
                sub({'dset.autoreject': True, 'norm.max_scale': 1e12})
