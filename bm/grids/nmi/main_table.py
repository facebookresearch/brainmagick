# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product  # noqa
from .._explorers import ClipExplorer
from ...train import main  # noqa
"""Results from the main table."""


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200,
        partition="learnlab",
        constraint="volta32gb",
    )
    launcher.bind_({
        'model': 'clip_conv',
    })

    seeds = [2036, 2037, 2038]
    audio_sets = [
        'audio_mous',
        'gwilliams2022',
        'broderick2019',
        'brennan2019',
    ]

    # Results from Table 2.
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset]}, seed=seed)
            if dset in ['broderick2019']:
                # Faster eval during training, this is only during training, not
                # for the final eval!
                sub.bind_({'test.wer_recordings': 100})

            if dset == 'audio_mous':
                # audio_mous present sentences in random orders to subjects,
                # we use the sequence uid to assign to train / valid / test splits.
                sub.bind_({'dset.force_uid_assignement': True})
            sub()
            # Following XP is just to get a noise level baseline
            sub({'optim.max_batches': 1, 'optim.epochs': 1, 'test.wer_random': True})
            # # Variations with different input speech-related representations.
            sub({'dset.features': ['MelSpectrum']})
            sub({'dset.features': ['MelSpectrum'], 'feature_model': 'deep_mel'})  # DeepMel
            # Then we train a regression model.
            ssub = sub.bind({'optim.loss': 'mse', 'dset.features': ['MelSpectrum']})
            ssub()
            # Uncomment once the first XP has done training.
            # xp = main.get_xp(ssub._argv)
            # sub({'dset.features': ['MelSpectrum'], 'optim.max_batches': 1, 'optim.lr': 0,
            #      'optim.epochs': 1},
            #     continue_sig=xp.sig, continue_best=True)
