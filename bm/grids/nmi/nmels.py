# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Experimenting with the number of mel bands."""

from itertools import product  # noqa
from .._explorers import ClipExplorer
from bm.train import main  # noqa


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200, constraint="volta32gb",
        partition="devlab,learnlab",
    )
    launcher.bind_({'model': 'clip_conv'})

    seeds = [2036, 2037, 2038]
    audio_sets = [
        'audio_mous',
        'gwilliams2022',
        'broderick2019',
        'brennan2019',
    ]
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset]}, seed=seed)
            if dset in ['broderick2019']:
                sub.bind_({'test.wer_recordings': 100})
            if dset in audio_sets[:1]:
                # audio_mous present sentences in random orders to subjects,
                # we use the sequence uid to assign to train / valid / test splits.
                sub.bind_({'dset.force_uid_assignement': True})
            for nmel in [20, 40, 80, 120]:  # comparing number of mel bands.
                mel = sub.bind({'dset.features': ['MelSpectrum']})
                mel.bind_({'dset.features_params.MelSpectrum.n_mels': nmel})
                mel()
                mel({'feature_model': 'deep_mel'})
                mse = mel.bind({'optim.loss': 'mse'})
                mse()
                # Uncomment once the first XP has done training.
                # xp = main.get_xp(mse._argv)
                # mel({'dset.features': ['MelSpectrum'], 'optim.max_batches': 1, 'optim.lr': 0,
                #      'optim.epochs': 1},
                #     continue_sig=xp.sig, continue_best=True)
