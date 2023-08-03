# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Fair comparison between EEG and MEG by dropping channels / training data."""
from itertools import product  # noqa
from .._explorers import ClipExplorer
from ...train import main  # noqa


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200,
        partition="learnlab",
        constraint="volta32gb",
    )
    launcher.bind_({'model': 'clip_conv'})

    seeds = [2036, 2037, 2038]
    audio_sets = [
        'audio_mous',
        'gwilliams2022',
        'broderick2019',
    ]
    # For a fair comparison with Broderick, we keep 19 subjects
    # per dataset (nb of recordings will depend on dataset).
    # For MOUS, we will only have 15 hours of data left,
    # so we remove some data to align everyone on that.
    train_ratio = 0.7
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset]}, seed=seed)
            if dset in ['broderick2019']:
                sub.bind_({'test.wer_recordings': 100})

            if dset in audio_sets[:1]:
                # audio_mous present sentences in random orders to subjects,
                # we use the sequence uid to assign to train / valid / test splits.
                sub.bind_({'dset.force_uid_assignement': True})

            if dset == 'audio_mous':
                # Fair comparison to Broderick
                sub.bind_({'dset.n_recordings': 19,
                           '+simpleconv.subsample_meg_channels': 128,
                           'dset.remove_ratio': 0.})
            elif dset == 'gwilliams2022':
                # Fair comparison to Broderick
                sub.bind_({'dset.n_recordings': 140,
                           '+simpleconv.subsample_meg_channels': 128,
                           'dset.remove_ratio': 0.62 * train_ratio})
            elif dset == 'broderick2019':
                # Fair comparison to Broderick
                sub.bind_({'dset.n_recordings': 380,
                           '+simpleconv.subsample_meg_channels': 128,
                           'dset.remove_ratio': 0.21 * train_ratio})
            else:
                assert False
            sub()
            # Following XP is just to get a noise level baseline
            sub({'optim.max_batches': 1, 'optim.epochs': 1, 'test.wer_random': True})
            sub({'dset.features': ['MelSpectrum']})
            deep_mel = sub.bind({'clip.arch': True, 'clip.arch_dim': 768,
                                 'dset.features': ['MelSpectrum']})
            deep_mel({'clip.sync_grad': True, 'clip.save_best': True})
            ssub = sub.bind({'optim.loss': 'mse', 'dset.features': ['MelSpectrum']})
            ssub()
            # Uncomment once the first XP has done training.
            # xp = main.get_xp(ssub._argv)
            # sub({'dset.features': ['MelSpectrum'], 'optim.max_batches': 1, 'optim.lr': 0,
            #      'optim.epochs': 1},
            #     continue_sig=xp.sig, continue_best=True)
