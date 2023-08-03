# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Ablation grid."""
from itertools import product  # noqa
from .._explorers import ClipExplorer


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200,
        partition="learnlab",
    )
    # See conf/model/clip_conv.yaml for the configuration used.
    launcher.bind_({'model': 'clip_conv', 'optim.batch_size': 256})

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
            if dset == 'broderick2019':
                sub.bind_({'test.wer_recordings': 100})
            if dset in audio_sets[:1]:
                # audio_mous present sentences in random orders to subjects,
                # we use the sequence uid to assign to train / valid / test splits.
                sub.bind_({'dset.force_uid_assignement': True})

            # Reference model
            sub()
            # Then the experiments for Table 4
            # merger is spatial attention
            sub({'simpleconv.merger': False})
            sub({'simpleconv.merger_dropout': 0.})
            sub({'simpleconv.glu': 0})
            sub({'simpleconv.initial_linear': 0})
            sub({'simpleconv.gelu': False})
            sub({'simpleconv.skip': False})
            sub({'simpleconv.complex_out': False})
            sub({'simpleconv.subject_layers': False})
            sub({'simpleconv.subject_layers': False, 'simpleconv.subject_dim': 64})
            sub({'norm.max_scale': 100})
            sub({'norm.max_scale': 1e12})
