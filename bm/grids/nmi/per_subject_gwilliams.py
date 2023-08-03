# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Getting models trained on varying number of subjects."""

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
        'gwilliams2022',
    ]
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset]}, seed=seed)
            sub.bind_({'dset.n_subjects_test': 3})

            for n_subj in range(3, 28, 3):
                sub({'dset.n_subjects': n_subj})
