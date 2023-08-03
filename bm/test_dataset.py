# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import warnings
from unittest import mock

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from . import env
from . import studies
from . import dataset as dset
from .studies import schoffelen2019


def test_epochs_definition_on_fake_schoffelen2019_recording() -> None:
    recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    epoch_def_w = dset.SegmentDataset.Factory(condition="word", tmax=1.0)
    epoch_def_3 = dset.SegmentDataset.Factory(condition=3.0, tmax=1.0)
    with schoffelen2019.mock.data():
        output = epoch_def_w.apply(recording)
        assert output is not None
        assert len(output) == 2
        output = epoch_def_3.apply(recording)
        assert output is not None
        assert len(output) == 27


def test_epochs_dataset_on_fake_schoffelen2019_recording() -> None:
    recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    recording._subject_index = 0  # needs to be initialized
    recording._recording_index = 0  # needs to be initialized
    fact = dset.SegmentDataset.Factory(condition="word", tmin=-0.5, tmax=0.5)
    with schoffelen2019.mock.data():
        ds = fact.apply(recording)
        assert ds is not None
        feature_0, mask_0, _ = ds._get_feature(0)
        full_feature = ds._get_full_feature()
        batch = next(iter(ds))
        assert full_feature.shape == (2, 99_999)
    np.testing.assert_array_almost_equal(feature_0, batch.features)
    n_times = feature_0.shape[1]
    assert feature_0[0, n_times // 2 - 1] == 0
    assert feature_0[0, n_times // 2] >= 1
    assert batch.meg.shape[1] == n_times
    assert batch.features_mask.shape[1] == batch.features.shape[1]
    assert batch.features.shape[0] == sum([f.dimension for f in ds.features.values()])


def test_epochs_dataset_slicing(tmp_path: Path) -> None:
    with env.temporary(cache=tmp_path / "fake_cache_slice"):
        recording = studies.register["fake"]("sub-A2002")  # type: ignore
    recording._subject_index = 0  # needs to be initialized
    recording._recording_index = 0  # needs to be initialized
    fact = dset.SegmentDataset.Factory(
        condition="word", decim=1, tmin=-0.5, tmax=0.5, sample_rate=200)
    meg_features = fact.apply(recording)
    assert meg_features is not None
    meg = meg_features[0].meg
    assert isinstance(meg, torch.Tensor)
    meg_features_except0 = meg_features[1:]  # all samples but the first one
    assert isinstance(meg_features_except0, dset.SegmentDataset)
    assert len(meg_features_except0) == len(meg_features) - \
        1, "Only 1 sample should have been removed"
    assert meg_features_except0._get_bounds_times(0) == meg_features._get_bounds_times(1), (
        "Bounds indices for sample 0 of meg_features_expect0 should corresponds to "
        "sample 1 of meg_features."
    )
    for mf in meg_features:
        assert mf.meg.shape[1] == mf.features.shape[1], \
            "features & epochs should have same n_times"
    # collate
    num = len(meg_features)
    out = dset.SegmentBatch.collate_fn([batch for batch in meg_features])
    assert isinstance(out, dset.SegmentBatch)
    assert out.subject_index.shape == (num,)
    assert out.meg.dim() == 3
    assert out.features.shape[0] == num


def test_get_datasets(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # make sure no warning is triggerred
        with env.temporary(cache=tmp_path / "fake_cache_dataset"):
            datasets = dset.get_datasets(
                [dict(study="fake")], n_recordings=2, test_ratio=0.2,
                valid_ratio=0.1, condition=.2, split_assign_seed=87,
                min_n_blocks_per_split=1,
                # min_block_duration=4.0,  # FIXME should not be needed
            )  # TODO: for some reason, the CI does not like filtering here :s
            assert len(datasets) == 3
            assert all(isinstance(ds, ConcatDataset) for ds in datasets)
            assert all(len(ds) > 0 for ds in datasets)
            assert len(datasets.train) > len(datasets.valid)
            assert len(datasets.train) > len(datasets.test)


def test_extract_recordings() -> None:
    selections = [dict(study="schoffelen2019", modality="visual"),
                  dict(study="schoffelen2019", modality="audio")]
    with mock.patch.object(schoffelen2019.paths.StudyPaths, 'is_valid', return_value=True):
        recordings = dset._extract_recordings(selections, n_recordings=4)
    uids = [recording.subject_uid for recording in recordings]
    assert uids == ["sub-V1001", "sub-A2002", "sub-V1002", "sub-A2003"]
