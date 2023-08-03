# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import mock, SkipTest
from pathlib import Path
import pandas as pd
from bm import env
from .. import fake
from .. import api
from .. import test_api
from . import preproc
from . import schoffelen2019
from . import mock as schoffelenmock


def test_env() -> None:
    string = str(env)
    assert "studies" in string


def test_set_env() -> None:
    with fake.set_env(blublu=1):
        assert os.environ.get("blublu", None) == "1"
    assert os.environ.get("blublu", None) is None


def test_recording_dict() -> None:
    assert "schoffelen2019" in api.register
    assert "" not in api.register


def test_tgrid_to_df() -> None:
    data_path = Path(__file__).parent.with_name("mockdata") / "full_example.TextGrid"
    if not data_path.exists():
        raise SkipTest("Full text grid example not commited.")
    df = preproc.tgrid_to_dict(str(data_path))
    test_api.cached_assert_df_equal(df, "expected_full_example.csv")


# To bo fixed
def test_get_all_recordings() -> None:
    with mock.patch.object(schoffelen2019.StudyPaths, 'is_valid', return_value=True):
        recording_list = list(schoffelen2019.Schoffelen2019Recording.iter())
    assert len(recording_list) == 195
    recording = recording_list[0]
    assert recording.modality == "visual"
    assert recording.subject_uid == "sub-V1001"


def test_fake_recording_cache(tmp_path: Path) -> None:
    with env.temporary(cache=tmp_path):
        recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
        with schoffelenmock.data():
            recording.events()
        # loaded out of context thanks to cache
        meta = schoffelen2019.Schoffelen2019Recording("sub-A2002").events()
    assert isinstance(meta, pd.DataFrame)


def test_recording_empty_copy() -> None:
    recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    with schoffelenmock.data():
        recording.events()
        recording.raw()
        recording2 = recording.empty_copy()
        assert recording2.subject_uid == recording.subject_uid
        # cache exists for base
        assert recording._events is not None
        assert recording._arrays
        # cache does not exist for copy
        assert recording2._events is None
        assert not recording2._arrays


def test_fake_data_on_recording() -> None:
    # Note: if test fails you changed the data schema, please delete the expected_meta.csv file
    # from the system the test runs on, rerun the test to recreate it, and once more to make sure
    # it passes.
    recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    with schoffelenmock.data():
        raw = recording.raw()
        assert not raw.preload
        assert (len(raw.ch_names), raw.n_times) == (273, 99999)
        assert raw is recording.raw(), "This is an assumption used in the factory"
        recording.events()
    events = recording.events(clean=False)  # should be cached
    #
    # check meta data against cache
    events["filepath"] = events["filepath"].apply(
        lambda x: Path(x).name if isinstance(x, str) else x
    )
    test_api.cached_assert_df_equal(events, "expected_meta.csv")


def test_actual_recording_events() -> None:
    recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    try:
        events = recording.events(clean=False)
    except (RuntimeError, KeyError) as e:
        raise SkipTest("Data unavailable") from e
    test_api.cached_assert_df_equal(events, "expected_actual_schoffelen2019_audio_meta.csv")


def test_preprocess_mne() -> None:
    recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    with schoffelenmock.data():
        raw = recording.raw()
        out = api.preprocess_mne(raw, 600, 200)
    assert out.info["sfreq"] == 600
    assert out.n_times == 49999


def test_preprocessed(tmp_path: Path) -> None:
    with env.temporary(cache=tmp_path):
        recording = schoffelen2019.Schoffelen2019Recording("sub-A2002")
    with schoffelenmock.data():
        out = recording.preprocessed(600, 200)
    assert out.n_times == 49999
