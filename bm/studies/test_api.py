# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import tempfile
import typing as tp
import unittest
from pathlib import Path
import pytest
import pandas as pd
from bm import env
from bm import studies
from . import api
from . import gwilliams2022 as gw


logger = logging.Logger(__file__)


def cached_assert_df_equal(df: tp.Union[pd.DataFrame, tp.List[tp.Dict[str, tp.Any]]],
                           filename: str) -> None:
    """Compare a dataframe to a cached csv file created the first time
    this is called
    If events is expected to have change, delete the file and rerun
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # remove full tempory path
    if "filepath" in df.columns:
        inds = [isinstance(x, str) for x in df.filepath]
        df.loc[inds, "filepath"] = ["PATH/" + Path(x).name for x in df.loc[inds, "filepath"]]
    expect_path = Path(__file__).parent.with_name("mockdata") / filename
    if not expect_path.exists():
        df.to_csv(expect_path, index=True)
        raise RuntimeError("A new events cache has been created, please rerun.")
    # csv fails to keep the exact types so lets compare from reading the csv generated df
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tmp.csv"
        df.to_csv(path, index=True)
        df = pd.read_csv(path, index_col=0)
    expected = pd.read_csv(expect_path, index_col=0)
    # preliminary column check for clarity
    exp_col = set(expected.columns)
    got_col = set(df.columns)
    add = got_col - exp_col
    miss = exp_col - got_col
    try:
        assert not add, (
            f"Got additional columns: {add} This may happen if the schema has changed, "
            f"if that's the case, remove file {filename} from the host running "
            "this test and rerun the test.")
        assert not miss, (
            f"Got missing columns: {miss} This may happen if the schema has changed, "
            f"if that's the case, remove file {filename} from the host running "
            "this test and rerun the test.")
        # full check
        pd.testing.assert_frame_equal(df, expected, check_dtype=False, check_exact=False)
    except AssertionError as e:
        logger.warning("Cache file (delete to update): %s", expect_path)
        raise e


@pytest.mark.parametrize("name,shape", [
    # ("brennan2019", (60, 363725)),
    # ("broderick2019", (128, 22949)),
    # ("gwilliams2022", (208, 396000)),
    # ("schoffelen2019", (273, 4832825)),
])
def test_actual_data(name: str, shape: tp.Tuple[int, int]) -> None:
    # This test only runs if data is available
    # files are automatically generated on the first run and should not be commited.
    # if the preprocessing changes, feel free to delete the files to create new ones
    assert name in api.register
    if name not in env.studies:
        raise unittest.SkipTest("Data unavailable")
    if "MOCK_CACHE" in str(env.studies[name]):
        raise unittest.SkipTest("Mock cache leaked, has a test failed somewhere?")
    recording = next(api.register[name].iter())
    events = recording.events(clean=False)
    cached_assert_df_equal(events, f"expected_actual_{name}_meta.csv")
    raw = recording.raw()
    assert (len(raw.ch_names), raw.n_times) == shape


@ pytest.mark.parametrize("name", list(env.studies))
def test_env(name: str) -> None:
    if name not in studies.register:
        raise ValueError(f"Study path {name!r} is not a study name")


def test_paths(tmp_path: Path) -> None:
    with env.temporary(studies=dict(gwilliams2022=tmp_path)):
        path = gw.StudyPaths()
    assert path.megs.name == "MEG"
    assert path.events.name == "events"


def test_no_iterrows() -> None:
    studypath = Path(__file__).parent
    assert studypath.name == "studies"
    detected: tp.List[str] = []
    for fp in studypath.rglob("*.py"):
        if not fp.name.startswith("test_"):
            if ".iterrows(" in fp.read_text():
                detected.append(str(fp))
    if detected:
        detected_str = "\n - ".join(detected)
        raise AssertionError(
            "Following files use iterrows, you must use itertuples instead"
            f"for faster processing:\n - {detected_str}"
        )


def test_list_selections() -> None:
    selections = api.list_selections()
    assert len(selections) > 5
