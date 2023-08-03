# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import random
import tempfile
import contextlib
import typing as tp
from pathlib import Path
from unittest import mock
from _pytest.monkeypatch import MonkeyPatch
from bm import env
from . import schoffelen2019
from . import preproc
from .. import fake

# %%%% FAKING MOUS STRUCTURE FOR TESTING


TEST_FILENAME = "testmeg-raw.fif"  # for test only


# pylint: disable=unused-argument
def _mocked_recording_paths(id_: int) -> schoffelen2019.StudyPaths:
    return schoffelen2019.StudyPaths("sub-A2002")


def add_fake_sequence_uid(log):
    """Replacement function  for preproc_metadata.add_sequence_uid"""
    rng = random.Random(42)
    candidates = list(range(1, 100))
    rng.shuffle(candidates)
    uid_map = dict(enumerate(candidates))

    block_ids = log['block']
    sequence_uid = block_ids.map(uid_map)
    log['sequence_uid'] = sequence_uid
    return log


# pylint:disable=too-many-locals
@contextlib.contextmanager
def data() -> tp.Generator[Path, None, None]:
    """Context manager which mocks the structure of the dataset by creating
    fake datafiles.
    This is a monster which plugs as deeply as possible in the pipeline.
    """
    pid = "A2002"
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        tmp = enter(tempfile.TemporaryDirectory())
        # update paths with a temporary folder
        tmp_path = Path(tmp) / "MOCK_CACHE"
        monkeypatch = MonkeyPatch()
        monkeypatch.setattr(env, "_studies", dict(schoffelen2019=tmp_path))
        monkeypatch.setattr(schoffelen2019.StudyPaths, "create", _mocked_recording_paths)
        # create meg data as a fif file (in a meg.ds folder to mock actual data)
        # the name of the fake file will be recognized at reading time
        # NOTE: n_times seems to require being odd (processing bugs otherwise)
        meg = fake.create_fake_meg()
        meg_folder = tmp_path / f"sub-{pid}" / "meg"
        meg_folder.mkdir(parents=True)
        meg.save(meg_folder / TEST_FILENAME)
        # Mock logs with a sample of audio logs
        mockdata = Path(__file__).parents[2] / "mockdata"
        assert mockdata.exists()
        log_folder = tmp_path / "sourcedata" / "meg_task"
        log_folder.mkdir(parents=True)
        shutil.copy(mockdata / "audio.log", log_folder / f"MEG-MOUS-Aud{pid}.log")
        # Mock textgrid files by copying one file and create multiple symlinks to it
        base = schoffelen2019.StudyPaths.phoneme_file(0)
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(mockdata / "example.TextGrid", base)
        for k in range(1, 1000):
            schoffelen2019.StudyPaths.phoneme_file(k).symlink_to(base)

        enter(mock.patch.object(preproc,
                                'add_sequence_uid',
                                new=add_fake_sequence_uid))
        enter(fake.set_env(**{fake.FAKE_ENV: 1}))
        # return the temporary folder if needed
        try:
            yield tmp_path
        finally:
            # undo the paths mocking
            monkeypatch.undo()
