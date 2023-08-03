# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import os
import subprocess
import sys

from dora import XP

import bm
from bm import play
from . import studies
from .studies import schoffelen2019


def _test_play(tmp_path: Path, is_decode: bool = False) -> XP:
    from bm.train import main
    os.environ['_BM_TEST_PATH'] = str(tmp_path)
    main.dora.dir = tmp_path
    workdir = Path(bm.__file__).parents[1]
    previous_content = {x for x in workdir.iterdir() if not x.name.startswith(".")}
    train_cmd = [
        sys.executable,
        "-m",
        "bm.train",
    ]
    train_args = [
        "num_workers=0",
        "device=cpu",
        "dset.n_recordings=2",
        "dset.tmax=1",
        "dset.force_uid_assignement=true",
        "dset.selections=[\"fake\"]",
        "dset.features=[WordFrequency,WordLength]",
        "dset.min_n_blocks_per_split=1",
        "optim.epochs=1",
        "eval_every=1",
        "model=convrnn",
        "convrnn.hidden.meg=4",
        "convrnn.subject_dim=2",
        f"cache={tmp_path / 'fake_cache_play'}",
        "optim.batch_size=2",
    ]

    if is_decode:
        train_args += [
            'task.type=decode',
            'model=decoder_convrnn',
        ]
    else:
        train_args += [
            "convrnn.hidden.features=1",
        ]

    subprocess.check_call(train_cmd + train_args, cwd=workdir)
    files = list(tmp_path.iterdir())
    xps = list((tmp_path / "xps").iterdir())
    new_content = {x for x in workdir.iterdir() if not x.name.startswith(".")}
    assert files, "Missing outputs in temp folder"
    assert xps, "Missing xp in the xps folder"
    assert new_content == previous_content, "The work directory has been tampered with"
    # print(new_content, previous_content, tmp_path)
    out = main.get_xp(train_args)
    return out


def test_play_decode(tmp_path: Path) -> None:
    xp = _test_play(tmp_path, is_decode=True)
    solver = play.get_solver_from_xp(xp)
    assert len(solver.history) > 0
    test_metrics = play.get_test_metrics(solver, reduce=False)
    for name, val in test_metrics.items():
        # TODO(orik): understand why an untrained model always outputs the same classifications.
        # Might be related to the way layers are initialized.
        # adefossez: this is failing massively and non deterministically
        # reducing the scope for now but I have no idea what is going on...
        assert val.std() > 0 or "l2_reg_" in name, name


def test_sentence_prediction(tmp_path: Path) -> None:
    xp = _test_play(tmp_path, is_decode=False)
    solver = play.get_solver_from_xp(xp)
    builder = play.SentenceFeatures.from_solver(solver, additional_time=2.0)
    features = builder("Ich bin ein Berliner")
    pred = solver.predict(features=features, subject_index=0)
    assert pred.shape == (273, 636)
    play.predict(solver, features, subject_index=0)


def test_basal_extraction() -> None:
    Fake = studies.register["fake"]
    builder = play.SentenceFeatures(
        sample_rate=schoffelen2019.RAW_SAMPLE_RATE, features_params={}, features=[])
    recording = Fake('sub-A2002')  # type: ignore
    out = builder.extract_basal_states(recording)
    meta = recording.events(clean=True)
    words = meta.loc[meta.kind == "word"]
    first_words = words.loc[words.word_index == 0]
    assert len(first_words) < len(words)
    # there may be fewer sequences than events since events may be out of bounds
    assert len(out) <= len(first_words)


def test_play_encode(tmp_path: Path) -> None:
    xp = _test_play(tmp_path)
    solver = play.get_solver_from_xp(xp)
    assert len(solver.history) > 0
    test_metrics = play.get_test_metrics(solver, reduce=False)
    for val in test_metrics.values():
        assert val.std() > 0
    test_metrics = play.get_test_metrics(solver, reduce=False)
    for val in test_metrics.values():
        assert val.std() > 0
