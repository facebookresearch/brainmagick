# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl

from .events import (
    Event, DataSlice, Sound, extract_sequence_info, split_wav_as_block, assign_blocks)


@pytest.fixture
def event_dicts() -> tp.List[dict]:
    """List of fake event dictionaries that contain two sequences of words and an audio file.
    """
    events = [
        {'start': 0.0, 'duration': 0.9, 'kind': 'word', 'word': 'ceci', 'modality': 'audio',
         'language': 'fr', 'condition': 'sentence', 'sequence_id': 0},
        {'start': 0.0, 'duration': 0.2, 'kind': 'phoneme', 'phoneme': 'b', 'modality': 'audio',
         'language': 'fr', 'condition': 'sentence', 'sequence_id': 0, 'word_index': 0},
        {'start': 1.0, 'duration': 0.9, 'kind': 'word', 'word': 'est', 'modality': 'audio',
         'language': 'fr', 'condition': 'sentence', 'sequence_id': 0},
        {'start': 2.0, 'duration': 0.9, 'kind': 'word', 'word': 'un', 'modality': 'audio',
         'language': 'fr', 'condition': 'sentence', 'sequence_id': 1},
        {'start': 3.0, 'duration': 0.9, 'kind': 'word', 'word': 'test', 'modality': 'audio',
         'language': 'fr', 'condition': 'sentence', 'sequence_id': 1},
        {'start': 0.0, 'duration': 5.0, 'kind': 'sound', 'filepath':
         'MOCK_CACHE/ceci_est_un_test.wav', 'offset': 0.0, 'modality': 'audio', 'language': 'fr'},
    ]
    return events


def test_extract_sequence_info(event_dicts) -> None:
    events_df = pd.DataFrame(event_dicts)
    out = extract_sequence_info(events_df, word=True, phoneme=True)

    assert 'word_index' in out
    assert 'word_sequence' in out
    assert (out.loc[out.kind == 'word', 'word_index'] == [0, 1, 0, 1]).all()
    assert (out.loc[out.kind == 'word', 'word_sequence'] ==
            ['ceci est'] * 2 + ['un test'] * 2).all()

    assert 'phoneme_id' in out
    assert (out.loc[out.kind == 'phoneme', 'phoneme_id'] == 0).all()


def test_event_accessor_validate(event_dicts) -> None:
    events_df = pd.DataFrame(event_dicts)
    events_df = extract_sequence_info(events_df)
    valid_events_df = events_df.event.validate()

    assert valid_events_df.shape == valid_events_df.shape
    assert valid_events_df.loc[2, 'filepath'] != event_dicts[-1]['filepath']  # Updated filepath


@pytest.fixture
def events_df(event_dicts) -> pd.DataFrame:
    """DataFrame of fake events.
    """
    events = pd.DataFrame(event_dicts)
    events = extract_sequence_info(events)
    return events.event.validate()


def test_event_accessor_list_required_fields(events_df) -> None:
    events_df.event.list_required_fields()


def test_event_accessor_unsupported_kind(events_df) -> None:
    events_df.loc[0, 'kind'] = 'unknown'
    with pytest.raises(ValueError):
        events_df.event.validate()


def test_event_accessor_missing_field(events_df) -> None:
    with pytest.raises(TypeError):
        events_df.drop(columns=['start']).event.validate()


def test_event_accessor_invalid_duration(events_df) -> None:
    events_df.duration -= events_df.duration.max()
    with pytest.raises(ValueError):
        events_df.event.validate()


def test_event_accessor_iter(events_df) -> None:
    for i, event in enumerate(events_df.event.iter()):
        assert isinstance(event, Event)
    assert i == len(events_df) - 1


def test_event_accessor_create_blocks(events_df) -> None:
    out = events_df.event.create_blocks(groupby='sentence')
    blocks = out[out.kind == 'block']

    assert len(blocks) == 2
    assert blocks.iloc[0].start == 0
    assert blocks.iloc[0].duration == 2.0
    assert np.isinf(blocks.iloc[-1].duration)  # For compatibility with previous Events API
    assert blocks.iloc[0].uid == 'ceci est'


def test_event_accessor_merge_blocks(events_df) -> None:
    blocks = events_df.event.create_blocks(groupby='sentence')
    out = blocks.event.merge_blocks(min_block_duration_s=3)
    assert len(out) == 1
    assert out.iloc[0].start == 0.0
    assert np.isinf(out.iloc[0].duration)  # For compatibility with previous Events API
    assert out.iloc[0].uid == 'ceci est,un test'


def test_assign_blocks(events_df) -> None:
    out = events_df.event.create_blocks(groupby='sentence')
    blocks = out[out.kind == 'block']
    ratios = [0.5]

    assigned_blocks = assign_blocks(blocks, ratios=ratios, seed=12, min_n_blocks_per_split=1)
    assert 'split' in assigned_blocks.columns
    assert len(assigned_blocks) == len(blocks)
    assert (assigned_blocks.split.values == [1, 0]).all()


def test_assign_blocks_remove_ratio(events_df) -> None:
    out = events_df.event.create_blocks(groupby='sentence')
    blocks = out[out.kind == 'block']
    ratios = [0.5]

    assigned_blocks = assign_blocks(
        blocks, ratios=ratios, seed=12, remove_ratio=0.2, min_n_blocks_per_split=1)
    assert 'split' in assigned_blocks.columns
    assert len(assigned_blocks) == len(blocks)
    assert (assigned_blocks.split.values == [1, 0]).all()


def test_event_accessor_plot(events_df) -> None:
    fig, ax = events_df.event.plot()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_split_wav_as_blocks(events_df) -> None:
    blocks = [(0.0, 1.5), (1.5, 2.5), (2.5, 3.5)]
    new_frame = split_wav_as_block(events_df, blocks, margin=0.1)
    pd.testing.assert_frame_equal(
        events_df[events_df.kind != 'sound'],
        new_frame[new_frame.kind != 'sound'].reset_index(drop=True))

    sounds = new_frame[new_frame.kind == 'sound']
    assert len(sounds) == 4
    assert (sounds.start == [0.0, 1.5, 2.5, 3.5]).all()
    assert (sounds.duration == [1.5, 1.0, 1.0, 1.5]).all()
    assert (sounds.filepath.values == events_df[events_df.kind == 'sound'].filepath.values).all()


def test_dslices_indices() -> None:
    event = DataSlice(start=0.5, duration=0.2, sample_rate=12, modality=None, language=None)
    assert event.start_ind == 6
    assert event.stop_ind == 8
    assert event.duration_ind == 2


@pytest.mark.parametrize("start_end,expected", [
    ((10, 20), (10, 20)),
    ((0, 21), (10, 20)),
    ((5, 15), (10, 15)),
    ((15, 25), (15, 20)),
    ((25, 35), ()),
])
def test_dslice_overlap(start_end: tp.Tuple[float, float], expected: tp.Tuple[float, ...]) -> None:
    dslice = DataSlice(start=10, duration=10, sample_rate=12,
                       modality=None, language=None)  # start: 10, end: 20
    event = Event(start=start_end[0], duration=start_end[1] - start_end[0],
                  modality=None, language=None)
    if expected:
        overlap = dslice.overlap(event)
        assert (overlap.start, overlap.start + overlap.duration) == (expected)
    else:
        with pytest.raises(ValueError):
            overlap = dslice.overlap(event)


def test_sound_duration() -> None:
    filepath = str(Path(__file__).parent / "mockdata" / "one_two.wav")
    sound = Sound(
        start=1.0, duration=0, modality='audio', language='test', filepath=filepath, offset=0.0)

    assert sound.duration == pytest.approx(1.322086)  # duration of the example wave file
