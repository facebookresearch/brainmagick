# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event handling classes and functions.

To manipulate events when e.g. integrating a new study into brainmagick, use the `EventAccessor` to
directly handle events stored inside a standard pandas DataFrame. For more information about events
and the `EventAccessor`, see `doc/recordings_and_events.md`.

The class `Event` and its children (e.g. `Sound`, `Word`, etc.) define the expected fields for each
event kind.
"""

import random
import hashlib
import typing as tp
from pathlib import Path
from dataclasses import dataclass, fields, asdict

import pandas as pd
import numpy as np
import torchaudio
from dora import to_absolute_path

from . import utils


@dataclass
class Event:
    """Base class for all event kinds with the bare minimum common fields.

    If the event is instantiated with `from_dict()`, additional non-required fields that are
    provided will be ignored instead of causing an error.
    """
    start: float
    duration: float
    modality: tp.Optional[str]  # Move to keyword arguments when updating to 3.10
    language: tp.Optional[str]  # Move to keyword arguments when updating to 3.10
    # See https://www.trueblade.com/blogs/news/python-3-10-new-dataclass-features

    def __post_init__(self):
        if self.duration < 0:
            raise ValueError("Negative durations are not allowed for events.")
        # self._sample_rate: tp.Optional[utils.Frequency] = None  # XXX Necessary?

    @classmethod
    def from_dict(cls, row: dict) -> "Event":
        """Create event from dictionary while ignoring extra parameters.
        """
        return cls(**{k: v for k, v in row.items() if k in [f.name for f in fields(cls)]})

    @classmethod
    def _kind(cls) -> str:
        """Convenience method to get the name from the class."""
        return cls.__name__.lower()

    @property
    def kind(self) -> str:
        return self.__class__._kind()

    @property
    def stop(self) -> float:
        return self.start + self.duration


@dataclass
class DataSlice(Event):
    """Event describing a slice of data with methods to find out overlap with other events.
    """
    sample_rate: float

    def __post_init__(self):
        super().__post_init__()
        self._sample_rate = utils.Frequency(self.sample_rate)
        self._parent: tp.Optional["DataSlice"] = None

    def overlap(self, event: Event) -> "DataSlice":
        """Creates new DataSlice that only contains the overlap with the provided event.
        """
        start = max(self.start, event.start)
        stop = min(self.stop, event.stop)
        out = self.__class__(
            start=start, duration=stop - start, sample_rate=self.sample_rate,
            language=self.language, modality=self.modality)
        out._sample_rate = self._sample_rate  # XXX Necessary?
        out._parent = self
        return out

    def slice_in_parent(self) -> slice:
        """Provides slice with respect to the parent DataSlice to position the feature correctly.
        """
        assert self._parent is not None
        start = self.start_ind - self._parent.start_ind
        return slice(start, start + self.duration_ind)

    # Convenience methods for computing indices when a sample rate is provided

    @property
    def start_ind(self) -> int:
        return self._sample_rate.to_ind(self.start)

    @property
    def stop_ind(self) -> int:
        return self._sample_rate.to_ind(self.stop)

    @property
    def duration_ind(self) -> int:
        return self.stop_ind - self.start_ind


@dataclass
class Sound(Event):
    """Event corresponding to an audio recording saved as a WAV file.
    """
    filepath: str
    offset: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.filepath = str(to_absolute_path(Path(self.filepath)))
        if np.isnan(self.offset):  # can happen if not specified
            # NOTE: Offset is used when splitting WAVs to avoid block boundary overlap.
            self.offset = 0.0
        if 'MOCK_CACHE' in self.filepath:
            assert self.duration is not None
            actual_duration = self.duration
        else:
            assert Path(self.filepath).exists(), f'{self.filepath} does not exist.'
            info = torchaudio.info(self.filepath)
            actual_duration = float(info.num_frames / info.sample_rate) - self.offset
            if self.duration is None or self.duration == 0:
                self.duration = actual_duration
            else:
                self.duration = min(actual_duration, self.duration)


@dataclass
class Word(Event):
    """Event corresponding to a word.

    Note
    ----
    See function `bm.events::extract_word_sequence_info` to compute fields `word_index` and
    `word_sequence` from a `sequence_id` field.
    """
    word: str  # the actual word
    word_index: int  # index of the word in the sequence
    word_sequence: str  # sequence of words the word is part of

    def __post_init__(self):
        super().__post_init__()
        assert self.modality in ['audio', 'visual']
        self.word_index = int(self.word_index)


@dataclass
class Phoneme(Event):
    """Event corresponding to a phoneme utterance.
    """
    phoneme_id: int


@dataclass
class MultipleWords(Event):
    """Event corresponding to multiple words presented at once on the screen.
    """
    words: str


@dataclass
class Motor(Event):
    """Event corresponding to a behavior."""


@dataclass
class Special(Event):
    """Special event for things we don't know yet how to deal with.
    """
    name: str


@dataclass
class Block(Event):
    """Event corresponding to a block, defined by a start, a duration and a unique identifier.
    """
    uid: str

    def __post_init__(self):
        super().__post_init__()
        self.uid = str(self.uid)


# Functions for processing events into blocks

def extract_sequence_info(events: pd.DataFrame, word: bool = True,
                          phoneme: bool = True) -> pd.DataFrame:
    """Extract word and/or phoneme sequence-related information from an events DataFrame.

    Extract information about word and/or phoneme sequences from an events DataFrame. Columns
    'sequence_id' and 'word' must be available for extracting word-based information, and column
    'word_index' for phoneme-based information. The following columns are created if they don't
    already exist:
        'word_index': index of a word in the sequence, e.g. in a sentence
        'word_sequence': actual sequence of words a word belongs to, e.g. a sentence
        'phoneme_id': index of a phoneme in a word

    Parameters
    ----------
    events :
        DataFrame of events (not modified in-place).
    word :
        If True, extract word sequence information (word_index, word_sequence).
    phoneme :
        If True, extract phoneme sequence information (phoneme_id).

    Returns
    -------
    Updated DataFrame of events.
    """
    def is_missing(df, key):
        return key not in df.columns or all(df[key].isnull())

    events_out = events.copy()

    if word and (events.kind == 'word').any():
        missing_cols = [col for col in ['sequence_id', 'word'] if col not in events.columns]
        if missing_cols:
            raise ValueError(f'Columns \"{missing_cols}\" are required but were not found.')

        is_word = events.kind.isin(['word', 'multiplewords'])
        words = events.loc[is_word]

        if words.sequence_id.nunique() < 2:
            raise ValueError('Only one word sequence ID found.')

        for _, d in words.groupby("sequence_id"):
            # define word indices by making it compatible for multiple words
            if is_missing(d, "word_index"):  # Index of the word in the sequence
                indices = np.cumsum([0] + [len(w.split()) for w in d.word])
                events_out.loc[d.index, "word_index"] = indices[:-1]

            if is_missing(d, "word_sequence"):  # Sequence of words
                for uid in d.index:
                    events_out.loc[uid, "word_sequence"] = " ".join(d.word.values)

    if phoneme and (events.kind == 'phoneme').any():
        phonemes = events_out[events_out.kind == 'phoneme']
        if is_missing(phonemes, 'word_index'):
            raise ValueError('Column \"word_index\" is required but was not found.')

        for _, group in phonemes.groupby(['sequence_id', 'word_index']):
            if is_missing(group, 'phoneme_id'):
                events_out.loc[group.index, 'phoneme_id'] = range(len(group))

    return events_out


def _get_block_uid(events: pd.DataFrame) -> str:
    """Get block unique IDs for the events contained in a DataFrame.

    The unique ID of a block is either the concatenation of the words or filepaths it contains, or,
    if available and unique, the value in the 'sequence_uid' column.
    """
    if 'sequence_uid' in events.columns:  # Use existing sequence_uid, e.g. with Schoffelen2019
        unique_sequence_uids = events.sequence_uid.unique()
        if len(unique_sequence_uids) == 1:
            uid = unique_sequence_uids[0]
            return uid

    # Use concatenation of words or filepaths
    has_words = \
        events.condition.isin(EventAccessor.WORD_CONDITIONS) & (events.kind != 'phoneme')
    if not any(has_words):  # Use filepaths if there are no words in the block
        uid_ = [f for f in events.filepath.unique() if isinstance(f, str)]
        assert len(uid_), 'No filepath information available for defining block unique ID.'
        uid_ += [str(events.start.min())]
    else:
        uid_ = events.loc[has_words].word.astype(str)

    uid = ' '.join(uid_)

    return uid


def _create_blocks(events: pd.DataFrame, groupby: str) -> pd.DataFrame:
    """Create blocks from an events DataFrame.

    Blocks have a start, a duration, and a unique ID that can be used to identify its content.
    Blocks are used when splitting examples into training, validation and test sets to avoid
    creating segments that end in the middle of a sequence.

    Parameters
    ----------
    events :
        Events DataFrame (not modified in-place).
    groupby :
        Group events by this category to create blocks. E.g., grouping by 'sentence' will create
        blocks that start with the first word of each sentence.

    Returns
    -------
    Updated events DataFrame that contains the created blocks.
    """
    assert groupby in EventAccessor.VALID_BLOCK_TYPES, \
        f'by={groupby} not supported, must be one of {EventAccessor.VALID_BLOCK_TYPES}.'

    # Find events that are valid block starts
    blocks = list()
    for event in events.event.iter():
        if groupby == "sentence":
            block_start = (event.kind == "word") and (event.word_index == 0)
        elif groupby == "sound":
            block_start = event.kind == "sound"
        elif groupby == "fixation":
            block_start = event.condition == "fixation"
        elif groupby == 'sentence_or_sound':  # Used for Schoffelen2019
            block_start = (event.kind == 'sound') or (
                (event.kind == 'word') and (event.modality == 'visual') and
                (event.word_index == 0))
        else:
            block_start = False

        if block_start:
            blocks.append(event)

    eps = 1e-7
    event_stops = events.start + events.duration
    events_end = event_stops.max() + eps
    assert all(np.diff([b.start for b in blocks]) > 0), "events not sorted"
    block_stops = [b.start for b in blocks[1:]] + [events_end]

    # Add boundary unique ID
    block_events = list()
    for block, stop in zip(blocks, block_stops):
        # Create block unique ID based on all events contained in the block
        mask = (events.start >= block.start) & ((events.start + events.duration) < stop)
        uid = _get_block_uid(events[mask])
        block_info = asdict(  # Convert to Block object to apply checks
            Block(start=block.start, duration=stop - block.start, uid=uid,
                  language=block.language, modality=block.modality))
        block_events.append(block_info)

    blocks_df = pd.DataFrame(block_events)
    blocks_df['kind'] = 'block'
    blocks_df.duration.iat[-1] = float('inf')  # For compatibility with old API - last block has
    # infinite duration

    # Sort by start time
    events = pd.concat([events, blocks_df], axis=0, ignore_index=True)
    events.loc[events.kind == "block", "start"] -= eps  # Make sure blocks come before their events
    events = events.sort_values("start", ignore_index=True)
    events.loc[events.kind == "block", "start"] += eps  # Move back to real start time

    return events


def _merge_blocks(blocks: pd.DataFrame, min_block_duration_s: float = 60) -> pd.DataFrame:
    """Merge consecutive blocks until the minimum duration has been reached.

    Parameters
    ----------
    blocks :
        DataFrame of blocks (not modified in-place).
    min_block_duration_s :
        Minimum block duration, in seconds. Smaller blocks will be merged until they are at least
        as long as this value.

    Returns
    -------
    DataFrame of merged blocks.

    Note
    ----
    The last block might be smaller than min_block_duration.
    """
    new_blocks, uids, start = list(), list(), 0.0
    for k, block in enumerate(blocks.event.iter()):
        uids.append(block.uid)
        is_last = k == len(blocks) - 1
        stop = block.start + block.duration
        if is_last or stop > start + min_block_duration_s:  # Record a merged block
            uid = ','.join(uids)
            new_block = asdict(  # Convert to Block object to apply checks
                Block(start=start, duration=stop - start, uid=uid, language=block.language,
                      modality=block.modality))
            new_blocks.append(new_block)
            uids, start = list(), stop
    assert not uids, "All blocks should have been included"
    new_blocks_df = pd.DataFrame(new_blocks)
    new_blocks_df['kind'] = 'block'
    assert hasattr(new_blocks_df, 'duration')  # For mypy
    if any(new_blocks_df.duration[:-1] < min_block_duration_s):
        raise ValueError(f'Some blocks are smaller than {min_block_duration_s}.')

    return new_blocks_df


def assign_blocks(blocks: pd.DataFrame, ratios: tp.List[float], seed: int,
                  remove_ratio: float = 0, min_n_blocks_per_split: int = 20) -> pd.DataFrame:
    """Randomly assign blocks to subsets approximately respecting the given `ratios`.

    This will return `len(ratios) + 1` subsets, with the last subset containing whatever wasn't
    assigned so far.

    Parameters
    ----------
    blocks :
        DataFrame of blocks.
    ratios :
        Ratios to aim for when creating the subsets. An additional subset will be created with
        whatever blocks have not been assigned.
    seed :
        Seed for random assignment.
    remove_ratio :
        If > 0, create a subset of blocks that respects this approximate ratio and drop it.
    min_n_blocks_per_split :
        Minimum number of blocks that are expected in a split. If at least one of the split has
        fewer than this number of blocks, a ValueError will be raised.

    Returns
    -------
    Updated DataFrame of blocks, with additional column 'split' indicating which subset each block
    was assigned to.
    """
    if remove_ratio > 0.:
        ratios = ratios + [remove_ratio]

    assert all(ratio > 0 for ratio in ratios)
    assert sum(ratios) < 1., "last dataset has negative ratio size"
    ratios.append(1. - sum(ratios))
    cdf = np.cumsum(ratios)

    split = list()
    for block in blocks.event.iter():
        uid = block.uid
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + seed)
        score = rng.random()
        for idx, cdf_val in enumerate(cdf):
            if score < cdf_val:
                split.append(idx)
                break

    assert len(split) == len(blocks)
    assigned_blocks = blocks.copy()
    assigned_blocks['split'] = split

    if (assigned_blocks.split.value_counts() < min_n_blocks_per_split).any():
        raise ValueError(
            f'At least one of the splits has fewer than {min_n_blocks_per_split} blocks.')

    if remove_ratio > 0.:  # Drop blocks to be removed and adjust split numbers
        remove_ratio_ind = len(ratios) - 2
        assigned_blocks = assigned_blocks[assigned_blocks.split != remove_ratio_ind]
        assigned_blocks.split = assigned_blocks.split.map(
            lambda x: x - 1 if x > remove_ratio_ind else x)

    return assigned_blocks


def split_wav_as_block(events: pd.DataFrame, blocks: tp.List[tp.Tuple[float, float]],
                       margin: float = 0.1) -> pd.DataFrame:
    """Split sound events so that they do not overlap block boundaries.

    This makes sure there is no contamination across train/valid/test splits of audio features.

    Parameters
    ----------
    events :
        DataFrame of events.
    blocks :
        List of tuples (start_time, end_time) that define the blocks.
    margin :
        Margin, in seconds, to use around the block boundaries to avoid creating empty chunks and
        float rounding errors.

    Returns
    ------
    Updated DataFrame containing split sound events.
    """
    if 'offset' not in events:
        events['offset'] = 0.

    sound_mask = events.kind == 'sound'
    other_events = events[~sound_mask]
    events_queue = [event for _, event in events[sound_mask].iterrows()]  # Benchmarked - iterrows
    # is fine here

    new_events = list()
    for start, stop in blocks:
        while events_queue:
            if events_queue[0].start >= stop - margin:
                # we go to the next block.
                break
            event = events_queue.pop(0)
            if event.start + event.duration <= start + margin:
                # almost no overlap with current block
                pass
            elif event.start <= start - margin:
                # a significant portion of the audio is before the block
                new_event = event.copy(deep=True)
                event.duration = start - event.start
                new_event.offset += event.duration
                new_event.start += event.duration
                new_event.duration -= event.duration
                events_queue.insert(0, new_event)
            elif event.start + event.duration > stop + margin:
                new_event = event.copy(deep=True)
                event.duration = stop - event.start
                new_event.start += event.duration
                new_event.offset += event.duration
                new_event.duration -= event.duration
                # we push new event for further processing as it might overlap many blocks.
                events_queue.insert(0, new_event)
            new_events.append(event)
    events = pd.concat([pd.DataFrame(new_events + events_queue), other_events])
    events = events.sort_values('start', ignore_index=True)

    return events


@pd.api.extensions.register_dataframe_accessor('event')
class EventAccessor:
    """Accessor for event information stored as a pandas DataFrame.

    To know what fields are required for a specific kind of event, use
    >>> frame.event.list_required_fields()
    Alternatively, the definitions of the Event (sub)classes can be inspected in `bm/events.py`.

    For more information about events and the `EventAccessor`, see `doc/recordings_and_events.md`.
    """
    CLASS_KIND_MAPPING = {
        'word': Word,
        'multiple_words': MultipleWords,
        'sound': Sound,
        'phoneme': Phoneme,
        'motor': Motor,
        'special': Special,
        'block': Block
    }
    WORD_CONDITIONS = {'sentence', 'context', 'question', 'fixation', 'word_list'}
    VALID_BLOCK_TYPES = {'sentence', 'sound', 'sentence_or_sound'}

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self._frame = self.validate()

    @classmethod
    def list_required_fields(cls, kind: tp.Optional[str] = None) -> None:
        """Helper method so users can get a summary of the expected fields for an event kind.

        Parameters
        ----------
        kind :
            Event kind for which to list required fields. If None, list required fields for all
            registered event kinds.

        XXX: Move out of accessor so validation doesn't run for this?
        """
        if kind is not None:
            event_class = cls.CLASS_KIND_MAPPING[kind]
            required_fields = ['kind'] + [field.name for field in fields(event_class)]
            msg = f'{kind} event: {required_fields}'
            print(msg)
        else:
            for kind in cls.CLASS_KIND_MAPPING.keys():
                cls.list_required_fields(kind)

    def _validate_event(self, event: pd.Series) -> dict:
        """Validate event, i.e. check fields and values are as expected, and update it accordingly.

        This is done by instantiating an event object of the corresponding kind, which carries out
        the validation, and then updating the input with the applied changes (if any).
        """
        # Check kinds are valid
        if event['kind'] not in self.CLASS_KIND_MAPPING.keys():
            raise ValueError(
                f'Unexpected kind \"{event["kind"]}\". Support for new event kinds can be added by'
                ' creating new `Event` classes in `bm.events`.')

        # Build event object to run the checks inside the kind-specific Event class
        event_class: tp.Type[Event] = self.CLASS_KIND_MAPPING[event.kind]
        event_obj = event_class.from_dict(event)

        # Add back fields that were ignored by the Event class
        # event.update(asdict(event_obj))  # Very slow, use dict updating instead
        event = {**event, **asdict(event_obj)}

        return event

    def validate(self) -> pd.DataFrame:
        """Validate the DataFrame of events.

        Returns
        -------
        pd.DataFrame
            DataFrame in which each row has been validated and updated accordingly.
        """
        if not self._frame.empty:
            return pd.DataFrame(self._frame.apply(self._validate_event, axis=1).tolist())
        else:
            return self._frame.copy()

    def iter(self) -> tp.Iterator[Event]:
        """Iterate over rows of the DataFrame to yield Event objects.
        """
        for row in self._frame.itertuples(index=False):
            event_class: tp.Type[Event] = self.CLASS_KIND_MAPPING[row.kind]
            yield event_class.from_dict(row._asdict())

    def create_blocks(self, groupby: str) -> pd.DataFrame:
        """Create blocks from an events DataFrame.

        See `_create_blocks`.
        """
        return _create_blocks(self._frame, groupby=groupby)

    def merge_blocks(self, min_block_duration_s: float = 60) -> pd.DataFrame:
        """Merge consecutive blocks until the minimum duration has been reached.

        See `_merge_blocks`.
        """
        blocks_df = self._frame[self._frame.kind == 'block']
        return _merge_blocks(blocks_df, min_block_duration_s=min_block_duration_s)

    def plot(self, window_s: float = 30.0, ax: tp.Optional[tp.Any] = None, show_desc: bool = True,
             desc_cropping_s: float = 0, desc_fontsize: float = 7, figsize: tuple = (10, 10),
             print_summary: bool = True) -> tp.Tuple[tp.Any, tp.Any]:
        """Plot events for visual assessment of alignment.

        See `bm.viz.plot_events`.
        """
        from .viz import plot_events

        fig, ax = plot_events(
            self._frame, window_s=window_s, ax=ax, show_desc=show_desc,
            desc_cropping_s=desc_cropping_s, desc_fontsize=desc_fontsize, figsize=figsize,
            print_summary=print_summary)
        return fig, ax
