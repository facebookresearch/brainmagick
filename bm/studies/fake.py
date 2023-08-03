# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import itertools
import contextlib
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from . import api
from . import schoffelen2019
from bm.lib.phonemes import ph_dict


FAKE_ENV = "BRAIN_MAGICK_FAKE_DATA"


@contextlib.contextmanager
def set_env(**environ: tp.Any) -> tp.Generator[None, None, None]:
    """Temporarily changes environment variables.
    """
    old_environ = {x: os.environ.get(x, None) for x in environ}
    os.environ.update({x: str(y) for x, y in environ.items()})
    try:
        yield
    finally:
        for k, val in old_environ.items():
            os.environ.pop(k)
            if val is not None:
                os.environ[k] = val


def create_fake_meg() -> mne.io.RawArray:
    """Creates a fake meg with random noise in it
    """
    n_channels, n_times = 273 + 28, 99_999
    ch_names = ["MISC", "STI101"] + ["c" + str(k) for k in range(n_channels - 2)]
    # NOTE: actually MISC and STI101 should have specific ch_types, but nevermind for now
    ch_names = ch_names[:n_channels]  # rectification
    content = np.random.randn(len(ch_names), n_times)
    content[:2, :] = 0  # non meg channels  # TODO what was that again?
    sfreq = schoffelen2019.RAW_SAMPLE_RATE
    for time_s, value in [(10, 10), (20, 20), (30, 10)]:
        t_ms = int(time_s * sfreq)
        content[1, t_ms: t_ms + 3000] = value  # adds events to the stimuli channel
    return mne.io.RawArray(content, info=mne.create_info(ch_names, sfreq=sfreq, ch_types='mag'))


def make_fake_events(total_duration: float = 83, seed: int = 1234) -> pd.DataFrame:
    """Create a fake event DataFrame with multiple events and precomputed blocks.
    """
    rng = random.Random(seed)
    event_dicts = list()
    wavpath = Path(__file__).parent.parent / 'mockdata' / 'one_two.wav'
    word_sequence = ['Toen', 'barkeeper', 'de']
    language = 'nl'

    time = 0.0
    for block_index in itertools.count():
        time += rng.uniform(0.5, 1.0)
        block_start_time = time

        # Need blocks to be approximately 3.0 s or more, otherwise they can't be used
        n_repeats = rng.randint(2, 3)
        sequence = word_sequence * n_repeats
        for word_index, word in enumerate(sequence):
            duration = rng.uniform(0.1, 0.2)
            time += duration + rng.uniform(0.1, 0.3)
            modality = rng.choice(['audio', 'visual'])

            # Add a word
            word_event = dict(kind='word', start=time, duration=duration, modality=modality,
                              language=language, word=word, word_index=word_index,
                              word_sequence=' '.join(sequence), condition='sentence')
            event_dicts.append(word_event)

            # Add a phoneme
            if modality == 'audio':
                ph_id = rng.choice(list(ph_dict.values()))
                phoneme_event = dict(kind='phoneme', start=time, duration=duration,
                                     phoneme_id=ph_id, modality=modality, language=language)
                event_dicts.append(phoneme_event)

        # Create corresponding sound event and block that cover the last events
        block_end_time = time + duration
        sound = dict(kind='sound', start=block_start_time,
                     duration=block_end_time - block_start_time, filepath=wavpath)
        event_dicts.append(sound)
        block = dict(kind='block', start=block_start_time,
                     duration=block_end_time - block_start_time, uid='block' + str(block_index))
        event_dicts.append(block)

        if time > total_duration:
            break

    events = pd.DataFrame(event_dicts).event.validate()
    return events


class FakeRecording(api.Recording):

    data_url = 'http://fake.com'
    paper_url = 'http://fake.com'
    doi = ''
    licence = ''
    modality = ''
    language = ''
    device = 'meg'
    description = 'Fake recording used for testing.'

    # pylint: disable=arguments-differ
    @classmethod
    def iter(  # type: ignore
            cls, seed: int = 1234
    ) -> tp.Iterator["FakeRecording"]:
        for k in range(4):
            yield cls(str(k), seed=seed + k)

    def __init__(self, subject_uid: str, seed: int = 1234) -> None:
        super().__init__(subject_uid=subject_uid, recording_uid=subject_uid)
        self.seed = seed
        # make sure the cache does not get contaminated
        if self._cache_folder is not None:
            if "fake_cache" not in str(self._cache_folder):
                raise RuntimeError("Mock recording cache must contain 'fake_cache' string")

    def _load_events(self) -> pd.DataFrame:
        total_duration = self.raw().times[-1]
        return make_fake_events(total_duration=total_duration, seed=self.seed)

    def _load_raw(self) -> mne.io.RawArray:
        """Loads, caches and returns the raw for the subject
        """
        raw = create_fake_meg()
        picks = mne.pick_types(
            raw.info, meg=True, eeg=False, stim=False, eog=False, ecg=False
        )[28: (28 + 273)]
        raw.pick(picks)  # only keep data channels
        return raw
