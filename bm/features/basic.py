# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic features like word pulse (short burst of 1s on word start).
"""

import typing as tp

import torch

from .base import Feature
from ..utils import Frequency
from .. import events

from wordfreq import zipf_frequency
from bm.lib.phonemes import ph_dict


class WordPulse(Feature):
    event_kind = "word"
    normalizable = False

    def __init__(self, sample_rate: Frequency, duration_ms: float = 50.) -> None:
        super().__init__(sample_rate)
        self.duration_ms = duration_ms

    def get(self, event: events.Word) -> tp.Any:
        length = max(1, self.sample_rate.to_ind(event.duration))
        pulse_length = self.sample_rate.to_ind(self.duration_ms / 1000)
        out = torch.zeros((1, length), dtype=torch.float32)
        out[:, :pulse_length] = 1
        return out


class PhonemePulse(Feature):
    event_kind = "phoneme"
    normalizable = False

    def __init__(self, sample_rate: Frequency, duration_ms: float = 16) -> None:
        super().__init__(sample_rate)
        self.duration_ms = duration_ms

    def get(self, event: events.Phoneme) -> int:
        # Return the phoneme value and convert phonemes to phoneme pulses later at post_process()
        return int(event.phoneme_id) + 1  # 0 is saved for silence

    def post_process(self, tensor: torch.Tensor) -> None:
        """
        Phonemes appear several times in a row in our data. Marks phoneme pulses with '1' only for
        locations where the phoneme has changed.

        For example, the following phoneme sequence (0 is the silent/no-phoneme):
        [0,0,2,2,2,2,2,2,5,5,5,5,5,5,5,7,7,7,7,3,3,3,3,0,0,0,0]
        Will be transformed to (with pulse_len_samples=2):
        [0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0]
        """
        SILENT_PHONEME = 0
        last_phoneme = SILENT_PHONEME
        samples_num = tensor.shape[1]
        for i in range(samples_num):
            curr_phoneme = tensor[0, i].item()
            tensor[0, i] = 0
            if curr_phoneme != last_phoneme:
                last_phoneme = curr_phoneme
                if curr_phoneme != SILENT_PHONEME:
                    tensor[0, i] = 1

        pulse_len = max(1, int(self.duration_ms * self.sample_rate / 1000))
        convert_to_pulse_count = 0
        for i in range(samples_num - (pulse_len - 1)):
            if convert_to_pulse_count > 0:
                tensor[0, i] = 1
                convert_to_pulse_count -= 1
            if tensor[0, i] == 1:
                convert_to_pulse_count = pulse_len - 1


class WordSegment(Feature):
    """
    Categorical feature that marks '1' with places where word stimulus exists
    """
    cardinality = 2
    event_kind = "word"

    def get(self, event: events.Word) -> int:
        return 1


class Modality(Feature):
    """Categorical task feature"""
    cardinality = 3
    event_kind = "word"

    def get(self, event: events.Word) -> int:
        if event.modality == "audio":
            return 1
        if event.modality == "visual":
            return 2
        raise RuntimeError("Only audio and visual modalities are supported")


class WordLength(Feature):
    event_kind = "word"

    def get(self, event: events.Word) -> int:
        return len(event.word)


class WordIndex(Feature):
    event_kind = "word"

    def get(self, event: events.Word) -> int:
        return event.word_index + 1


class WordFrequency(Feature):
    event_kind = "word"

    def get(self, event: events.Word) -> float:
        assert event.language is not None
        return float(zipf_frequency(event.word, event.language))


class Phoneme(Feature):
    """
    Outputs the phoneme category for a given feature.
    """
    cardinality = len(ph_dict) + 1  # + 1 for silent phoneme
    event_kind = "phoneme"

    def get(self, event: events.Phoneme) -> int:
        assert 0 <= int(event.phoneme_id) < self.cardinality - 1, \
            f"Phoneme ID={int(event.phoneme_id)} while cardinality is {self.cardinality}"
        return int(event.phoneme_id) + 1  # 0 is saved for silence


class WordHash(Feature):
    """
    Return a word hash. If `buckets` is provided, will distribute
    the word hashes into that many buckets and be a categorical feature.
    This is useful for word prediction without requiring
    a precise vocabulary.
    """

    normalizable = False
    event_kind = "word"

    def __init__(self, sample_rate: Frequency, buckets: tp.Optional[int] = None):
        super().__init__(sample_rate)
        self.buckets = buckets
        if buckets is not None:
            self.cardinality = 1 + buckets

    def get(self, event: events.Word) -> float:
        hsh = hash(event.word.lower().strip('.').encode())
        if self.buckets is not None:
            hsh = 1 + (hsh % self.buckets)
        return hsh
