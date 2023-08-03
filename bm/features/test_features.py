# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bm.utils import Frequency
from bm import events, play, env
from bm.studies.fake import make_fake_events
from . import FeaturesBuilder, audio

logger = logging.getLogger(__name__)

# register contextual embeddings features
# pylint: disable=unused-import


def _compare_dict(a, b):
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    diff = keys_a.symmetric_difference(keys_b)
    assert diff == set()

    for key in a.keys():
        assert a[key] == b[key], key


@pytest.fixture()
def events_df() -> pd.DataFrame:
    return make_fake_events(total_duration=10)


@pytest.mark.parametrize("start,stop",  # type: ignore
                         [(0, 3), (1., 2.)]
                         )
def test_builder(start: float, stop: float, events_df) -> None:
    sample_rate = Frequency(100)
    # Xlm requires extra downloads, so let's forget about it
    features = [x for x in sorted(FeaturesBuilder._FEATURE_CLASSES) if (not x.startswith(
        "Xlm") and not ("Wav2Vec" in x))]
    features_params = {
        "MelSpectrum": {"n_fft": 100, "n_mels": 8},
        "WordHash": {"buckets": 100},
    }
    builder = FeaturesBuilder(events_df, features, features_params, sample_rate=sample_rate)
    out, mask, _ = builder(start, stop)
    assert out.shape == (builder.dimension, int((stop - start) * sample_rate))
    assert features == list(builder.keys())
    max_amp = np.max(np.abs(out.numpy()), axis=1)
    if np.min(max_amp) == 0:
        for feature in features:
            entry = out[builder.get_slice(feature)]
            if entry.std() == 0:
                assert entry.std() > 0, f"Feature {feature} is inactive"
        assert "Some feature is inactive but couldnt find it."
    dimensions = {name: f.dimension for name, f in builder.items()}
    reference = dict([
        ("Phoneme", 1),
        ("PhonemePulse", 1),
        ("MelSpectrum", 8),
        ("Pitch", 1),
        ("WordSegment", 1),
        ("WordHash", 1),
        ("WordPulse", 1),
        ("Modality", 1),
        ("WordEmbedding", 300),
        ("WordEmbeddingSmall", 96),
        ("WordFrequency", 1),
        ("WordLength", 1),
        ("WordIndex", 1),
        ("PartOfSpeech", 1),
        ("BertEmbedding", 768),
    ])
    _compare_dict(dimensions, reference)

    out_dimensions = {name: f.output_dimension for name, f in builder.items()}
    reference = dict([
        ("Phoneme", 44),
        ("PhonemePulse", 1),
        ("MelSpectrum", 8),
        ("Pitch", 1),
        ("WordSegment", 2),
        ("WordPulse", 1),
        ("WordHash", 101),
        ("Modality", 3),
        ("WordEmbedding", 300),
        ("WordEmbeddingSmall", 96),
        ("WordFrequency", 1),
        ("WordLength", 1),
        ("WordIndex", 1),
        ("PartOfSpeech", 21),
        ("BertEmbedding", 768),
    ])
    _compare_dict(out_dimensions, reference)

    with pytest.raises(KeyError):
        builder.get_slice("NotAFeature")

    # Check feature params are set correctly
    assert builder["MelSpectrum"].n_fft == features_params["MelSpectrum"]["n_fft"]
    assert builder["MelSpectrum"].dimension == features_params["MelSpectrum"]["n_mels"]


@pytest.mark.parametrize("start,stop,num", [
    (0.0, 0.8, 0), (0.0, 1.5, 1), (0.0, 3.0, 6), (1.5, 3.0, 5), (3.0, 10.0, 16)
])
def test_event_filtering(start: float, stop: float, num: int, events_df) -> None:
    sample_rate = Frequency(100)
    builder = FeaturesBuilder(
        events_df, ["WordLength"], features_params={}, sample_rate=sample_rate)
    _, _, events = builder(start, stop)

    assert [event.start >= start and event.stop <= stop for event in events]
    assert len(events) == num + 1


@pytest.mark.parametrize("name", list(FeaturesBuilder._FEATURE_CLASSES))
def test_event_kind_exists(name: str) -> None:
    cls = FeaturesBuilder._FEATURE_CLASSES[name]
    assert cls.event_kind in events.EventAccessor.CLASS_KIND_MAPPING.keys()


@pytest.mark.parametrize(  # type: ignore
    "name,expected", [
        ("WordFrequency", 5.95),
        ("WordLength", 4),
    ]
)
def test_features(name: str, expected: float, events_df) -> None:
    event = next(events_df.itertuples(index=False))
    out = FeaturesBuilder._FEATURE_CLASSES[name](sample_rate=Frequency(10)).get(event)
    assert out == pytest.approx(expected, abs=0.01)


def test_sentence_builder() -> None:
    builder = play.SentenceFeatures(
        ["WordPulse", "WordFrequency", "WordLength"], features_params={}, sample_rate=20)
    sentence = builder("My name is Earl")
    assert sentence.shape == (3, 86)


@pytest.mark.parametrize(  # type: ignore
    "sentence", ["De barkeeper.", "De bar's keeper"]
)
def test_xlr_embedding(sentence: str) -> None:
    cache = Path(torch.hub.get_dir())
    if not (cache / "pytorch_fairseq_main").exists():
        pytest.skip("Deactivated since model is not already downloaded")
    word = sentence.split()[1]
    event = events.Word(start=0, duration=1, word=word, word_sequence=sentence, word_index=1,
                        modality="visual", language="nl")
    feature = FeaturesBuilder._FEATURE_CLASSES["XlmEmbedding"](sample_rate=Frequency(10))
    out = feature.get(event)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (feature.dimension,)


@pytest.mark.parametrize("cls", [audio.Wav2VecConvolution, audio.Wav2VecTransformer])
def test_wav2vec(cls, tmp_path: Path) -> None:
    if os.environ.get("CIRCLECI", ""):
        pytest.skip("Models do not run in the CI")
    wavpath = str(Path(__file__).parent.parent / "mockdata" / "one_two.wav")
    event = events.Sound(start=1, duration=1, filepath=wavpath, modality=None, language=None)
    overlap = events.DataSlice(
        start=event.start + 0.1, duration=1.0, sample_rate=100, modality=None, language=None)
    with env.temporary(cache=tmp_path):
        feature = cls(sample_rate=Frequency(100))
        feature.get_on_overlap(event, overlap)
        caches = list(tmp_path.iterdir())
        assert len(caches) == 1
        # reload (expected to use the cache)
        out = feature.get_on_overlap(event, overlap)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (feature.dimension, 100)
