# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools  # pylint: disable=broad-except
from pathlib import Path
import typing as tp
from urllib.request import urlretrieve
from zipfile import ZipFile

from dora import to_absolute_path
import pandas as pd
import mne

from .. import api
from ..download import download_donders
from . import preproc
from .paths import StudyPaths as StudyPaths
from bm import env


RAW_SAMPLE_RATE = 1200  # FIXME


def read_raw(raw_fname: str, *args: tp.Any, **kwargs: tp.Any) -> mne.io.RawArray:
    """Calls to ctf reader or standard mne (fif) reader depending on the extension
    """
    # The database uses CTF but tests uses fif.
    # The aim is to be able to test the pipeline more easily through mocked data.
    if raw_fname.endswith('.ds'):
        return mne.io.read_raw_ctf(raw_fname, *args, **kwargs)
    return mne.io.read_raw(raw_fname, *args, **kwargs)


def _prepare():
    if 'schoffelen2019' not in env.studies:
        # hopefully in a unit test
        return
    path = env.studies['schoffelen2019']
    path = to_absolute_path(path)
    url = Schoffelen2019Recording.data_url
    parent, study = url.split('/')[-2:]
    download_donders(study, path, parent=parent)
    derivatives = Path(path) / 'download' / 'derivatives'
    if not derivatives.exists():
        zip_derivatives = derivatives.parent / "derivatives.zip"
        if not zip_derivatives.exists():
            print("Downloading Broderick_2019 private files...")
            url = "https://ai.honu.io/papers/brainmagick/derivatives.zip"
            urlretrieve(url, zip_derivatives)
        print("Extracting Broderick_2019 private files...")
        with ZipFile(str(zip_derivatives), "r") as zip:
            zip.extractall(derivatives.parent)


class Schoffelen2019Recording(api.Recording):
    """Subject instance providing access to all recording information.
    All computations are lazzy so instantiating takes no time.

    Parameter
    ---------
    uid: int
        the ID/name of the recording (Eg.: 1004). Visual are in the range 1000, and audio 2000.

    Attributes
    ----------
    modality: str
        either "visual" or "audio"
    uid: int
        the uid in the initial dataset Eg: 2101 for a visual recording.
    """

    data_url = "https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236_v1"
    paper_url = "https://www.nature.com/articles/s41597-019-0020-y"
    doi = "https://doi.org/10.1038/s41597-019-0020-y"
    licence = "Donders"
    modality = "all"  # FIXME insufficiently specific
    language = "nl"
    device = "meg"
    description = "204 subjects listened or read context-less sentences."

    # pylint disable=arguments-differ
    @classmethod
    def iter(  # type: ignore
        cls,
        events_filter: tp.Optional[str] = None,
        modality: str = "all"
    ) -> tp.Iterator["Schoffelen2019Recording"]:
        """Returns a generator of all subjects (except 2063 which is incomplete)

        Parameters
        ----------
        events_filter: str
            ???
        modality: str
            either "visual", "audio" or "all"

        Note
        ----
        recordings are ordered by uid
        """
        _prepare()
        # The following subjects have 2 separate MEG runs, which would require
        # specific code to handle.
        # 1115 has nans in unexpected places.
        bad_nums = [2011, 2036, 2062, 2063, 2076, 2084, 1006, 1014, 1090, 1115]
        no_subject = [1014, 1018, 1021, 1023, 1041, 1043, 1047, 1051, 1056]
        no_subject += [1060, 1067, 1082, 1091, 1096, 1112]
        no_subject += [2012, 2018, 2022, 2023, 2026, 2043, 2044, 2045, 2048]
        no_subject += [2054, 2060, 2074, 2081, 2082, 2087, 2093, 2100, 2107]
        no_subject += [2112, 2115, 2118, 2123]
        if modality not in ["visual", "audio", "all"]:
            raise ValueError(f"Unknown modality: {modality}")
        for num in itertools.chain(range(1001, 1118), range(2002, 2126)):
            if num in bad_nums + no_subject:
                continue  # incomplete data
            subject_uid = f"sub-{'V' if num < 2000 else 'A'}{num}"
            subject = cls(subject_uid, events_filter)
            if subject.modality == modality or modality == "all":
                yield subject

    def __init__(self, subject_uid: str, events_filter: tp.Optional[str] = None) -> None:
        super().__init__(subject_uid=subject_uid, recording_uid=subject_uid)
        num = int(subject_uid[-4:])
        self.modality = "visual" if num < 2000 else "audio"
        assert subject_uid == f"sub-{self.modality[0].upper()}{num}"
        self.paths = StudyPaths.create(subject_uid)  # same as init, but easier to mock in tests
        self._events_filter = events_filter

    def _load_raw(self) -> mne.io.RawArray:
        """Loads raw file from the dataset
        """
        # having a separate function is helpful for mocking
        raw = read_raw(str(self.paths.raw), preload=False)
        if raw.info["sfreq"] != RAW_SAMPLE_RATE:
            raise RuntimeError("Raw has an unexpected sample rate, breaking code assumptions")
        picks = mne.pick_types(
            raw.info, meg=True, eeg=False, stim=False, eog=False, ecg=False
        )[28: (28 + 273)]
        raw.pick(picks)  # only keep data channels
        return raw

    def _load_events(self) -> pd.DataFrame:
        """Loads events from the dataset
        """
        # having a separate function is helpful for mocking
        raw = read_raw(str(self.paths.raw), preload=False)
        sfreq = raw.info["sfreq"]
        events = mne.find_events(raw, shortest_event=1)
        # Read Event data
        metadata = preproc.read_log(str(self.paths.metadata))
        # Align MEG and events
        metadata = preproc.get_log_times(metadata, events, sfreq)
        # rename
        metadata = metadata.rename(columns=dict(
            start="offset", meg_time="start", stop="legacy_stop", condition="kind"
        ))

        # Clean up dataframe
        events_df = metadata.drop(
            columns=[x for x in metadata.columns if x.startswith("legacy_")])
        cols_to_keep = [
            'start', 'duration', 'kind', 'context', 'word', 'filepath', 'sequence_id',
            'word_index', 'phoneme', 'phoneme_id', 'word_sequence', 'sequence_uid']
        events_df = events_df.loc[events_df.kind.isin(['word', 'phoneme', 'sound']), cols_to_keep]

        # Create blocks
        events_df[['language', 'modality']] = self.language, self.modality
        events_df = events_df.event.create_blocks(groupby='sentence_or_sound')

        return events_df

    def events(self, clean: bool = False) -> pd.DataFrame:
        events = super().events(clean)
        # TODO(orik): move events filtering into FeatureMaker
        if clean and self._events_filter is not None:
            events = events.query(self._events_filter)
        return events
