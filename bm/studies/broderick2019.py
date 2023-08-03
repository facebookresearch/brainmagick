# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Force speech alignement manually performed on privately shared audio
files, using Gentle. There seems to a significant discrepency on the
result of run_15 around 170 s.
"""
import json
import typing as tp
from urllib.request import urlretrieve
from zipfile import ZipFile

import mne
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
import spacy

from . import api, utils
from ..events import extract_sequence_info


SPACY_MODEL = "en_core_web_md"


def get_paths() -> utils.StudyPaths:
    return utils.StudyPaths(Broderick2019Recording.study_name())


def _prepare():
    paths = get_paths()
    paths.download.mkdir(exist_ok=True, parents=True)
    url = "http://datadryad.org/api/v2/datasets/"
    url += "doi%253A10.5061%252Fdryad.070jc/download"
    zip_dset = paths.download / "doi_10.5061_dryad.070jc__v3.zip"

    # download public files
    if not zip_dset.exists():
        print("Downloading Broderick_2019 dataset...")
        urlretrieve(url, zip_dset)

    # extract
    if not any([f.name == "N400.zip" for f in paths.download.iterdir()]):
        print("Extracting Broderick_2019 dataset...")
        with ZipFile(str(paths.download / zip_dset), "r") as zip:
            zip.extractall(str(paths.download))
    dsets = [
        "Cocktail Party",
        "N400",
        "Natural Speech - Reverse",
        "Natural Speech",
        "Speech in Noise",
    ]
    for dset in dsets:
        subfolder = paths.download / dset
        if not subfolder.exists():
            print(f"Extracting {dset}...")
            with ZipFile(str(subfolder) + ".zip", "r") as zip:
                zip.extractall(str(paths.download))

    # download audio files
    zip_private = paths.download / "private.zip"
    if not zip_private.exists():
        print("Downloading Broderick_2019 private files...")
        url = "https://ai.honu.io/papers/brainmagick/private.zip"
        urlretrieve(url, zip_private)

    # extract private files
    folder_private = paths.download / "private"
    if not folder_private.exists():
        print("Extracting Broderick_2019 private files...")
        with ZipFile(str(zip_private), "r") as zip:
            zip.extractall(paths.download)


class _BroderickMetadata:
    def _parse_json(self, run_id: str) -> pd.DataFrame:
        """parse json to flatten word and phoneme into a dataframe"""
        folder = get_paths().download

        with open(folder / "private" / f"align{run_id}.json", "r") as f:
            align = json.load(f)

        meta = list()
        for entry in align["words"]:
            # for each event
            entry.pop("endOffset")
            entry.pop("startOffset")
            success = entry.pop("case") == "success"
            if not success:
                continue
            if entry["alignedWord"] == "<unk>":
                success = False
            entry["success"] = success

            # word level
            txt = entry.pop("word")
            entry["string"] = txt
            phones = entry.pop("phones")
            entry["phone"] = " ".join([k["phone"] for k in phones])
            entry["duration"] = entry["end"] - entry["start"]

            aligned = entry.pop("alignedWord")
            entry["aligned"] = aligned
            meta.append(entry)
            meta[-1]["kind"] = "word"

            # phoneme level
            start = entry["start"]
            for phone in phones:
                phone["start"] = start
                start += phone["duration"]
                phone["end"] = start
                phone["kind"] = "phoneme"
                phone["success"] = success
                phone["aligned"] = phone["phone"]
                phone["string"] = phone["phone"]
                meta.append(phone)
        # add audio entry
        wav = folder / "private" / f"audio{run_id}.wav"
        sound = dict(start=0, kind="sound", filepath=wav)

        df = pd.DataFrame([sound] + meta)
        df["duration"] = df["end"] - df["start"]

        return df

    def _parse_txt(self, run_id: str) -> pd.DataFrame:
        # read text
        txt_file = get_paths().download / "private" / f"oldman_run{run_id}.txt"
        with open(txt_file, "r") as f:
            txt = f.read()

        # tokenize text
        doc = self.nlp(txt)

        # retrieve word and sentences
        df = []
        for sequence_id, sent in enumerate(doc.sents):
            seq_uid = str(sent)
            for word_id, word in enumerate(sent):
                word_ = re.sub(r"\W+", "", str(word))
                if not len(word_):
                    continue
                df.append(
                    dict(
                        word=word_,
                        original_word=word,
                        word_id=word_id,
                        sequence_id=sequence_id,
                        sequence_uid=seq_uid,
                    )
                )
        df = pd.DataFrame(df)
        return df

    def __call__(self, run_id: str) -> pd.DataFrame:

        # lazy init
        if not hasattr(self, "nlp"):
            self.nlp = spacy.load(SPACY_MODEL)
            self._cache: dict = dict()

        if run_id not in self._cache.keys():
            self._cache[run_id] = self._process(run_id)
        return self._cache[run_id].copy()

    def _process(self, run_id: str) -> pd.DataFrame:
        # read json file
        json = self._parse_json(run_id)

        # read text and parse with spacy
        text = self._parse_txt(run_id)

        # compare words in json and in text
        trans_words = json.query('kind=="word"')

        i, j = utils.match_list(trans_words.string.str.lower(), text.word.str.lower())
        assert len(i) > 450

        # add sequence information
        fields = ("sequence_id", "sequence_uid", "word_id")
        for k in fields:
            json.loc[trans_words.iloc[i].index, k] = text.iloc[j][k].values
        missed = np.setdiff1d(range(len(json)), trans_words.index[i])

        # fill-up missing information for phoneme and missed words
        prev = None
        indices = []
        for curr, sequ in enumerate(json.sequence_id):
            if curr in missed:
                indices.append(json.index[curr])
            else:
                if len(indices) and prev is not None:
                    for k in fields:
                        json.loc[indices, k] = json.iloc[prev][k]
                    indices = []
                prev = curr

        json["condition"] = "sentence"
        for kind in ("word", "phoneme"):
            idx = json.kind == kind
            json.loc[idx, kind] = json.loc[idx].string

        json.loc[json.kind == 'phoneme', 'phoneme_id'] = 0  # Add dummy phoneme_id

        return json


class Broderick2019Recording(api.Recording):

    data_url = "https://datadryad.org/stash/dataset/doi:10.5061/dryad.070jc"
    paper_url = "https://pubmed.ncbi.nlm.nih.gov/29478856/"
    doi = "https://doi.org/10.5061/dryad.070jc"
    licence = "CC0 1.0"
    modality = "audio"
    language = "english"
    device = "eeg"
    description = """
    """
    _metadata = _BroderickMetadata()

    @classmethod
    def iter(cls) -> tp.Iterator["Broderick2019Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        # download, extract, organize
        paths = get_paths()
        _prepare()
        if not spacy.util.is_package(SPACY_MODEL):
            spacy.cli.download(SPACY_MODEL)

        files = list((paths.download / "Natural Speech" / "EEG").iterdir())
        subjects = [
            int(f.name.split("Subject")[1])
            for f in files
            if "Subject" in f.name
        ]
        subjects = sorted(subjects)

        for subject in subjects:
            for run_id in range(1, 21):
                recording = cls(subject_uid=str(subject), run_id=str(run_id))
                yield recording

    def __init__(self, subject_uid: str, run_id: str) -> None:
        recording_uid = f"{subject_uid}_run{run_id}"
        super().__init__(subject_uid=subject_uid, recording_uid=recording_uid)
        self.run_id = run_id

    def _load_raw(self) -> mne.io.RawArray:
        paths = get_paths()

        eeg_fname = (
            paths.download
            / "Natural Speech"
            / "EEG"
            / f"Subject{self.subject_uid}"
            / f"Subject{self.subject_uid}_Run{self.run_id}.mat"
        )
        mat = loadmat(str(eeg_fname))

        assert mat["fs"][0][0] == 128
        ch_types = ["eeg"] * 128
        # FIXME montage?
        montage = mne.channels.make_standard_montage("biosemi128")
        info = mne.create_info(montage.ch_names, 128.0, ch_types)
        eeg = mat["eegData"].T * 1e6
        assert len(eeg) == 128
        raw = mne.io.RawArray(eeg, info)
        raw.set_montage(montage)

        # TODO make mastoids EEG and add layout position
        info_mas = mne.create_info(
            ["mastoids1", "mastoids2"], 128.0, ["misc", "misc"]
        )
        mastoids = mne.io.RawArray(mat["mastoids"].T * 1e6, info_mas)
        raw.add_channels([mastoids])

        raw = raw.pick_types(
            meg=False, eeg=True, misc=False, eog=False, stim=False
        )
        return raw

    def _load_events(self) -> pd.DataFrame:
        # read and preprocess events from external log file
        # the files were shared manually and aligned with gentle
        events = self._metadata(self.run_id)

        events[['language', 'modality']] = self.language, self.modality
        events = extract_sequence_info(events, phoneme=False)
        events = events.event.create_blocks(groupby='sentence')

        return events
