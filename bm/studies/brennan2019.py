# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TODO
- [ ] gentle to get phonemes
"""
import typing as tp
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
import cgi


import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from . import api, utils
from ..events import extract_sequence_info

SFREQ = 500.0


def get_paths() -> utils.StudyPaths:
    return utils.StudyPaths(Brennan2019Recording.study_name())


def _download_file(url, target_folder):
    """automatically detect file name"""
    urlretrieve(url)
    remotefile = urlopen(url)
    hdr = remotefile.info()["Content-Disposition"]
    value, params = cgi.parse_header(hdr)
    target = target_folder / params["filename"]
    target_folder.mkdir(exist_ok=True, parents=True)
    urlretrieve(url, filename=str(target))
    return


def _prepare():
    paths = get_paths()

    files = ["vm40xs661", "t435gf09p", "0v8381376", "6h440t36j", "qv33rx36x"]
    files += ["7w62f925w", "5425kb76p", "g445cf216", "df65v8733", "41687j32q"]
    files += ["r207tq17h", "pn89d748r", "41687j330", "xg94hq37z", "fj2362955"]
    files += ["1r66j195h", "j098zc06b", "3n203z903", "gx41mj79g", "mp48sd64h"]
    files += ["p2676w56p", "dn39x2566", "pv63g1045", "r207tq18s", "qr46r1659"]
    files += ["wd375x18w", "td96k336b", "6q182m27b", "ms35t936k", "02870w66d"]
    files += ["cj82k821x", "9k41zf376", "bk128b81j", "q524jp737", "37720d60h"]
    files += ["ks65hd14w", "b5644s476", "3t945r72w", "bn999773b", "4t64gp10r"]
    files += ["qr46r166k", "h415pb60j", "sq87bv504", "ht24wk29w", "p2676w57z"]
    files += ["2514nm49h", "41687j348", "tq57ns04w", "4t64gp111", "5712m736z"]
    files += ["f1881m88g", "2b88qd012", "2b88qd00s", "bn999775w", "h415pb59s"]
    files += ["q524jp72z"]

    success = paths.download / "success_download.txt"
    if not success.exists():

        url = "https://deepblue.lib.umich.edu/data/downloads/"
        print(f"Downloading `brennan2019` files to {paths.download}...")
        for file in tqdm(files):
            _download_file(url + file, paths.download)

        with open(success, "w") as f:
            f.write("success")

    # extract
    success = paths.download / "success_extract.txt"
    if not success.exists():
        print(f"Extracting `brennan2019` audio to {paths.download}/audio...")
        with zipfile.ZipFile(str(paths.download / "audio.zip"), "r") as zip:
            zip.extractall(str(paths.download))

        print(f"Extracting `brennan2019` proc to {paths.download}/proc...")
        with zipfile.ZipFile(str(paths.download / "proc.zip"), "r") as zip:
            zip.extractall(str(paths.download))

        with open(success, "w") as f:
            f.write("success")


def _read_meta(fname):
    proc = loadmat(
        fname,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )["proc"]

    # ref = proc["implicitref"]
    # ref_channels = proc["refchannels"]

    # subject_id = proc["subject"]
    meta = proc["trl"]

    # TODO artefacts, ica, rejected components etc
    assert len(meta) == proc["tot_trials"]
    assert proc["tot_chans"] == 61
    bads = list(proc["impedence"]["bads"])
    bads += list(proc["rejections"]["badchans"])
    # proc['varnames'], ['segment', 'tmin', 'Order']

    # summary = proc["rejections"]["final"]["artfctdef"]["summary"]
    # bad_segments = summary["artifact"]

    #     meta = pd.DataFrame(meta[:, 0].astype(int), columns=['start'])

    #     meta['start_offset'] = meta[:, 1].astype(int) # wave?
    #     meta['wav_file'] = meta[:, 3].astype(int)
    #     meta['start_sec'] = meta[:, 4]
    #     meta['mat_index'] = meta[:, 5].astype(int)
    columns = list(proc["varnames"])
    if len(columns) != meta.shape[1]:
        columns = ["start_sample", "stop_sample", "offset"] + columns
        assert len(columns) == meta.shape[1]
    meta = pd.DataFrame(meta, columns=["_" + i for i in columns])
    assert len(meta) == 2129  # FIXME retrieve subjects who have less trials?

    # Add Brennan's annotations
    paths = get_paths()
    story = pd.read_csv(paths.download / "AliceChapterOne-EEG.csv")
    events = meta.join(story)

    events["kind"] = "word"
    events["condition"] = "sentence"
    events["duration"] = events.offset - events.onset
    columns = dict(Word="word", Position="word_id", Sentence="sequence_id")
    events = events.rename(columns=columns)
    events["start"] = events["_start_sample"] / SFREQ

    # add audio events
    wav_file = (
        paths.download / "audio" / "DownTheRabbitHoleFinal_SoundFile%i.wav"
    )
    sounds = []
    for segment, d in events.groupby("Segment"):
        # Some wav files start BEFORE the onset of eeg recording...
        start = d.iloc[0].start - d.iloc[0].onset
        sound = dict(
            kind="sound", start=start, filepath=str(wav_file) % segment
        )
        sounds.append(sound)
    events = pd.concat([events, pd.DataFrame(sounds)], ignore_index=True)
    events = events.sort_values("start").reset_index()

    # clean up
    keep = [
        "start",
        "duration",
        "kind",
        "word",
        "word_id",
        "sequence_id",
        "condition",
        "filepath",
    ]
    events = events[keep]
    events[['language', 'modality']] = 'english', 'audio'
    events = extract_sequence_info(events)
    events = events.event.create_blocks(groupby='sentence')
    events = events.event.validate()

    return events


def _read_eeg(fname):
    fname = Path(fname)
    assert fname.exists()
    assert str(fname).endswith(".mat")
    mat = loadmat(
        fname,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )
    mat = mat["raw"]

    # sampling frequency
    sfreq = mat["hdr"]["Fs"]
    assert sfreq == 500.0
    assert mat["fsample"] == sfreq

    # channels
    n_chans = mat["hdr"]["nChans"]
    n_samples = mat["hdr"]["nSamples"]
    ch_names = list(mat["hdr"]["label"])
    assert len(ch_names) == n_chans

    # vertical EOG
    assert ch_names[60] == "VEOG"

    # audio channel
    add_audio_chan = False
    if len(ch_names) == 61:
        ch_names += ["AUD"]
        add_audio_chan = True
    assert ch_names[61] in ("AUD", "Aux5")

    # check name
    for i, ch in enumerate(ch_names[:-2]):
        assert ch == str(i + 1 + (i >= 28))

    # channel type
    assert set(mat["hdr"]["chantype"]) == set(["eeg"])
    ch_types = ["eeg"] * 60 + ["eog", "misc"]
    assert set(mat["hdr"]["chanunit"]) == set(["uV"])

    # create MNE info
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    subject_id = fname.name.split(".mat")[0]
    info["subject_info"] = dict(his_id=subject_id, id=int(subject_id[1:]))

    # time
    diff = np.diff(mat["time"]) - 1 / sfreq
    tol = 1e-5
    assert np.all(diff < tol)
    assert np.all(diff > -tol)
    start, stop = mat["sampleinfo"]
    assert start == 1
    assert stop == n_samples
    assert mat["hdr"]["nSamplesPre"] == 0
    assert mat["hdr"]["nTrials"] == 1

    # data
    data = mat["trial"]
    assert data.shape[0] == n_chans
    assert data.shape[1] == n_samples
    if add_audio_chan:
        data = np.vstack((data, np.zeros_like(data[0])))

    # create mne objects
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    montage = mne.channels.make_standard_montage("easycap-M10")
    raw.set_montage(montage)

    assert raw.info["sfreq"] == SFREQ
    assert len(raw.ch_names) == 62

    return raw


class Brennan2019Recording(api.Recording):

    data_url = "https://deepblue.lib.umich.edu/data/concern/data_sets/"
    data_url += "bg257f92t"
    paper_url = "https://journals.plos.org/plosone/"
    paper_url += "article?id=10.1371/journal.pone.0207741"
    doi = "https://doi.org/10.1371/journal.pone.0207741"
    licence = "CC BY 4.0"
    modality = "audio"
    language = "english"
    device = "eeg"
    description = """EEG of Alice in WonderLand, By Brennan and Hale 2019.
    The eeg data was bandpassed between 0.1 and 200. Hz"""

    @classmethod
    def iter(cls) -> tp.Iterator["Brennan2019Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        # download, extract, organize
        paths = get_paths()
        _prepare()

        subjects = [
            f.name
            for f in (paths.download / "proc").iterdir()
            if (f.name.startswith("S") and f.name.endswith(".mat"))
        ]
        assert len(subjects) == 42
        # remove bad subject s24 (metadata does not have enough trials)
        # FIXME retrieve these subjects?
        bads = [
            "S24.mat",
            "S26.mat",
            "S27.mat",
            "S30.mat",
            "S32.mat",
            "S34.mat",
            "S35.mat",
            "S36.mat",
        ]
        bads += ["S02.mat"]  # bad proc.trl?
        subjects = [s.split(".")[0] for s in subjects if s not in bads]

        for subject in subjects:
            recording = cls(subject_uid=str(subject))
            yield recording

    def __init__(self, subject_uid: str) -> None:
        super().__init__(subject_uid=subject_uid, recording_uid=subject_uid)

    def _load_raw(self) -> mne.io.RawArray:
        paths = get_paths()
        raw = _read_eeg(paths.download / f"{self.subject_uid}.mat")
        return raw

    def _load_events(self) -> pd.DataFrame:
        file = get_paths().download / "proc" / f"{self.subject_uid}.mat"
        events = _read_meta(file)
        return events
