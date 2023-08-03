# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
from pathlib import Path
import typing as tp
import mne
import yaml
import pandas as pd
import numpy as np
import torch
import julius
import bm
from bm import env


def _give_permission(filepath: tp.Optional[Path]) -> None:
    """Set 777 permissions for sharing more easily
    """
    if filepath is not None and filepath.exists():  # for sharing
        try:
            filepath.chmod(0o777)
        except PermissionError:
            pass


register: tp.Dict[str, tp.Type["Recording"]] = {}
R = tp.TypeVar("R", bound="Recording")


def from_selection(selection: tp.Dict[str, tp.Any]) -> tp.Iterator["Recording"]:
    """Instantiate the iterator of recording from a selection dict
    Selections are dict with at least one key "study" corresponding to the study name,
    additional keys will be used as parameters of the Recording.iter method.

    Example
    -------
    from_selection({"study": "schoffelen2019, "modality": "audio"})
    """
    params = dict(selection)
    name = params.pop("study")
    return register[name].iter(**params)


class Recording:
    """
    Parameter
    ---------
    subject_uid: str
        unique uid of the subject, as a string

    Attributes
    ----------
    subject_index: int
        the index of the subject, across all recordings and studies.
    recording_index: int
        the index of the recording, across all recordings and studies.
    """
    data_url: str
    paper_url: str
    doi: str
    licence: str
    modality: str
    language: str
    device: str
    description: str

    # TO BE IMPLEMENTED FOR EACH STUDY #

    @classmethod
    def iter(cls: tp.Type[R], **kwargs: tp.Any) -> tp.Iterator[R]:
        """List all the recordings in the study"""
        raise NotImplementedError

    def _load_events(self) -> pd.DataFrame:
        """Loads the events of this recording as a csv"""
        raise NotImplementedError

    def _load_raw(self) -> mne.io.RawArray:
        """Loads the raw MEG array of this recording"""
        raise NotImplementedError

    # the following is shared between all studies

    @classmethod
    def study_name(cls) -> str:
        return cls.__name__.replace("Recording", "").lower()

    @classmethod
    def __init_subclass__(cls) -> None:
        """Record all existing recording classes"""
        super().__init_subclass__()
        if cls.__name__.startswith('_'):
            return  # for base classes
        name = cls.study_name()
        register[name] = cls
        expected_name = cls.__module__.rsplit('.', maxsplit=1)[-1]
        assert name == expected_name, (
            f"Study {name} is defined in {expected_name} "
            "instead of its using name.")
        register[cls.study_name()] = cls
        # check that Recording has correct information
        compulsory_keys = (
            'data_url', 'paper_url', 'doi', 'licence', 'modality',
            'language', 'device', 'description'
        )
        for key in compulsory_keys:
            assert isinstance(getattr(cls, key), str)

        # check that Recording.list API is correct
        params = list(inspect.signature(cls.iter).parameters.keys())
        assert "study" not in params, ('"study" is a reserved name which cannot be used as '
                                       f'a parameter of {cls.__name__}.iter.')

    def __init__(self, *, subject_uid: str, recording_uid: str) -> None:
        if not isinstance(subject_uid, str):
            raise TypeError(f"Recording.subject_uid needs to be a str instance, got: {subject_uid}")
        self.subject_uid = subject_uid
        self.recording_uid = recording_uid
        self._subject_index: tp.Optional[int] = None  # specified during training
        self._recording_index: tp.Optional[int] = None  # specified during training
        self._mne_info: tp.Optional[mne.Info] = None
        # cache system
        self._arrays: tp.Dict[tp.Tuple[int, float], mne.io.RawArray] = {}
        self._events: tp.Optional[pd.DataFrame] = None

        if env.cache is None:
            self._cache_folder: tp.Optional[Path] = None
        else:
            self._cache_folder = env.cache / "studies" / self.study_name() / recording_uid
            self._cache_folder.mkdir(parents=True, exist_ok=True, mode=0o777)

    def empty_copy(self: R) -> R:
        """Creates a copy of the instance, without cached information
        (for fast transfer)
        """
        out = copy.copy(self)
        out._events = None
        out._arrays = {}
        return out

    @property
    def subject_index(self) -> int:
        if self._subject_index is None:
            raise RuntimeError("Recording.subject_index has not been initialized")
        return self._subject_index

    @property
    def recording_index(self) -> int:
        if self._recording_index is None:
            raise RuntimeError("Recording.recording_index has not been initialized")
        return self._recording_index

    @property
    def meg_dimension(self) -> int:
        # take any available raw array to identify final time
        raw_array = self.any_raw()
        return len(raw_array.ch_names)

    @property
    def mne_info(self) -> mne.Info:
        """Return the MNE Info object. Note that the sample rate might not be correct
        due to resampling, but all other informations should be preserved."""
        if self._mne_info is None:
            self._mne_info = self.any_raw().info
        return self._mne_info

    def any_raw(self) -> mne.io.RawArray:
        """Return any raw currently cached, including preprocessed."""
        return next(iter(self._arrays.values())) if self._arrays else self.raw()

    def raw(self) -> mne.io.RawArray:
        """Loads, caches and returns the raw for the subject
        """
        key = (0, 0.0)  # 0 for raw
        if key not in self._arrays:
            raw = self._load_raw()
            raw = raw.pick_types(eeg=True, meg=True, ref_meg=True, stim=False)
            self._arrays[key] = raw
            self.mne_info  # populate mne info cache
        return self._arrays[key]

    # below is automatically handled and should not be reimplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recording_uid!r})"

    def preprocessed(self, sample_rate: tp.Optional[float] = None,
                     highpass: float = 0) -> mne.io.RawArray:
        """Creates and/or loads the data at a given sampling rate.
        1200Hz sample_rate with no highpass (0) would returns the raw,
        different values would create a subsampled fif file and load it from the cache.

        Parameter
        ---------
        sample_rate: int
            The wanted sample rate of the data.
        highpass: float
            the frequency of the highpass filter (no high pass filter if 0)
        """
        if sample_rate is not None and sample_rate != int(sample_rate):
            raise ValueError(
                "For simplicity's sake, only integer sampling rates in Hz are allowed")
        sample_rate = int(sample_rate) if sample_rate is not None else 0
        key: tp.Tuple[int, float] = (sample_rate, highpass)
        if key in self._arrays:
            return self._arrays[key]
        name = f"meg-sr{sample_rate}-hp{highpass}-raw.fif"
        filepath = None if self._cache_folder is None else self._cache_folder / name
        # check is frequency matches raw
        if filepath is None or not filepath.exists():
            raw = self.raw()
            if raw.info["sfreq"] == sample_rate:
                key = (0, highpass)
        if key == (0, 0.0):
            return self.raw()
        if key in self._arrays:
            return self._arrays[key]
        # still not available, so preprocess
        if self._cache_folder is None:
            raise RuntimeError("No cache folder provided for intermediate "
                               f"(subsampled at {sample_rate}Hz) storage.")
        assert filepath is not None
        if not filepath.exists():
            low_mne = preprocess_mne(self.raw(), sample_rate=sample_rate, highpass=highpass)
            low_mne.save(str(filepath), overwrite=True)
            _give_permission(filepath)  # for sharing
        self._arrays[key] = mne.io.read_raw_fif(str(filepath), preload=False)
        self.mne_info  # populate mne info cache.
        return self._arrays[key]

    @staticmethod
    def _read_from_cache(cache_file: Path) -> pd.DataFrame:
        return pd.read_csv(cache_file, index_col=None)

    @staticmethod
    def _write_to_cache(events: pd.DataFrame, cache_file: Path) -> None:
        assert isinstance(events, pd.DataFrame)
        events.to_csv(cache_file, index=False)
        _give_permission(cache_file)

    # pylint: disable=unused-argument
    def events(self, clean: bool = True) -> pd.DataFrame:
        """Loads, caches and returns the events for the subject

        Parameters
        ----------
        clean: bool
            only returns lines which can be cast as an Event type
        """

        if self._events is None:
            if self._cache_folder is None:
                self._events = self._load_events()
            else:
                cache_file = self._cache_folder / "events.csv"
                if cache_file.exists():
                    self._events = self._read_from_cache(cache_file)
                else:
                    self._events = self._load_events()
                    self._write_to_cache(self._events, cache_file)
        events = self._events

        return events

#     def get_super_blocks(self, num_super_blocks: tp.Optional[int] = None,
#                          sentence_blocks: bool = False) -> tp.Dict[int, tp.Tuple[float, float]]:
#         """ Return contiguous super blocks defined as (start, end) meg times tuples, splitting
#         between blocks (e.g. groups of sentences). All super blocks will have roughly the same
#         number of blocks.
#
#         If `sentence_blocks` is True, then blocks are defined as single sentences,
#         using their unique sentence id as key in the output dict. In that case, `num_blocks`
#         is ignored.
#
#         Otherwise, blocks will contain multiple contiguous sentences, and the block key
#         in the output dict will be increasing indexes.
#         """
#         meta = self.events(clean=False)
#         if sentence_blocks:
#             assert num_super_blocks is None
#             block_ids = meta['sequence_uid'].values
#         else:
#             assert num_super_blocks is not None
#             block_ids = meta['block'].values
#
#         boundaries = (block_ids[:-1] != block_ids[1:]).nonzero()[0]
#
#         if sentence_blocks:
#             blocks_in_super_block = 1
#             num_super_blocks = len(boundaries) + 1
#             assert len(boundaries) + 1 == len(np.unique(block_ids))  # type: ignore
#         else:
#             assert num_super_blocks is not None
#             blocks_in_super_block = len(boundaries + 1) // num_super_blocks
#
#         super_blocks = {}
#         assert blocks_in_super_block > 0, "Not enough blocks"
#
#         start = 0.0
#         last_meta_index = 0
#         for idx in range(num_super_blocks - 1):
#             if sentence_blocks:
#                 meta_index = boundaries[idx]
#             else:
#                 # Keeping old behavior, I think the code below is off by 1
#                 # but changing it now would break XP.
#                 meta_index = boundaries[(idx + 1) * blocks_in_super_block]
#             meta_row = meta.iloc[meta_index + 1]
#             end = meta_row["start"]
#             if sentence_blocks:
#                 key = block_ids[meta_index]
#                 assert (block_ids[last_meta_index:meta_index + 1] == key).all()
#             else:
#                 # Old behavior
#                 key = idx
#             super_blocks[key] = (start, end)
#             start = end
#             last_meta_index = meta_index + 1
#         # take any available array to identify final time
#         mne_array = next(iter(self._arrays.values())) if self._arrays else self.raw()
#         if sentence_blocks:
#             super_blocks[block_ids[last_meta_index]] = (start, mne_array.times[-1])
#         else:
#             super_blocks[num_super_blocks - 1] = (start, mne_array.times[-1])
#         assert len(super_blocks) == num_super_blocks, (len(super_blocks), num_super_blocks)
#         return super_blocks


def preprocess_mne(
    raw: mne.io.RawArray,
    sample_rate: int = 200,
    highpass: float = 0,
) -> mne.io.RawArray:
    """Creates a new mne.io.RawArray at another sampling rate
    Parameter:
    raw: mne.io.RawArray
        the mne array to resample
    sample_rate: int
        the required sample rate for the new mne array
    highpass: int
        the frequency of the highpass filter (no high pass filter if 0)
    """
    data = torch.Tensor(raw.get_data())
    old_sr = int(np.round(raw.info["sfreq"]))
    if sample_rate > old_sr:
        raise ValueError("The sample rate should be below "
                         f"{old_sr}Hz, got {sample_rate}")
    resamp = julius.ResampleFrac(old_sr=old_sr, new_sr=sample_rate)
    data = resamp(data)
    if highpass:
        data -= julius.lowpass_filter(data, highpass / sample_rate)

    info_kwargs = dict(raw.info)
    info_kwargs['sfreq'] = sample_rate
    info = mne.Info(**info_kwargs)
    # check that layout works
    layout = mne.find_layout(info)  # noqa
    return mne.io.RawArray(data.numpy(), info=info)


def list_selections() -> tp.List[tp.Tuple[tp.Type[Recording], tp.Dict[str, tp.Any]]]:
    """Lists all the preselections in selections_definitions.yaml

    Returns
    -------
    list
        list of elements (RecordingType, subparameters)
    """
    fp = Path(bm.__file__).parent / "conf" / "selections" / "selections_definitions.yaml"
    assert fp.exists(), f"Unknown file {fp}"
    with fp.open() as f:
        content = yaml.safe_load(f)
    out = []
    for params in content.values():
        study = params.pop("study")
        if study != "fake":
            out.append((register[study], params))
    return out
