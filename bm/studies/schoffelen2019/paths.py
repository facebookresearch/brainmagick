# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import typing as tp
from dora import to_absolute_path
from bm import env


TEST_FILENAME = "testmeg-raw.fif"  # for test only


class StudyPaths:

    def __init__(self, subject_uid: str) -> None:
        self._subject_uid = subject_uid
        self._raw: tp.Optional[Path] = None
        self._metadata: tp.Optional[Path] = None

    @classmethod
    def create(cls, subject_uid: str) -> "StudyPaths":
        # used for mocking
        return cls(subject_uid)

    def is_valid(self) -> bool:
        try:
            # pylint: disable=pointless-statement
            self.raw
            self.metadata
        except RuntimeError:
            return False
        return True

    @property
    def raw(self) -> Path:
        if self._raw is None:
            # list meg folders
            meg_folder = self.dataset() / self._subject_uid / "meg"
            meg_files = list(meg_folder.glob("*.ds"))
            meg_files = [x for x in meg_files if "rest" not in x.name]
            test_file = meg_folder / TEST_FILENAME
            if test_file.exists():
                # hack for testing: if we find the test file instead, we use it!
                meg_files = [test_file]
            if not meg_files:
                if not meg_folder.exists():
                    raise RuntimeError(
                        f"No MEG folder for recording {self._subject_uid} at path\n{meg_folder}")
                raise RuntimeError(f"No MEG file for recording {self._subject_uid}")
            self._raw = meg_files[-1]
        return self._raw

    @property
    def metadata(self) -> Path:
        if self._metadata is None:
            # list meg folders
            metadata_folder = self.dataset() / "sourcedata" / "meg_task"
            search_string = f"*{self._subject_uid.replace('sub-', '')}*"
            metadata_files = list(metadata_folder.glob(search_string))
            if not metadata_files:
                raise RuntimeError(f"No metadata file for recording {self._subject_uid}")
            self._metadata = metadata_files[-1]
        return self._metadata

    # USER INDEPENDENT PATHS OF THE DATASET

    @staticmethod
    def dataset() -> Path:
        # used for mocking
        path = to_absolute_path(Path(env.studies["schoffelen2019"]))  # TODO: move logic here
        dl = path / 'download'
        if dl.exists():
            return dl
        return path

    @staticmethod
    def wave_file(name: str) -> Path:
        # eg name: 186.wav
        if name.startswith('/'):
            return Path(name)  # already a full absolute path  (useful for tests)
        return StudyPaths.dataset() / "stimuli" / "audio_files" / f"EQ_Ramp_Int2_Int1LPF{name}"

    @staticmethod
    def stimuli_file() -> Path:
        return StudyPaths.dataset() / 'stimuli' / 'stimuli.txt'

    @staticmethod
    def phoneme_file(sequ_id: int) -> Path:
        return (StudyPaths.dataset() / "derivatives" / "phonemes" /
                ("EQ_Ramp_Int2_Int1LPF%.3i.TextGrid" % sequ_id))
