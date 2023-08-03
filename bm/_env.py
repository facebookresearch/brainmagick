# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: kingjr, 2020

import contextlib
import multiprocessing
import logging
import typing as tp
from pathlib import Path
import yaml

from .utils import identify_host


logger = logging.getLogger(__name__)


class Env:
    """Global environment variable providing study and cache paths if available
    This is called as bm.env
    """

    _instance: tp.Optional["Env"] = None

    def __new__(cls) -> "Env":
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._studies: tp.Dict[str, Path] = self.study_default_paths()
        self.cache: tp.Optional[Path] = None  # cache for precomputation
        # models used to create features (Eg: word embeddings)
        self.feature_models: tp.Optional[Path] = None

        # Hijacking this part of the code as it is one of the first
        # to be executed, so a great place to set the start method.
        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            logger.warning("Could not set start method, cache might not work properly.")

    @staticmethod
    def _get_host_study_paths(all_study_paths) -> tp.Dict[str, str]:
        """Get study paths for the current host.
        """
        hostname = identify_host()
        logger.debug(f'Identified host {hostname}.')
        study_paths = all_study_paths.get(hostname)
        if study_paths is None:  # Use default paths
            logger.warning(
                f'Hostname {hostname} not defined in '
                '/conf/study_paths/study_paths.yaml. Using default paths.')
            study_paths = all_study_paths['default']

        return study_paths

    @classmethod
    def study_default_paths(cls) -> tp.Dict[str, Path]:
        """Fills the study paths with their default value in study_paths.yaml"""
        fp = Path(__file__).parent / "conf" / "study_paths" / "study_paths.yaml"
        assert fp.exists()
        with fp.open() as f:
            content = yaml.safe_load(f)
        logger.debug(content)
        study_paths = cls._get_host_study_paths(content)

        return {x: Path(y) for x, y in study_paths.items() if Path(y).exists()}

    @contextlib.contextmanager
    def temporary_from_args(self, args: tp.Any, wipe_studies: bool = False) -> tp.Iterator[None]:
        """Update cache, features and study paths

        Parameters
        ----------
        wipe_studies: if True, the studies paths currently in the env will be wiped, if False
            only the specified keys will override the current paths.
        """
        kwargs: tp.Dict[str, tp.Any] = dict(studies={} if wipe_studies else self.studies)
        for name, val in args.items():
            if val is not None:
                if name in ('cache', 'feature_models'):
                    kwargs[name] = Path(val)
                elif name == "study_paths" and val is not None:
                    study_paths = self._get_host_study_paths(val)
                    kwargs["studies"].update(
                        {x: Path(y) for x, y in study_paths.items()})
        with self.temporary(**kwargs):
            yield

    @property
    def studies(self) -> tp.Dict[str, Path]:
        return dict(self._studies)

    @studies.setter
    def studies(self, paths: tp.Dict[str, tp.Union[str, Path]]) -> None:
        self._studies = {name: Path(path) for name, path in paths.items()}

    @contextlib.contextmanager
    def temporary(self, **kwargs: tp.Any) -> tp.Iterator[None]:
        """Temporarily replaces a path by the provided one
        for the duration of the "with" context.
        """
        currents: tp.Dict[str, tp.Any] = {}
        for key, val in kwargs.items():
            if isinstance(val, str):
                val = Path(val)
            currents[key] = getattr(self, key)
            setattr(self, key, val)
        try:
            yield
        finally:
            for key, val in currents.items():
                setattr(self, key, val)

    def __repr__(self) -> str:
        vals = {k: x for k, x in self.__dict__.items() if not k.startswith("_")}
        vals["studies"] = self._studies
        string = ",".join(f"{x}={y}" for x, y in sorted(vals.items()))
        return f"{self.__class__.__name__}({string})"


env = Env()
