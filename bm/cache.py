# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Caching utility."""
import hashlib
import json
import logging
from pathlib import Path
import typing as tp

import numpy as np
from omegaconf.basecontainer import BaseContainer
from omegaconf import OmegaConf
import torch

from ._env import env
from .utils import write_and_rename


logger = logging.getLogger(__name__)


def jsonable(value):
    if isinstance(value, dict):
        lst = [(jsonable(k), jsonable(v)) for k, v in value.items()]
        lst.sort()
        return dict(lst)
    elif isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    elif isinstance(value, Path):
        return str(value)
    elif value is None or isinstance(value, (int, float, str, bool)):
        return value
    elif isinstance(value, BaseContainer):
        return OmegaConf.to_container(value, resolve=True)
    else:
        raise ValueError(f"{repr(value)} is not jsonable.")


def _get_signature(value):
    value = jsonable(value)
    return hashlib.sha1(json.dumps(value).encode()).hexdigest()[:16]


class Cache:
    def __init__(self, name: str, args: tp.Any = None, *, mode: str = "torch"):
        """
        Caching mechanism, will automatically save content if a cache path
        is available in a subfolder `name`. `args` should be any arguments
        that will be common to all keys stored. For safety reasons, it should
        always be jsonable, as pickle might end up pickling various things
        that will change across runs and is harder to debug.
        """
        self._suffix = {"torch": ".pkl", "memmap": ".npy"}[mode]
        if env.cache is None:
            self.path = None
        else:
            args_sig = _get_signature(args)
            self.path = env.cache / name / args_sig
            self.path.mkdir(exist_ok=True, parents=True)

    def cache_path(self, key: tp.Any) -> tp.Optional[Path]:
        if self.path is None:
            return None
        name = _get_signature(key)
        return self.path / (name + self._suffix)

    def get(self, _computation, **kwargs) -> tp.Any:
        path = self.cache_path(kwargs)
        if path is not None and path.exists():
            try:
                if self._suffix == ".pkl":
                    return torch.load(path)
                else:
                    return np.lib.format.open_memmap(path)
            except OSError as error:
                logger.warning("Error while loading cache file: %r", error)
        result = _computation(**kwargs)
        if path is not None:
            with write_and_rename(path, pid=True) as tmp:
                if self._suffix == ".pkl":
                    torch.save(result, tmp)
                else:
                    assert isinstance(result, np.ndarray), "Only np.ndarrays are allowed"
                    np.save(tmp, result)
        return result


class MemoryCache:
    """Same as Cache but in memory, used for sharing a model between multiple
    instances of features for instance.
    """
    _CACHE: tp.Dict[str, tp.Dict[str, tp.Dict[str, tp.Any]]] = {}

    def __init__(self, name: str, args: tp.Any = None):
        self.args_sig = _get_signature(args)
        self.name = name
        self._CACHE.setdefault(name, {}).setdefault(self.args_sig, {})

    @property
    def _cache_dict(self):
        return self._CACHE[self.name][self.args_sig]

    def cache_key(self, key: tp.Any) -> str:
        return _get_signature((self.args_sig, key))

    def get(self, _computation, *args, **kwargs) -> tp.Any:
        key = self.cache_key((args, kwargs))
        if key in self._cache_dict:
            return self._cache_dict[key]
        else:
            value = _computation(*args, **kwargs)
            self._cache_dict[key] = value
            return value
