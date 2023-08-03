# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import itertools
from contextlib import contextmanager
import functools
import logging
from pathlib import Path
import os
import time
import socket

import numpy as np


logger = logging.getLogger(__name__)


X = tp.TypeVar("X")


class Frequency(float):
    """A float representing a frequency, with extra helpers to
    help convert from seconds to samples and vice-versa
    """

    @tp.overload
    def to_ind(self, seconds: float) -> int:
        ...

    @tp.overload  # noqa
    def to_ind(self, seconds: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_ind(self, seconds: tp.Any) -> tp.Any:  # noqa
        """Converts a time in seconds (or multiple times in an array)
         to a sample index
         """
        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self).astype(int)
        return int(round(seconds * self))

    @tp.overload
    def to_sec(self, index: int) -> float:
        ...

    @tp.overload  # noqa
    def to_sec(self, index: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_sec(self, index: tp.Any) -> tp.Any:  # noqa
        """Converts a sample index to a time in seconds"""
        return index / self


def timer(prefix):
    current = time.time()

    def _step(name):
        nonlocal current
        now = time.time()
        delta = now - current
        current = now

        print(prefix + name + f": {delta * 1000:.1f}ms")
    return _step


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the **kwargs passed to it in `self._init_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, **kwargs):
        self._init_kwargs = kwargs
        init(self, **kwargs)

    return __init__


class CaptureInit:
    _init_kwargs: dict

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__init__ = capture_init(cls.__init__)  # type: ignore


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


def colorize(text: str, color: str) -> str:
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text: str) -> str:
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


def roundrobin(*iterables: tp.Iterable[X]) -> tp.Iterable[X]:
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis, copied from the itertools documentation
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def flatten_dict(dict_to_flatten: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    """
    Flattens a dict of dicts in the following manner:
    {"test": {"metric1", 0.1, "metric2": 0.2},
     "validation": {"metric3": 0.3}} --->
    {"text/metric1": 0.1, "test/metric2": 0.2, "validation/metric3": 0.3}
    """
    flattened_dict = {}
    for key_name, sub_dict in dict_to_flatten.items():
        items_to_add = {
            f"{key_name}/{key}": val for key, val in sub_dict.items()
        }
        flattened_dict.update(items_to_add)
    return flattened_dict


@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp", pid=False):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


def identify_host() -> str:
    """Identify host, e.g. compute cluster, based on hostname.
    """
    hostname = socket.gethostname()
    if hostname.startswith('devfair') or hostname.startswith('learnfair'):
        name = 'faircluster'
    else:
        name = hostname

    return name
