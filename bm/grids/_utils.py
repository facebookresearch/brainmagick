# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass
import itertools
import types
from pathlib import Path
import inspect
from numbers import Number
import logging

logger = logging.getLogger(__name__)


class _GridParamGroup:
    """
    Keeps pairs of parameter and possible values and functionality to be used by the
    SimpleGridSearcher convenient class.
    """
    @dataclass
    class GridParam:
        cfg_key: str
        values: tp.List

    def __init__(self, args_dict: tp.Dict[str, tp.Union[tp.List, str, Number]]):
        self.grid_params: tp.List[_GridParamGroup.GridParam] = []

        for cfg_key, values in args_dict.items():
            if not isinstance(values, list):
                values = [values]
            grid_param = _GridParamGroup.GridParam(cfg_key, values)
            self.grid_params.append(grid_param)

            assert len(self.grid_params[0].values) == len(self.grid_params[-1].values), \
                "Params defined must have same number of value options" + str(self.grid_params)

    @property
    def param_len(self):
        return len(self.grid_params[0].values)

    def get_params_dict(self, idx):
        assert idx < self.param_len
        params_dict = {}
        for grid_param in self.grid_params:
            params_dict[grid_param.cfg_key] = grid_param.values[idx]
        return params_dict


class SimpleGridSearcher:
    """
    A convenient class that allows to define a set of parameters and possible values, and performs
    a naive grid search over the params.

    Usage:
      - After creating an instance, define grid search parameters using define_grid_param() and
        providing a dict of param names and their possible values.
      - Call grid_search() to launch the grid experiments. The code will iterate over all possible
        combinations of parameters defined previously and launch a relevant experiment.

    NOTE: parameters defined at the same define_grid_param() call will run without grid search
          of all combinations among them.
    NOTE: providing a non-list value will create a list of 1 parameter internally for the search
    NOTE: providing None as possible value will delete the parameter from the launch flags.

    Example:
        searcher = SimpleGridSearcher()
        searcher.define_grid_param({"together1": [1,2], "together2": [0.1, 0.2]})
        searcher.define_grid_param({"alone": ["one", None]})
        searcher.define_grid_param({"fixed": "constant1"})
        searcher.grid_search(launcher)

    Exps running are: {"together1", 1, "together2": 0.1, "alone": "one", "fixed": "constant1"}
                      {"together1", 2, "together2": 0.2,                 "fixed": "constant1"}
                      {"together1", 1, "together2": 0.1, "alone": "one", "fixed": "constant1"}
                      {"together1", 2, "together2": 0.2,                 "fixed": "constant1"}
    """
    def __init__(self) -> None:
        self.all_params: tp.List[_GridParamGroup] = []

    @staticmethod
    def _remove_dict_none_vals(exp_params_dict):
        return {k: v for k, v in exp_params_dict.items() if v is not None}

    def define_grid_param(self, args_dict: tp.Dict[str, tp.Union[tp.List, str, Number]]):
        self.all_params.append(_GridParamGroup(args_dict))

    def grid_search(self, launcher):
        all_grid_params_len = [list(range(param.param_len)) for param in self.all_params]

        for exp_params_indices in itertools.product(*all_grid_params_len):
            exp_params_dict = {}
            for param_idx, permutation_idx in enumerate(exp_params_indices):
                param_dict = self.all_params[param_idx].get_params_dict(permutation_idx)
                assert not any(key in exp_params_dict.keys() for key in param_dict.keys()), \
                    f"Key redefined at {param_dict.keys()}"
                exp_params_dict.update(param_dict)

            exp_params_dict = SimpleGridSearcher._remove_dict_none_vals(exp_params_dict)
            sub = launcher.bind(exp_params_dict)

            # Hack to use sub__call__() and not sub() as this method is patched when called
            # from augmentation_decoders and it doesn't work with explicit sub() call.
            sub.__call__()


def get_all_explorer_sigs(explorer, launcher) -> tp.List[str]:
    """
    Hack to return a list of signature strings for all experiments from a given explorer method.
    Using function
    """
    xp_sigs = []

    def launcher_call_patch(self, *args, **kwargs):
        launcher2 = self.bind(*args, **kwargs)
        sheep = self._shepherd.get_sheep_from_argv(launcher2._argv)
        if sheep.state() != "COMPLETED":
            logger.warning(f"Returning XP that is not completed. State={sheep.state()}.")
        xp_sigs.append(str(sheep.xp.folder.name))

    def _copy_patch(self, *args, **kwargs):
        new_launcher = self.launcher_copy(*args, **kwargs)
        new_launcher.__call__ = types.MethodType(launcher_call_patch, new_launcher)
        new_launcher.launcher_copy = new_launcher._copy
        new_launcher._copy = types.MethodType(_copy_patch, new_launcher)
        return new_launcher

    # Hack, use a tmp launcher with patched __call__ to extract encoder XPs
    tmp_launcher = launcher.bind()
    if tmp_launcher._copy.__name__ != "_copy_patch":
        tmp_launcher.launcher_copy = tmp_launcher._copy

    tmp_launcher.__call__ = types.MethodType(launcher_call_patch, tmp_launcher)
    tmp_launcher._copy = types.MethodType(_copy_patch, tmp_launcher)

    # Call the encoders grid with hacked launcher to get encoder XP sigs
    explorer(tmp_launcher)

    return xp_sigs


def get_dummy_version(version_num):
    """
    Returns a unique string composed of the calling grid file name and a version to be set as
    a dummy flag for an experiment grid. Assuming the caller is a grid file.
    """
    frame = inspect.stack()[1]
    filename = frame[0].f_code.co_filename
    return Path(filename).stem + f"-v{version_num}"
