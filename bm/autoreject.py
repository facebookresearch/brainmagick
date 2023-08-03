# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Autoreject wrapper, that simply replace bad sensors with the estimate.
"""


# Adapted from https://github.com/hubertjb/dynamic-spatial-filtering/blob/master/transforms.py
# Released under the BSD 3-Clause License
# Here after the original License terms:
# BSD 3-Clause License

# Copyright (c) 2021, InteraXon Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from autoreject import AutoReject
from autoreject.autoreject import _check_data, _apply_interp, _apply_drop


def ensure_valid_positions(epochs):
    """Make sure the EEG channel positions are valid.

    If channels are bipolar and referenced to M1 or M2, rename them to just the
    first derivation so that autoreject can be used.
    """
    ch_names = epochs.info['ch_names']
    if all(['-' not in c for c in ch_names]):  # Only monopolar channels
        pass
    elif all([c.endswith('-M1') or c.endswith('-M2') for c in ch_names]):
        ch_mapping = {c: c.split('-')[0] for c in ch_names}
        epochs.rename_channels(ch_mapping)
        epochs.set_montage('standard_1020')
    else:
        raise ValueError('Bipolar channels are referenced to another channel '
                         'than M1 or M2.')


class AutoRejectDrop(AutoReject):
    """Callable AutoReject with inplace processing and optional epoch dropping.

    See `autoreject.AutoReject`.
    """
    def __init__(self, drop=False, inplace=False, **kwargs):
        super().__init__(**kwargs)
        self.drop = drop
        self.inplace = inplace

    def __getstate__(self):
        """Necessary because the `AutoReject` object implements its own version.
        """
        state = super().__getstate__()
        for param in ['inplace', 'drop']:
            state[param] = getattr(self, param)
        return state

    def __setstate__(self, state):
        """Necessary because the `AutoReject` object implements its own version.
        """
        super().__setstate__(state)
        for param in ['inplace', 'drop']:
            setattr(self, param, state[param])

    def transform(self, epochs, return_log=False):
        """Same as AutoReject.transform(), but with inplace processing and
        optional epoch dropping.
        """
        if not hasattr(self, 'n_interpolate_'):
            raise ValueError('Please run autoreject.fit() method first')

        _check_data(epochs, picks=self.picks_, verbose=self.verbose)

        reject_log = self.get_reject_log(epochs)
        # First difference with the original code:
        epochs_clean = epochs if self.inplace else epochs.copy()
        _apply_interp(reject_log, epochs_clean, self.threshes_,
                      self.picks_, self.dots, self.verbose)

        if self.drop:  # Second difference with the original code
            _apply_drop(reject_log, epochs_clean, self.threshes_, self.picks_,
                        self.verbose)

        if return_log:
            return epochs_clean, reject_log
        else:
            return epochs_clean

    def __call__(self, epochs):
        epochs = self.fit_transform(epochs)
