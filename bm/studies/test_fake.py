# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mne
import pandas as pd
from . import fake


def test_fake_recording() -> None:
    recording = fake.FakeRecording('sub-A2002')
    assert isinstance(recording.events(), pd.DataFrame)
    assert isinstance(recording.raw(), mne.io.RawArray)
