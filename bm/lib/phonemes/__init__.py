# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

ph_dict: dict = {}
dir_path = Path(__file__).parent

with open(dir_path / "phonemes.json", 'r') as f:
    ph_dict = json.load(f)
