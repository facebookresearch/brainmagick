# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hiplot explorers, see Dora documentations.
"""
from dora.hiplot import HiPlotExplorer
import hiplot


class MainHP(HiPlotExplorer):
    def process_metrics(self, xp, metrics):
        valid = metrics["valid"]
        train = metrics["train"]
        out = {
            "train_loss": round(train["loss"], 5),
            "valid_loss": round(valid["loss"], 5),
            "best_loss": round(valid["best"], 5),
        }
        if 'test' in metrics:
            test = metrics['test']
            if 'wer_vocab' in test:
                out.update({'wer_vocab': round(100 * test['wer_vocab'], 2)})
        return out

    def postprocess_exp(self, exp: hiplot.Experiment):
        exp.display_data(hiplot.Displays.XY).update({"lines_thickness": 1.0, "lines_opacity": 1.0})
        exp.display_data(hiplot.Displays.XY).update({"axis_x": "epoch", "axis_y": "valid_loss"})
