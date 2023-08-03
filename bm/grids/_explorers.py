# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Explorers for Dora, see Dora's documentation."""
import typing as tp

from dora import Explorer
import treetable as tt


class BMExplorer(Explorer):
    test_metrics: tp.List[str] = []

    def get_grid_metrics(self) -> list:
        """Return the metrics that should be displayed in the tracking table.
        See https://github.com/adefossez/treetable#usage-and-examples for
        more information on how to define the layout.

        Each XP is represented by a dict, that is returned by `process_history()`.
        A `group` represents a sub-dictionary inside that dict. A `leaf`
        represents a value inside the given dict or sub-dict, along with
        a formatting rule. It is possible to specify alignment
        at a group level (applies to all sub groups and leaves) or
        for a single leaf.

        """
        # Given this table layout, the dict returned by `process_history()`
        # should be of the shape
        # {"train": {"epoch": ..., "train": ..., "valid": ..., "best": ...},
        #  "test": {"test_metric_1": ..., ...}}
        return [
            tt.group("train", [
                tt.leaf("epoch"),
                tt.leaf("loss", ".4f"),
            ], align=">"),
            tt.group("valid", [
                tt.leaf("loss", ".4f"),
                tt.leaf("best", ".4f"),
            ], align=">"),
            tt.group("test", [
                tt.leaf(name, ".3f")
                for name in self.test_metrics
             ], align=">")
        ]

    def process_history(self, history: tp.List[dict]) -> dict:
        """Process the history, typically loaded from the
        `history.json` file as a list of dict, one entry per epoch.
        You get a chance to reorganize stuff here, or maybe perform
        some extra processing, and should return a single dict,
        potentially with nested dict.
        """
        stages: tp.Dict[str, tp.Dict[str, tp.Any]] = {
            'train': {'epoch': len(history)}
        }
        best = float('inf')
        for metrics in history:
            for stage_name, stage_metrics in metrics.items():
                if stage_name not in stages:
                    stages[stage_name] = {}
                stages[stage_name].update(stage_metrics)
            best = min(best, stages['valid']['loss'])
            stages['valid']['best'] = best

        return stages


class ClipExplorer(BMExplorer):
    test_metrics: tp.List[str] = ['wer', 'wer_vocab']
