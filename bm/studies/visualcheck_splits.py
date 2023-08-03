# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from time import gmtime, strftime

import git
import mne
import matplotlib.pyplot as plt

import bm
from ..viz import plot_events
from .api import list_selections

mne.set_log_level(False)


def main():
    dsets = list_selections()

    start_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    repo = git.Repo(search_parent_directories=True)
    title = start_time + "_commit-" + repo.head.object.hexsha

    report = mne.Report(title=title)
    report.add_sys_info('sys info')
    report_file = Path(os.getcwd()) / f"diagnosis_events_{title}.html"

    for dset, kwargs in dsets:
        # FIXME get rid off hard-coded cache
        with bm.env.temporary(cache='/checkpoint/jeanremi/bm/cache'):
            n_recordings = 2
            recordings = list(dset.iter(**kwargs))
            for recording in recordings[:n_recordings]:
                events = recording.events()

                # plot old and new boundaries
                window = 300.
                fig_height = (int(events.start.max()/window)+1) * 1.5
                fig, axes = plt.subplots(1, 2, figsize=[15, fig_height])

                plot_events(events, window, ax=axes[0])
                plot_events(events, window, ax=axes[1], crop_text=10.)

                report.add_figs_to_section(
                    fig,
                    f'{recording.study_name()}_{recording.recording_uid}',
                    recording.study_name()
                )
                report.save(report_file, open_browser=False, overwrite=True)

    stop_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    success = f'success: start at {start_time}, ended at {stop_time}'
    report.add_html(success, 'success', tags=('success',))
    report.save(report_file, open_browser=False, overwrite=True)
    print('success')


if __name__ == '__main__':
    main()
