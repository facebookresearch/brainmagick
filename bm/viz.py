# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import mne
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mne.set_log_level(False)


EVENT_COLOR_AND_HEIGHT: tp.Dict[str, tp.Tuple[str, float]] = dict(
    sound=('k', 0.1),
    phoneme=('tab:orange', 0.2),
    word=('tab:blue', 0.5),
    multiplewords=('royalblue', 0.5),
    motor=('gray', 0.6),
    block=('tab:red', 0.7)
)


def plot_events(events: pd.DataFrame, window_s: float = 30.0,
                ax: tp.Optional[mpl.axes.Axes] = None, show_desc: bool = True,
                desc_cropping_s: float = 0, desc_fontsize: float = 7, figsize: tuple = (10, 10),
                print_summary: bool = True) -> mpl.axes.Axes:
    """Plot events for visual assessment of alignment.

    Plot events as square waves, with their kind indicated by color and height, and their
    associated word printed at their top. The time axis is wrapped over multiple rows to allow
    visualizing more events into a single plot.

    Parameters
    ----------
    events :
        DataFrame of events for the recording, created by the study class upon loading the data.
    window_s :
        Number of seconds to display on the x-axis, i.e. if events span a longer duration than this
        the figure will be wrapped over multiple rows.
    ax :
        Matplotlib axes to plot into.
    show_desc :
        If True, display event descriptions along with the event lines.
    desc_cropping_s :
        Number of seconds (length of x-axis) after which to crop event descriptions. If 0, defaults
        to `window_s`.
    desc_fontsize :
        Font size of the event descriptions.
    print_summary :
        If True, print count and duration for each event kind.

    Returns
    -------
    Figure and axes into which the events are plotted.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()

    events['stop'] = events.start + events.duration
    if desc_cropping_s == 0:
        desc_cropping_s = window_s

    seen_kinds = set()  # To know when to add to the legend
    y_ticks, y_tick_labels = list(), list()
    view_stop, view_nb = 0., -1
    while view_stop < events.stop.max():
        view_nb += 1
        # Start and stop define the current window to be plotted
        view_start = view_nb * window_s
        view_stop = (view_nb + 1) * window_s

        # Find events that overlap with current window
        in_view = (
            (events.start >= view_start) & (events.start <= view_stop) |  # Contained in window
            (events.start < view_start) & (events.stop >= view_start))  # Starts before window
        events_in_view = events.loc[in_view]
        if events_in_view.empty:
            continue

        y_ticks.append(-view_nb)
        y_tick_labels.append(view_start)

        for kind, d in events_in_view.groupby('kind'):
            if d.empty:
                continue
            color, height = EVENT_COLOR_AND_HEIGHT[kind]
            start, stop = d.start - view_start, d.stop - view_start
            t = np.ravel(np.c_[start, start, stop, stop])
            zeros = np.zeros(len(d))
            ones = np.ones(len(d))
            v = -view_nb + np.ravel(np.c_[zeros, ones, ones, zeros]) * height
            lw = .5 if kind != 'split' else 2.

            ax.plot(t, v, color=color, lw=lw, label='_nolabel_' if kind in seen_kinds else kind)
            seen_kinds.update([kind])

        # Add event description
        if show_desc:
            words = events_in_view.query('kind in ("word", "multiplewords")')
            for kind, ds in words.groupby('kind'):
                if ds.empty:
                    continue
                color, height = EVENT_COLOR_AND_HEIGHT[kind]

                for d in ds.itertuples(index=False):
                    start = d.start - view_start
                    if start > desc_cropping_s:
                        break
                    word = d.word if kind == 'word' else d.words
                    ax.text(start, -view_nb + height - 0.2, word, color=color,
                            fontsize=desc_fontsize, clip_on=True)

    # Tidy up axes
    ax.set_xlim(0, desc_cropping_s if show_desc else window_s)
    ax.set_xlabel('Window offset (s)')
    ax.set_ylabel('Recording offset (s)')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Events')
    ax.legend(loc='lower right')

    fig.tight_layout()

    if print_summary:
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print(events.groupby('kind').duration.agg(['count', 'mean', 'std', 'min', 'max']))

    return fig, ax
