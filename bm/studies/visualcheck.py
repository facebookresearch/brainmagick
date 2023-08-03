# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
For each study, preprocess and epochs recordings, to decode word frequency (for
visual stimuli) or phonetic voicing (for audio), which helps verify whether the
events and the brain signals are well aligned.

This is a slow test.
"""
import os
from time import gmtime, strftime
from pathlib import Path

import git
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wordfreq import zipf_frequency as zipf

from .api import list_selections
import bm


def fast_percentile(x, pct, size=10_000):
    x = x.ravel()
    x = np.random.choice(x, size=size)
    return np.percentile(x, pct)


def get_subjects(dset, kwargs):
    recordings = list(dset.iter(**kwargs))
    subjects_uid = set([r.subject_uid for r in recordings])
    subjects = []
    for subject_uid in subjects_uid:
        subject = []
        for recording in recordings:
            if recording.subject_uid != subject_uid:
                continue
            subject.append(recording)
        subjects.append(subject)
    return subjects


def preprocess(recordings, kind, clip=True):
    epochs = []
    for r in recordings:
        # filter and decimate data
        raw = r.preprocessed(sample_rate=50, highpass=.1)

        # segment data
        sfreq = raw.info['sfreq']
        words = r.events().query(f'kind=="{kind}"')
        events = np.c_[words.start * sfreq,
                       np.ones((len(words), 2))].astype(int)

        epo = mne.Epochs(
            raw,
            events,
            metadata=words,
            preload=True,
            event_repeated='drop'
        )
        epochs.append(epo)
    if len(epochs) > 1:
        epochs = mne.concatenate_epochs(epochs, on_mismatch='ignore')
    else:
        epochs = epochs[0]

    # clip to limit impact of artefacts
    epochs = epochs.apply_baseline()
    if clip:
        data = epochs.get_data().transpose(2, 0, 1)
        threshold = fast_percentile(np.abs(data), 95)
        data = np.clip(data, -threshold, threshold)
        epochs._data[:] = data.transpose(1, 2, 0)
        epochs = epochs.apply_baseline()
    return epochs


def decod(epochs, kind):
    assert kind in ('word', 'phoneme')
    epochs = epochs[f'kind=="{kind}"']
    # meg data
    data = epochs.get_data().transpose(2, 0, 1)

    # word frequency
    if kind == 'word':
        words = epochs.metadata.word.values
        language = epochs.metadata.language.unique()
        assert len(language) == 1
        language = language[0]
        assert isinstance(language, str) and len(language) == 2
        print(language)

        y = np.array([zipf(w, language) for w in words])
    elif kind == 'phoneme':
        # phoneme is voiced
        y = epochs.metadata.phoneme.apply(lambda s: s[0] in 'aeiouy')
        y = y.astype(float)

    y -= y.mean()
    y /= y.std()

    # model
    scoring = make_scorer(lambda yt, yp: pearsonr(yt, yp)[0])
    model = make_pipeline(StandardScaler(), Ridge())
    cv = KFold(5, shuffle=True)

    # decode at each time point
    scores = np.zeros(len(data))
    for t, X in enumerate(data):
        print('.', end='')
        score = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        scores[t] = np.nanmean(score)
    return scores


def main():
    # FIXME avoid hardcoded paths
    with bm.env.temporary(cache='/checkpoint/jeanremi/bm/cache'):
        repo = git.Repo(search_parent_directories=True)
        start_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        title = start_time + "_commit-" + repo.head.object.hexsha

        report = mne.Report(title=title)
        report.add_sys_info('sys info')

        report_file = Path(os.getcwd()) / f"studies_diagnosis_{title}.html"

        dsets = list_selections()

        # Study summary
        for dset, kwargs in dsets:
            subjects = get_subjects(dset, kwargs)
            # name
            dset_name = dset.study_name()
            if len(kwargs):
                dset_name += f'_{kwargs}'

            html = f'name: {dset_name}<br>'
            html += f'paper: <a href={dset.paper_url}>{dset.paper_url}</a><br>'
            html += f'data: <a href={dset.data_url}>{dset.data_url}</a><br>'
            html += f'doi: <a href={dset.doi}>{dset.doi}</a><br>'
            html += f'licence: {dset.licence}<br>'
            html += f'language: {dset.language}<br>'
            html += f'device: {dset.device}<br>'
            html += f'description: {dset.description}<br>'

            # modality
            recordings = [r for s in subjects for r in s]
            modality = list(set([r.modality for r in recordings]))
            modality_text = ','.join(modality)
            html += f'modality: {modality_text}<br>'

            # number of subjects and recordings
            uids = list(set([r.subject_uid for r in recordings]))
            uids_text = ', '.join(uids)
            html += f'subjects (n={len(subjects)}): {uids_text}<br>'
            html += f'recordings: n={len(recordings)}<br>'

            report.add_html(html, dset_name, tags=('summary',))
            report.save(report_file, open_browser=False, overwrite=True)

        # Plot average decoding across subjects
        # run in second loop to be able to inspect quick figures in the report
        # even if the diagnosis is not finished
        for n_subjects, n_recordings in ((1, 1), (10, 4)):
            for dset, kwargs in dsets:
                dset_name = dset.study_name()
                if len(kwargs):
                    dset_name += f'_{kwargs}'

                fig, (ax1, ax2) = plt.subplots(2, 1)

                # only decode 10 subjects to increase speed
                subjects = get_subjects(dset, kwargs)[:n_subjects]
                scores = []
                evokeds = []
                for recordings in subjects:
                    # only take up to 4 recordings to increase speed
                    recordings = recordings[:n_recordings]
                    modality = recordings[0].modality
                    assert modality in ('visual', 'audio')
                    kind = 'word' if modality == 'visual' else 'phoneme'
                    epochs = preprocess(recordings, kind=kind)

                    # aggregate average response
                    evo = epochs.average(method='median')
                    evokeds.append(evo)

                    # compute decoding score
                    score = decod(epochs, kind=kind)
                    ax2.plot(epochs.times, score, color='gray', lw=.5)
                    scores.append(score)
                ax2.fill_between(epochs.times, np.mean(scores, 0), color='r', lw=2)
                ax2.set_title(str(dset))

                # plot evoked response
                evo = mne.EvokedArray(
                    np.mean([e.get_data() for e in evokeds], 0),
                    info=evokeds[0].info,
                    tmin=evokeds[0].times[0]
                )
                evo.plot(spatial_colors=True, axes=ax1, show=False)

                # save in report
                tag = f"{n_subjects}_subjects"
                report.add_figs_to_section(fig, dset_name, tag)
                report.save(report_file, open_browser=False, overwrite=True)

        # add success to explicit that nothing crashed
        stop_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        success = f'success: start at {start_time}, ended at {stop_time}'
        report.add_html(success, 'success', tags=('success',))
        report.save(report_file, open_browser=False, overwrite=True)


if __name__ == "__main__":
    main()
