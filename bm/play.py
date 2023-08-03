# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Jupyter notebook related utilities.
"""

import logging
import random
import typing as tp

import pandas as pd
import mne
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import hydra
import flashy
from dora import XP
from dora.log import LogProgress

from . import utils
from . import features as _features
from . import dataset as dset
from . import env
from .dataset import SegmentBatch
from .studies import api
from .solver import Solver
from .metrics import TestMetric


logger = logging.getLogger(__name__)


def get_solver_from_xp(xp: XP, override_cfg: tp.Optional[dict] = None):
    """
    Given a XP, return the solver. The best state will be automatically
    loaded. `override_cfg` can be used to overrides some hyper-params,
    while still using the same XP folder and checkpoints.
    """
    # Lazy imports due to weird things with Dora and Hydra.
    from .train import get_solver, override_args_

    logger.info(f"Loading solver from XP {xp.sig}. "
                f"Overrides used: {xp.argv}")
    args = xp.cfg
    override_args_(args)
    if override_cfg is not None:
        args = OmegaConf.merge(args, DictConfig(override_cfg))
    with env.temporary_from_args(args):
        try:
            with xp.enter():
                solver = get_solver(args, training=False)
            solver.model.eval()
            return solver
        finally:
            hydra.core.global_hydra.GlobalHydra.instance().clear()


def get_solver_from_sig(sig: str, override_cfg: tp.Optional[dict] = None):
    """
    Same as `get_solver_from_xp`, but providing only the XP signature.
    """
    # Lazy imports due to weird things with Dora and Hydra.
    from .train import main
    xp = main.get_xp_from_sig(sig)
    return get_solver_from_xp(xp, override_cfg=override_cfg)


def get_solver_from_args(args: tp.List = [], override_cfg: tp.Optional[dict] = None):
    """
    Same as `get_solver_from_xp`, but providing only command line args used.

    ..Warning:: `args` is used to determine the XP signature and checkpoints.
        On the other hand, override_cfg is applied at a later stage, in order
        to overrride some configs without changing the XP folder and checkpoints.
    """
    # Lazy imports due to weird things with Dora and Hydra.
    from .train import main
    xp = main.get_xp(args)
    main.init_xp(xp)
    return get_solver_from_xp(xp, override_cfg)


def get_test_metrics(
        solver: Solver,
        trim_offset: int = 0,
        metrics_constructor: tp.Optional[tp.List[tp.Callable[..., TestMetric]]] = None,
        reduce: bool = True,
        datasets: tp.Optional[tp.List[dset.SegmentDataset]] = None):
    """
    Given a solver, which you can obtain with `get_solver_from_sig` for instance,
    compute test metrics bewteen the estimated and ground truth output, per recording,
    across epochs (i.e. dataset entries).

    Args:
        solver (Solver): solver to use for evaluation.
        trim_offset (int): when evaluating correlation, remove that many
            samples first. This avoid evaluating a model too close to the initial MEG.
        metrics_constructor: a list of test metrics to be calculated on the test dataset. Creates
            the metrics internally from solver if None value is given
        reduce (int): whether to return the final aggregated metric result, or an intermediary value
            (usually used for debugging and explainability).
        datasets (None or List): list of recording datasets to calculate metrics on. Use given
            solve test datasets if None is given

    Shape:
        - Output: `[S, C, T]` with `S` the number of recordings, `C` the output channels,
            and `T` the number of timesteps.

    """
    test_datasets = datasets or solver.datasets.test.datasets
    # Compute correlation recording by recording
    # when doing multi modal task, some gpus can be underused because they
    # will always get tasks of the same type (e.g. visual is faster than audio).
    # so we randomize datasets order before sharding.
    dataset_order = list(range(len(test_datasets)))
    random.shuffle(dataset_order)
    rank = flashy.distrib.rank()
    world_size = flashy.distrib.world_size()
    these_datasets = [test_datasets[i] for i in dataset_order[rank::world_size]]
    logprog = LogProgress(logger, these_datasets,
                          updates=solver.args.num_prints, name='Correlations')
    if metrics_constructor is None:
        metrics_constructor = solver.get_metric_constructors()
    test_metrics: tp.Dict[str, tp.List[float]] = {
        metric_constructor().name: []
        for metric_constructor in metrics_constructor
    }
    for recording_index, recording in enumerate(logprog):
        loader = DataLoader(recording, batch_size=solver.args.optim.batch_size,
                            num_workers=solver.args.num_workers, collate_fn=SegmentBatch.collate_fn)
        metrics = []
        for metric_constructor in metrics_constructor:
            metrics.append(metric_constructor())

        for batch in loader:
            batch = batch.to(solver.device)
            with torch.no_grad():
                estimate, gt, features_mask, _ = solver._process_batch(batch)
            if estimate is None:
                continue
            estimate = estimate[..., trim_offset:]
            gt = gt[..., trim_offset:]
            features_mask = features_mask[..., trim_offset:]
            dtype = torch.cdouble if estimate.dtype.is_complex else torch.double
            for metric in metrics:
                metric.update(estimate.to(dtype), gt.to(dtype), features_mask)

        for metric in metrics:
            test_metrics[metric.name].append(metric.get().cpu().float())

    all_correlations = {
        metric_name: [None] * len(test_datasets) for metric_name in test_metrics
    }

    for rank in range(world_size):
        for metric_name in all_correlations:
            shared = flashy.distrib.broadcast_object(test_metrics[metric_name], src=rank)
            for dset_index, result in zip(dataset_order[rank::world_size], shared):
                all_correlations[metric_name][dset_index] = result
    for results in all_correlations.values():
        assert all(x is not None for x in results)

    for metric_constructor in metrics_constructor:
        metric = metric_constructor()
        metric_name = metric.name
        if reduce:
            all_correlations[metric_name] = metric.reduce(all_correlations[metric_name])
        else:
            all_correlations[metric_name] = torch.stack(all_correlations[metric_name])
    return all_correlations


class SentenceFeatures:
    """Creates features from a sequence of words.

    Parameters
    ----------
    features: list
        list of features to generate
    sample_rate: float
        sample rate of the features to generate
    highpass: float
        highpass filter to use for basal state extraction
    task: str
        name of the task (for the features)
    additional_time: float
        time in seconds to append at the end of the features
        (where nothing happens)

    Note
    ----
    The features will be generated with a heuristic:
    - first word starts at 1s
    - as a default, each word has a duration of 0.1s per letter, with a maximum of 1.0s
    - as a default, time between words lasts 0.3s
    - 1s is added at the end
    You can use the `generate` method for more flexibility (providing words and their durations)

    Example
    -------
    >>> builder = play.SentenceFeatures([WordFrequency", "WordLength"], sample_rate=20)
    >>> sentence = builder("de kat slaapt in de woonkamer")
    >>> assert sentence.shape == (2, 17)
    """

    @classmethod
    def from_solver(cls, solver: tp.Any, **kwargs) -> "SentenceFeatures":
        dst = solver.args.dset
        features_params = {}
        if hasattr(dst, 'features_params') and dst.features_params is not None:
            features_params = dict(dst.features_params)
        return cls(dst.features, features_params, sample_rate=dst.sample_rate,
                   highpass=dst.highpass, **kwargs)

    def __init__(self, features: tp.List[str], features_params: dict,
                 sample_rate: float, highpass: float = 0.0,
                 modality: str = "visual", additional_time: float = 1.0) -> None:
        self._highpass = highpass
        self._sample_rate = utils.Frequency(sample_rate)
        self._features = features
        self._features_params = features_params
        self._modality = modality
        self._additional_time = additional_time

    def _generate_events(
        self, word_durations: tp.List[tp.Tuple[str, float]], interword: float = 0.3
    ) -> pd.DataFrame:
        """Generate features based on word durations.

        Parameters
        ----------
        word_durations: list of tuples
            list of tuples of type (word, duration)
        interword: float
            duration between two words
        """
        time = 1.0
        events: tp.List[dict] = []
        sentence = " ".join(x[0] for x in word_durations)
        for k, (word, duration) in enumerate(word_durations):
            events.append(dict(
                kind='word', word=word,  sequence_uid=12, modality=self._modality, start=time,
                duration=duration, word_index=k, word_sequence=sentence, language='nl'))
            time += duration + interword
        events = pd.DataFrame(events).event.validate()
        return events

    def generate(
            self, word_durations: tp.List[tp.Tuple[str, float]], interword: float = 0.3
    ) -> torch.Tensor:
        events = self._generate_events(word_durations=word_durations, interword=interword)
        return self._generate_from_events(events)

    def _generate_from_events(self, events: pd.DataFrame) -> torch.Tensor:
        last = events.iloc[-1]
        duration = last.start + last.duration + self._additional_time  # assuming ordered
        builder = _features.FeaturesBuilder(
            events, self._features, features_params=self._features_params,
            sample_rate=self._sample_rate)
        return builder(0, duration)[0]

    def __call__(self, sentence: str) -> torch.Tensor:
        word_durations = [(word, max(0.3, min(0.8, 0.1 * len(word))))
                          for word in sentence.strip().split()]
        return self.generate(word_durations)

    def extract_basal_states(self, recording: api.Recording, duration: float = 0.5) -> mne.Epochs:
        """Extract an mne.Epochs instance of basal state.
        Basal state are selected as chuncks of the provided duration,
        just before the occurence of the first word of a word sequence/sentence

        Parameters
        ----------
        recording: Subject
            the recording to extract basal state from
        duration: float
            duration of the basal state to extract

        Note
        ----
        This assumes default baseline removal
        """
        query = "kind=='word' & word_index==0"
        fact = dset.SegmentDataset.Factory(
            condition=query, tmin=-duration, tmax=0.0,
            highpass=self._highpass, sample_rate=self._sample_rate
        )
        ds = fact.apply(recording)
        assert ds is not None
        return ds.epochs


def predict(solver: tp.Any, features: torch.Tensor,
            subject_index: tp.Optional[int] = None,
            meg_init: bool = False) -> mne.EvokedArray:
    """Predict on a given subject (or the average if subject_index=None) from a
    solver and an feature.

    Note: this API should be updated, since it's not very convenient.
    """
    dst = solver.args.dset
    selections = [solver.args.selections[x] for x in dst.selections]
    recordings = dset._extract_recordings(selections, n_recordings=dst.n_recordings)
    indices = list(range(dst.n_recordings)) if subject_index is None else [subject_index]
    recordings = [recordings[k] for k in indices]
    outs: tp.List[tp.Any] = []
    base = 0 * features
    for recording in recordings:
        meg = torch.zeros([273, features.shape[1]])
        if meg_init:
            builder = SentenceFeatures.from_solver(solver)
            meg_inits = builder.extract_basal_states(
                duration=solver.args.task.meg_init, recording=recording)
            meg_chunck = torch.from_numpy(next(meg_inits[2])).float()
            meg[:, :meg_chunck.shape[1]] = meg_chunck
        predictions = [
            solver.predict(features=f, meg=meg, subject_index=recording.subject_index)
            for f in (features, base)]
        outs.append(predictions[1] - predictions[0])
    data = (sum(outs) / len(outs)).cpu().detach().numpy()  # type: ignore
    info = solver.datasets.test.datasets[0].epochs.info
    return mne.EvokedArray(data, info=info, tmin=-1.)
