# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import flashy
from dora.log import LogProgress
import torch
from torch.utils.data import ConcatDataset, Dataset

from .losses import ClipLoss
from .solver import Solver

logger = logging.getLogger(__name__)


def get_wer(
        solver: Solver,
        dataset: tp.Optional[Dataset] = None):
    solver.model.eval()
    solver.loss.eval()
    test_args = solver.args.test
    if dataset is None:
        datasets = solver.datasets.test.datasets
        if test_args.wer_study is not None:
            datasets = [
                dset for dset in datasets
                if dset.recording.study_name() == test_args.wer_study]
        if test_args.wer_recordings is not None:
            datasets = datasets[:test_args.wer_recordings]
        dataset = ConcatDataset(datasets)
    # we shuffle the loader so that sharding doesn't impact negatives.
    loader = solver.make_loader(dataset, shuffle=True)
    logprog = LogProgress(logger, loader, updates=solver.args.num_prints, name="WER")
    test_features = solver.datasets.test.datasets[0].features
    estimates_list = []
    outputs_list = []
    word_hashes_list = []
    tmin = solver.args.dset.test.tmin
    if tmin is None:
        tmin = solver.args.dset.tmin
    check_at_time = int((-tmin) * solver.args.dset.sample_rate) + 2
    for batch in logprog:
        word_hash = batch.features[:, test_features.get_slice('WordHash')][:, 0]
        features = test_features.extract_features(
            batch.features, solver.used_features.keys())
        with torch.no_grad():
            estimate, output, features_mask, reject_mask = solver._process_batch(
                batch.replace(features=features))
        reject_mask = reject_mask.to(word_hash.device)
        if estimate is not None:
            estimates_list.append(estimate.cpu())
            outputs_list.append(output.cpu())
            wh = word_hash[reject_mask][:, check_at_time]
            if check_at_time > 0:
                wh = torch.where(wh == 0, word_hash[reject_mask][:, check_at_time - 1], wh)
            wh = torch.where(wh == 0, word_hash[reject_mask][:, check_at_time + 1], wh)
            if check_at_time > 1:
                wh = torch.where(wh == 0, word_hash[reject_mask][:, check_at_time - 2], wh)
            wh = torch.where(wh == 0, word_hash[reject_mask][:, check_at_time + 2], wh)
            assert (wh != 0).all()
            word_hashes_list.append(wh)
    estimates = torch.cat(estimates_list, dim=0)
    outputs = torch.cat(outputs_list, dim=0)
    word_hashes = torch.cat(word_hashes_list, dim=0).int()

    if solver.args.test.wer_negatives:
        perm = torch.randperm(len(outputs))
        kept = perm[:solver.args.test.wer_negatives]
        negatives = outputs[kept]
        negative_hashes = word_hashes[kept]
    else:
        negatives = outputs
        negative_hashes = word_hashes
    logger.info("wer %d negatives selected", len(negatives))

    negatives = negatives.to(solver.device)
    negative_hashes = negative_hashes.to(solver.device)
    correct = 0.
    soft_correct = 0.
    correct_vocab = 0.
    clip = solver.loss
    assert isinstance(clip, ClipLoss)
    logprog = LogProgress(
        logger, zip(estimates, word_hashes, outputs),
        name="WER Rank", total=len(outputs), updates=solver.args.num_prints)
    for estimate, word_hash, output in logprog:
        estimate = estimate.to(solver.device)
        negatives[-1] = output.to(solver.device)
        negative_hashes[-1] = word_hash.to(solver.device)
        if solver.args.test.wer_random:
            estimate = torch.randn_like(estimate)

        # Probability distribution over negative samples
        probas = clip.get_probabilities(estimate[None], negatives)[0]

        # Probability distribution over negative vocabulary words
        negative_hashes_vocab, indices = torch.unique(negative_hashes, return_inverse=True)
        probas_vocab = torch.zeros(len(negative_hashes_vocab), dtype=probas.dtype).to(probas.device)
        probas_vocab.scatter_add_(0, indices, probas)

        # Extract Top k
        _, bests = probas.topk(solver.args.test.wer_topx)
        _, bests_vocab = probas_vocab.topk(solver.args.test.wer_topx)

        # Count correct
        correct += (negative_hashes[bests] == word_hash).any().item()
        correct_vocab += (negative_hashes_vocab[bests_vocab] == word_hash).any().item()

        # Soft wer
        right_answers = negative_hashes == word_hash
        soft_correct += probas[right_answers].sum().item()
    correct /= len(estimates)
    correct_vocab /= len(estimates)
    soft_correct /= len(estimates)
    metrics = {'wer': 1 - correct, 'wer_vocab': 1 - correct_vocab}
    return flashy.distrib.average_metrics(metrics)
