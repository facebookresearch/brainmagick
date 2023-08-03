# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import defaultdict
from pathlib import Path

import flashy.logging
import flashy.utils
import mne
import numpy as np
import pandas as pd
import torch
from bm import play
from bm.losses import ClipLoss
from bm.train import main
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _get_extra_info(batch, sample_rate):
    """
    Extract word, word_index, sequence_index from event_lists
    """
    # first dim of data is the word index, second dim is hash of word sequence.
    data = torch.ones_like(batch.features)[:, :2].float()*-1
    words = np.empty_like(batch.features[:, 0], dtype="<U30")
    word_segs = []
    assert len(data) == len(batch._event_lists)
    for k, events in enumerate(batch._event_lists):
        segment = ""
        start = events[0].start
        n_times = data.shape[-1]
        # sid = -1
        for event in events:
            if event.kind == "word":
                estart_ind = sample_rate * (event.start - start)
                estop_ind = estart_ind + sample_rate * event.duration
                estart_ind = max(0, int(estart_ind))
                estop_ind = min(n_times, int(estop_ind))
                data[k, 0, estart_ind: estop_ind] = event.word_index
                if event.word_sequence:
                    data[k, 1, estart_ind: estop_ind] = hash(event.word_sequence.encode())
                else:
                    raise RuntimeError("Could not get the word sequence.")
                if estart_ind >= 0 and (estop_ind - estart_ind) > 0:
                    words[k, estart_ind: estop_ind] = event.word
                    segment += (" " + event.word)
        word_segs.append(segment.strip())
    word_segs = np.array(word_segs)
    return data, words, word_segs


def _load_test_data(
        solver, batch_size=100, n_recordings=None, shuffle=True,
        test_study=None):
    """
    Outputs dictionnary containing the test dataset, with keys:
        - word_hashes (tensor)
        - word_indices (long tensor)
        - seq_indices (long tensor)
        - word_strings (numpy array)
        - word_segment_strings (numpy array)
    for each key, the value is of size n_samples
    """
    datasets = solver.datasets.test.datasets
    if test_study is not None:
        print([dset.recording.study_name() for dset in datasets])
        datasets = [
            dset for dset in datasets
            if dset.recording.study_name() == test_study]
    if n_recordings is not None:
        logging.info(
            f"Restrincting WER computation to the first {n_recordings} recordings")
        datasets = datasets[:n_recordings]
    dataset = ConcatDataset(datasets)

    loader = solver.make_loader(dataset, shuffle=shuffle, batch_size=batch_size)
    test_features = solver.datasets.test.datasets[0].features

    outs = defaultdict(list)
    tmin = solver.args.dset.test.tmin
    if tmin is None:
        tmin = solver.args.dset.tmin
    check_at_time = int((-tmin) * solver.args.dset.sample_rate) + 2
    seen_segment_hashes = set()

    logger.info("Extracting test data")
    for k, batch in enumerate(flashy.logging.LogProgressBar(
            logger, loader, updates=20, name='extract')):
        features = test_features.extract_features(
            batch.features, solver.used_features.keys())
        extra_info, word_str, word_segs_str = _get_extra_info(
            batch, solver.args.dset.sample_rate)
        with torch.no_grad():
            preds, trues, features_mask, reject_mask = solver._process_batch(
                batch.replace(features=features))
            if preds is not None:
                # Select the word, seq_id coresponding to the word onset
                if 'WordHash' in test_features:
                    word_hash = batch.features[:, test_features.get_slice(
                        'WordHash')][:, 0]
                else:
                    word_hash = [hash(s.encode().lower())
                                 for s in word_str.reshape(-1)]
                    word_hash = torch.LongTensor(
                        word_hash).reshape(word_str.shape)

                reject_mask = reject_mask.cpu()
                wh = word_hash[reject_mask][:, check_at_time]
                wi = extra_info[reject_mask, 0][:, check_at_time]
                si = extra_info[reject_mask, 1][:, check_at_time]
                ws = word_str[reject_mask][:, check_at_time]
                wseg = word_segs_str[reject_mask]

                if check_at_time > 0:
                    wh = torch.where(wh == 0,
                                     word_hash[reject_mask]
                                     [:, check_at_time - 1],
                                     wh)
                wh = torch.where(wh == 0,
                                 word_hash[reject_mask]
                                 [:, check_at_time + 1],
                                 wh)
                # assert (wh != 0).all()
                assert len(wh) == len(preds) == len(
                    trues) == len(wseg) == len(si)

                # Update
                outs["preds"].append(preds.cpu())
                segment_hashes = [
                    hash(f"{sid}_{wid}".encode())
                    for (sid, wid) in zip(si.cpu().long(), wi.cpu().long().tolist())]
                outs["segment_hashes"].append(torch.tensor(segment_hashes))
                mask = []
                for segment_hash in segment_hashes:
                    if segment_hash in seen_segment_hashes:
                        mask.append(False)
                    else:
                        seen_segment_hashes.add(segment_hash)
                        mask.append(True)
                outs["trues"].append(trues[mask].cpu())
                outs["trues_segment_hashes"].append(outs["segment_hashes"][-1][mask])
                outs["word_hashes"].append(wh.cpu().long())
                outs["word_indices"].append(wi.cpu().long())
                outs["seq_indices"].append(si.cpu().long())
                outs["word_strings"].append(ws)
                outs["word_segment_strings"].append(wseg)
                outs["subject_id"].append(
                    batch.subject_index[reject_mask].cpu().long())
                outs["recording_id"].append(
                    batch.recording_index[reject_mask].cpu().long())
                study = "-".join([k.study_name() for k in batch._recordings])
                subject_uid = "-".join([k.subject_uid for k in
                                        batch._recordings])
                recording_uid = "-".join([k.recording_uid
                                          for k in batch._recordings])
                outs["study"].append(np.array([study]*len(wh)))
                outs["subject_uid"].append(np.array([subject_uid]*len(wh)))
                outs["recording_uid"].append(
                    np.array([recording_uid]*len(wh)))

    del preds
    del trues
    del batch
    del loader

    for k, v in outs.items():
        if k in ["word_strings", "word_segment_strings", "study",
                 "recording_uid", "subject_uid"]:
            outs[k] = np.concatenate(v, 0)
        else:
            outs[k] = torch.cat(v, dim=0)
    return outs


class Evaluator(object):

    def __init__(self, sig, solver=None,
                 shuffle_test_data=False):
        self.sig = sig
        if solver is None:
            self.solver = self.load_solver()
        self.dset_args = self.solver.args.dset
        self.shuffle_test_data = shuffle_test_data
        self.metadata_keys = [
            "segment_hashes",
            "word_hashes",
            "word_indices",
            "seq_indices",
            "word_segment_strings",
            "word_strings",
            "subject_id",
            "subject_uid",
            "recording_uid",
            "recording_id",
            "study"]
        self.probs = None
        self.preds = None
        self.trues = None
        self.trues_segment_hashes = None
        self.word_hashes = None
        self.metadata = None

    def load_solver(self):
        logger.info(f"Loading solver {self.sig}")
        mne.set_log_level(False)
        flashy.logging.setup_logging(with_file_log=False)
        override_cfg = {}
        os.chdir(main.dora.dir.parent)  # Dirty ?
        solver = play.get_solver_from_sig(self.sig, override_cfg=override_cfg)
        solver.model.eval()
        solver.loss.eval()
        return solver

    def load_test_data(
            self, batch_size=100, n_recordings=None,
            test_study=None):
        data = _load_test_data(
            self.solver,
            batch_size=batch_size,
            n_recordings=n_recordings,
            shuffle=self.shuffle_test_data, test_study=test_study)
        self.trues = data["trues"]
        self.trues_segment_hashes = data["trues_segment_hashes"]
        self.preds = data["preds"]
        self.word_hashes = data["word_hashes"]
        self.metadata = {k: data[k] for k in self.metadata_keys}


def _get_accuracy_from_probs(probs, target_labels, vocab_labels, topk=10):
    """
    probs: for each row, the probability distribution over a vocab
    returns the topk accuracy that the topk best predicted labels
    match the target_labels
    Inputs:
        probs: of shape [B, V] probability over vocab, each row sums to 1
        target_labels: of shape [B] true word for each row
        vocab_labels: [V] word that correspond to each column
        topk: int
    Returns: float scalar, topk accuracy
    """
    assert len(target_labels) == len(probs)
    assert len(vocab_labels) == probs.shape[1]

    # Extract topk indices
    idx = probs.topk(topk, dim=1).indices

    # Get the corresponding topk labels
    whs = vocab_labels[idx.view(-1)].reshape(idx.shape)

    # 1 if the labels matches with the targets
    correct = ((whs == target_labels[:, None]).any(1)).float()

    # Average across samples
    acc = correct.mean()

    return acc.item()


def builds_probs(clip,
                 preds,
                 trues,
                 dset_args,
                 batch_size=100,
                 tmin=None,
                 tmax=None):
    """
    Build probability on the distribution of unique word_hasehs (vocab)
    """

    # Trim to tmin, tmax if necessary
    if tmin is not None:
        trim_min = int((tmin - dset_args.tmin)
                       * dset_args.sample_rate)
    else:
        trim_min = None
    if tmax is not None:
        trim_max = int((tmax - dset_args.tmin)
                       * dset_args.sample_rate)
    else:
        trim_max = None
    preds = preds[..., trim_min:trim_max]
    trues = trues[..., trim_min:trim_max]

    # Setup negatives
    candidates = trues.cuda()

    # Loop over samples
    loader = DataLoader(
        TensorDataset(preds, torch.arange(0, len(preds)),),
        batch_size=batch_size)
    probs = torch.zeros(len(preds), len(trues))
    lp = flashy.logging.LogProgressBar(logger, loader, updates=20, name='probs')
    for preds_, idx_ in lp:
        # Compute probabilities
        probs_ = clip.get_probabilities(preds_.cuda(), candidates).cpu()
        # Update
        probs[idx_] = probs_

    return probs


def run_eval(evaluator, output_dir, n_negatives=20_000,
             probs_batch_size=100):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Probs (vocab)
    logger.info("Compute probabilities")
    if evaluator.solver.args.optim.loss != "clip":
        print(
            f"Initialising CLIP LOSS because loss= {evaluator.solver.args.optim.loss}")
        clip = ClipLoss(**evaluator.solver.args.clip,
                        dset_args=evaluator.solver.args.dset)
    else:
        clip = evaluator.solver.loss

    # Solver config
    OmegaConf.save(config=evaluator.solver.args,
                   f=output_dir / "solver_config.yaml")
    logger.info(f"Saving config output to {output_dir}")

    # Probs (segment)
    logger.info("Compute probabilities for segments")
    segment_hashes = evaluator.metadata['segment_hashes']
    vocab_segment = evaluator.trues_segment_hashes
    probs_segment = builds_probs(
        clip,
        evaluator.preds,
        evaluator.trues,
        evaluator.dset_args,
        batch_size=probs_batch_size,
        tmin=None,
        tmax=None)
    logger.info(f"Saving probs output to {output_dir}")
    with flashy.utils.write_and_rename(output_dir / "probs_segment.pth", pid=True) as fo:
        torch.save(probs_segment, fo)
    with flashy.utils.write_and_rename(output_dir / "vocab_segment.pth", pid=True) as fo:
        torch.save(vocab_segment, fo)
    # Metadata
    logger.info(f"Saving metadata output to {output_dir}")
    with flashy.utils.write_and_rename(output_dir / "metadata.csv", pid=True) as fo:
        pd.DataFrame(evaluator.metadata).to_csv(fo)

    # Acc and baseline
    logger.info("Compute accuracies")
    df = []
    for k in (1, 5, 10):
        acc_segment = _get_accuracy_from_probs(
            probs_segment, segment_hashes, vocab_segment, topk=k)
        df.append(dict(
            topk=k,
            # acc_word=acc_word,
            acc_segment=acc_segment,
        ))
        logger.info("Top-%d acc: %.2f", k, 100 * acc_segment)
    df = pd.DataFrame(df).set_index("topk")
    logger.info(f"Saving acc output to {output_dir}")
    with flashy.utils.write_and_rename(output_dir / "acc.csv", pid=True) as fo:
        df.to_csv(fo)

    # Test statistics
    negative_stats = {
        "n_test_samples": len(evaluator.word_hashes),
        "n_test_vocab": len(torch.unique(evaluator.word_hashes)),
        "n_test_segments": len(torch.unique(segment_hashes)),
        "n_neg_samples": len(evaluator.word_hashes[:n_negatives]),
        "n_neg_segments": len(torch.unique(segment_hashes[:n_negatives])),
    }
    for k, v in negative_stats.items():
        logger.info(f"{k}: {v:.0f}")
    logger.info(f"Saving stats output to {output_dir}")
    with flashy.utils.write_and_rename(output_dir / "negative_stats.csv", pid=True) as fo:
        pd.Series(negative_stats).to_csv(fo)


class EvalJob(object):

    def __init__(self, conf, save_dir):
        self.conf = conf
        self.save_dir = Path(save_dir)

    def __call__(self, sig):

        # Init evaluator
        evaluator = Evaluator(str(sig), shuffle_test_data=False)
        logger.info(f"test studies: {evaluator.dset_args.selections}")

        if self.conf.multistudy:
            studies = evaluator.dset_args.selections
        else:
            studies = [None]

        for study in studies:

            if study == "audio_mous":
                study = "schoffelen2019"

            # Setup output path
            save_path = self.save_dir / sig
            if self.conf.multistudy:
                save_path = save_path / study
            if self.conf.n_recordings is not None:
                save_path = save_path / f"nrec_{self.conf.n_recordings}"
            save_path.mkdir(exist_ok=True, parents=True)
            output_file = save_path / "metadata.csv"

            if self.conf.overwrite or not output_file.is_file():

                logger.info(
                    f"Running Evaluation on selected test dataset {study} in {output_file}")

                # Load test data
                evaluator.load_test_data(
                    n_recordings=self.conf.n_recordings,
                    batch_size=self.conf.load_batch_size,
                    test_study=study)

                # Compute and save probs
                run_eval(evaluator, save_path,
                         n_negatives=self.conf.n_negatives,
                         probs_batch_size=self.conf.probs_batch_size)


if __name__ == "__main__":

    logger.setLevel(logging.INFO)

    # ======= Setup config =======

    conf_default = OmegaConf.create(dict(
        sigs=None,
        grid_name=None,
        exclude_sigs=None,
        n_negatives=20000,
        load_batch_size=100,
        probs_batch_size=1000,
        distributed=False,
        n_recordings=None,
        multistudy=False,

        overwrite=False,

        # Slurm params
        slurm_partition="devlab,learnlab",
        slurm_array_parallelism=100,
        gpus_per_node=1,
        cpus_per_task=4,
        timeout_min=60*72,
        slurm_job_name="eval_bm",
        slurm_mem_per_gpu='500GB',
    ))

    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_default, conf_cli)

    # ======= Setup signatures to run and save dir =======

    signatures = list()
    if conf.grid_name is not None:
        assert conf.sigs is None
        grid_dir = main.dora.dir / "grids" / conf.grid_name
        assert grid_dir.exists(), f"{grid_dir} does not exists"
        sigs = [k.parent.stem for k in grid_dir.glob("*/checkpoint.th")]
        signatures += sigs
    if conf.sigs is not None:
        signatures += list(conf.sigs)
    if conf.exclude_sigs is not None:
        signatures = [k for k in signatures if k not in conf.exclude_sigs]

    save_dir = main.dora.dir / "eval" / "signatures"
    signatures_to_run = []
    for sig in signatures:
        if conf.multistudy:
            exists = len(list((save_dir / str(sig)).glob("*/metadata.csv"))) > 0
        else:
            output_file = save_dir / str(sig) / "metadata.csv"
            exists = output_file.is_file()
        if conf.overwrite or not exists:
            signatures_to_run.append(sig)

    print(
        f"Running eval on {len(signatures_to_run)} signatures: {signatures_to_run}\n\
        saving to {save_dir}")

    # ======= Launch job =======
    eval_job = EvalJob(conf, save_dir)

    if conf.distributed:
        from submitit import AutoExecutor
        logger.info("Running with submitit")

        executor = AutoExecutor("submitit_jobs")
        executor.update_parameters(
            slurm_partition=conf.slurm_partition,
            slurm_comment="",
            slurm_array_parallelism=conf.slurm_array_parallelism,
            timeout_min=conf.timeout_min,
            cpus_per_task=conf.cpus_per_task,
            name=conf.slurm_job_name,
            slurm_mem_per_gpu=conf.slurm_mem_per_gpu,
            gpus_per_node=conf.gpus_per_node,
        )

        jobs = executor.map_array(eval_job, signatures_to_run)
        # import pdb
        # pdb.set_trace()

    else:
        for sig in signatures_to_run:
            eval_job(sig)
