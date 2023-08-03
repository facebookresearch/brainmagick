# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hashlib import sha1
import logging
import os
from pathlib import Path
import typing as tp
import sys

from dora import hydra_main, to_absolute_path
import flashy
import hydra
import mne
from omegaconf import OmegaConf
import torch

from . import dataset as dset
from .models import ConvRNN, SimpleConv, DeepMel
from .solver import Solver

logger = logging.getLogger(__name__)


def model_hash(model: torch.nn.Module) -> str:
    hasher = sha1()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def get_solver(args: tp.Any, training=True):
    # Dataset and loading
    assert args.optim.batch_size % flashy.distrib.world_size() == 0
    logger.debug("Building datasets")
    args.optim.batch_size //= flashy.distrib.world_size()

    if args.num_workers is None:
        args.num_workers = min(10, 2 * args.optim.batch_size)

    assert args.dset.sample_rate is not None, "sample rate <= 1200 required"
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    dsets = dset.get_datasets(
        num_workers=args.num_workers, progress=True,
        **kwargs,
    )
    if args.download_only:
        sys.exit(0)

    meg_dimension = dsets.train[0].meg.shape[0]
    used_features = dsets.train.datasets[0].features
    if args.task.type == 'decode':
        in_channels = dict(meg=meg_dimension)
        chout = used_features.output_dimension

    elif args.task.type == 'encode':
        in_channels = dict(meg=meg_dimension, features=used_features.dimension)
        chout = in_channels["meg"]

    # Brain module creation
    if args.override_n_subjects_model is not None:
        n_subjects = args.override_n_subjects_model
    else:
        n_subjects = 1 + max(dset.recording.subject_index for dset in dsets.train.datasets)

    assert n_subjects > 0
    torch.manual_seed(args.seed)
    model_chout = chout
    if args.feature_model_name is not None:
        if args.task.type == "decode":
            model_chout = args.feature_model_params.n_out_channels
    if args.model_name == "convrnn":
        model = ConvRNN(in_channels=in_channels, out_channels=model_chout,
                        n_subjects=n_subjects, **args.convrnn)
    elif args.model_name == "simpleconv":
        model = SimpleConv(in_channels=in_channels, out_channels=model_chout,
                           n_subjects=n_subjects, **args.simpleconv)
    else:
        raise ValueError(f"Invalid model {args.model}")
    model.to(args.device)

    # Feature model creation
    if args.feature_model_name is not None:
        feature_model_params = args.feature_model_params
        if args.feature_model_name == "deep_mel":
            feature_model = DeepMel(**feature_model_params, n_in_channels=chout)
        else:
            raise ValueError(f"Invalid feature model {args.feature_model_name}.")

        if "device" in feature_model_params and feature_model_params.device:
            # Option to have the brain and speech modules on different GPUs
            feature_model.to(feature_model_params.device)
        else:
            feature_model.to(args.device)
        logger.debug("Feature model: %r", feature_model)
        logger.info("Feature model hash: %s", model_hash(feature_model))
    else:
        feature_model = None
    logger.debug('Model: %r', model)
    logger.info('Model hash: %s', model_hash(model))

    optimizer = None
    if training:
        # optimizer
        optargs = args.optim
        params = list(model.parameters())
        if feature_model is not None:
            params += list(feature_model.parameters())
        if optargs.name == "adam":
            optimizer = torch.optim.Adam(params, lr=optargs.lr, betas=(0.9, optargs.beta2))
        else:
            raise ValueError(f'Invalid optimizer {args.optim}')

    return Solver(
        args=args,
        datasets=dsets,
        model=model,
        feature_model=feature_model,
        optimizer=optimizer)


def run(args: tp.Any) -> float:
    level = logging.DEBUG if args.verbose else logging.INFO
    flashy.setup_logging(level=level)
    mne.set_log_level(False)
    if args.verbose:
        logging.getLogger("bm").setLevel(logging.DEBUG)
        logging.getLogger("dora").setLevel(logging.DEBUG)

    flashy.distrib.init()

    solver = get_solver(args)

    # Construct Solver
    if args.show:
        logger.info(solver.model)
        mb = sum(p.numel() for p in solver.model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        return 0.0

    return solver.train()


def override_args_(args: tp.Any):
    """Override hydra config with simple code logic that can't be described in a config.yaml file.

    TODO: For every hydra param we change, we should set a new param in the config (so original
    param will stay the same). This provides better backward compatibility.
    """
    for selection in args.selections:
        events_filter = getattr(selection, "events_filter", None)
        if events_filter is not None:
            # TODO next line is BUGGY (and untested)
            event_filter_key_to_value = dict(
                hydra.compose(config_name=args.events_filter_file))  # type: ignore
            if events_filter in event_filter_key_to_value:  # should not happen by default with None
                selection.events_filter = event_filter_key_to_value[events_filter]
    if args.cache is not None:
        args.cache = to_absolute_path(args.cache)


@hydra_main(config_name="config", config_path="conf", version_base="1.1")
def main(args: tp.Any) -> float:
    override_args_(args)

    global __file__  # pylint: disable=global-statement,redefined-builtin
    # Fix bug when using multiprocessing with Hydra
    __file__ = hydra.utils.to_absolute_path(__file__)

    from . import env  # we need this here otherwise submitit pickle does crazy stuff.
    # Updating paths in config that should stay relative to the original working dir
    with env.temporary_from_args(args):
        torch.set_num_threads(1)
        logger.info(f"For logs, checkpoints and samples, check {os.getcwd()}.")
        logger.info(f"Caching intermediate data under {args.cache}.")
        logger.debug(args)
        return run(args)


if '_BM_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_BM_TEST_PATH'])

if __name__ == "__main__":
    main()
