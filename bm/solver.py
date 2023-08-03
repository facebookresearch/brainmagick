# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from functools import partial

import flashy
import julius
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cache import Cache
from .dataset import SegmentBatch
from .losses import ClipLoss, FeatureDecodingLoss, L1Loss, L2Loss
from .metrics import ClassificationAcc, L2Reg, OnlineCorrelation
from .norm import BatchScaler, ScaleReject
from .svd import svd_penalty
from .utils import bold, copy_state, swap_state

logger = logging.getLogger(__name__)


class Solver(flashy.BaseSolver):
    def __init__(self, args, datasets, model, feature_model=None, optimizer=None) -> None:
        super().__init__()
        self.args = args
        self.device: str = args.device
        self.datasets = datasets
        self.used_features = datasets.train.datasets[0].features
        self.model = model
        self.feature_model = feature_model
        self.all_models = nn.ModuleList([self.model, self.feature_model])  # Simplify checkpointing
        self.optimizer = optimizer
        self.best_state: tp.Optional[dict] = None
        self.best_state_loss: tp.Optional[dict] = None
        self.loss = self._create_loss(args.optim.loss).to(self.device)

        # Scalers
        self.scaler: tp.Optional[BatchScaler] = None
        self.scale_reject: tp.Optional[ScaleReject] = None
        self.scaler_cache = Cache("scaler", (args.dset, args.norm))

        self.last_test_epoch = 0
        self.best_epoch = 0
        self.best_loss = float('inf')
        self.register_stateful(
            'all_models', 'optimizer', 'best_state', 'scaler', 'loss', 'last_test_epoch',
            'best_epoch', 'best_loss')
        self.init_tensorboard()
        if self.args.wandb.use_wandb:
            wandb_kwargs: tp.Dict[str, tp.Any] = dict(self.args.wandb)
            wandb_kwargs.pop('use_wandb')
            self.init_wandb(**wandb_kwargs)

        # Load back from the checkpoint
        self.restore()
        if optimizer is None and self.best_state is not None:
            self.all_models.load_state_dict(self.best_state)

        if datasets is not None:
            self._init_loader()
            self._init_scaler()
            self.scale_reject = ScaleReject(
                self.scaler, self.args.norm.max_scale,
                self.args.norm.exclude_empty_features,
                self.args.norm.clip)

        self.negative_pool = self._make_negative_pool()

    def _create_loss(self, loss: str):
        if loss == 'l1':
            return L1Loss()
        elif loss == 'mse':
            return L2Loss()
        elif loss == 'regression_classification':
            return FeatureDecodingLoss(
                self.used_features,
                self.scaler if self.args.optim.use_weighting else None)
        elif loss == 'clip':
            kw = dict(self.args.clip)
            kw.pop('save_best', None)
            kw.pop('sync_grad', None)
            loss = ClipLoss(**kw, dset_args=self.args.dset)
            if self.optimizer is not None:
                self.optimizer.add_param_group({"params": loss.parameters()})
            return loss
        else:
            raise ValueError(f"Unsupported loss {loss}")

    def _init_scaler(self) -> None:
        if self.scaler is None:
            if flashy.distrib.is_rank_zero():
                self.scaler = self.scaler_cache.get(self._fit_scaler)
                path = self.scaler_cache.cache_path({})
                logger.info("Scaler cache file %s", path)
            self.scaler = flashy.distrib.broadcast_object(self.scaler)

    def restore(self) -> bool:
        """
        Resume from checkpoint
        """
        if self.checkpoint_path.exists():
            return super().restore()
        elif self.args.continue_sig:
            path = self.folder.parent / self.args.continue_sig / self.checkpoint_path.name
            assert path.exists(), "Could not find checkpoint " + str(path)
            state = torch.load(path, 'cpu')
            if self.args.continue_best:
                self.all_models.load_state_dict(state['best_state'])
            else:
                self.all_models.load_state_dict(state['model'])
        return False

    def make_loader(self, dataset, can_be_distributed=True, **kwargs):
        defaults = {
            'batch_size': self.args.optim.batch_size,
            'num_workers': self.args.num_workers,
            'collate_fn': SegmentBatch.collate_fn,
        }
        defaults.update(kwargs)
        if can_be_distributed:
            return flashy.distrib.loader(dataset, **defaults)
        else:
            return DataLoader(dataset, **defaults)

    def _fit_scaler(self):
        logger.info(f"Fitting scaler. Dataset size={len(self.datasets.train)} samples.")
        loader_factory = partial(
            self.make_loader, can_be_distributed=False, shuffle=True, persistent_workers=False)
        scaler = BatchScaler(
            features_builder=self.used_features,
            device=self.device,
            **self.args.norm.scaler)
        scaler.fit([loader_factory(dset) for dset in self.datasets.train.datasets])
        return scaler

    def _init_loader(self):
        datasets = self.datasets
        shuffled = ["train"]
        if self.args.optim.max_batches:
            shuffled.append("valid")
        self.loaders = {
            name: self.make_loader(getattr(datasets, name), shuffle=name in shuffled)
            for name in ["train", "valid", "test"]}

    def _make_negative_pool(self):
        # Check negative_pool_size
        if self.args.optim.negatives is not None:
            if self.args.optim.negative_pool_size is None:
                self.args.optim.negative_pool_size = 2 * self.args.optim.negatives
                logger.info(f"Setting negative_pool_size to 2 * {self.args.optim.negatives}")
            assert self.args.optim.negative_pool_size >= self.args.optim.negatives, \
                "Pool of negatives should be larger than the number of negatives"
        # Intialize negative pool
        negative_pool = {
            name: torch.Tensor([], device="cpu") for name in ["train", "valid"]
        }
        return negative_pool

    def get_formatter(self, stage_name: str):
        return flashy.Formatter({
            'loss': '.4f',
            'wer*': '.3%',
        }, default_format='.4f')

    def predict(self, meg: torch.Tensor = None, features: torch.Tensor = None,
                subject_index: int = 0, recording_index: int = 0) -> torch.Tensor:
        """Perform one prediction. If the MEG is not provided, it is set to 0.
        """
        subjects = torch.tensor([subject_index])  # type: ignore
        recordings = torch.tensor([recording_index])  # type: ignore
        assert features is not None
        if meg is None:
            meg = torch.zeros([273, features.shape[1]])
        mask = torch.ones_like(features).to(features).bool()
        estimate = self._process_batch(
            SegmentBatch(
                meg.unsqueeze(0), features.unsqueeze(0), mask.unsqueeze(0), subjects,
                recordings),
            training=False)[0]
        return estimate[0]

    def train(self) -> None:
        if len(self.history) > 0:
            logger.info("Replaying past metrics...")
            for epoch, stages in enumerate(self.history):
                for stage_name, metrics in stages.items():
                    self.result_logger._log_summary(
                        stage_name, metrics, step=epoch,
                        formatter=self.get_formatter(stage_name))

        for epoch in range(self.epoch, self.args.optim.epochs + 1):
            # Stages are used for automatic metric reporting to Dora, and it also
            # allows tuning how metrics are formatted.
            self.run_stage('train', self._run_one_epoch, training=True)

            with torch.no_grad():
                self.run_stage('valid', self._run_one_epoch, training=False)

            # determine if we stop this epoch
            will_stop = epoch == self.args.optim.epochs
            if self.args.early_stop_patience:
                if self.epoch >= self.best_epoch + self.args.early_stop_patience:
                    logger.warning("Model valid_loss did not improve for "
                                   f"{self.args.early_stop_patience} epochs. "
                                   "Stopping the training.")
                    will_stop = True

            if epoch % self.args.eval_every == 0 or will_stop:
                if self.best_epoch > self.last_test_epoch:
                    assert self.best_state is not None
                    with swap_state(self.all_models, self.best_state):
                        with torch.no_grad():
                            self.run_stage('test', self._test_one_epoch)
                    self.last_test_epoch = epoch
            if self.scale_reject is not None:
                logger.info(f"Scale Reject | Ratio {self.scale_reject.rejection_rate:.3%}")
            # Commit will send the metrics to Dora and save checkpoints by default.
            self.commit()

            if will_stop:
                break

    def _process_batch(self, batch: SegmentBatch, training=False):
        """
        Runs model with a batch of data. Supports both the encoder and decoder tasks.

        Args:
            training - whether to run this method in training mode (perform data augmentation, etc)
        Returns:
            tuple containing the estimated of the model and ground-truth output.
            Output type for the encoder task is MEG, and for the decoder task is features.
        """
        args = self.args
        task = args.task
        sample_rate = args.dset.sample_rate
        batch = batch.to(self.device)

        if self.scale_reject:
            batch, reject_mask = self.scale_reject(batch)
        else:
            reject_mask = torch.ones(len(batch.meg), dtype=torch.bool).to(self.device)
        meg = batch.meg
        features = batch.features
        features_mask = batch.features_mask
        if not task.mask_loss:
            features_mask = torch.ones_like(features_mask)

        if len(meg) == 0:
            return None, None, None, None

        assert torch.isfinite(meg).all()
        assert torch.isfinite(features).all()
        assert torch.isfinite(features_mask).all()

        if 'offset_meg_ms' in args.task and args.task.offset_meg_ms:
            offset_meg_samples = int(args.task.offset_meg_ms / 1000 * sample_rate)
            meg = meg[..., offset_meg_samples:]

            if ("parameters" in args) and ("input_sample_rate" in args.feature_model_params):
                offset_features_samples = int(
                    args.task.offset_meg_ms / 1000 * args.feature_model_params.input_sample_rate
                )
            else:
                offset_features_samples = offset_meg_samples

            features = features[..., :-offset_features_samples]
            features_mask = features_mask[..., :-offset_features_samples]

        meg_gt = meg.clone()

        if task.lowpass:
            meg = julius.lowpass_filter(meg, task.lowpass / sample_rate, zeros=5)
            if (task.lowpass_gt and training) or task.lowpass_gt_test:
                meg_gt = meg.clone()

        if task.type == "decode":
            limit = 0
            inputs = dict(meg=meg)
            output = features
        elif task.type == "encode":
            limit = int(task.meg_init * sample_rate)
            length = meg.shape[-1]
            mask = torch.zeros(length).to(meg)
            mask[:limit] = 1
            inputs = dict(meg=mask * meg, features=features)
            output = meg_gt
        else:
            assert False, f"Unknown task {task.type}"

        estimate = self.model(inputs, batch)

        # We remove the initial part of the signal, to prevent learning there.
        estimate = estimate[..., limit:]
        output = output[..., limit:]
        features_mask = features_mask[..., limit:]

        if self.feature_model is not None:
            if args.feature_model_name == "wav2vec2":
                # In case multi GPUs: Move "output" on the same device as the feature model
                output = output.to(self.feature_model.device)

                output = self.feature_model(output, output_dim=estimate.shape[-1])

                # In case multi GPUs: Moves "output" on the same device
                #  as "estimate" before the loss computation                                    )
                output = output.to(self.device)

                # TODO: Test if interpolation is good enough for mask
                features_mask = F.interpolate(
                        features_mask.float(), estimate.shape[-1]
                    ).bool()
            else:
                output = self.feature_model(output)
        return estimate, output, features_mask, reject_mask

    # NOTE: "epoch" in neuro dataset terminology means a recording session where user is presented
    # with a stimulus and their neuro activity is recorded (confusing, I know)
    def _run_one_epoch(self, training=False):
        self.all_models.train(training)
        self.loss.train(training)
        data_loader = self.loaders['train'] if training else self.loaders['valid']

        # get a different order for distributed training, otherwise this will get ignored
        if training and flashy.distrib.is_distributed():
            # flashy counts epoch as 1-based, while we used to be 0 based.
            # going back to 0-based here for compat.
            data_loader.sampler.set_epoch(self.epoch - 1)

        total = len(data_loader)
        if self.args.optim.max_batches:
            total = min(total, self.args.optim.max_batches)
        logprog = self.log_progress(self.current_stage, data_loader,
                                    updates=self.args.num_prints, total=total)
        last_batch = None
        averager = flashy.averager()
        for idx, batch in enumerate(logprog):
            estimate, output, features_mask, _ = self._process_batch(batch, training)
            # Shitty hack for distributed to work properly
            if estimate is None:  # batch contained only invalid elements
                if last_batch is None:
                    raise RuntimeError("Empty batch and last batch is none")
                else:
                    estimate, output, features_mask, _ = self._process_batch(last_batch, training)
            else:
                last_batch = batch

            if not features_mask.any():
                logger.error('no mask! %r %r', estimate.shape, features_mask.shape)
                assert False

            # Complete outputs with negatives here
            with torch.no_grad():
                n_negatives = self.args.optim.negatives
                if n_negatives is not None:
                    assert self.args.optim.loss == 'clip'
                    if len(output) < n_negatives:
                        phase = 'train' if training else 'valid'
                        buf = self.negative_pool[phase]
                        n_kept = n_negatives - len(output)
                        kept = torch.randperm(len(buf))[:n_kept]
                        output = torch.cat([output, buf[kept].to(output.device)], dim=0)
                        # Update negative pool with current batch
                        buf = torch.cat([output.to(buf.device), buf])
                        self.negative_pool[phase] = buf[:self.args.optim.negative_pool_size]

            loss = self.loss(estimate, output, features_mask)

            if training:
                for mod in self.model.modules():
                    if hasattr(mod, 'training_penalty'):
                        loss += mod.training_penalty
                if self.args.optim.svd:
                    loss += self.args.optim.svd * svd_penalty(self.model)

            # optimize model in training mode
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                flashy.distrib.sync_model(self.all_models)
                self.optimizer.step()

            metrics = averager({'loss': loss})
            logprog.update(**metrics)
            # Just in case, clear some memory
            del loss, estimate
            if idx + 1 == self.args.optim.max_batches:
                break
        metrics = flashy.distrib.average_metrics(metrics, idx + 1)
        if not training and metrics['loss'] < self.best_loss:
            self.best_loss = metrics['loss']
            self.best_epoch = self.epoch
            logger.info(bold('New best valid loss %.4f'), self.best_loss)
            self.best_state = copy_state(self.all_models.state_dict())
        return metrics

    def get_metric_constructors(self):
        """
        Returns a list of test metrics used to evaluate the model. Each element contains a
        constructor of a class used to calculate the metric
        """
        if self.args.task.type == "encode":
            return [OnlineCorrelation.get_constructor(slice(None), slice(None), 'corr_meg')]
        elif self.args.task.type == "decode":
            metric_constructors = []
            for feature in self.used_features.values():
                feature_name = feature.name
                feature_slice = self.used_features.get_slice(feature_name)
                model_out_slice = self.used_features.get_slice(feature_name, model_output=True)

                if feature.categorical:
                    metric_constructors += [
                        ClassificationAcc.get_constructor(
                            model_out_slice, feature_slice,
                            name=f"acc_{feature_name}")
                    ]
                else:
                    metric_constructors += [
                        L2Reg.get_constructor(feature_slice, model_out_slice,
                                              name=f"l2_{feature_name}"),
                        OnlineCorrelation.get_constructor(
                            model_out_slice, feature_slice,
                            name=f"corr_{feature_name}")
                    ]

            return metric_constructors
        assert False

    def _test_one_epoch(self, datasets=None):
        if self.args.task.type == 'encode':
            time_offset = -self.args.dset.tmin
            time_offset -= self.args.task.meg_init
            trim_offset = int(self.args.dset.sample_rate * time_offset)
        else:
            trim_offset = 0
        if isinstance(self.loss, ClipLoss):
            from .wer import get_wer
            metrics = get_wer(self)
        else:
            from .play import get_test_metrics  # todo: make that nice one day
            metrics = get_test_metrics(self, trim_offset, datasets=datasets)
        return metrics
