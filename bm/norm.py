# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from collections import OrderedDict, defaultdict
import logging
import random
import typing as tp

from dora.log import LogProgress
import torch
from torch.utils.data import DataLoader

from .features import FeaturesBuilder, Feature
from .dataset import SegmentBatch

logger = logging.getLogger(__name__)


def _as_nd(x):
    """Converts`x` from [B, C, T] to [B * T, C].
    """
    return x.permute(0, 2, 1).reshape(-1, x.shape[1])


def _from_nd(x, shape):
    """Converts from `[B * prod(shape), C]` to `[B, C, *shape]`.
    """
    return x.view(shape[0], shape[2], -1).permute(0, 2, 1).contiguous()


class Scaler(ABC):

    def fit(self, X, mask):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def inverse_transform(self, X):
        raise NotImplementedError()


class RobustScaler(Scaler):
    """
    Similar to RobustScaler for sklearn but can run on GPU
    """

    def __init__(self, lowq=0.25, highq=0.75, subsample=1., device="cpu"):
        self.lowq = lowq
        self.highq = highq
        self.subsample = subsample
        self.device = device

    def fit(self, X):
        samples, dimension = X.shape
        self.scale_ = torch.empty(dimension)
        self.center_ = torch.empty(dimension)
        X = X.to(self.device)
        for d in range(dimension):
            # to dimension one at a time to fit on the GPU memory
            col = X[:, d]
            keep = torch.rand_like(col) < self.subsample
            col = col[keep]
            # torch 1.7.0 has a quantile function but it is not really faster than sorting.
            col, _ = col.sort()
            quantiles = [self.lowq, 0.5, self.highq]
            low, med, high = [col[int(q * len(col))].item() for q in quantiles]
            self.scale_[d] = high - low
            self.center_[d] = med
            if self.scale_[d] == 0:
                # this will happen as we are padding some recordings
                # so that all recordings have the same number of channels.
                self.scale_[d] = 1
        assert (self.scale_ != 0).any()
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        return (X - self.center_.to(X)) / self.scale_.to(X)

    def inverse_transform(self, X):
        return (X * self.scale_.to(X)) + self.center_.to(X)


class StandardScaler(Scaler):
    """Same as sklearn, reimplemented here to avoid going back and forth between numpy
    and torch tensor.
    """
    def __init__(self, per_channel):
        self.per_channel = per_channel

    def fit(self, X, mask):
        samples, dimension = X.shape
        if self.per_channel:
            # Scales for each feature channel separately
            self.center_ = X[mask.expand_as(X)].reshape(-1, dimension).mean(dim=0)
            self.scale_ = X[mask.expand_as(X)].reshape(-1, dimension).std(dim=0)
        else:
            self.center_ = X[mask.expand_as(X)].mean(dim=0)
            self.scale_ = X[mask.expand_as(X)].std(dim=0)
        return self

    def transform(self, X):
        return (X - self.center_.to(X)) / self.scale_.to(X)

    def inverse_transform(self, X):
        return (X * self.scale_.to(X)) + self.center_.to(X)


class NoOpScaler(Scaler):
    """Dummy scaler that just returns the given input.
    """

    def fit(self, X, mask):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class NoOpCategoryCountScaler(NoOpScaler):
    """Dummy scaler that just returns the given input, but also counts
    histogram of categories, to be used later in adjusting CCE loss weights.
    """

    def __init__(self, cardinality):
        self.cardinality = cardinality

    def fit(self, X, mask):
        X_max_item = self.cardinality - 1
        X_min_item = 0
        assert all(X == X.int()) and X.min().item() == 0 and X.max().item() < self.cardinality
        self.categories_count_ = torch.histc(
            X[mask], bins=X_max_item - X_min_item + 1, min=X_min_item, max=X_max_item)
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class BatchScaler:
    def __init__(self, features_builder: FeaturesBuilder,
                 n_samples_per_recording=200, per_channel=False, device: str = 'cpu',
                 n_samples_features: tp.Optional[int] = None):
        self.n_samples_per_recording = n_samples_per_recording
        self.n_samples_features = n_samples_features
        self.device = device
        self.meg_scalers: tp.Dict[int, Scaler] = {}
        self.features_builder = features_builder
        self.feature_scalers: tp.Dict[str, Scaler] = OrderedDict()
        for name, feature in self.features_builder.items():
            self.feature_scalers[name] = self._create_feature_scaler(
                feature, per_channel)

    def _create_feature_scaler(self, feature: Feature, per_channel: bool) -> Scaler:
        scaler: Scaler
        if feature.normalizable:
            scaler = StandardScaler(per_channel)
        elif feature.categorical:
            scaler = NoOpCategoryCountScaler(feature.cardinality)
        else:
            scaler = NoOpScaler()
        return scaler

    def fit(self, loaders: tp.Sequence[tp.Iterable[DataLoader]]):
        all_meg = defaultdict(list)
        all_mask = []
        all_features = []
        progress = iter(LogProgress(logger, loaders, name="Fitting scalers", level=logging.INFO))
        loader = next(progress)
        next_loader: tp.Optional[tp.Iterable]
        while True:
            # We pre iterate on the next loader to have PyTorch create threads and warmup
            # the pipeline as we process the previous one.
            try:
                next_loader = iter(next(progress))
                # next_loader = next(progress)
            except StopIteration:
                next_loader = None

            remaining = self.n_samples_per_recording
            for batch in loader:
                remaining -= len(batch.meg)
                recording_index = batch.recording_index[0].item()
                assert (batch.recording_index == recording_index).all()
                all_meg[recording_index].append(batch.meg)
                all_features.append(batch.features)
                all_mask.append(batch.features_mask)
                if remaining <= 0:
                    break
            if next_loader is None:
                break
            loader = next_loader

        if self.n_samples_features is not None:
            rand_indexes = list(range(len(all_features)))
            rng = random.Random(1234)
            rng.shuffle(rand_indexes)
            all_features = [all_features[idx] for idx in rand_indexes]
            all_mask = [all_mask[idx] for idx in rand_indexes]
            remaining = self.n_samples_features
            for idx, features in enumerate(all_features):
                remaining -= len(features)
                if remaining <= 0:
                    all_features = all_features[:idx + 1]
                    all_mask = all_mask[:idx + 1]
                    break
        features = _as_nd(torch.cat(all_features))
        features_mask = _as_nd(torch.cat(all_mask))
        logger.info("features collected for norm: %r", features.shape)

        for recording_index, recording_meg in all_meg.items():
            meg = _as_nd(torch.cat(recording_meg))
            meg_scaler = RobustScaler(device=self.device)
            meg_scaler.fit(meg)
            assert recording_index not in self.meg_scalers
            self.meg_scalers[recording_index] = meg_scaler

        for name, feature_scaler in self.feature_scalers.items():
            slice_ = self.features_builder.get_slice(name)
            feature_data = features[:, slice_]
            feature_scaler.fit(feature_data, features_mask)
            if isinstance(feature_scaler, StandardScaler):
                assert (feature_scaler.scale_ > 0).all(), \
                    f"Annotation embedding {name} could not be normalized as the " \
                    "values were all the same. Are there relevant event annotations" \
                    " to be embedded?"

    def _transform(self, batch: SegmentBatch, inverse_transform: bool) -> SegmentBatch:
        """
        Performs normalization or inverse normalization transform on batch.

        Args:
            batch: batch to normalize
            inverse_transform: If True, run the de-normalization inverse transform on given data,
                otherwise run the normalization transform
        """
        meg = batch.meg
        features = batch.features
        recording_index = batch.recording_index
        if features.shape[1] != self.features_builder.dimension:
            raise ValueError(f"Invalid channel dim {features.shape[1]} for features, "
                             f"expected {self.features_builder.dimension}")

        all_meg = []
        for entry_meg, entry_recording in zip(meg, recording_index):
            scaler = self.meg_scalers[entry_recording.item()]
            transform_func = scaler.inverse_transform if inverse_transform else scaler.transform
            entry_meg = transform_func(entry_meg.t()).t()
            all_meg.append(entry_meg)
        meg = torch.stack(all_meg)

        transformed_features = []
        for name, feature_scaler in self.feature_scalers.items():
            slice_ = self.features_builder.get_slice(name)
            feature_data = features[:, slice_]
            # Transform/Inverse-transform
            transform_func = feature_scaler.inverse_transform if inverse_transform \
                else feature_scaler.transform
            transformed_feature = _from_nd(
                transform_func(_as_nd(feature_data)), feature_data.shape)

            transformed_features.append(transformed_feature)
        features = torch.cat(transformed_features, dim=1)
        return batch.replace(meg=meg, features=features)

    def transform(self, batch: SegmentBatch):
        return self._transform(batch, inverse_transform=False)

    def inverse_transform(self, batch: SegmentBatch):
        return self._transform(batch, inverse_transform=True)

    def inverse_transform_feature(self, feature_name, feature_data):
        """
        Inverse transform only one feature. Used for debug in notebook files.
        """
        feature_scaler = self.feature_scalers[feature_name]
        return _from_nd(
            feature_scaler.inverse_transform(_as_nd(feature_data)), feature_data.shape)

    def get_categorical_feature_weights(self, feature_name) -> torch.Tensor:
        """
        Returns a set of weights Tensor whose values are inversely proportional to the frequency
        of each category in a categorical feature.
        These weights can be used when applying CCE loss to compensate for class imbalancing.
        """
        scaler = self.feature_scalers[feature_name]
        assert isinstance(scaler, NoOpCategoryCountScaler)
        probs = scaler.categories_count_ / scaler.categories_count_.sum()
        # We use sqrt to "smooth" the weights given to classes, to avoid a situation
        # where a class is given extremely high or low weight value due to data imbalance.
        # Basically, we only partially rebalance, in order not to mess up optimization too much.
        weights = 1 / torch.sqrt(probs)
        weights[probs == 0] = 0.
        # Now to make sure we stay balanced with respect to other losses, we can make sure that
        # E[weights] = 1, i.e. sum p / sqrt(p) = sum sqrt(p)
        weights /= torch.sqrt(probs).sum()
        return weights


class ScaleReject:
    """
    Rescales the input MEG and features. If the MEG after rescaling
    still contains large values (e.g. more than `limit`) rejects the offending item.
    """

    def __init__(self, scaler, limit=16, exclude_empty_features=False, clip=False):
        self.scaler = scaler
        self.limit = limit
        self.clip = clip
        self.exclude_empty_features = exclude_empty_features
        self._rejection_count = 0
        self._count = 0

    def __call__(self, batch: SegmentBatch) -> tp.Tuple[SegmentBatch, torch.Tensor]:
        batch = self.scaler.transform(batch)
        self._count += len(batch.meg)

        meg = batch.meg
        features_mask = batch.features_mask

        if self.clip:
            meg.clamp_(-self.limit, self.limit)
        meg_max_sample_per_batch = meg.abs().view(len(meg), -1).max(-1)[0]
        reject = meg_max_sample_per_batch > self.limit
        if self.exclude_empty_features:
            features_is_none = features_mask.view(len(features_mask), -1).sum(-1) == 0
            reject |= features_is_none  # Reject all trials without features
        self._rejection_count += reject.long().sum().item()
        keep = ~reject
        return batch[keep], keep

    @property
    def rejection_rate(self):
        return self._rejection_count / max(self._count, 1)
