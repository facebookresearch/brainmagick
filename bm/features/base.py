# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic audio features."""

from collections import OrderedDict
import typing as tp
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bm.utils import Frequency
from bm.events import Event, DataSlice

logger = logging.getLogger(__name__)


class FeaturesBuilder(OrderedDict):  # type: ignore
    """Creates array of features on-the-fly.
    """
    # stores all features classes
    _FEATURE_CLASSES: tp.Dict[str, tp.Type["Feature"]] = {}

    def __init__(self, events: pd.DataFrame, features: tp.Sequence[str],
                 features_params: dict,
                 sample_rate: Frequency, event_mask: bool = False) -> None:
        super().__init__()
        features = list(features)
        self.features_params = features_params
        self.sample_rate = sample_rate
        self.event_mask = event_mask

        try:
            self.update([
                (feature,
                 self._FEATURE_CLASSES[feature](  # type: ignore
                     sample_rate=self.sample_rate,
                     **features_params.get(feature, {})))
                for feature in features])
        except KeyError as e:
            available = set(self._FEATURE_CLASSES)
            missing = ", ".join(set(features) - available)
            options = ", ".join(set(available) - set(features))
            raise KeyError(f"Cound not find feature(s): {missing}. "
                           f"Did you mean one of: {options}?") from e
        # prepare events
        event_kinds = {f.event_kind for f in self.values()}
        if self.event_mask:
            from .basic import WordSegment  # lazy import
            self.word_seg_feature = WordSegment(self.sample_rate)
            event_kinds.add(self.word_seg_feature.event_kind)

        # copy otherwise this is a view and we can't assign _stop
        self.events = events.loc[[c in event_kinds for c in events.kind], :].copy()
        self.events.loc[:, "_stop"] = self.events.start + self.events.duration  # TODO move
        missing_events = event_kinds - set(events.kind)
        missing_events -= set(['sound'])  # too many warnings for multimodal models.
        if missing_events and len(events) > 0:
            logger.warning("Could not find any event for feature(s) "
                           "with kind(s): %s", missing_events)

    # pylint: disable=too-many-locals
    def __call__(self, start: float, stop: float
                 ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.List[Event]]:
        if len(self.values()) == 1:
            # If there is only 1 feature, let's use directly its sample_rate
            sample_rate = list(self.values())[0].sample_rate
        else:
            # Otherwise use the FeatureBuilder global sample_rate (ie 120)
            sample_rate = self.sample_rate

        n_times = sample_rate.to_ind(stop - start)
        data = torch.zeros((self.dimension, n_times), dtype=torch.float32)
        mask = torch.zeros((1, n_times), dtype=torch.float32)
        select = np.logical_and(self.events._stop >= start, self.events.start < stop)
        events = self.events.loc[select, :]

        # Init data with features default vals
        for feature in self.values():
            assert data[self.get_slice(feature.name)].shape[0] == feature.dimension
            data[self.get_slice(feature.name)] = feature.default_value

        dslice = DataSlice(
            start=start, duration=stop - start, sample_rate=sample_rate,
            language=None, modality=None)  # XXX To remove when migrating to Python 3.10
        event_list: tp.List[Event] = [dslice]  # keep total duration for debug
        for event in events.event.iter():
            # indices relative to the feature start
            event_list.append(event)
            # figure out overlaps
            overlap = dslice.overlap(event)
            if overlap.duration_ind < 1:
                continue

            for feature in self.values():
                if feature.event_kind == event.kind:
                    feature._curr_epoch_stop = stop  # Saved for visualization in notebooks
                    feature._curr_epoch_start = start  # Saved for visualization in notebooks

                    assert overlap.duration_ind >= 1
                    val = feature.get_on_overlap(event, overlap)
                    data[self.get_slice(feature.name), overlap.slice_in_parent()] = val

            # Populates mask which indicates non-silent parts of the epoch (ones that contain
            # a stimulus).
            if self.event_mask:
                if self.word_seg_feature.event_kind == event.kind:
                    val = self.word_seg_feature.get(event)  # type: ignore
                    mask[:, overlap.slice_in_parent()] = val

        for feature in self.values():
            feature.post_process(data[self.get_slice(feature.name)])

        if not self.event_mask:
            mask[:, :] = 1

        return data, mask.bool(), event_list

    def get_slice(self, name: str, model_output: bool = False) -> slice:
        """Returns the slice matching the given feature in the features Tensor.

        Parameters
        ----------
        name :
            Name of the feature.
        model_output :
            If true, returns the slice matching feature in the model output. This can be different
            from the features Tensor if the model outputs predictions used for classification.
        """
        if name not in self:
            raise KeyError(f"Could not find feature {name}.")
        start = 0
        for key, feature in self.items():
            feature_dim = feature.output_dimension if model_output else feature.dimension
            if name == key:
                break
            start += feature_dim
        return slice(start, start + feature_dim)

    def extract_features(
            self, features: torch.Tensor, feature_names: tp.Sequence[str]) -> torch.Tensor:
        """Returns a tensor of features containing only the given
        input names from another given feature builder object. Make
        sure to return the tensor in the correct order.
        """
        assert features.shape[1] == self.dimension, "Input should contain all features"
        assert all([name in self for name in feature_names])

        extracted_features: tp.List[torch.Tensor] = []
        for feature_name in feature_names:
            feature_slice = self.get_slice(feature_name)
            feature = features[:, feature_slice]
            extracted_features.append(feature)

        return torch.cat(extracted_features, dim=1)

    @property
    def dimension(self) -> int:
        """Returns the sum dimension of all features registered in
        this class object.
        """
        return sum(feature.dimension for feature in self.values())

    @property
    def output_dimension(self) -> int:
        """Returns the sum dimension of all features when predicted by the model.
        """
        return sum(feature.output_dimension for feature in self.values())

    def __reduce__(self) -> tp.Any:
        """This fixes pickling, because we inherit from OrderedDict.
        """
        return object.__reduce__(self)


class Feature:
    """Base class for defining features value based on a name.
    """
    event_kind = ""
    dimension = 1
    cardinality: tp.Optional[int] = None  # if not None, this is treated as a categorical feature
    default_value = 0.
    sample_rate = Frequency(float('NaN'))  # will be overriden

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        assert cls.event_kind, "Missing event_kind"
        if cls.__name__.startswith('_'):
            # use for feature base classes that are not features themselves.
            return
        FeaturesBuilder._FEATURE_CLASSES[cls.__name__] = cls

    @property
    def output_dimension(self) -> int:
        """Returns the dimension of the feature when predicted by the model.
        """
        return self.dimension if self.cardinality is None else self.cardinality

    @property
    def categorical(self) -> bool:
        """Whether this feature represents a categorical feature.
        """
        return self.cardinality is not None

    @property
    def normalizable(self) -> bool:
        """Whether this feature can be normalized.
        """
        return not self.categorical

    def __init__(self, sample_rate: Frequency) -> None:
        self.sample_rate = sample_rate
        assert self.dimension >= 1
        assert self.cardinality is None or self.dimension == 1, \
            "We do not see a case where the dimension should be larger than one for categorical " \
            "features."

    def __repr__(self) -> str:
        return f"{self.name}({self.sample_rate})"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get(self, event: tp.Any) -> tp.Union[float, torch.Tensor]:
        """Provides the data associated with the event, without taking into
        account which part overlaps with the data slice / epoch. The overlapping
        is managed in get_on_overlap.
        """
        raise NotImplementedError

    def get_on_overlap(self, event: tp.Any, overlap: DataSlice) -> tp.Union[float, torch.Tensor]:
        """Get the feature ready for inclusion into the epoch data.

        This default implementations truncates the output of "get" to fit into the DataSlice.
        """
        if not isinstance(event, Event):
            raise TypeError(f"Inconsistent event type {type(event)}")
        val = self.get(event)
        if not isinstance(val, (torch.Tensor, float, int)):
            raise TypeError(f"Invalid type {type(val)} for feature {self}")
        if isinstance(val, torch.Tensor):
            if len(val.shape) == 2:
                assert val.shape[-1] > 0
                first = max(0, -overlap._sample_rate.to_ind(event.start - overlap.start))
                first = min(first, val.shape[-1] - 1)
                val = val[:, first: first + overlap.duration_ind]
                # data_slice_len is calculated from self.sample_rate.to_ind and can
                # produce rounding diffe when compared to the output feature dim
                if (overlap.duration_ind - val.shape[-1]) == 1:
                    # Replicate padding works only on 3dim or more tensor (that's
                    # why squeeze/unsqueeze was used.
                    val = F.pad(
                        val.unsqueeze(0), (0, 1), mode="replicate").squeeze(0)
                else:
                    assert val.shape[-1] == overlap.duration_ind
            while len(val.shape) < 2:
                val = val.unsqueeze(-1)
            if len(val.shape) > 2:
                raise RuntimeError(f"Weird shape {val.shape}")
        return val

    def post_process(self, tensor: torch.Tensor) -> None:
        pass
