# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC
from functools import partial
import torch
import typing as tp

logger = logging.getLogger(__name__)


class TestMetric(ABC):
    def __init__(self, left_slice: slice, right_slice, name: str = "metric"):
        self.name = name
        self.left_slice = left_slice
        self.right_slice = right_slice

    @classmethod
    def get_constructor(cls, *args: tp.Any, **kwargs: tp.Any) -> tp.Callable[..., "TestMetric"]:
        return partial(cls, *args, **kwargs)

    def update(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> "TestMetric":
        raise NotImplementedError()

    def get(self) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def reduce(cls, stats: tp.List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(stats).mean().item()


class OnlineCorrelation(TestMetric):
    def __init__(self, left_slice: slice, right_slice: slice,
                 name: str = "correlation", dim: int = 0, tol: float = 1e-8):
        """
        Compute online correlations between `left` and `right` tensors,
        along the given dimension. Tensors can be provided as small batches
        with `update`.
        For centered complex variables, the correlation is extended to:
            Re[(conj(x)^T y)] / (x.abs() * y.abs())
        Args:
            dim (int): dimension along which to compute the correlation.
                Multiple calls to `update()` should stream chunks along
                this dimension.
            tol (float): tolerence used for numerical stability.
        """
        super().__init__(left_slice, right_slice, name)
        self.dim = dim
        self.tol = tol
        assert tol >= 0
        self._count = torch.Tensor([0])
        self._sum_dot: torch.Tensor
        self._sum_left: torch.Tensor
        self._sum_right: torch.Tensor
        self._sum_left_squared: torch.Tensor
        self._sum_right_squared: torch.Tensor

    def update(self, left: torch.Tensor, right: torch.Tensor,
               mask: torch.Tensor) -> "OnlineCorrelation":
        left = left[:, self.left_slice]
        right = right[:, self.right_slice]

        dim = self.dim
        if self._count.sum() == 0:
            index: tp.List[tp.Union[int, slice]] = [slice(None) for _ in left.size()]
            index[dim] = 0
            ref = left[index]
            self._sum_dot = torch.zeros_like(ref)
            self._sum_left = torch.zeros_like(ref)
            self._sum_right = torch.zeros_like(ref)
            if ref.dtype.is_complex:
                ref = ref.real
            self._sum_left_squared = torch.zeros_like(ref)
            self._sum_right_squared = torch.zeros_like(ref)
            self._count = torch.zeros_like(ref)

        # .conj() and .abs() are required for proper treatment
        # of complex numbers
        self._sum_dot += (left.conj() * right * mask).sum(dim)
        self._sum_left += (left * mask).sum(dim)
        self._sum_right += (right * mask).sum(dim)
        self._sum_left_squared += (left * mask).abs().pow(2).sum(dim)
        self._sum_right_squared += (right * mask).abs().pow(2).sum(dim)
        self._count += mask.sum(dim)

        return self

    def get(self) -> torch.Tensor:
        """
        Return the correlation tensor.
        """
        def _norm_centered(sum_, sum_squared):
            norm_squared = sum_squared - sum_.abs().pow(2) / self._count
            if norm_squared.min() < -self.tol:
                raise ValueError(
                    f"Numerical instabilities when computing the correlation. "
                    f"Expected {sum_squared} - {sum_}**2 / {self._count} to be positive "
                    f"but got {norm_squared.min()}")
            return norm_squared.clamp_(0, float('inf')).sqrt_()

        norm_left = _norm_centered(self._sum_left, self._sum_left_squared)
        norm_right = _norm_centered(self._sum_right, self._sum_right_squared)

        dot = self._sum_dot - self._sum_left.conj() * self._sum_right / self._count
        if dot.dtype.is_complex:
            dot = dot.real
        correlation = dot / (norm_left * norm_right).clamp(self.tol, float('inf'))
        assert not torch.isnan(correlation).any(), "Tensor contain nans. Perhaps division by " \
                                                   f"zero cause that? {correlation}"
        return correlation


class AccumulativeMetric(TestMetric):
    def __init__(self,
                 left_slice: slice,
                 right_slice: slice,
                 name: str = "N/A", dim: int = 0):

        super().__init__(left_slice, right_slice, name)
        self.dim = dim
        self._count = torch.Tensor([0])
        self._accum_metric: torch.Tensor

    def update(self, left: torch.Tensor, right: torch.Tensor,
               mask: torch.Tensor) -> "AccumulativeMetric":
        left = left[:, self.left_slice]
        right = right[:, self.right_slice]

        dim = self.dim
        if self._count.sum() == 0:
            index: tp.List[tp.Union[int, slice]] = [slice(None) for _ in right.size()]
            index[dim] = 0
            ref = right[index]
            self._accum_metric = torch.zeros_like(ref)
            self._count = torch.zeros_like(ref)

        self._accum_metric += self.accum_func(left, right, mask)
        self._count += mask.sum(dim)

        return self

    def get(self) -> torch.Tensor:
        if self._count.sum() == 0:
            return torch.Tensor([0.])
        ret = self._accum_metric / self._count
        assert not torch.isnan(ret).any(), "Tensor contain nans. Perhaps division by " \
                                           f"zero cause that? {ret}"
        return ret

    def accum_func(self, left, right, mask):
        raise NotImplementedError()


class L1Reg(AccumulativeMetric):
    def accum_func(self, left, right, mask):
        return abs((left - right) * mask).sum(self.dim)


class L2Reg(AccumulativeMetric):
    def accum_func(self, left, right, mask):
        return (((left - right) * mask)**2).sum(self.dim)

    @classmethod
    def reduce(cls, stats: torch.Tensor) -> torch.Tensor:
        return torch.stack(stats).mean().sqrt().item()


class ClassificationAcc(AccumulativeMetric):
    def accum_func(self, left, right, mask):
        preds = left.argmax(1, keepdim=True)
        expected = right.clone()
        # Use two different invalid classes for positions not in mask, so not to count them in the
        # accuracy predictions
        preds[~mask], expected[~mask] = -1, -2
        return (preds == expected).sum(self.dim)
