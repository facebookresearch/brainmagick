# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import numpy as np

from bm.metrics import OnlineCorrelation, ClassificationAcc


def test_online_correlation():
    size = 2048
    dim = 4

    def _test(a, b, mask, descr):
        tol = 1e-10 if a.dtype is torch.double else 1e-5
        ac = a - a.mean(dim=0, keepdim=True)
        bc = b - b.mean(dim=0, keepdim=True)
        ref = (ac * bc).mean(0) / (a.std(0, False) * b.std(0, False))

        online = OnlineCorrelation(dim=0, left_slice=slice(None), right_slice=slice(None))
        online.update(a, b, mask)
        assert torch.norm(ref - online.get()) < tol, descr

        online = OnlineCorrelation(dim=0, left_slice=slice(None), right_slice=slice(None))
        cs = 16
        for k in range(0, size, cs):
            online.update(a[k:k + cs], b[k: k + cs], mask[k: k + cs])
        assert torch.norm(ref - online.get()) < tol, descr

    for dtype in [torch.float, torch.double]:
        for sigma in [1e-6, 1e-2, 1, 100, 1e4]:
            a = torch.randn(size, dim, dtype=dtype)
            b = sigma * torch.randn(size, dim, dtype=dtype) + a
            mask = torch.ones_like(a).bool()

            _test(a, b, mask, (dtype, sigma))


def test_online_complex_correlation():
    sr = 120
    dur = 4
    t = torch.arange(sr * dur).float() / sr

    freq = 12

    for phase in [0, math.pi / 3, math.pi / 4, math.pi / 2, math.pi]:
        a = torch.exp(2j * math.pi * t * freq).unsqueeze(1)
        b = torch.exp(2j * math.pi * t * freq - 1j * phase).unsqueeze(1)
        mask = torch.ones_like(a).bool()

        res = OnlineCorrelation(
            dim=0, left_slice=slice(None), right_slice=slice(None)
        ).update(a, b, mask).get().item()
        assert abs(res - math.cos(phase)) < 1e-5, phase / math.pi


def test_classification_accuracy():
    probabilities = torch.randn(2, 2, 10)
    labels = torch.randint(0, 2, (2, 1, 10))
    mask = torch.ones_like(labels).bool()

    res = ClassificationAcc(
        left_slice=slice(0, 2), right_slice=slice(0, 1)
    ).update(probabilities, labels, mask).get()
    res = ClassificationAcc.reduce([res])
    expected = (probabilities.argmax(1, keepdim=True) == labels).float().mean().item()
    np.testing.assert_almost_equal(res, expected, decimal=5)
