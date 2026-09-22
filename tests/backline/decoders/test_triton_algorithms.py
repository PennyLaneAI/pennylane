# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Triton decoder algorithms."""

# pylint: disable=protected-access,wrong-import-position,broad-exception-caught

import pytest

triton = pytest.importorskip("triton")
import triton.language as tl


def _has_cuda_target() -> bool:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _has_cuda_target(), reason="Triton decoder tests require a CUDA device"),
]

from pennylane.backline.decoders.triton import algorithms


@triton.jit
def _load_posteriors(posteriors_ptr, N: tl.constexpr):
    # builds a tuple of posteriors
    posteriors = ()
    for i in tl.static_range(N):
        posteriors += (tl.load(posteriors_ptr + i),)
    return posteriors


@triton.jit
def _postprocess_kernel(
    posteriors_ptr, out_ptr, syndrome, POSTPROCESS: tl.constexpr, N: tl.constexpr
):
    posteriors = _load_posteriors(posteriors_ptr, N)
    if POSTPROCESS == "osd":
        correction = algorithms._osd(posteriors, syndrome)
    else:
        correction = algorithms._hard_decision(posteriors)
    tl.store(out_ptr, correction)


@triton.jit
def _decode_one_kernel(out_ptr, syndrome, POSTPROCESS: tl.constexpr):
    tl.store(
        out_ptr,
        algorithms._decode_one(syndrome, ((1,),), postprocess=POSTPROCESS, prob=0.1, num_iters=2),
    )


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Triton decoder tests require a CUDA device")
    return torch


class TestTritonAlgorithms:
    """Tests for helper algorithms used by the Triton decoder."""

    def test_hard_decision_packs_negative_posteriors(self):
        """Negative posteriors should set their bit in the correction mask."""
        torch = _torch()

        posteriors = torch.tensor([1.0, -0.5, 0.0, -2.0], device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.uint64)
        _postprocess_kernel[(1,)](posteriors, out, 0, POSTPROCESS="hard", N=4)

        assert out.cpu().item() == 0b1010

    def test_osd_returns_zero_for_zero_syndrome(self):
        """Order-zero OSD should not flip anything for a zero syndrome."""
        torch = _torch()

        posteriors = torch.tensor([-1.0, -3.0, -2.0], device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.uint64)
        _postprocess_kernel[(1,)](posteriors, out, 0, POSTPROCESS="osd", N=3)

        assert out.cpu().item() == 0

    def test_osd_selects_most_negative_posterior(self):
        """Order-zero OSD should flip the bit with the smallest posterior."""
        torch = _torch()

        posteriors = torch.tensor([0.1, -0.7, -0.6], device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.uint64)
        _postprocess_kernel[(1,)](posteriors, out, 1, POSTPROCESS="osd", N=3)

        assert out.cpu().item() == 0b010

    @pytest.mark.parametrize(("syndrome", "expected"), [(0, 0), (1, 1)])
    def test_decode_one_hard_decodes_single_bit_code(self, syndrome, expected):
        """The hard-decision decoder should handle the one-check one-bit code."""
        torch = _torch()

        out = torch.empty(1, device="cuda", dtype=torch.uint64)
        _decode_one_kernel[(1,)](out, syndrome, POSTPROCESS="hard")

        assert out.cpu().item() == expected

    @pytest.mark.parametrize(("syndrome", "expected"), [(0, 0), (1, 1)])
    def test_decode_one_osd_decodes_single_bit_code(self, syndrome, expected):
        """The OSD decoder should handle the one-check one-bit code."""
        torch = _torch()

        out = torch.empty(1, device="cuda", dtype=torch.uint64)
        _decode_one_kernel[(1,)](out, syndrome, POSTPROCESS="osd")

        assert out.cpu().item() == expected
