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

"""Tests for Triton decoder helper utilities."""

# pylint: disable=protected-access,wrong-import-position,broad-exception-caught

import math

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
    pytest.mark.skipif(
        not _has_cuda_target(), reason="Triton decoder tests require a CUDA device"
    ),
]

from pennylane.backline.decoders.triton.bp_iters import (
    _bp_c2v_msg,
    _bp_tanh_half,
    _get_syndrome_signs,
    _llr_from_p,
)


@triton.jit
def _get_syndrome_signs_kernel(out_ptr, syndrome, NCHECKS: tl.constexpr):
    signs = _get_syndrome_signs(syndrome, NCHECKS)
    for i in tl.static_range(NCHECKS):
        tl.store(out_ptr + i, signs[i])


@triton.jit
def _bp_tanh_half_kernel(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    tl.store(out_ptr, _bp_tanh_half(x))


@triton.jit
def _bp_c2v_msg_kernel(ssign_ptr, prod_ptr, out_ptr):
    ssign = tl.load(ssign_ptr)
    prod = tl.load(prod_ptr)
    tl.store(out_ptr, _bp_c2v_msg(ssign, prod))


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Triton decoder tests require a CUDA device")
    return torch


class TestTritonUtils:
    """Tests for helper utilities used by the Triton decoder."""

    def test_get_syndrome_signs_decodes_bits_to_bipolar_signs(self):
        """Syndrome bits should map to +1/-1 in least-significant-bit order."""
        torch = _torch()

        out = torch.empty(4, device="cuda", dtype=torch.float32)
        _get_syndrome_signs_kernel[(1,)](out, 0b1010, NCHECKS=4)

        expected = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("x", [-2.0, 0.0, 2.0])
    def test_bp_tanh_half_matches_math_tanh(self, x):
        """The Triton helper should compute tanh(x / 2)."""
        torch = _torch()

        x_tensor = torch.tensor([x], device="cuda", dtype=torch.float64)
        out = torch.empty(1, device="cuda", dtype=torch.float64)
        _bp_tanh_half_kernel[(1,)](x_tensor, out)

        assert out.cpu().item() == pytest.approx(math.tanh(x / 2.0))

    def test_bp_c2v_msg_matches_log_formula(self):
        """The message helper should match the expected log formula."""
        torch = _torch()

        ssign = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
        prod = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        _bp_c2v_msg_kernel[(1,)](ssign, prod, out)

        expected = -math.log((1.0 + 0.5) / (1.0 - 0.5))
        assert out.cpu().item() == pytest.approx(expected)

    def test_bp_c2v_msg_clamps_out_of_range_products(self):
        """Products outside [-1, 1] should be clamped before taking the log."""
        torch = _torch()

        ssign = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        prod = torch.tensor([2.0], device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        _bp_c2v_msg_kernel[(1,)](ssign, prod, out)

        hi = float(torch.tensor(1.0 - 1e-6, dtype=torch.float32).item())
        expected = math.log((1.0 + hi) / (1.0 - hi))
        assert math.isfinite(out.cpu().item())
        assert out.cpu().item() == pytest.approx(expected)

    def test_llr_from_p_matches_log_odds(self):
        """The LLR helper should return the standard log-odds transform."""
        assert _llr_from_p(0.2) == pytest.approx(math.log(4.0))
