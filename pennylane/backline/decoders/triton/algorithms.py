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

"""Postprocessing kernels for Triton-based syndrome decoding."""

import triton
import triton.language as tl

from .bp_iters import _sum_product_posteriors


@triton.jit
def _decode_one(
    syndrome,
    H: tl.constexpr,
    postprocess: tl.constexpr = "hard",
    prob: tl.constexpr = 0.1,
    num_iters: tl.constexpr = 10,
):
    """Decode one packed syndrome into a packed correction mask.

    Args:
        syndrome (u64): Packed syndrome bitmask. Bit ``i`` stores check ``i``.
        H (tuple[tuple[int]]): Binary parity-check matrix. Row ``i`` matches
            syndrome bit ``i``, and column ``j`` matches correction bit ``j``.
        postprocess (str): Postprocessing rule to apply to the posterior LLRs.
        prob (float): Prior error probability assigned to each variable.
        num_iters (int): Number of belief-propagation iterations.

    Returns:
        u64: Packed correction mask. Bit ``j`` targets variable ``j``.
    """
    syndrome = tl.cast(syndrome, tl.uint64)

    posterior_llrs = _sum_product_posteriors(syndrome, H, prob, num_iters)
    if postprocess == "osd":
        return _osd(posterior_llrs, syndrome)
    return _hard_decision(posterior_llrs)


@triton.jit
def _hard_decision(posterior_llrs):
    """Pack negative posterior LLRs into a correction mask.

    Args:
        posterior_llrs (tuple[float]): Posterior LLRs, one per variable.

    Returns:
        u64: Packed correction mask with bit ``i`` set when ``posterior_llrs[i] < 0``.
    """
    one = tl.cast(1, tl.uint64)
    zero = tl.cast(0, tl.uint64)
    mask = zero
    for i in tl.static_range(len(posterior_llrs)):
        mask = mask | (tl.where(posterior_llrs[i] < 0.0, one, zero) << i)
    return mask


@triton.jit
def _osd(posterior_llrs, syndrome):
    """Build an order-zero one-bit correction for a nonzero syndrome.

    Args:
        posterior_llrs (tuple[float]): Posterior LLRs, one per variable.
        syndrome (u64): Packed syndrome bitmask.

    Returns:
        u64: One-hot correction mask for the most likely variable, or ``0`` when
            the syndrome is zero.
    """
    one = tl.cast(1, tl.uint64)
    zero = tl.cast(0, tl.uint64)
    best_index = zero
    best_llr = posterior_llrs[0]
    for i in tl.static_range(1, len(posterior_llrs)):
        is_better = posterior_llrs[i] < best_llr
        best_index = tl.where(is_better, tl.cast(i, tl.uint64), best_index)
        best_llr = tl.where(is_better, posterior_llrs[i], best_llr)
    return tl.where(syndrome != 0, one << best_index, zero)
