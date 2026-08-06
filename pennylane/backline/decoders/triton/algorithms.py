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

import triton
import triton.language as tl

from .bp_iters import _sum_product_posteriors

# ================== Single Syndrome Decoder ====================


@triton.jit
def _decode_one(
    syndrome,
    H: tl.constexpr,
    postprocess: tl.constexpr = "hard",
    prob: tl.constexpr = 0.1,
    NITER: tl.constexpr = 10,
):
    """Decode one syndrome into a correction."""
    syndrome = tl.cast(syndrome, tl.uint64)

    P = _sum_product_posteriors(syndrome, H, prob, NITER)
    if postprocess == "osd":
        return _osd(P, syndrome)
    return _hard_decision(P)


# ========================= Post Process ========================


@triton.jit
def _hard_decision(P):
    """Pack negative posterior values into a correction mask."""
    one = tl.cast(1, tl.uint64)
    zero = tl.cast(0, tl.uint64)
    mask = zero
    for i in tl.static_range(len(P)):
        mask = mask | (tl.where(P[i] < 0.0, one, zero) << i)
    return mask


@triton.jit
def _osd(P, syndrome):
    """Build an order-zero one-bit correction for a nonzero syndrome."""
    one = tl.cast(1, tl.uint64)
    zero = tl.cast(0, tl.uint64)
    bi = zero
    best = P[0]
    for i in tl.static_range(1, len(P)):
        c = P[i] < best
        bi = tl.where(c, tl.cast(i, tl.uint64), bi)
        best = tl.where(c, P[i], best)
    return tl.where(syndrome != 0, one << bi, zero)
