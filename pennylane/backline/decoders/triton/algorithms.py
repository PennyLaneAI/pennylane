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

from .posteriors import _norm_min_sum_posteriors, _sum_product_posteriors
from .postprocess import _apply_postprocess


@triton.jit
def _decode_one(
    syndrome,
    H: tl.constexpr,
    bp_variant: tl.constexpr = "sum_product",
    postprocess: tl.constexpr = "hard",
    prob: tl.constexpr = 0.1,
    NITER: tl.constexpr = 10,
    ALPHA: tl.constexpr = 0.75,
):
    """Decode one syndrome into a correction."""
    NCHECKS: tl.constexpr = len(H)
    NVARS: tl.constexpr = len(H[0])
    tl.static_assert(
        (bp_variant == "sum_product") or (bp_variant == "norm_min_sum"),
        "unknown belief-propagation variant",
    )
    tl.static_assert(NCHECKS <= 64, "decoder supports <= 64 check bits")
    tl.static_assert(NVARS <= 64, "decoder supports <= 64 variable bits")

    syndrome = tl.cast(syndrome, tl.uint64)

    if bp_variant == "sum_product":
        P = _sum_product_posteriors(syndrome, H, prob, NITER)
    else:
        P = _norm_min_sum_posteriors(syndrome, H, prob, NITER, ALPHA)
    return _apply_postprocess(P, syndrome, postprocess=postprocess)
