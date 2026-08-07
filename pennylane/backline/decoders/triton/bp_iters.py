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

import math

import triton
import triton.language as tl

from .utils import _bp_c2v_msg, _bp_tanh_half, _get_syndrome_signs, _llr_from_p


# Adapted from Pennylane Blog: https://pennylane.ai/demos/tutorial_bp_catalyst
@triton.jit
def _sum_product_posteriors(
    syndrome,
    H: tl.constexpr,
    prob: tl.constexpr = 0.1,
    NITER: tl.constexpr = 10,
):
    """Compute posterior LLRs with sum-product belief propagation."""
    L0: tl.constexpr = _llr_from_p(prob)
    NCHECKS: tl.constexpr = len(H)
    NVARS: tl.constexpr = len(H[0])

    s = _get_syndrome_signs(syndrome, NCHECKS)

    E = ()
    for _ in tl.static_range(NCHECKS):
        row = ()
        for _ in tl.static_range(NVARS):
            row += (0.0,)
        E += (row,)

    for _ in range(NITER):
        T = ()
        for c in tl.static_range(NCHECKS):
            row = ()
            for v in tl.static_range(NVARS):
                if H[c][v]:
                    msg = L0
                    for c2 in tl.static_range(NCHECKS):
                        if c2 != c and H[c2][v]:
                            msg += E[c2][v]
                    row += (_bp_tanh_half(msg),)
                else:
                    row += (0.0,)
            T += (row,)

        newE = ()
        for c in tl.static_range(NCHECKS):
            row = ()
            for v in tl.static_range(NVARS):
                if H[c][v]:
                    prod = 1.0
                    for v2 in tl.static_range(NVARS):
                        if v2 != v and H[c][v2]:
                            prod *= T[c][v2]
                    row += (_bp_c2v_msg(s[c], prod),)
                else:
                    row += (0.0,)
            newE += (row,)
        E = newE

    P = ()
    for v in tl.static_range(NVARS):
        post = L0
        for c in tl.static_range(NCHECKS):
            if H[c][v]:
                post += E[c][v]
        P += (post,)
    return P
