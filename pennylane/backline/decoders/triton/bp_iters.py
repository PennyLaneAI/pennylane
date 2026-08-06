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

# ========== Math Utils =================
@triton.jit
def _get_syndrome_signs(syndrome, NCHECKS: tl.constexpr):
    """Convert syndrome bits to bipolar signs."""
    s = ()
    for i in tl.static_range(NCHECKS):
        s += (tl.where(((syndrome >> i) & 1) != 0, -1.0, 1.0),)
    return s


@triton.jit
def _bp_tanh_half(x):
    """Compute tanh(x / 2) from exponentials."""
    e = tl.exp(x)
    return (e - 1.0) / (e + 1.0)


@triton.jit
def _bp_c2v_msg(ssign, prod, EPS: tl.constexpr = 1e-9):
    """Compute a numerically bounded check-to-variable message."""
    hi = 1.0 - EPS
    p = tl.maximum(-hi, tl.minimum(prod, hi))
    return ssign * tl.log((1.0 + p) / (1.0 - p))


@triton.constexpr_function
def _llr_from_p(p):
    """Convert an error probability to a log-likelihood ratio."""
    return math.log((1.0 - p) / p)
