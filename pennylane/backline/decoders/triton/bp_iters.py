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

"""Belief-propagation helper kernels for Triton decoders."""

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
    """Compute posterior LLRs for one packed syndrome.

    Args:
        syndrome (u64): Packed syndrome bitmask. Bit ``i`` stores check ``i``.
        H (tuple[tuple[int]]): Binary parity-check matrix. Row ``i`` matches
            syndrome bit ``i``, and column ``j`` matches variable ``j``.
        prob (float): Prior error probability assigned to each variable.
        NITER (int): Number of belief-propagation iterations.

    Returns:
        tuple[float]: Posterior LLRs, one per variable.
    """
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


@triton.jit
def _get_syndrome_signs(syndrome, NCHECKS: tl.constexpr):
    """Convert syndrome bits into bipolar check signs.

    Args:
        syndrome (u64): Packed syndrome bitmask.
        NCHECKS (int): Number of checks to unpack from ``syndrome``.

    Returns:
        tuple[float]: Tuple containing ``+1.0`` for a zero bit and ``-1.0`` for
            a one bit, in least-significant-bit order.
    """
    s = ()
    for i in tl.static_range(NCHECKS):
        s += (tl.where(((syndrome >> i) & 1) != 0, -1.0, 1.0),)
    return s


@triton.jit
def _bp_tanh_half(x):
    """Compute ``tanh(x / 2)`` for a Triton scalar.

    Args:
        x (float): Input value.

    Returns:
        float: The value of ``tanh(x / 2)``.
    """
    e = tl.exp(x)
    return (e - 1.0) / (e + 1.0)


@triton.jit
def _bp_c2v_msg(ssign, prod, EPS: tl.constexpr = 1e-6):
    """Compute a bounded check-to-variable message.

    Args:
        ssign (float): Bipolar sign derived from the packed syndrome bit.
        prod (float): Product of neighbouring variable-to-check messages.
        EPS (float): Margin used to clamp ``prod`` away from ``±1``.

    Returns:
        float: Check-to-variable message in LLR form.
    """
    hi = 1.0 - EPS
    p = tl.maximum(-hi, tl.minimum(prod, hi))
    return ssign * tl.log((1.0 + p) / (1.0 - p))


@triton.constexpr_function
def _llr_from_p(p):
    """Convert a compile-time error probability into a prior LLR.

    Args:
        p (float): Error probability in the open interval ``(0, 1)``.

    Returns:
        float: Log-likelihood ratio ``log((1 - p) / p)``.
    """
    return math.log1p(-p) - math.log(p)
