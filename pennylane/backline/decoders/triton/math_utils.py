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


@triton.jit
def _sign1(x):
    """Return -1 for negative values and 1 otherwise."""
    return tl.where(x < 0.0, -1.0, 1.0)


@triton.constexpr_function
def _llr_from_p(p):
    """Convert an error probability to a log-likelihood ratio."""
    return math.log((1.0 - p) / p)
