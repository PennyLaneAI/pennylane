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

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Triton decoders require installed `triton` Python package.") from exc


@triton.jit
def _sum_product_posteriors(  # pylint: disable=too-many-branches,too-many-nested-blocks
    syndrome,
    H: tl.constexpr,
    prob: tl.constexpr,
    num_iters: tl.constexpr,
):
    """Compute posterior LLRs for one packed syndrome.

    Adapted from Pennylane Blog: https://pennylane.ai/demos/tutorial_bp_catalyst

    Args:
        syndrome (u64): Packed syndrome bitmask. Bit ``i`` stores check ``i``.
        H (tuple[tuple[int]]): Binary parity-check matrix. Row ``i`` matches
            syndrome bit ``i``, and column ``j`` corresponds to qubit ``j``.
        prob (float): Prior error probability assigned to each qubit.
        num_iters (int): Number of belief-propagation iterations.

    Returns:
        tuple[float]: Posterior LLRs, one per qubit.
    """
    prior_llr: tl.constexpr = _llr_from_p(prob)
    num_checks: tl.constexpr = len(H)
    num_qubits: tl.constexpr = len(H[0])

    syndrome_signs = _get_syndrome_signs(syndrome, num_checks)

    check_to_var_msgs = ()
    for _ in tl.static_range(num_checks):
        row = ()
        for _ in tl.static_range(num_qubits):
            row += (0.0,)
        check_to_var_msgs += (row,)

    for _ in range(num_iters):
        var_to_check_msgs = ()
        for c in tl.static_range(num_checks):
            row = ()
            for v in tl.static_range(num_qubits):
                if H[c][v]:
                    message = prior_llr
                    for c2 in tl.static_range(num_checks):
                        if c2 != c and H[c2][v]:
                            message += check_to_var_msgs[c2][v]
                    row += (_tanh_half(message),)
                else:
                    row += (0.0,)
            var_to_check_msgs += (row,)

        next_check_to_var_msgs = ()
        for c in tl.static_range(num_checks):
            row = ()
            for v in tl.static_range(num_qubits):
                if H[c][v]:
                    message_product = 1.0
                    for v2 in tl.static_range(num_qubits):
                        if v2 != v and H[c][v2]:
                            message_product *= var_to_check_msgs[c][v2]
                    row += (_bp_c2v_msg(syndrome_signs[c], message_product),)
                else:
                    row += (0.0,)
            next_check_to_var_msgs += (row,)
        check_to_var_msgs = next_check_to_var_msgs

    posterior_llrs = ()
    for v in tl.static_range(num_qubits):
        posterior = prior_llr
        for c in tl.static_range(num_checks):
            if H[c][v]:
                posterior += check_to_var_msgs[c][v]
        posterior_llrs += (posterior,)
    return posterior_llrs


@triton.jit
def _get_syndrome_signs(syndrome, num_checks: tl.constexpr):
    """Convert syndrome bits into bipolar check signs.

    Args:
        syndrome (u64): Packed syndrome bitmask.
        num_checks (int): Number of checks to unpack from ``syndrome``.

    Returns:
        tuple[float]: Tuple containing ``+1.0`` for a zero bit and ``-1.0`` for
            a one bit, in least-significant-bit order.
    """
    signs = ()
    for i in tl.static_range(num_checks):
        signs += (tl.where(((syndrome >> i) & 1) != 0, -1.0, 1.0),)
    return signs


@triton.jit
def _tanh_half(value):
    """Compute ``tanh(value / 2)`` for a Triton scalar.

    Args:
        value (float): Input value.

    Returns:
        float: The value of ``tanh(value / 2)``.
    """
    half_value = 0.5 * value

    if tl.target_info.is_cuda():
        return tl.extra.cuda.libdevice.tanh(half_value)
    if tl.target_info.is_hip():
        return tl.extra.hip.libdevice.tanh(half_value)

    # overflow robust tanh formula
    exp_abs = tl.exp(-tl.abs(value))
    return tl.where(
        value >= 0,
        (1.0 - exp_abs) / (1.0 + exp_abs),
        (exp_abs - 1.0) / (exp_abs + 1.0),
    )


@triton.jit
def _bp_c2v_msg(syndrome_sign, message_product):
    """Compute a bounded check-to-variable message.

    Args:
        syndrome_sign (float): Bipolar sign derived from the packed syndrome bit.
        message_product (float): Product of neighbouring variable-to-check messages.

    Returns:
        float: Check-to-variable message in LLR form.
    """
    eps = 1e-6
    clamp_limit = 1.0 - eps
    clamped_product = tl.maximum(-clamp_limit, tl.minimum(message_product, clamp_limit))
    return syndrome_sign * tl.log((1.0 + clamped_product) / (1.0 - clamped_product))


@triton.constexpr_function
def _llr_from_p(error_probability):
    """Convert a compile-time error probability into a prior log-likelihood ratio (LLR).

    Args:
        error_probability (float): Error probability in the open interval ``(0, 1)``.

    Returns:
        float: Log-likelihood ratio ``log((1 - error_probability) / error_probability)``.
    """
    return math.log1p(-error_probability) - math.log(error_probability)
