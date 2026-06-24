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
"""A modified signed out-place multiplier that expects only one signed input, together with
an unsigned input and an unsigned output."""

import pennylane as qp
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import _twos_complement_helper


def semi_signed_out_multiplier(x_wires, y_wires, output_wires, work_wires):
    r"""Multiplier of an unsigned register and a signed register into an unsigned output register

    Args:
        x_wires (WiresLike): register storing the unsigned integer :math:`x` to be multiplied
        y_wires (WiresLike): register storing the signed integer :math:`y` to be multiplied
        output_wires (WiresLike): register storing the unsigned integer :math:`z` before the
            calculation and the output :math:`(z+xy)\mod 2^k` afterwards, where :math:`k` is the
            number of wires in ``output_wires``.
        work_wires (WiresLike): work wires to use for the calculation.

    This is a very specific setup of a multiplier that is useful for the
    :func:`~.pennylane.labs.templates.trotter_vibronic` function. Note that due to the structure
    of the circuit, there is no benefit in knowing the output wires to start out in the zero state.

    **Number of work wires**

    In principle, we would only need one work wire for this function. However, in order to
    achieve the best gate counts, we require more work wires:

    - 1 work wire for the cache of the sign bit of :math:`y`,
    - :math:`m-1` work wires for the two's complement helper (containing a singly-controlled
      :class:`~.Incrementer`), where :math:`m` is ``len(y_wires)``
    - :math:`k` work wires for caching unsigned multiplication, in order to use
      :class:`~.OutMultiplier` with ``output_wires_zeroed=True``. Here, :math:`k` is
      ``len(output_wires)``.
    - :math:`k+1` work wires for the unsigned ``OutMultiplier``.
    - :math:`k-1` work wires for adding the cached multiplication into the output wires

    The two's complement helper function, the multiplier, and the addition return their work wires
    in a clean state, so they can use the same work wires. Thus, we overall
    require :math:`1+\max(m-1, 2k+1, k-1)` work wires for lowest-gate-count decompositions.
    At the cost of additional gates, the multiplier work wires can be reduced from :math:`2k+1` to
    :math:`\min(2k-1, k+m+1)` (cache + adder-based decomposition), :math:`k+1` (no cache +
    controlled add-subtract decomposition), or :math:`k` (no cache + adder-based decomposition).
    Accordingly, the achievable total work wire counts are
    :math:`\max(m, 2k+2)`, :math:`\max(m, \min(2k, k+m+2))`, :math:`\max(m, k+2)` or
    :math:`\max(m, k+1)`.

    """
    y_aux, work_wires = work_wires[0], work_wires[1:]

    # Sign extension
    qp.CNOT([y_wires[0], y_aux])

    # Take 2s complement
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # at this point the sign is only kept in the auxiliary qubit's state

    # Multiply the magnitudes into the output register
    # If y was negative, flip all output qubits before and after (unsigned) multiplication onto
    # the output wires. This effects that we are subtracting the product if y was negative, and
    # add it otherwise.
    qp.ctrl(qp.BasisState([1] * len(output_wires), output_wires), control=y_aux)
    qp.OutMultiplier(
        x_wires,
        y_wires,
        output_wires,
        work_wires=work_wires,
        output_wires_zeroed=False,
    )
    qp.ctrl(qp.BasisState([1] * len(output_wires), output_wires), control=y_aux)

    # Return input y to original state
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # Uncompute sign extension
    qp.CNOT([y_wires[0], y_aux])
