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
"""Contains the MultiX template."""

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import PauliX, cond
from pennylane.wires import Wires


class MultiX(Operator2):
    r"""
    Conditionally applies PauliX gates according to a bitstring.

    ``MultiX`` applies ``PauliX'' wire `i` when the `i`-th bit of ``bitstring``
    is :math:`b_i=1`, and otherwise applies the identity to wire `i`.

    Args:
        bitstring: A one-dimensional array containing either `1` or `0` entries.
        wires: The wires onto which ``MultiX`` acts. The number of wires used
               must match the length of ``bitstring``.
    """

    dynamic_argnames = "bitstring"
    wire_argnames = "wires"

    def __init__(self, bitstring, wires):

        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)

        MultiX._validate_inputs(bitstring, wires)

        super().__init__(bitstring, wires)

    @property
    def num_wires(self):
        """Returns the number of wires the operation acts on."""
        return len(self.wires)

    @staticmethod
    def _canonicalize_inputs(bitstring, wires):
        """Canonicalize types for arguments bitstring and wires."""

        if isinstance(bitstring, (list, tuple)):
            bitstring = math.array(bitstring)
        wires = Wires(wires)

        return (bitstring, wires)

    @staticmethod
    def _validate_inputs(bitstring, wires):
        """Validate the bitstring shapes, values, and length matching the length of wires."""

        if math.ndim(bitstring) != 1:
            raise ValueError("The bitstring argument must be a one-dimensional array.")

        if len(wires) == 0:
            raise ValueError("The wires arugment must contain at least one wire.")

        bitstring_length = math.shape(bitstring)[0]
        if bitstring_length != len(wires):
            raise ValueError("The bitstring and wires arguments must have equal lengths.")

        if not math.all(math.logical_or(bitstring == 0, bitstring == 1)):
            raise ValueError("The bitstring must contain only binary 0 and 1 values.")

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_matrix(bitstring, wires):
        r"""
        Representation of a MultiX operator as a concrete matrix in the computational basis.

        Args:
            bitstring: A one-dimensional array containing either `1` or `0` entries.
            wires: The wires onto which ``MultiX`` acts. The number of wires used
                   must match the length of ``bitstring``.
        Returns:
            Concrete matrix representing the operator MultiX(bitstring, wires).
        """
        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)
        MultiX._validate_inputs(bitstring, wires)

        identity = math.eye(2, like=bitstring)
        pauli_x = math.convert_like(PauliX.compute_matrix(), bitstring)

        matrix = math.ones((1, 1), like=bitstring)
        for i in range(len(wires)):
            local_matrix = pauli_x if bitstring[i] == 1 else identity
            matrix = math.kron(matrix, local_matrix)

        return matrix


# Resources function for MultiX
def _multix_resources(bitstring, wires):  # pylint: disable=unused-argument
    # The total number of PauliX gates used depends on the bitstring.
    # Specifically, sum(bitstring) gates are used by MultiX, not len(bitstring).
    # However, if bitrsring is an AbstractArray, only the shape of bitstring is known.
    # Therefore, the resource count can only supply the *worst-case* scenario instead.
    # Hence why exact=False is used when registering the resource.
    return {PauliX: len(wires)}


# Decomposition function for MultiX
@register_resources(_multix_resources, exact=False)
def _multix_decomposition(bitstring, wires):

    @for_loop(0, len(wires), 1)
    def locally_apply_paulix(i):
        cond(bitstring[i], PauliX)(wires=wires[i])

    locally_apply_paulix()  # pylint: disable=no-value-for-parameter


add_decomps(MultiX, _multix_decomposition)
