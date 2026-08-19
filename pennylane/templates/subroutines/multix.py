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

from pennylane import compiler, math
from pennylane.capture import enabled
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import PauliX, cond
from pennylane.typing import AbstractArray, AbstractWires, Int, TensorLike, Wire
from pennylane.wires import Wires, WiresLike


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

    dynamic_argnames = ("bitstring",)
    wire_argnames = ("wires",)

    arg_specs = {
        "bitstring": Int[-1],
        "wires": Wire[-1],
    }

    grad_method = None

    def __init__(self, bitstring: TensorLike, wires: WiresLike) -> None:

        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)

        MultiX._validate_inputs(bitstring, wires)

        super().__init__(bitstring, wires)

    # pylint: disable-next=arguments-differ
    def __abstract_init__(
        self, bitstring: AbstractArray | TensorLike, wires: AbstractWires | WiresLike
    ) -> None:
        super().__abstract_init__(bitstring, wires)

        MultiX._validate_inputs(self.bitstring, self.wires)

    @property
    def num_wires(self) -> int:
        """Returns the number of wires the operation acts on."""
        return len(self.wires)

    @staticmethod
    def _canonicalize_inputs(bitstring: TensorLike, wires: WiresLike) -> tuple[TensorLike, Wires]:
        """Canonicalize types for arguments bitstring and wires."""

        if isinstance(bitstring, (list, tuple)):
            bitstring = math.array(bitstring)
        wires = Wires(wires)

        return (bitstring, wires)

    @staticmethod
    def _validate_inputs(
        bitstring: AbstractArray | TensorLike, wires: AbstractWires | WiresLike
    ) -> None:
        """Validate the bitstring shapes, values, and length matching the length of wires."""

        if math.ndim(bitstring) != 1:
            raise ValueError("The bitstring argument must be a one-dimensional array.")

        if len(wires) == 0:
            raise ValueError("The wires arugment must contain at least one wire.")

        bitstring_length = math.shape(bitstring)[0]
        if bitstring_length != len(wires):
            raise ValueError("The bitstring and wires arguments must have equal lengths.")

        # ensure bitstring is either abstract or has binary entries
        is_bitstring_abstract = isinstance(bitstring, AbstractArray) or math.is_abstract(bitstring)
        if not is_bitstring_abstract:
            is_bitstring_binary = math.all(math.logical_or(bitstring == 0, bitstring == 1))
            if not is_bitstring_binary:
                raise ValueError("The bitstring must contain only binary 0 and 1 values.")

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_matrix(bitstring: TensorLike, wires: WiresLike) -> TensorLike:
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
            local_matrix = identity + bitstring[i] * (pauli_x - identity)
            matrix = math.kron(matrix, local_matrix)

        return matrix

    def adjoint(self) -> "MultiX":
        """Returns the adjoint of the operator."""
        return MultiX(self.bitstring, wires=self.wires)


# Resources function for MultiX
def _multix_resources(bitstring: TensorLike, wires: WiresLike):  # pylint: disable=unused-argument
    # The total number of PauliX gates used depends on the bitstring.
    # Specifically, sum(bitstring) gates are used by MultiX, not len(bitstring).
    # However, if bitrsring is an AbstractArray, only the shape of bitstring is known.
    # Therefore, the resource count can only supply the *worst-case* scenario instead.
    # Hence why exact=False is used when registering the resource.
    return {PauliX: len(wires)}


# Decomposition function for MultiX
@register_resources(_multix_resources, exact=False)
def _multix_decomposition(bitstring: TensorLike, wires: WiresLike) -> None:

    if compiler.active() or enabled():
        wires = math.array(wires, like="jax")

    @for_loop(0, len(wires), 1)
    def locally_apply_paulix(i):
        cond(bitstring[i], PauliX)(wires=wires[i])

    locally_apply_paulix()  # pylint: disable=no-value-for-parameter


add_decomps(MultiX, _multix_decomposition)
