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
from pennylane.ops import Hadamard, PauliX, cond
from pennylane.typing import AbstractArray, AbstractWires, Int, TensorLike, Wire
from pennylane.wires import Wires, WiresLike


class MultiX(Operator2):
    r"""
    Conditionally applies PauliX gates according to a bitstring.

    For a bitstring :math:`\mathbf{b} = (b_0, \ldots, b_{n-1})`,
    ``MultiX`` applies the operator

    .. math::

        \operatorname{MultiX}(\mathbf{b}) = X^{b_0} \otimes X^{b_1} \otimes
        \cdots \otimes X^{b_{n-1}}, \qquad b_i \in \{0, 1\},

    where :math:`X^0 = I` and :math:`X^1 = X`.
    The position of each bit corresponds to the wire at the same position in ``wires``.

    Args:
        bitstring (TensorLike): A one-dimensional array containing only ``0`` or ``1`` entries.
        wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
            match the length of ``bitstring``.

    **Examples**

    The bitstring ``[1, 0, 1]`` applies a :class:`~.PauliX` gate to the first and third
    wires. This can be seen directly from the decomposition:

    >>> import pennylane as qp
    >>> op = qp.MultiX([1, 0, 1], wires=["a", "b", "c"])
    >>> op.decomposition()
    [X('a'), X('c')]

    Applying the same operation to :math:`|000\rangle` produces the computational basis
    state :math:`|101\rangle`:

    >>> dev = qp.device("default.qubit", wires=3)
    >>> @qp.qnode(dev)
    ... def circuit():
    ...     qp.MultiX([1, 0, 1], wires=range(3))
    ...     return qp.probs(wires=range(3))
    >>> circuit()
    array([0., 0., 0., 0., 0., 1., 0., 0.])
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
        """Return the number of wires on which the operation acts."""
        return len(self.wires)

    @staticmethod
    def _canonicalize_inputs(bitstring: TensorLike, wires: WiresLike) -> tuple[TensorLike, Wires]:
        """Canonicalize the ``bitstring`` and ``wires`` arguments."""

        if isinstance(bitstring, (list, tuple)):
            bitstring = math.array(bitstring)
        wires = Wires(wires)

        return (bitstring, wires)

    @staticmethod
    def _validate_inputs(
        bitstring: AbstractArray | TensorLike, wires: AbstractWires | WiresLike
    ) -> None:
        """Validate the bitstring's shape and values and its length relative to the wires."""

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
        Representation of the operator as a canonical matrix in the computational basis.
        Assumes the wire order is the order provided by ``wires``.

        Args:
            bitstring (TensorLike): A one-dimensional array containing only ``0`` or ``1`` entries.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.

        Returns:
            TensorLike: The canonical matrix representing ``MultiX(bitstring, wires)``.
        """
        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)
        MultiX._validate_inputs(bitstring, wires)

        # The local matrices are either identity or pauli_x matrices
        identity = math.eye(2, like=bitstring)
        pauli_x = math.convert_like(PauliX.compute_matrix(), bitstring)

        matrix = math.ones((1, 1), like=bitstring)
        for i in range(len(wires)):
            local_matrix = identity + bitstring[i] * (pauli_x - identity)
            matrix = math.kron(matrix, local_matrix)

        return matrix

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_eigvals(bitstring: TensorLike, wires: WiresLike) -> TensorLike:
        r"""
        Eigenvalues of the operator.

        Args:
            bitstring (TensorLike): A one-dimensional array containing only ``0`` or ``1`` entries.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.

        Returns:
            TensorLike: The eigenvalues of ``MultiX(bitstring, wires)``.
        """
        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)
        MultiX._validate_inputs(bitstring, wires)

        identity_eigvals = math.convert_like([1, 1], bitstring)
        pauli_x_eigvals = math.convert_like([1, -1], bitstring)

        eigvals = math.ones(1, like=bitstring)
        for i in range(len(wires)):
            local_eigvals = identity_eigvals + bitstring[i] * (pauli_x_eigvals - identity_eigvals)
            eigvals = math.kron(eigvals, local_eigvals)

        return eigvals

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_diagonalizing_gates(bitstring: TensorLike, wires: WiresLike):
        r"""
        A sequence of local Hadamard gates that diagonalizes a sequence of local PauliX gates.

        Args:
            bitstring (TensorLike): A one-dimensional array containing only ``0`` or ``1`` entries.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.

        Returns:
            A Hadamard gate for each position in ``bitstring`` containing a 1.
        """
        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)
        MultiX._validate_inputs(bitstring, wires)

        return [Hadamard(wire) for wire in wires]

    def adjoint(self) -> "MultiX":
        """Return the adjoint of the operator."""
        return MultiX(self.bitstring, wires=self.wires)


# Resources function for MultiX
def _multix_resources(bitstring: TensorLike, wires: WiresLike):  # pylint: disable=unused-argument
    # The total number of PauliX gates used depends on the bitstring.
    # Specifically, sum(bitstring) gates are used by MultiX, not len(bitstring).
    # However, if bitstring is an AbstractArray, only the shape of bitstring is known.
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
