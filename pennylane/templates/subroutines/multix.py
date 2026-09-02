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

import numpy as np
from scipy import sparse

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.symbolic_decomposition import self_adjoint
from pennylane.ops import Hadamard, PauliX, cond
from pennylane.ops.op_math.pow2 import pow_involutory
from pennylane.typing import AbstractArray, AbstractWires, Bool, TensorLike, Wire
from pennylane.wires import Wires, WiresLike


class MultiX(Operator2):
    r"""
    Conditionally applies PauliX gates according to a non-empty bitstring.

    For a bitstring :math:`\mathbf{b} = (b_0, \ldots, b_{n-1})`,
    ``MultiX`` applies the operator

    .. math::

        \operatorname{MultiX}(\mathbf{b}) = X^{b_0} \otimes X^{b_1} \otimes
        \cdots \otimes X^{b_{n-1}}, \qquad b_i \in \{0, 1\},

    where :math:`X^0 = I` and :math:`X^1 = X`.
    The position of each bit corresponds to the wire at the same position in ``wires``.

    Args:
        bitstring (TensorLike): A one-dimensional Boolean array. Integer arrays containing only
            ``0`` or ``1`` entries are accepted and cast to Boolean.
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

    arg_specs = {"bitstring": Bool[-1], "wires": Wire[-1]}

    grad_method = None

    def __init__(self, bitstring: TensorLike, wires: WiresLike) -> None:
        bitstring, wires = MultiX._canonicalize_validate_and_cast_inputs(bitstring, wires)
        super().__init__(bitstring, wires)

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
        return bitstring, wires

    @staticmethod
    def _validate_inputs(
        bitstring: AbstractArray | TensorLike, wires: AbstractWires | WiresLike
    ) -> None:
        """Validate the bitstring's shape and values and its length relative to the wires."""

        if math.ndim(bitstring) != 1:
            raise ValueError("The bitstring argument must be a one-dimensional array.")

        if len(wires) == 0:
            raise ValueError("The wires argument must contain at least one wire.")

        bitstring_length = math.shape(bitstring)[0]
        if bitstring_length != len(wires):
            raise ValueError("The bitstring and wires arguments must have equal lengths.")

        bitstring_dtype = (
            bitstring.dtype
            if isinstance(bitstring, AbstractArray)
            else math.get_dtype_name(bitstring)
        )

        is_bitstring_integer_or_bool = np.issubdtype(bitstring_dtype, np.integer) or np.issubdtype(
            bitstring_dtype, bool
        )

        if not is_bitstring_integer_or_bool:
            raise ValueError("The bitstring must be an integer or boolean array.")

        is_bitstring_abstract = isinstance(bitstring, AbstractArray) or math.is_abstract(bitstring)
        if not is_bitstring_abstract:
            is_bitstring_binary = math.all(math.logical_or(bitstring == 0, bitstring == 1))
            # is_bitstring_binary evaluates to True at least when all entries belong to (0, 1, False, True)
            if not is_bitstring_binary:
                raise ValueError(
                    "The bitstring must contain only integer or boolean binary (0,1,False,True) values."
                )

    @staticmethod
    def _canonicalize_validate_and_cast_inputs(
        bitstring: TensorLike, wires: WiresLike
    ) -> tuple[TensorLike, Wires]:
        """Runs the full pipeline for handling input the input arguments ``bitstring`` and ``wires`` arguments."""
        # canonicalize (standardize formats/containers)
        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)
        # validate (throw error if inputs have problems)
        MultiX._validate_inputs(bitstring, wires)
        # cast (convert bitstring to Boolean dtype)
        bitstring = (
            Bool[len(bitstring)]
            if isinstance(bitstring, AbstractArray)
            else math.cast(bitstring, bool)
        )
        return bitstring, wires

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_matrix(bitstring: TensorLike, wires: WiresLike) -> TensorLike:
        r"""
        Representation of the operator as a canonical matrix in the computational basis.
        Assumes the wire order is the order provided by ``wires``.

        Args:
            bitstring (TensorLike): A one-dimensional Boolean array. Integer arrays containing only
                ``0`` or ``1`` entries are accepted and cast to Boolean.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.

        Returns:
            TensorLike: The canonical matrix representing ``MultiX(bitstring, wires)``.
        """

        bitstring, wires = MultiX._canonicalize_validate_and_cast_inputs(bitstring, wires)

        # The local matrices are either identity or pauli_x matrices
        interface = math.get_interface(bitstring)
        identity = math.eye(2, like=interface)
        pauli_x = math.asarray(PauliX.compute_matrix(), like=interface)

        def _local_matrix(i):
            # helper function for computing the i-th local matrix according to the bitstring
            numerical_bit = math.cast(bitstring[i], int)
            return numerical_bit * pauli_x + (1 - numerical_bit) * identity

        # Here we require len(bitstring) >= 1 as validated on input above
        matrix = _local_matrix(0)
        for i in range(1, len(wires)):
            matrix = math.kron(matrix, _local_matrix(i))

        return matrix

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_sparse_matrix(
        bitstring: TensorLike, wires: WiresLike, format: str = "csr"
    ) -> sparse.spmatrix:
        r"""
        Representation of the operator as a canonical sparse matrix in the computational basis.

        Args:
            bitstring (TensorLike): A one-dimensional Boolean array. Integer arrays containing only
                ``0`` or ``1`` entries are accepted and cast to Boolean.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.
            format (str): Format of the returned SciPy sparse matrix, for example ``"csr"``.

        Returns:
            scipy.sparse.spmatrix: The canonical sparse matrix representing
            ``MultiX(bitstring, wires)``.
        """
        # Example: MultiX([1,1], wires=[0,1]) is the operator X @ X
        # In the computational basis, X @ X performs the map
        #
        #     |00>,|01>,|10>,|11> --- X @ X ---> |11>,|10>,|01>,|00>
        #
        # which just permutes the computational basis states.
        # The canonical matrix representation is therefore
        #              [[0 0 0 1],
        #      X @ X =  [0 0 1 0],
        #               [0 1 0 0],
        #               [1 0 0 0]]

        # Note MultiX(bitstring) is always a permutation matrix sending |x> to |x (+) b>
        # where x = (x0, ..., xn-1) is a basis state and b = (b0, ..., bn-1) is the bitstring.
        #
        # The only indices (row,col) where the canonical matrix for MultiX(bitstring) contains
        # a non-zero entry of '1' are when the 'column index' is the bit-wise OR '^' between
        # the 'row index' and the provided 'bitstring'. Hence the operation below
        #    flipped_states = bit_mask ^ basis_states
        #
        # Also note MultiX(bitstring) is not only Hermitian but symmetric meaning the canonical
        # matrix will equal its own transpose.
        # Therefore, row and column indices, respectively 'basis_states' and 'flipped_states',
        # can safely be considered interchangable.

        bitstring, wires = MultiX._canonicalize_validate_and_cast_inputs(bitstring, wires)

        dimension = 2 ** len(wires)
        bit_mask = sum(int(bit) << (len(wires) - index - 1) for index, bit in enumerate(bitstring))

        basis_states = np.arange(0, dimension, 1, dtype=int)
        flipped_states = bit_mask ^ basis_states
        matrix = sparse.csr_matrix(
            (np.ones(dimension, dtype=int), (basis_states, flipped_states)),
            shape=(dimension, dimension),
        )
        return matrix.asformat(format)

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_eigvals(bitstring: TensorLike, wires: WiresLike) -> TensorLike:
        r"""
        Eigenvalues of the operator.

        Args:
            bitstring (TensorLike): A one-dimensional Boolean array. Integer arrays containing only
                ``0`` or ``1`` entries are accepted and cast to Boolean.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.

        Returns:
            TensorLike: The eigenvalues of ``MultiX(bitstring, wires)``.
        """
        bitstring, wires = MultiX._canonicalize_validate_and_cast_inputs(bitstring, wires)

        interface = math.get_interface(bitstring)
        identity_eigvals = math.asarray([1, 1], like=interface)
        pauli_x_eigvals = math.asarray([1, -1], like=interface)

        def _local_eigvals(i):
            # helper function for computing the i-th local eigvals according to the bitstring
            numerical_bit = math.cast(bitstring[i], int)
            return numerical_bit * pauli_x_eigvals + (1 - numerical_bit) * identity_eigvals

        # Here we require len(bitstring) >= 1 as validated on input above
        eigvals = _local_eigvals(0)
        for i in range(1, len(wires)):
            eigvals = math.kron(eigvals, _local_eigvals(i))

        return eigvals

    @staticmethod
    # pylint: disable-next=arguments-differ
    def compute_diagonalizing_gates(bitstring: TensorLike, wires: WiresLike):
        r"""
        A sequence of local Hadamard gates that diagonalizes a sequence of local PauliX gates.

        Args:
            bitstring (TensorLike): A one-dimensional Boolean array. Integer arrays containing only
                ``0`` or ``1`` entries are accepted and cast to Boolean.
            wires (WiresLike): The wires on which ``MultiX`` acts. The number of wires must
                match the length of ``bitstring``.

        Returns:
            A Hadamard gate for each position in ``bitstring`` containing a 1.
        """
        bitstring, wires = MultiX._canonicalize_inputs(bitstring, wires)
        MultiX._validate_inputs(bitstring, wires)

        return [Hadamard(wire) for i, wire in enumerate(wires) if bitstring[i]]

    def __repr__(self) -> str:
        if not (isinstance(self.bitstring, AbstractArray) or isinstance(self.wires, AbstractWires)):
            return f"MultiX({math.cast(self.bitstring, int)}, wires={self.wires})"
        return super().__repr__()

    def adjoint(self) -> "MultiX":
        """Return the adjoint of the operator."""
        return MultiX(self.bitstring, wires=self.wires)

    def pow(self, z: int | float) -> list[Operator2]:
        # Only encodes the involutive property: MultiX^2 = I
        return super().pow(z % 2)


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

    if compiler.active() or capture.enabled():
        bitstring = math.array(bitstring, like="jax")
        wires = math.array(wires, like="jax")

    @for_loop(0, len(wires), 1)
    def locally_apply_paulix(i):
        cond(bitstring[i], PauliX)(wires=wires[i])

    locally_apply_paulix()  # pylint: disable=no-value-for-parameter


add_decomps(MultiX, _multix_decomposition)
add_decomps("Adjoint(MultiX)", self_adjoint)
add_decomps("Pow(MultiX)", pow_involutory)
