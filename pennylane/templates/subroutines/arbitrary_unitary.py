# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the ArbitraryUnitary template.
"""
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot

_PAULIS = ["I", "X", "Y", "Z"]


def _tuple_to_word(index_tuple):
    """Convert an integer tuple to the corresponding Pauli word.

    The Pauli operators are converted as ``0 -> I``, ``1 -> X``,
    ``2 -> Y``, ``3 -> Z``.

    Args:
        index_tuple (Tuple[int]): An integer tuple describing the Pauli word

    Returns:
        str: The corresponding Pauli word
    """
    return "".join([_PAULIS[i] for i in index_tuple])


def _n_k_gray_code(n, k, start=0):
    """Iterates over a full n-ary Gray code with k digits.

    Args:
        n (int): Base of the Gray code. Needs to be greater than one.
        k (int): Number of digits of the Gray code. Needs to be greater than zero.
        start (int, optional): Optional start of the Gray code. The generated code
            will be shorter as the code does not wrap. Defaults to 0.
    """
    for i in range(start, n**k):
        codeword = [0] * k

        base_repesentation = []
        val = i

        for j in range(k):
            base_repesentation.append(val % n)
            val //= n

        shift = 0
        for j in reversed(range(k)):
            codeword[j] = (base_repesentation[j] + shift) % n
            shift += n - codeword[j]

        yield codeword


def _all_pauli_words_but_identity(num_wires):
    # Start at 1 to ignore identity
    yield from (_tuple_to_word(idx_tuple) for idx_tuple in _n_k_gray_code(4, num_wires, start=1))


class ArbitraryUnitary(Operation):
    """Implements an arbitrary unitary on the specified wires.

    An arbitrary unitary on :math:`n` wires is parametrized by :math:`4^n - 1`
    independent real parameters. This templates uses Pauli word rotations to
    parametrize the unitary.

    **Example**

    ArbitraryUnitary can be used as a building block, e.g. to parametrize arbitrary
    two-qubit operations in a circuit:

    .. code-block:: python

        def arbitrary_nearest_neighbour_interaction(weights, wires):
            qml.broadcast(unitary=ArbitraryUnitary, pattern="double", wires=wires, parameters=weights)

    Args:
        weights (tensor_like): The angles of the Pauli word rotations, needs to have length :math:`4^n - 1`
            where :math:`n` is the number of wires the template acts upon.
        wires (Iterable): wires that the template acts on
    """

    num_wires = AnyWires
    grad_method = None
    num_params = 1
    ndim_params = (1,)

    def __init__(self, weights, wires, id=None):
        shape = qml.math.shape(weights)
        dim = 4 ** len(wires) - 1
        if len(shape) not in (1, 2) or shape[-1] != dim:
            raise ValueError(
                f"Weights tensor must be of shape {(dim,)} or (batch_dim, {dim}); got {shape}."
            )

        super().__init__(weights, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(weights, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.ArbitraryUnitary.decomposition`.

        Args:
            weights (tensor_like): The angles of the Pauli word rotations, needs to have length :math:`4^n - 1`
                    where :math:`n` is the number of wires the template acts upon.
            wires (Any or Iterable[Any]): wires that the operator acts on


        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        for i, pauli_word in enumerate(_all_pauli_words_but_identity(len(wires))):
            op_list.append(PauliRot(weights[..., i], pauli_word, wires=wires))

        return op_list

    @staticmethod
    def shape(n_wires):
        """Compute the expected shape of the weights tensor.

        Args:
            n_wires (int): number of wires that template acts on
        """
        return (4**n_wires - 1,)
