# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the template for the Fast Approximate BLock Encoding (FABLE) technique.
"""
import warnings
import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires


class FABLE(Operation):
    r"""
    Construct a unitary with the fast approximate block encoding method.

    The FABLE method allows to simplify block encoding circuits without reducing accuracy,
    for matrices of specific structure [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_].


    Args:
        input_matrix (tensor_like): an :math:`(N \times N)` matrix to be encoded,
            where :math:`N` should have dimension equal to :math:`2^n` where :math:`n` is an integer
        tol (float): rotation gates that have an angle value smaller than this tolerance are removed
        id (str or None): string representing the operation (optional)

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    We can define a matrix and a block-encoding circuit as follows:

    .. code-block:: python

        input_matrix= np.array([[0.1,0.2],[0.2,0.1]])
        dev = qml.device('default.qubit')
        @qml.qnode(dev)
        def example_circuit():
            qml.FABLE(input_matrix, wires=range(3), tol=0)
            return qml.state()

    We can see that the input_matrix has been block encoded in the matrix of the circuit:

    >>> ancilla = [0]
    >>> s = int(qml.math.ceil(qml.math.log2(max(len(input_matrix), len(input_matrix[0])))))
    >>> wires_i = list(range(1, 1 + s))
    >>> wires_j = list(range(1 + s, 1 + 2 * s))
    >>> wire_order = ancilla + wires_i[::-1] + wires_j[::-1]
    >>> expected = (
            2**s
            * qml.matrix(example_circuit, wire_order=wire_order)().real[0 : 2**s, 0 : 2**s]
        )
    ... print(f"Block-encoded matrix:\n{expected}", "\n")
    Block-encoded matrix:
    [[0.1 0.2]
    [0.2 0.1]]

    .. note::
        By default it is assumed that the matrix is an :math:`(N \times N)` square matrix,
        where :math:`N` is a power of 2. However, for matrices of arbitrary size,
        we add zeros to reach the correct dimension. It is also assumed that the values
        of the input matrix are within [-1, 1]. Apply a subnormalization factor if needed.
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, input_matrix, wires, tol=0, id=None):
        wires = Wires(wires)

        if not qml.math.is_abstract(input_matrix):
            if qml.math.any(qml.math.iscomplex(input_matrix)):
                raise ValueError("Support for imaginary values has not been implemented.")

            alpha = qml.math.linalg.norm(qml.math.ravel(input_matrix), np.inf)
            if alpha > 1:
                raise ValueError(
                    "The subnormalization factor should be lower than 1."
                    + "Ensure that the values of the input matrix are within [-1, 1]."
                )
        else:
            if tol != 0:
                raise ValueError(
                    "JIT is not supported for tolerance values greater than 0. Set tol = 0 to run."
                )

        row, col = qml.math.shape(input_matrix)
        if row != col:
            warnings.warn(
                f"The input matrix should be of shape NxN, got {input_matrix.shape}."
                + "Zeroes were padded automatically."
            )
            dimension = max(row, col)
            input_matrix = qml.math.pad(input_matrix, ((0, dimension - row), (0, dimension - col)))
            row, col = qml.math.shape(input_matrix)

        n = int(qml.math.ceil(qml.math.log2(col)))

        if col < 2**n:
            input_matrix = qml.math.pad(input_matrix, ((0, 2**n - col), (0, 2**n - col)))
            col = 2**n
            warnings.warn(
                "The input matrix should be of shape NxN, where N is a power of 2."
                + f"Zeroes were padded automatically. Input is now of shape {input_matrix.shape}."
            )

        if len(wires) != 2 * n + 1:
            raise ValueError(f"Number of wires is incorrect, expected {2*n+1} but got {len(wires)}")

        self._hyperparameters = {"tol": tol}

        super().__init__(input_matrix, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(input_matrix, wires, tol=0):  # pylint:disable=arguments-differ
        r"""Sequence of gates that represents the efficient circuit produced by the FABLE technique

        Args:
            input_matrix (tensor_like): an :math:`(N \times N)` matrix to be encoded
            wires (Any or Iterable[Any]): wires that the operator acts on
            tol (float): tolerance

        Returns:
            list[.Operator]: list of gates for efficient circuit
        """
        op_list = []
        alphas = qml.math.arccos(input_matrix).flatten()
        thetas = compute_theta(alphas)

        ancilla = [wires[0]]
        wires_i = wires[1 : 1 + len(wires) // 2]
        wires_j = wires[1 + len(wires) // 2 : len(wires)]

        code = gray_code((2 * qml.math.log2(len(input_matrix))))
        n_selections = len(code)

        control_wires = [
            int(qml.math.log2(int(code[i], 2) ^ int(code[(i + 1) % n_selections], 2)))
            for i in range(n_selections)
        ]
        wire_map = dict(enumerate(wires_j + wires_i))

        for w in wires_i:
            op_list.append(qml.Hadamard(w))

        nots = {}
        for theta, control_index in zip(thetas, control_wires):
            if qml.math.is_abstract(theta):
                for c_wire in nots:
                    op_list.append(qml.CNOT(wires=[c_wire] + ancilla))
                op_list.append(qml.RY(2 * theta, wires=ancilla))
                nots[wire_map[control_index]] = 1
            else:
                if abs(2 * theta) > tol:
                    for c_wire in nots:
                        op_list.append(qml.CNOT(wires=[c_wire] + ancilla))
                    op_list.append(qml.RY(2 * theta, wires=ancilla))
                    nots = {}
                if wire_map[control_index] in nots:
                    del nots[wire_map[control_index]]
                else:
                    nots[wire_map[control_index]] = 1

        for c_wire in nots:
            op_list.append(qml.CNOT([c_wire] + ancilla))

        for w_i, w_j in zip(wires_i, wires_j):
            op_list.append(qml.SWAP(wires=[w_i, w_j]))

        for w in wires_i:
            op_list.append(qml.Hadamard(w))

        return op_list
