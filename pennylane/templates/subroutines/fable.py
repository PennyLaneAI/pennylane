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

    The FABLE method allows to simplify block encoding circuits without reducing accuracy, for matrices of specific structure [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_].


    Args:
        A (tensor_like): an :math:`(N \times N)` matrix to be encoded, where N should have dimension equal to 2^n where n is an integer
        tol (float): rotation gates that have an angle value smaller than this tolerance are removed
        id (str or None): string representing the operation (optional)

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    We can define a matrix and a block-encoding circuit as follows:

    .. code-block:: python

        A = np.array([[-0.51192128, -0.51192128,  0.6237114 ,  0.6237114 ],
                [ 0.97041007,  0.97041007,  0.99999329,  0.99999329],
                [ 0.82429855,  0.82429855,  0.98175843,  0.98175843],
                [ 0.99675093,  0.99675093,  0.83514837,  0.83514837]])
        ancilla = ["ancilla"]
        s = int(np.log2(A.shape[0]))
        wires_i = [f"i{index}" for index in range(s)]
        wires_j = [f"j{index}" for index in range(s)]
        wire_order = ancilla + wires_i[::-1] + wires_j[::-1]
        dev = qml.device('default.qubit')
        @qml.qnode(dev)
        def example_circuit():
            qml.FABLE(A, tol=0.01)
            return qml.state()

    We can see that :math:`A` has been block encoded in the matrix of the circuit:

    >>> M = len(A) * qml.matrix(circuit, wire_order=wire_order)().real[0 : len(A), 0 : len(A)]
    ... print(f"Block-encoded matrix:\n{M}", "\n")
    Block-encoded matrix:
    [[-0.51192128 -0.51192128  0.6237114   0.6237114 ]
    [ 0.97041007  0.97041007  0.99999329  0.99999329]
    [ 0.82429855  0.82429855  0.98175843  0.98175843]
    [ 0.99675093  0.99675093  0.83514837  0.83514837]]

    .. note::
        By default it is assumed that the matrix is an NxN square matrix, where N is a power of 2. However, for matrices of arbitrary size,
        we add zeros to reach the correct dimension. It is also assumed that the values of the input matrix are within [-1, 1]. Apply a subnormalization factor if needed.
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, A, tol=0, id=None):
        if qml.math.any(qml.math.iscomplex(A)):
            raise ValueError("Support for imaginary values has not been implemented.")

        alpha = qml.math.linalg.norm(qml.math.ravel(A), np.inf)
        if alpha > 1:
            raise ValueError(
                "The subnormalization factor should be lower than 1. Ensure that the values of the input matrix are within [-1, 1]."
            )

        M, N = qml.math.shape(A)
        if M != N:
            warnings.warn(
                f"The input matrix should be of shape NxN, got {A.shape}. Zeroes were padded automatically."
            )
            k = max(M, N)
            A = qml.math.pad(A, ((0, k - M), (0, k - N)))

        n = int(qml.math.ceil(qml.math.log2(N)))
        if N < 2**n:
            A = qml.math.pad(A, ((0, 2**n - N), (0, 2**n - N)))
            N = 2**n
            warnings.warn(
                f"The input matrix should be of shape NxN, where N is a power of 2. Zeroes were padded automatically. Input is now of shape {A.shape}."
            )

        self._hyperparameters = {"tol": tol}

        ancilla = ["ancilla"]
        s = int(qml.math.log2(qml.math.shape(A)[0]))
        wires_i = [f"i{index}" for index in range(s)]
        wires_j = [f"j{index}" for index in range(s)]

        all_wires = Wires(ancilla) + Wires(wires_i) + Wires(wires_j)
        super().__init__(A, wires=all_wires, id=id)

    @staticmethod
    def compute_decomposition(A, wires, tol=0):  # pylint:disable=arguments-differ
        r"""Sequence of gates that represents the efficient circuit produced by the FABLE technique

        Args:
            A (tensor_like): an :math:`(N \times N)` matrix to be encoded
            wires (Any or Iterable[Any]): wires that the operator acts on
            tol (float): tolerance

        Returns:
            list[.Operator]: list of gates for efficient circuit
        """
        op_list = []
        alphas = qml.math.arccos(A).flatten()
        thetas = compute_theta(alphas)

        ancilla = [wires[0]]
        wires_i = wires[1 : 1 + len(wires) // 2]
        wires_j = wires[1 + len(wires) // 2 : len(wires)]

        code = gray_code(2 * qml.math.sqrt(len(A)))
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

    def _flatten(self):
        return self.data, (self._hyperparameters["tol"])

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], tol=metadata)
