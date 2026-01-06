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
# pylint: disable=no-value-for-parameter
import warnings
from collections import Counter

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import CNOT, RY, SWAP, Hadamard, cond
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane.wires import Wires


class FABLE(Operation):
    r"""
    Construct a unitary with the fast approximate block encoding method.

    The FABLE method allows to simplify block encoding circuits without reducing accuracy,
    for matrices of specific structure [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_].


    Args:
        input_matrix (tensor_like): a :math:`(2^n \times 2^n)` matrix to be encoded,
            where :math:`n` is an integer
        wires (Iterable[int, str], Wires): the wires the operation acts on. The number of wires can
            be computed as :math:`(2 \times n + 1)`.
        tol (float): rotation gates that have an angle value smaller than this tolerance are removed
        id (str or None): string representing the operation (optional)

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    We can define a matrix and a block-encoding circuit as follows:

    .. code-block:: python

        input_matrix = np.array([[0.1, 0.2],[0.3, -0.2]])
        dev = qml.device('default.qubit', wires=3)
        @qml.qnode(dev)
        def example_circuit():
            qml.FABLE(input_matrix, wires=range(3), tol=0)
            return qml.state()

    We can see that the input matrix has been block encoded in the matrix of the circuit:

    >>> s = int(np.ceil(np.log2(max(len(input_matrix), len(input_matrix[0])))))
    >>> expected = 2**s * qml.matrix(example_circuit)().real[0 : 2**s, 0 : 2**s]
    >>> print(f"Block-encoded matrix:\n{expected}")
    Block-encoded matrix:
    [[ 0.1  0.2]
     [ 0.3 -0.2]]

    .. note::
        FABLE can be implemented for matrices of arbitrary shape and size.
        When given a :math:`(N \times M)` matrix, the matrix is padded with zeroes
        until it is of :math:`(N \times N)` dimension, where :math:`N` is equal to :math:`2^n`,
        and :math:`n` is an integer. It is also assumed that the values
        of the input matrix are within :math:`[-1, 1]`. Apply a subnormalization factor if needed.
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    resource_keys = {"wires", "thetas", "control_wires", "tol"}

    def __init__(self, input_matrix, wires, tol=0, id=None):
        wires = Wires(wires)

        if not math.is_abstract(input_matrix):
            if math.any(math.iscomplex(input_matrix)):
                raise ValueError("Support for imaginary values has not been implemented.")

            alpha = math.linalg.norm(math.ravel(input_matrix), np.inf)
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

        row, col = math.shape(input_matrix)
        if row != col:
            warnings.warn(
                f"The input matrix should be of shape NxN, got {input_matrix.shape}."
                + "Zeroes were padded automatically."
            )
            dimension = max(row, col)
            input_matrix = math.pad(input_matrix, ((0, dimension - row), (0, dimension - col)))
            row, col = math.shape(input_matrix)
        n = int(math.ceil(math.log2(col)))
        if n == 0:  ### For edge case where someone puts a 1x1 array.
            n = 1
        if col < 2**n:
            input_matrix = math.pad(input_matrix, ((0, 2**n - col), (0, 2**n - col)))
            col = 2**n
            warnings.warn(
                "The input matrix should be of shape NxN, where N is a power of 2."
                + f"Zeroes were padded automatically. Input is now of shape {input_matrix.shape}."
            )

        if len(wires) != 2 * n + 1:
            raise ValueError(f"Number of wires is incorrect, expected {2*n+1} but got {len(wires)}")

        self._hyperparameters = {"tol": tol}

        super().__init__(input_matrix, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        code = gray_code(int(2 * math.log2(len(self.parameters[0]))))
        control_wires = np.log2(code ^ np.roll(code, -1)).astype(int)
        alphas = math.arccos(self.parameters).flatten()
        thetas = compute_theta(alphas)
        return {
            "wires": self.wires,
            "thetas": thetas,
            "control_wires": control_wires,
            "tol": self.hyperparameters["tol"],
        }

    @staticmethod
    def compute_decomposition(input_matrix, wires, tol=0):  # pylint:disable=arguments-differ
        r"""Sequence of gates that represents the efficient circuit produced by the FABLE technique

        Args:
            input_matrix (tensor_like): an :math:`(N \times N)` matrix to be encoded
            wires (Any or Iterable[Any]): wires that the operator acts on
            tol (float): rotation gates that have an angle value smaller than this tolerance are removed

        Returns:
            list[.Operator]: list of gates for efficient circuit
        """
        alphas = math.arccos(input_matrix).flatten()
        thetas = compute_theta(alphas)

        auxiliary = [wires[0]]
        wires_i = wires[1 : 1 + len(wires) // 2][::-1]
        wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]

        code = gray_code(int(2 * math.log2(len(input_matrix))))
        control_wires = np.log2(code ^ np.roll(code, -1)).astype(int)

        wire_map = dict(enumerate(wires_j + wires_i))

        op_list = [Hadamard(w) for w in wires_i]

        nots = {}
        for theta, control_index in zip(thetas, control_wires):
            if math.is_abstract(theta):
                for c_wire in nots:
                    op_list.append(CNOT(wires=[c_wire] + auxiliary))
                op_list.append(RY(2 * theta, wires=auxiliary))
                nots = {}
                nots[wire_map[control_index]] = 1
                continue

            if math.abs(2 * theta) > tol:
                for c_wire in nots:
                    op_list.append(CNOT(wires=[c_wire] + auxiliary))
                op_list.append(RY(2 * theta, wires=auxiliary))
                nots = {}
            if wire_map[control_index] in nots:
                del nots[wire_map[control_index]]
            else:
                nots[wire_map[control_index]] = 1

        for c_wire in nots:
            op_list.append(CNOT([c_wire] + auxiliary))

        for w_i, w_j in zip(wires_i, wires_j):
            op_list.append(SWAP(wires=[w_i, w_j]))

        for w in wires_i:
            op_list.append(Hadamard(w))

        return op_list


def _fable_resources(wires, thetas, control_wires, tol):

    resources = Counter({resource_rep(Hadamard): len(wires) // 2 * 2})

    wires_i = wires[1 : 1 + len(wires) // 2][::-1]
    wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]

    wire_map = dict(enumerate(wires_j + wires_i))

    nots = {}
    for theta, control_index in zip(thetas, control_wires):

        if math.abs(2 * theta) > tol:
            for _ in nots:
                resources[resource_rep(CNOT)] += 1
            resources[resource_rep(RY)] += 1
            nots = {}

        if wire_map[control_index] in nots:
            del nots[wire_map[control_index]]
        else:
            nots[wire_map[control_index]] = 1

    for _ in range(len(nots.keys())):
        resources[resource_rep(CNOT)] += 1

    resources[resource_rep(SWAP)] = min(len(wires_i), len(wires_j))

    return dict(resources)


@register_resources(_fable_resources)
def _fable_decomposition(input_matrix, wires, tol=0):
    alphas = math.arccos(input_matrix).flatten()
    thetas = compute_theta(alphas)

    auxilliary = [wires[0]]
    wires_i = wires[1 : 1 + len(wires) // 2][::-1]
    wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]

    code = gray_code(int(2 * math.log2(len(input_matrix))))
    control_wires = np.log2(code ^ np.roll(code, -1)).astype(int)

    wire_map = dict(enumerate(wires_j + wires_i))

    @for_loop(len(wires_i))
    def wires_i_hadamard_loop(i):
        w = wires_i[i]
        Hadamard(w)

    wires_i_hadamard_loop()

    nots = {}

    @for_loop(min(len(thetas), len(control_wires)))
    def theta_controls_loop(j, nots):
        theta = thetas[j]
        control_index = control_wires[j]

        def in_tol_branch(nots):

            @for_loop(len(nots.keys()))
            def concrete_nots_loop(l, nots):
                c_wire = list(nots.keys())[l]
                CNOT(wires=[c_wire] + auxilliary)
                return nots

            concrete_nots_loop(nots)

            RY(2 * theta, wires=auxilliary)
            nots = {}

            return nots

        def trivial_branch(nots):
            return nots

        nots = cond(math.abs(2 * theta) > tol, in_tol_branch, trivial_branch)(nots)

        def del_branch(nots):
            del nots[wire_map[control_index]]
            return nots

        def add_branch(nots):
            nots[wire_map[control_index]] = 1
            return nots

        nots = cond(wire_map[control_index] in nots, del_branch, add_branch)(nots)

        return nots

    nots = theta_controls_loop(nots)

    @for_loop(len(nots.keys()))
    def outer_nots_loop(m, nots):
        c_wire = list(nots.keys())[m]
        CNOT([c_wire] + auxilliary)

    outer_nots_loop(nots)

    @for_loop(min(len(wires_i), len(wires_j)))
    def swap_loop(n):
        w_i = wires_i[n]
        w_j = wires_j[n]
        SWAP(wires=[w_i, w_j])

    swap_loop()

    wires_i_hadamard_loop()


add_decomps(FABLE, _fable_decomposition)
