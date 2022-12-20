# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the template for performing basis transformation defined by a set of fermionic ladder operators.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operation, AnyWires
from pennylane.qchem.givens_rotations import givens_decomposition


class BasisRotation(Operation):
    r"""Implement a circuit that provides the unitary that can be used to do an exact single-body basis rotation"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, wires, unitary_matrix, do_queue=True, id=None):

        M, N = unitary_matrix.shape
        if M != N:
            raise ValueError(
                f"The unitary matrix should be of shape NxN, got {unitary_matrix.shape}"
            )

        self._hyperparameters = {
            "unitary_matrix": unitary_matrix,
        }

        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, unitary_matrix):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.BasisRotation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the operator acts on
            unitary (array): matrix specifying the basis trasformation

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []
        phase_list, givens_list = givens_decomposition(unitary_matrix)

        for (grot_mat, indices) in reversed(givens_list):
            theta, phi = np.arccos(np.real(grot_mat[1, 1])), np.angle(grot_mat[0, 0])

            if not np.isclose(phi, 0.0):
                op_list.append(qml.PhaseShift(phi / np.pi, wires=wires[indices[1]]))

            op_list.append(qml.PhaseShift(0.25, wires=wires[indices[0]]))
            op_list.append(qml.PhaseShift(-0.25, wires=wires[indices[1]]))
            op_list.append(qml.exp(qml.ISWAP(wires=[wires[indices[0]], wires[indices[1]]]), theta))
            op_list.append(qml.PhaseShift(-0.25, wires=wires[indices[0]]))
            op_list.append(qml.PhaseShift(0.25, wires=wires[indices[1]]))

        for idx, phase in enumerate(phase_list):
            op_list.append(qml.PhaseShift(np.angle(phase), wires=wires[idx]))

        return op_list
