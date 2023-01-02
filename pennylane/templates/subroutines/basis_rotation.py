# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

    def __init__(self, wires, unitary_matrix, check=False, do_queue=True, id=None):

        M, N = unitary_matrix.shape
        if M != N:
            raise ValueError(
                f"The unitary matrix should be of shape NxN, got {unitary_matrix.shape}"
            )

        if check:
            umat = qml.math.toarray(unitary_matrix)
            if not np.allclose(umat @ umat.conj().T, np.eye(M, dtype=complex), atol=1e-6):
                raise ValueError("The provided transformation matrix should be unitary.")

        if len(wires) < 2:
            raise ValueError(f"This template requries at least two wires, got {len(wires)}")

        self._hyperparameters = {
            "unitary_matrix": unitary_matrix,
        }

        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(
        wires, unitary_matrix, check=False
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.BasisRotation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the operator acts on
            unitary_matrix (array): matrix specifying the basis trasformation
            check (bool): test unitarity of the provided `unitary_matrix`

        Returns:
            list[.Operator]: decomposition of the operator
        """

        M, N = unitary_matrix.shape
        if M != N:
            raise ValueError(
                f"The unitary matrix should be of shape NxN, got {unitary_matrix.shape}"
            )

        if check:
            umat = qml.math.toarray(unitary_matrix)
            if not np.allclose(umat @ umat.conj().T, np.eye(M, dtype=complex), atol=1e-4):
                raise ValueError("The provided transformation matrix should be unitary.")

        if len(wires) < 2:
            raise ValueError(f"This template requries at least two wires, got {len(wires)}")

        op_list = []
        phase_list, givens_list = givens_decomposition(unitary_matrix)

        for idx, phase in enumerate(phase_list):
            op_list.append(qml.PhaseShift(np.angle(phase), wires=wires[idx]))

        for (grot_mat, indices) in givens_list:
            theta = np.arccos(np.real(grot_mat[1, 1]))
            phi = np.angle(grot_mat[0, 0] / grot_mat[1, 1])

            op_list.append(
                qml.SingleExcitation(2 * theta, wires=[wires[indices[0]], wires[indices[1]]])
            )

            if not np.isclose(phi, 0.0):
                op_list.append(qml.PhaseShift(phi, wires=wires[indices[0]]))

        return op_list
