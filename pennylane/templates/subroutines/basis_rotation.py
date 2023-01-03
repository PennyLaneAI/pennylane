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

# pylint: disable-msg=too-many-arguments
class BasisRotation(Operation):
    r"""Implement a circuit that provides the unitary that can be used to do an exact single-body basis rotation

    The `BasisRotation` template performs a unitary transformation :math:`U(u)` determined by the single-particle fermionic
    generators as:

    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}

    This :math:`U(u)` is implemented efficiently by performing its Givens decomposition into a sequence of
    :class:`~.PhaseShift` and :class:`~.SingleExcitation` gates using the construction scheme given by
    `W. Clements et al. <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_ (2016).

    Args:
        wires (Iterable[Any]): wires that the operator acts on
        unitary_matrix (array): matrix specifying the basis trasformation
        check (bool): test unitarity of the provided `unitary_matrix`

    Raises:
        ValueError: if the provided matrix is not square.

    .. details::

        **Usage Details**

        The `BasisRotation` template can be used to implement evolution :math:`e^{iH}` where the Hamiltonian
        :math:`H = \sum_{pq} V_{pq} a^\dagger_p a_q` and :math:`V` is an :math:`N` x :math:`N` hermitation matrix.
        The unitary matrix :math:`u` in this case will be the transformation matrix that diagonalizes :math:`V` such that:

        .. math::

            e^{i \sum_{pq} V_{pq} a^\dagger_p a_q} = U(u)^\dagger \prod_k e^{i\lambda_k \sigma_z^k} U(u),

        where the :math:`\lambda_k` are the eigenvalues of the Hamiltonian :math:`H`.

        >>> V = np.array([[ 0.53672126+0.j        , -0.1126064 -2.41479668j],
        ...               [-0.1126064 +2.41479668j,  1.48694623+0.j        ]])
        >>> eigen_vals, eigen_vecs = np.linalg.eigh(V)
        >>> umat = eigen_vecs.T
        >>> wires = range(len(umat))
        >>> dev = qml.device("default.qubit", wires=wires)
        >>> @qml.qnode(dev)
        ... def circuit():
        ...    qml.adjoint(qml.BasisRotation(wires=wires, unitary_matrix=umat))
        ...    for idx, eigenval in enumerate(eigen_vals):
        ...        qml.RZ(eigenval, wires=[idx])
        ...    qml.BasisRotation(wires=wires, unitary_matrix=umat)
        ...    return qml.state()
        >>> circ_unitary = qml.matrix(circuit)()
        >>> np.round(circ_unitary/circ_unitary[0][0], 3)
        tensor([[ 1.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
                [ 0.   +0.j   , -0.516-0.596j, -0.302-0.536j,  0.   +0.j   ],
                [ 0.   +0.j   ,  0.35 +0.506j, -0.311-0.724j,  0.   +0.j   ],
                [ 0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   , -0.438+0.899j]])

        **Theory**

        The overall effect of :math:`U(u)` can be realized as perfoming a transformation to set of new basis
        that is defined by the linear combination of fermionic ladder operators:

        .. math::

            U(u) a_p^\dagger U(u)^\dagger = b_p^\dagger,

        where :math:`a_p^\dagger` and :math:`b_p^\dagger` are the originial and transformed creation operators, respectively,
        are related to each other by the following relation:

        .. math::

            b_p^\dagger = \sum_{q}u_{pq} a_p^\dagger.

    """

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
