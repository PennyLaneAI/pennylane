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

import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.qchem.givens_decomposition import givens_decomposition


# pylint: disable-msg=too-many-arguments
class BasisRotation(Operation):
    r"""Implement a circuit that provides a unitary that can be used to do an exact single-body basis rotation.

    The :class:`~.pennylane.BasisRotation` template performs the following unitary transformation :math:`U(u)` determined by the single-particle fermionic
    generators as given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ :

    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    The unitary :math:`U(u)` is implemented efficiently by performing its Givens decomposition into a sequence of
    :class:`~.PhaseShift` and :class:`~.SingleExcitation` gates using the construction scheme given in
    `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    Args:
        wires (Iterable[Any]): wires that the operator acts on
        unitary_matrix (array): matrix specifying the basis transformation
        check (bool): test unitarity of the provided `unitary_matrix`

    Raises:
        ValueError: if the provided matrix is not square.
        ValueError: if length of the wires is less than two.

    .. details::
        :title: Usage Details
        :href: usage-basis-rotation

        The :class:`~.pennylane.BasisRotation` template can be used to implement the evolution :math:`e^{iH}` where
        :math:`H = \sum_{pq} V_{pq} a^\dagger_p a_q` and :math:`V` is an :math:`N \times N` Hermitian matrix.
        When the unitary matrix :math:`u` is the transformation matrix that diagonalizes :math:`V`, the evolution is:

        .. math::

            e^{i \sum_{pq} V_{pq} a^\dagger_p a_q} = U(u)^\dagger \prod_k e^{i\lambda_k \sigma_z^k} U(u),

        where :math:`\lambda_k` denotes the eigenvalues of matrix :math:`V`, the Hamiltonian coefficients matrix.

        >>> V = np.array([[ 0.53672126+0.j        , -0.1126064 -2.41479668j],
        ...               [-0.1126064 +2.41479668j,  1.48694623+0.j        ]])
        >>> eigen_vals, eigen_vecs = np.linalg.eigh(V)
        >>> umat = eigen_vecs.T
        >>> wires = range(len(umat))
        >>> def circuit():
        ...    qml.adjoint(qml.BasisRotation(wires=wires, unitary_matrix=umat))
        ...    for idx, eigenval in enumerate(eigen_vals):
        ...        qml.RZ(eigenval, wires=[idx])
        ...    qml.BasisRotation(wires=wires, unitary_matrix=umat)
        >>> circ_unitary = qml.matrix(circuit)()
        >>> np.round(circ_unitary/circ_unitary[0][0], 3)
        tensor([[ 1.   -0.j   , -0.   +0.j   , -0.   +0.j   , -0.   +0.j   ],
                [-0.   +0.j   , -0.516-0.596j, -0.302-0.536j, -0.   +0.j   ],
                [-0.   +0.j   ,  0.35 +0.506j, -0.311-0.724j, -0.   +0.j   ],
                [-0.   +0.j   , -0.   +0.j   , -0.   +0.j   , -0.438+0.899j]], requires_grad=True)

    .. details::
        :title: Theory
        :href: theory-basis-rotation

        The overall effect of :math:`U(u)` can be viewed as performing a transformation from one basis to a new basis
        that is defined by the linear combination of fermionic ladder operators:

        .. math::

            U(u) a_p^\dagger U(u)^\dagger = b_p^\dagger,

        where :math:`a_p^\dagger` and :math:`b_p^\dagger` are the original and transformed creation operators, respectively.
        The operators :math:`a_p^\dagger` and :math:`b_p^\dagger` are related to each other by the following equation:

        .. math::

            b_p^\dagger = \sum_{q}u_{pq} a_p^\dagger.

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, wires, unitary_matrix, check=False, id=None):
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
            raise ValueError(f"This template requires at least two wires, got {len(wires)}")

        self._hyperparameters = {
            "unitary_matrix": unitary_matrix,
        }

        super().__init__(wires=wires, id=id)

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
            unitary_matrix (array): matrix specifying the basis transformation
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
            raise ValueError(f"This template requires at least two wires, got {len(wires)}")

        op_list = []
        phase_list, givens_list = givens_decomposition(unitary_matrix)

        for idx, phase in enumerate(phase_list):
            op_list.append(qml.PhaseShift(np.angle(phase), wires=wires[idx]))

        for grot_mat, indices in givens_list:
            theta = np.arccos(np.real(grot_mat[1, 1]))
            phi = np.angle(grot_mat[0, 0])

            op_list.append(
                qml.SingleExcitation(2 * theta, wires=[wires[indices[0]], wires[indices[1]]])
            )

            if not np.isclose(phi, 0.0):
                op_list.append(qml.PhaseShift(phi, wires=wires[indices[0]]))

        return op_list
