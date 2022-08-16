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
This module contains the Identity operation that is common to both
cv and qubit computing paradigms in PennyLane.
"""
import numpy as np
from scipy import sparse

from pennylane.operation import CVObservable, Operation


class Identity(CVObservable, Operation):
    r"""pennylane.Identity(wires)
    The identity observable :math:`\I`.

    The expectation of this observable

    .. math::
        E[\I] = \text{Tr}(\I \rho)

    corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.
    """
    num_params = 0
    num_wires = 1
    """int: Number of wires that the operator acts on."""

    grad_method = None
    """Gradient computation method."""

    _queue_category = "_ops"

    ev_order = 1

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "I"

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Identity.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Identity.compute_eigvals())
        [ 1 1]
        """
        return np.array([1, 1])

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Identity.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Identity.compute_matrix())
        [[1. 0.]
         [0. 1.]]
        """
        return np.eye(2)

    @staticmethod
    def compute_sparse_matrix(*params, **hyperparams):
        return sparse.csr_matrix([[1, 0], [0, 1]])

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([1, 0, 0])

    @staticmethod
    def compute_diagonalizing_gates(wires):  # pylint: disable=arguments-differ,unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Identity.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> qml.Identity.compute_diagonalizing_gates(wires=[0])
        []
        """
        return []

    @staticmethod
    def compute_decomposition(wires=None):  # pylint:disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Identity.decomposition`.

        Args:
            wires (Any, Wires): A single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Identity.compute_decomposition(wires=0)
        []

        """
        return []

    @staticmethod
    def identity_op(*params):
        """Alias for matrix representation of the identity operator."""
        return Identity.compute_matrix(*params)

    def adjoint(self):
        return Identity(wires=self.wires)

    def pow(self, _):
        return [Identity(wires=self.wires)]
