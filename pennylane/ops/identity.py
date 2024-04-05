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
from functools import lru_cache
from scipy import sparse

import pennylane as qml
from pennylane.operation import AnyWires, AllWires, CVObservable, Operation


class Identity(CVObservable, Operation):
    r"""
    The Identity operator

    The expectation of this observable

    .. math::
        E[\I] = \text{Tr}(\I \rho)

    .. seealso:: The equivalent short-form alias :class:`~I`

    Args:
        wires (Iterable[Any] or Any): Wire label(s) that the identity acts on.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.
    """

    num_params = 0
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    grad_method = None
    """Gradient computation method."""

    _queue_category = "_ops"

    ev_order = 1

    def _flatten(self):
        return tuple(), (self.wires, tuple())

    def __init__(self, wires=None, id=None):
        super().__init__(wires=[] if wires is None else wires, id=id)
        self._hyperparameters = {"n_wires": len(self.wires)}
        self._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({}): 1.0})

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "I"

    def __repr__(self):
        """String representation."""
        if len(self.wires) == 0:
            return "I()"
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"I('{wire}')"
        return f"I({wire})"

    @property
    def name(self):
        return "Identity"

    @staticmethod
    def compute_eigvals(n_wires=1):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.I.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.I.compute_eigvals())
        [ 1 1]
        """
        return qml.math.ones(2**n_wires)

    @staticmethod
    @lru_cache()
    def compute_matrix(n_wires=1):  # pylint: disable=arguments-differ
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
        return qml.math.eye(int(2**n_wires))

    @staticmethod
    @lru_cache()
    def compute_sparse_matrix(n_wires=1):  # pylint: disable=arguments-differ
        return sparse.eye(int(2**n_wires), format="csr")

    def matrix(self, wire_order=None):
        n_wires = len(wire_order) if wire_order else len(self.wires)
        return self.compute_matrix(n_wires=n_wires)

    @staticmethod
    def _heisenberg_rep(p):
        return qml.math.array([1, 0, 0])

    @staticmethod
    def compute_diagonalizing_gates(
        wires, n_wires=1
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

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
    def compute_decomposition(wires, n_wires=1):  # pylint:disable=arguments-differ,unused-argument
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
        return I.compute_matrix(*params)

    def adjoint(self):
        return I(wires=self.wires)

    def pow(self, _):
        return [I(wires=self.wires)]


I = Identity
r"""The Identity operator

The expectation of this observable

.. math::
    E[\I] = \text{Tr}(\I \rho)

.. seealso:: The equivalent long-form alias :class:`~Identity`

Args:
    wires (Iterable[Any] or Any): Wire label(s) that the identity acts on.
    id (str): custom label given to an operator instance,
        can be useful for some applications where the instance has to be identified.

Corresponds to the trace of the quantum state, which in exact
simulators should always be equal to 1.
"""


class GlobalPhase(Operation):
    r"""A global phase operation that multiplies all components of the state by :math:`e^{-i \phi}`.

    **Details:**

    * Number of wires: All (the operation acts on all wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        phi (TensorLike): the global phase
        wires (Iterable[Any] or Any): unused argument - the operator is applied to all wires
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi=None, return_state=False):
            qml.X(0)
            if phi:
                qml.GlobalPhase(phi)
            if return_state:
                return qml.state()
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

    The circuit yields the same expectation values with and without the global phase:

    >>> circuit()
    (tensor(-1., requires_grad=True), tensor(1., requires_grad=True))
    >>> circuit(phi=0.123)
    (tensor(-1., requires_grad=True), tensor(1., requires_grad=True))

    However, the states of the two systems differ by a global phase factor:

    >>> circuit(return_state=True)
    tensor([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j], requires_grad=True)
    >>> circuit(return_state=True, phi=0.123)
    tensor([0.        +0.j        , 0.        +0.j        ,
            0.99244503-0.12269009j, 0.        +0.j        ], requires_grad=True)

    The operator can be applied with a control to create a relative phase between terms:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.ctrl(qml.GlobalPhase(0.123), 0)
            return qml.state()

        >>> circuit()
        tensor([0.70710678+0.j        , 0.        +0.j        ,
                0.70176461-0.08675499j, 0.        +0.j        ], requires_grad=True)


    """

    grad_method = "A"
    num_params = 1
    num_wires = AllWires
    """int: Number of wires that the operator acts on."""

    def __init__(self, phi, wires=None, id=None):
        super().__init__(phi, wires=[] if wires is None else wires, id=id)

    @staticmethod
    def compute_eigvals(phi, n_wires=1):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.GlobalPhase.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> qml.GlobalPhase.compute_eigvals(np.pi/2)
        array([6.123234e-17+1.j, 6.123234e-17+1.j])
        """
        return qml.math.exp(-1j * phi) * qml.math.ones(2**n_wires)

    @staticmethod
    def compute_matrix(phi, n_wires=1):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).
        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.GlobalPhase.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> qml.GlobalPhase.compute_matrix(np.pi/4, n_wires=1)
        array([[0.70710678-0.70710678j, 0.        +0.j        ],
               [0.        +0.j        , 0.70710678-0.70710678j]])
        """
        interface = qml.math.get_interface(phi)
        if interface == "tensorflow":
            return qml.math.exp(-1j * qml.math.cast(phi, complex)) * qml.math.eye(int(2**n_wires))
        return qml.math.exp(-1j * qml.math.cast(phi, complex)) * qml.math.eye(
            int(2**n_wires), like=interface
        )

    @staticmethod
    def compute_sparse_matrix(phi, n_wires=1):  # pylint: disable=arguments-differ
        return qml.math.exp(-1j * phi) * sparse.eye(int(2**n_wires), format="csr")

    @staticmethod
    def compute_diagonalizing_gates(
        phi, wires, n_wires=1
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.GlobalPhase.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> qml.GlobalPhase.compute_diagonalizing_gates(1.2, wires=[0])
        []
        """
        return []

    @staticmethod
    def compute_decomposition(phi, wires=None):  # pylint:disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. note::

            The ``GlobalPhase`` operation decomposes to an empty list of operations.
            Support for global phase
            was added in v0.33 and was ignored in earlier versions of PennyLane. Setting
            global phase to decompose to nothing allows existing devices to maintain
            current support for operations which now have ``GlobalPhase`` in the
            decomposition pipeline.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.GlobalPhase.decomposition`.

        Args:
            phi (TensorLike): the global phase
            wires (Iterable[Any] or Any): unused argument - the operator is applied to all wires

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.GlobalPhase.compute_decomposition(1.23)
        []

        """
        return []

    def matrix(self, wire_order=None):
        n_wires = len(wire_order) if wire_order else len(self.wires)
        return self.compute_matrix(self.data[0], n_wires=n_wires)

    def adjoint(self):
        return GlobalPhase(-1 * self.data[0], self.wires)

    def pow(self, z):
        return [GlobalPhase(z * self.data[0], self.wires)]

    def generator(self):
        # needs to return a new_opmath instance regardless of whether new_opmath is enabled, because
        # it otherwise can't handle Identity with no wires, see PR #5194
        return qml.s_prod(-1, qml.I(self.wires))
