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
This submodule contains the qutrit quantum operations
that do not depend on any parameters.
"""
# pylint:disable=arguments-differ
import numpy as np

from pennylane.operation import Operation, AdjointUndefinedError
from pennylane.wires import Wires
from .parametric_ops import validate_subspace

OMEGA = np.exp(2 * np.pi * 1j / 3)
ZETA = OMEGA ** (1 / 3)  # ZETA will be used as a phase for later non-parametric operations


class TShift(Operation):
    r"""TShift(wires)
    The qutrit shift operator

    The construction of this operator is based on equation 1 from
    `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_.

    .. math:: TShift = \begin{bmatrix}
                        0 & 0 & 1 \\
                        1 & 0 & 0 \\
                        0 & 1 & 0
                    \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TShift.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TShift.compute_matrix())
        [[0 0 1]
         [1 0 0]
         [0 1 0]]
        """
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TShift.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TShift.compute_eigvals())
        [ -0.5+0.8660254j -0.5-0.8660254j 1. +0.j         ]
        """
        return np.array([OMEGA, OMEGA**2, 1])

    # TODO: Add compute_decomposition once parametric ops are added.

    def pow(self, z):
        return super().pow(z % 3)


class TClock(Operation):
    r"""TClock(wires)
    Ternary Clock gate

    The construction of this operator is based on equation 1 from
    `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_.

    .. math:: TClock = \begin{bmatrix}
                        1 & 0      & 0        \\
                        0 & \omega & 0        \\
                        0 & 0      & \omega^2
                    \end{bmatrix}

    where :math:`\omega = e^{2 \pi i / 3}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TClock.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TClock.compute_matrix())
        [[ 1. +0.j         0. +0.j         0. +0.j       ]
         [ 0. +0.j        -0.5+0.8660254j  0. +0.j       ]
         [ 0. +0.j         0. +0.j        -0.5-0.8660254j]]
        """
        return np.diag([1, OMEGA, OMEGA**2])

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TClock.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TClock.compute_eigvals())
        [ 1. +0.j        -0.5+0.8660254j -0.5-0.8660254j]
        """
        return np.array([1, OMEGA, OMEGA**2])

    # TODO: Add compute_decomposition() once parametric ops are added.

    def pow(self, z):
        return super().pow(z % 3)


class TAdd(Operation):
    r"""TAdd(wires)
    The 2-qutrit controlled add gate

    The construction of this operator is based on definition 7 from
    `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_.
    It performs the controlled :class:`~.TShift` operation, and sends
    :math:`\hbox{TAdd} \vert i \rangle \vert j \rangle = \vert i \rangle \vert i + j \rangle`,
    where addition is taken modulo 3. The matrix representation is

    .. math:: TAdd = \begin{bmatrix}
                        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0
                    \end{bmatrix}

    .. note:: The first wire provided corresponds to the **control qutrit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TAdd.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TAdd.compute_matrix())
        [[1 0 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0]
         [0 0 0 1 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 1 0]
         [0 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0 0]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TAdd.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TAdd.compute_eigvals())
        [-0.5+0.8660254j -0.5-0.8660254j  1. +0.j        -0.5+0.8660254j -0.5-0.8660254j  1. +0.j         1. +0.j         1. +0.j         1. +0.j       ]
        """
        return np.array([OMEGA, OMEGA**2, 1, OMEGA, OMEGA**2, 1, 1, 1, 1])

    # TODO: Add compute_decomposition() once parametric ops are added.

    def pow(self, z):
        return super().pow(z % 3)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class TSWAP(Operation):
    r"""TSWAP(wires)
    The ternary swap operator.

    This operation is analogous to the qubit SWAP and acts on two-qutrit computational basis states
    according to :math:`TSWAP\vert i, j\rangle = \vert j, i \rangle`. Its matrix representation is

    .. math:: TSWAP = \begin{bmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            \end{bmatrix}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "TSWAP"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TSWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TSWAP.compute_matrix())
        [[1 0 0 0 0 0 0 0 0]
         [0 0 0 1 0 0 0 0 0]
         [0 0 0 0 0 0 1 0 0]
         [0 1 0 0 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 1 0]
         [0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 0 0 1]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TSWAP.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TSWAP.compute_eigvals())
        [ 1. -1.  1. -1.  1. -1.  1.  1.  1.]
        """
        return np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])

    # TODO: Add compute_decomposition()

    def pow(self, z):
        return super().pow(z % 2)

    def adjoint(self):
        return TSWAP(wires=self.wires)


class THadamard(Operation):
    r"""THadamard(wires, subspace)
    The ternary Hadamard operator

    Performs the Hadamard operation on a 2D subspace, if specified. The subspace is
    given as a keyword argument and determines which two of three single-qutrit basis states the
    operation applies to. When a subspace is not specified, the generalized Hadamard operation
    is used.

    The construction of this operator is based on section 2 of
    `Di et al. (2012) <https://arxiv.org/abs/1105.5485>`_ when the subspace is specified, and
    definition 4 and equation 5 from `Yeh et al. (2022) <https://arxiv.org/abs/2204.00552>`_
    when no subspace is specified. The operator definition of the ``subspace=None`` case is

    .. math:: \text{THadamard} = \frac{-i}{\sqrt{3}}\begin{bmatrix}
                    1 & 1 & 1 \\
                    1 & \omega & \omega^2 \\
                    1 & \omega^2 & \omega \\
                \end{bmatrix}

    where :math:`\omega = \exp(2 \pi i / 3)`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Optional[Sequence[int]]): the 2D subspace on which to apply the operation.
            This should be `None` for the generalized Hadamard.

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.THadamard(wires=0, subspace=(0, 1)).matrix()
    array([[ 0.70710678+0.j,  0.70710678+0.j,  0.        +0.j],
           [ 0.70710678+0.j, -0.70710678+0.j,  0.        +0.j],
           [ 0.        +0.j,  0.        +0.j,  1.        +0.j]])

    >>> qml.THadamard(wires=0, subspace=(0, 2)).matrix()
    array([[ 0.70710678+0.j,  0.        +0.j,  0.70710678+0.j],
           [ 0.        +0.j,  1.        +0.j,  0.        +0.j],
           [ 0.70710678+0.j,  0.        +0.j, -0.70710678+0.j]])

    >>> qml.THadamard(wires=0, subspace=(1, 2)).matrix()
    array([[ 1.        +0.j,  0.        +0.j,  0.        +0.j],
           [ 0.        +0.j,  0.70710678+0.j,  0.70710678+0.j],
           [ 0.        +0.j,  0.70710678+0.j, -0.70710678+0.j]])

    >>> qml.THadamard(wires=0, subspace=None).matrix()
    array([[ 0. -0.57735027j,  0. -0.57735027j,  0. -0.57735027j],
           [ 0. -0.57735027j,  0.5+0.28867513j, -0.5+0.28867513j],
           [ 0. -0.57735027j, -0.5+0.28867513j,  0.5+0.28867513j]])
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "TH"

    def __init__(self, wires, subspace=None):
        self._subspace = validate_subspace(subspace) if subspace is not None else None
        self._hyperparameters = {
            "subspace": self.subspace,
        }

        super().__init__(wires=wires)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This property returns the 2D subspace on which the operator acts if specified,
        or None if no subspace is defined. This subspace determines which two single-qutrit
        basis states the operator acts on. The remaining basis state is not affected by the
        operator.

        Returns:
            tuple[int] or None: subspace on which operator acts, if specified, else None
        """
        return self._subspace

    @staticmethod
    def compute_matrix(subspace=None):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.THadamard.matrix`

        Args:
            subspace (Sequence[int]): the 2D subspace on which to apply operation. This should be
            `None` for the generalized Hadamard.

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.THadamard.compute_matrix(subspace=(0, 2)))
        array([[ 0.70710678+0.j,  0.        +0.j,  0.70710678+0.j],
               [ 0.        +0.j,  1.        +0.j,  0.        +0.j],
               [ 0.70710678+0.j,  0.        +0.j, -0.70710678+0.j]])
        """

        if subspace is None:
            return (-1j / np.sqrt(3)) * np.array(
                [[1, 1, 1], [1, OMEGA, OMEGA**2], [1, OMEGA**2, OMEGA]]
            )

        mat = np.eye(3, dtype=np.complex128)

        unused_ind = list({0, 1, 2}.difference(set(subspace))).pop()

        mat[unused_ind, unused_ind] = np.sqrt(2)
        mat[subspace[0], subspace[1]] = 1
        mat[subspace[1], subspace[0]] = 1
        mat[subspace[1], subspace[1]] = -1

        return mat / np.sqrt(2)

    @property
    def has_adjoint(self):  # pylint: disable=arguments-renamed, invalid-overridden-method
        return self.subspace is not None

    def adjoint(self):
        if self.subspace is None:
            raise AdjointUndefinedError
        return THadamard(wires=self.wires, subspace=self.subspace)

    def pow(self, z):
        new_exp = z % 4 if self.subspace is None else z % 2
        return super().pow(new_exp)
