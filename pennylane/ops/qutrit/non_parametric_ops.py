import cmath
import warnings
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires

OMEGA = np.exp(2 * np.pi * 1j / 3)
ZETA = OMEGA**(1 / 3)


class QutritHadamard(Operation):
    r"""QutritHadamard(wires)
    The qutrit Hadamard operator

    .. math:: H = \frac{-i}{\sqrt{3}}\begin{bmatrix}
            1 &     1    &    1     \\
            1 &  \omega  & \omega^2 \\
            1 & \omega^2 &  \omega
        \end{bmatrix}
        \omega = \exp{2 * \pi * i / 3}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): The wire the operation acts on
    """
    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "H"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.QutritHadamard.compute_matrix())
        [[ 0. -0.57735027j  0. -0.57735027j  0. -0.57735027j]
         [ 0. -0.57735027j  0.5+0.28867513j -0.5+0.28867513j]
         [ 0. -0.57735027j -0.5+0.28867513j  0.5+0.28867513j]]
        """
        global_phase = -1j / np.sqrt(3)

        H = [[1, 1, 1],
            [1, OMEGA, OMEGA**2],
            [1, OMEGA**2, OMEGA]]

        return np.multiply(H, global_phase)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.QutritHadamard.compute_eigvals())
        [-0.-1.j  0.+1.j  1.+0.j]
        """
        return np.array([-1j, 1j, 1])

    # @staticmethod
    # def compute_decomposition(wires):
    #     r"""Representation of the operator as a product of other operators (static method).

    #     .. math:: O = O_1 O_2 \dots O_n.

    #     Args:
    #         wires (Any, Wires): Single wire that the operator acts on.

    #     Returns:
    #         list[Operator]: decomposition into lower level operations

    #     **Example:**

    #     >>> print(qml.QutritHadamard.compute_decomposition(0))
    #     [PhaseShift(1.5707963267948966, wires=[0])]

    #     """
    #     return [qml.PhaseShift(np.pi / 2, wires=wires)]

    def adjoint(self):
        op = QutritHadamard(wires=self.wires)
        op.inverse = not self.inverse
        return op


class QutritS(Operation):
    r"""QutritS(wires)
    The single-qutrit T gate

    .. math:: S = \zeta^{8} \begin{bmatrix}
                1 & 0 & 0 \\
                0 & 1 & 0 \\
                0 & 0 & \omega
            \end{bmatrix}
            \zeta = \omega^{1/3}

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

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "S"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.QutritS.compute_matrix())
        [[0.76604444-0.64278761j 0.        +0.j         0.        +0.j        ]
         [0.        +0.j         0.76604444-0.64278761j 0.        +0.j        ]
         [0.        +0.j         0.        +0.j         0.17364818+0.98480775j]]
        """
        mat = np.diag([1, 1, OMEGA])
        return np.multiply(ZETA**8, mat)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.QutritS.compute_eigvals())
        [0.76604444-0.64278761j 0.76604444-0.64278761j 0.17364818+0.98480775j]
        """
        return np.array([ZETA**8, ZETA**8, OMEGA * ZETA**8])

    #TODO: Add compute_decomposition method

    def adjoint(self):
        op = QutritS(wires=self.wires)
        op.inverse = not self.inverse
        return op


class QutritCX(Operation):
    r"""QutritCX(wires)
    The 2-qutrit controlled-X gate

    .. math:: CX = \begin{bmatrix}
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

    .. note:: The first wire provided corresponds to the **control qubit**.

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

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "CX"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.QutritCX.compute_matrix())
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
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CZ.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.QutritCX.compute_eigvals())
        [-0.5+0.8660254j -0.5-0.8660254j  1. +0.j -0.5+0.8660254j -0.5-0.8660254j  1. +0.j  1. +0.j  1. +0.j  1. +0.j]
        """
        return np.array([-0.5+0.8660254j, -0.5-0.8660254j, 1, -0.5+0.8660254j, -0.5-0.8660254j, 1, 1, 1, 1])

    def adjoint(self):
        op = QutritCX(self.wires)
        op.inverse = not self.inverse
        return op

    @property
    def control_wires(self):
        return Wires(self.wires[0])
