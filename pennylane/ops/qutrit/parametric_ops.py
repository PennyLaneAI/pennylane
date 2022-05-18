import functools
import math
from operator import matmul

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

OMEGA = np.exp(2 * np.pi * 1j / 3)


class QutritPhaseZ(Operation):
    r"""
    Single qutrit Z gate

    .. math:: Z(a, b) = \begin{bmatrix}
                1 &     0      & 0 \\
                0 & \omega^{a} & 0 \\
                0 &     0      & \omega^{b}
            \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2

    Args:
        a (int): Exponent to apply to phase for :math:`\ket{1}`
        b (int): Exponent to apply to phase for :math:`\ket{2}`
        wires (Sequence[int] or int): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 2
    """int: Number of trainable parameters that the operator depends on."""

    def __init__(self, a, b, wires, do_queue=True, id=None):
        super().__init__(a, b, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(a, b):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Args:
            a (int): Exponent to apply to phase for :math:`\ket{1}`
            b (int): Exponent to apply to phase for :math:`\ket{2}`

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.QutritPhaseZ.compute_matrix(1, 2)
        array([[ 1. +0.j       ,  0. +0.j       ,  0. +0.j       ],
               [ 0. +0.j       , -0.5+0.8660254j,  0. +0.j       ],
               [ 0. +0.j       ,  0. +0.j       , -0.5-0.8660254j]])
        """
        return qml.math.diag([1, OMEGA**a, OMEGA**b])

    @staticmethod
    def compute_eigvals(a, b):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Args:
            a (int): Exponent to apply to phase for :math:`\ket{1}`
            b (int): Exponent to apply to phase for :math:`\ket{2}`

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.QutritPhaseZ.compute_eigvals(1, 2)
        array([ 1. +0.j       , -0.5+0.8660254j, -0.5-0.8660254j])
        """
        return qml.math.stack([1, OMEGA**a, OMEGA**b])

    def adjoint(self):
        return QutritPhaseZ(-self.data[0], -self.data[1], wires=self.wires)


class QutritRot(Operation):
    r"""
    Qutrit rotation gate

    [Insert description here]

    **Details:**

    * Number of wires: 1
    * Number of parameters: 8

    Args:
        thetas (Sequence[float]): Angles of rotation. An arbitrary rotation
            requires 8 angles
        wires (Sequence[int] or int): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 8  # TODO: Check if this should be 8 or 1 since parameters are passed into array

    def __init__(self, thetas, wires, do_queue=True, id=None):
        super().__init__(thetas, wires=wires, do_queue=do_queue, id=id)

