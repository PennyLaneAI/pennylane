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
This submodule contains the discrete-variable quantum operations that do
not depend on any parameters.
"""
# pylint: disable=arguments-differ
import cmath
from copy import copy
from functools import lru_cache

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import (
    flip_zero_control,
    make_pow_decomp_with_period,
    pow_involutory,
    self_adjoint,
)
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

INV_SQRT2 = 1 / qml.math.sqrt(2)


class Hadamard(Operation):
    r"""Hadamard(wires)
    The Hadamard operator

    .. math:: H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1\\ 1 & -1\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~H`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    is_hermitian = True
    _queue_category = "_ops"

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return base_label or "H"

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"H('{wire}')"
        return f"H({wire})"

    @property
    def name(self) -> str:
        return "Hadamard"

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Hadamard.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Hadamard.compute_matrix())
        [[ 0.70710678  0.70710678]
         [ 0.70710678 -0.70710678]]
        """
        return np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]).asformat(
            format=format
        )

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Hadamard.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Hadamard.compute_eigvals())
        [ 1. -1.]
        """
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Hadamard.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.Hadamard.compute_diagonalizing_gates(wires=[0]))
        [RY(-0.7853981633974483, wires=[0])]
        """
        return [qml.RY(-np.pi / 4, wires=wires)]

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Hadamard.decomposition`.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> print(qml.Hadamard.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0]),
        RX(1.5707963267948966, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        return [
            qml.PhaseShift(np.pi / 2, wires=wires),
            qml.RX(np.pi / 2, wires=wires),
            qml.PhaseShift(np.pi / 2, wires=wires),
        ]

    def _controlled(self, wire: WiresLike) -> "qml.CH":
        return qml.CH(wires=Wires(wire) + self.wires)

    def adjoint(self) -> "Hadamard":
        return Hadamard(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # H = RZ(\pi) RY(\pi/2) RZ(0)
        return [np.pi, np.pi / 2, 0.0]

    def pow(self, z: int | float):
        return super().pow(z % 2)


H = Hadamard
r"""H(wires)
The Hadamard operator

.. math:: H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1\\ 1 & -1\end{bmatrix}.

.. seealso:: The equivalent long-form alias :class:`~Hadamard`

**Details:**

* Number of wires: 1
* Number of parameters: 0

Args:
    wires (Sequence[int] or int): the wire the operation acts on
"""


def _hadamard_rz_rx_resources():
    return {qml.RZ: 2, qml.RX: 1, qml.GlobalPhase: 1}


@register_resources(_hadamard_rz_rx_resources)
def _hadamard_to_rz_rx(wires: WiresLike, **__):
    qml.RZ(np.pi / 2, wires=wires)
    qml.RX(np.pi / 2, wires=wires)
    qml.RZ(np.pi / 2, wires=wires)
    qml.GlobalPhase(-np.pi / 2, wires=wires)


def _hadamard_rz_ry_resources():
    return {qml.RZ: 1, qml.RY: 1, qml.GlobalPhase: 1}


@register_resources(_hadamard_rz_ry_resources)
def _hadamard_to_rz_ry(wires: WiresLike, **__):
    qml.RZ(np.pi, wires=wires)
    qml.RY(np.pi / 2, wires=wires)
    qml.GlobalPhase(-np.pi / 2)


add_decomps(Hadamard, _hadamard_to_rz_rx, _hadamard_to_rz_ry)
add_decomps("Adjoint(Hadamard)", self_adjoint)
add_decomps("Pow(Hadamard)", pow_involutory)


def _controlled_h_resources(*_, num_control_wires, num_work_wires, work_wire_type, **__):
    if num_control_wires == 1:
        return {qml.CH: 1}
    return {
        qml.H: 2,
        qml.RY: 2,
        controlled_resource_rep(
            qml.X,
            {},
            num_control_wires=num_control_wires,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_h_resources)
def _controlled_hadamard(wires, control_wires, work_wires, work_wire_type, **__):

    if len(control_wires) == 1:
        qml.CH(wires)
        return

    qml.RY(-np.pi / 4, wires=wires[-1])
    qml.H(wires=wires[-1])
    qml.ctrl(
        qml.X(wires[-1]),
        control=wires[:-1],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    qml.H(wires=wires[-1])
    qml.RY(np.pi / 4, wires=wires[-1])


add_decomps("C(Hadamard)", flip_zero_control(_controlled_hadamard))


class PauliX(Operation):
    r"""
    The Pauli X operator

    .. math:: \sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~X`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    is_hermitian = True

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "X"

    resource_keys = set()

    batch_size = None

    _queue_category = "_ops"
    is_hermitian = True

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {qml.pauli.PauliWord({self.wires[0]: "X"}): 1.0}
            )
        return self._pauli_rep

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return base_label or "X"

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"X('{wire}')"
        return f"X({wire})"

    @property
    def name(self) -> str:
        return "PauliX"

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.X.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.X.compute_matrix())
        [[0 1]
         [1 0]]
        """
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[0, 1], [1, 0]]).asformat(format=format)

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.X.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.X.compute_eigvals())
        [ 1. -1.]
        """
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.X.diagonalizing_gates`.

        Args:
           wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
           list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.X.compute_diagonalizing_gates(wires=[0]))
        [H(0)]
        """
        return [Hadamard(wires=wires)]

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.X.decomposition`.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.X.compute_decomposition(0))
        [RX(3.141592653589793, wires=[0]),
        GlobalPhase(-1.5707963267948966, wires=[0])]

        """
        return [qml.RX(np.pi, wires=wires), qml.GlobalPhase(-np.pi / 2, wires=wires)]

    def adjoint(self) -> "PauliX":
        return X(wires=self.wires)

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        z_mod2 = z % 2
        if abs(z_mod2 - 0.5) < 1e-6:
            return [SX(wires=self.wires)]
        return super().pow(z_mod2)

    def _controlled(self, wire: WiresLike) -> "qml.CNOT":
        return qml.CNOT(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # X = RZ(-\pi/2) RY(\pi) RZ(\pi/2)
        return [np.pi / 2, np.pi, -np.pi / 2]


X = PauliX
r"""The Pauli X operator

.. math:: \sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0\end{bmatrix}.

.. seealso:: The equivalent long-form alias :class:`~PauliX`

**Details:**

* Number of wires: 1
* Number of parameters: 0

Args:
    wires (Sequence[int] or int): the wire the operation acts on
"""


def _paulix_to_rx_resources():
    return {qml.GlobalPhase: 1, qml.RX: 1}


@register_resources(_paulix_to_rx_resources)
def _paulix_to_rx(wires: WiresLike, **__):
    qml.RX(np.pi, wires=wires)
    qml.GlobalPhase(-np.pi / 2, wires=wires)


@register_condition(lambda z, **_: math.allclose(z % 2, 0.5))
@register_resources(lambda **_: {qml.SX: 1})
def _pow_x_to_sx(wires, **_):
    qml.SX(wires=wires)


@register_resources(lambda **_: {qml.RX: 1, qml.GlobalPhase: 1})
def _pow_x_to_rx(wires, z, **_):
    z_mod2 = z % 2
    qml.RX(np.pi * z_mod2, wires=wires)
    qml.GlobalPhase(-np.pi / 2 * z_mod2, wires=wires)


add_decomps(PauliX, _paulix_to_rx)
add_decomps("Adjoint(PauliX)", self_adjoint)
add_decomps("Pow(PauliX)", pow_involutory, _pow_x_to_rx, _pow_x_to_sx)


def _controlled_x_resource(
    *_, num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, **__
):
    if num_control_wires == 1:
        return {qml.CNOT: 1, PauliX: num_zero_control_values}
    if num_control_wires == 2:
        return {qml.Toffoli: 1, PauliX: num_zero_control_values * 2}
    return {
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_x_resource)
def _controlled_x_decomp(
    *_, wires, control_wires, control_values, work_wires, work_wire_type, **__
):
    """The decomposition rule for a controlled PauliX."""

    if len(control_wires) == 1 and not control_values[0]:
        qml.CNOT(wires=wires)
        qml.X(wires[1])
        return

    if len(control_wires) == 1:
        qml.CNOT(wires=wires)
        return

    if len(control_wires) > 2:
        qml.MultiControlledX(
            wires=wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        return

    zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
    for w in zero_control_wires:
        qml.PauliX(w)
    qml.Toffoli(wires=wires)
    for w in zero_control_wires:
        qml.PauliX(w)


add_decomps("C(PauliX)", _controlled_x_decomp)


class PauliY(Operation):
    r"""
    The Pauli Y operator

    .. math:: \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~Y`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    is_hermitian = True

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    resource_keys = set()

    basis = "Y"

    batch_size = None

    _queue_category = "_ops"

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {qml.pauli.PauliWord({self.wires[0]: "Y"}): 1.0}
            )
        return self._pauli_rep

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"Y('{wire}')"
        return f"Y({wire})"

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return base_label or "Y"

    @property
    def name(self) -> str:
        return "PauliY"

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Y.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Y.compute_matrix())
        [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
        """
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[0, -1j], [1j, 0]]).asformat(format=format)

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Y.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Y.compute_eigvals())
        [ 1. -1.]
        """
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Y.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.Y.compute_diagonalizing_gates(wires=[0]))
        [Z(0), S(0), H(0)]
        """
        return [
            Z(wires=wires),
            S(wires=wires),
            Hadamard(wires=wires),
        ]

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Y.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.Y.compute_decomposition(0))
        [RY(3.141592653589793, wires=[0]),
        GlobalPhase(-1.5707963267948966, wires=[0])]

        """
        return [qml.RY(np.pi, wires=wires), qml.GlobalPhase(-np.pi / 2, wires=wires)]

    def adjoint(self) -> "PauliY":
        return Y(wires=self.wires)

    def pow(self, z: float | int) -> list[qml.operation.Operator]:
        return super().pow(z % 2)

    def _controlled(self, wire: WiresLike) -> "qml.CY":
        return qml.CY(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # Y = RZ(0) RY(\pi) RZ(0)
        return [0.0, np.pi, 0.0]


Y = PauliY
r"""The Pauli Y operator

.. math:: \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0\end{bmatrix}.

.. seealso:: The equivalent long-form alias :class:`~PauliY`

**Details:**

* Number of wires: 1
* Number of parameters: 0

Args:
    wires (Sequence[int] or int): the wire the operation acts on
"""


def _pauliy_to_ry_gp_resources():
    return {qml.GlobalPhase: 1, qml.RY: 1}


@register_resources(_pauliy_to_ry_gp_resources)
def _pauliy_to_ry_gp(wires: WiresLike, **__):
    qml.RY(np.pi, wires=wires)
    qml.GlobalPhase(-np.pi / 2, wires=wires)


@register_resources(lambda **_: {qml.RY: 1, qml.GlobalPhase: 1})
def _pow_y(wires, z, **_):
    z_mod2 = z % 2
    qml.RY(np.pi * z_mod2, wires=wires)
    qml.GlobalPhase(-np.pi / 2 * z_mod2, wires=wires)


add_decomps(PauliY, _pauliy_to_ry_gp)
add_decomps("Adjoint(PauliY)", self_adjoint)
add_decomps("Pow(PauliY)", pow_involutory, _pow_y)


def _controlled_y_resource(*_, num_control_wires, num_work_wires, work_wire_type, **__):
    if num_control_wires == 1:
        return {qml.CY: 1}
    return {
        qml.S: 1,
        adjoint_resource_rep(qml.S): 1,
        controlled_resource_rep(
            qml.X,
            {},
            num_control_wires=num_control_wires,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_y_resource)
def _controlled_y_decomp(*_, wires, control_wires, work_wires, work_wire_type, **__):

    if len(control_wires) == 1:
        qml.CY(wires=wires)
        return

    qml.adjoint(qml.S(wires[-1]))
    qml.ctrl(
        qml.X(wires[-1]), control=wires[:-1], work_wires=work_wires, work_wire_type=work_wire_type
    )
    qml.S(wires=wires[-1])


add_decomps("C(PauliY)", flip_zero_control(_controlled_y_decomp))


class PauliZ(Operation):
    r"""
    The Pauli Z operator

    .. math:: \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~Z`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    is_hermitian = True
    _queue_category = "_ops"
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    resource_keys = set()

    basis = "Z"

    batch_size = None

    resource_keys = set()

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {qml.pauli.PauliWord({self.wires[0]: "Z"}): 1.0}
            )
        return self._pauli_rep

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"Z('{wire}')"
        return f"Z({wire})"

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return base_label or "Z"

    @property
    def name(self) -> str:
        return "PauliZ"

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Z.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Z.compute_matrix())
        [[ 1  0]
         [ 0 -1]]
        """
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[1, 0], [0, -1]]).asformat(format=format)

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Z.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Z.compute_eigvals())
        [ 1. -1.]
        """
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(
        wires: WiresLike,
    ) -> list[qml.operation.Operator]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Z.diagonalizing_gates`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.Z.compute_diagonalizing_gates(wires=[0]))
        []
        """
        return []

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Z.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.Z.compute_decomposition(0))
        [PhaseShift(3.141592653589793, wires=[0])]

        """
        return [qml.PhaseShift(np.pi, wires=wires)]

    def adjoint(self) -> "PauliZ":
        return Z(wires=self.wires)

    def pow(self, z: float) -> list[qml.operation.Operator]:
        z_mod2 = z % 2
        if z_mod2 == 0:
            return []
        if z_mod2 == 1:
            return [copy(self)]

        if abs(z_mod2 - 0.5) < 1e-6:
            return [S(wires=self.wires)]
        if abs(z_mod2 - 0.25) < 1e-6:
            return [T(wires=self.wires)]

        return [qml.PhaseShift(np.pi * z_mod2, wires=self.wires)]

    def _controlled(self, wire: WiresLike) -> "qml.CZ":
        return qml.CZ(wires=wire + self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # Z = RZ(\pi) RY(0) RZ(0)
        return [np.pi, 0.0, 0.0]


Z = PauliZ
r"""The Pauli Z operator

.. math:: \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1\end{bmatrix}.

.. seealso:: The equivalent long-form alias :class:`~PauliZ`

**Details:**

* Number of wires: 1
* Number of parameters: 0

Args:
    wires (Sequence[int] or int): the wire the operation acts on
"""


def _pauliz_to_ps_resources():
    return {qml.PhaseShift: 1}


@register_resources(_pauliz_to_ps_resources)
def _pauliz_to_ps(wires: WiresLike, **__):
    qml.PhaseShift(np.pi, wires=wires)


@register_condition(lambda z, **_: math.allclose(z % 2, 0.5))
@register_resources(lambda **_: {qml.S: 1})
def _pow_z_to_s(wires, **_):
    qml.S(wires=wires)


@register_condition(lambda z, **_: math.allclose(z % 2, 0.25))
@register_resources(lambda **_: {qml.T: 1})
def _pow_z_to_t(wires, **_):
    qml.T(wires=wires)


@register_resources(lambda **_: {qml.PhaseShift: 1})
def _pow_z(wires, z, **_):
    z_mod2 = z % 2
    qml.PhaseShift(np.pi * z_mod2, wires=wires)


add_decomps(PauliZ, _pauliz_to_ps)
add_decomps("Adjoint(PauliZ)", self_adjoint)
add_decomps("Pow(PauliZ)", pow_involutory, _pow_z, _pow_z_to_s, _pow_z_to_t)


def _controlled_z_resources(*_, num_control_wires, num_work_wires, work_wire_type, **__):
    if num_control_wires == 1:
        return {qml.CZ: 1}
    if num_control_wires == 2:
        return {qml.CCZ: 1}
    return {
        qml.H: 2,
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=num_control_wires,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_z_resources)
def _controlled_z_decomp(*_, wires, control_wires, work_wires, work_wire_type, **__):

    if len(control_wires) == 1:
        qml.CZ(wires=wires)
        return

    if len(control_wires) == 2:
        qml.CCZ(wires=wires)
        return

    qml.H(wires=wires[-1])
    qml.MultiControlledX(wires=wires, work_wires=work_wires, work_wire_type=work_wire_type)
    qml.H(wires=wires[-1])


add_decomps("C(PauliZ)", flip_zero_control(_controlled_z_decomp))


class S(Operation):
    r"""S(wires)
    The single-qubit phase gate

    .. math:: S = \begin{bmatrix}
                1 & 0 \\
                0 & i
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None

    resource_keys = set()

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({self.wires[0]: "I"}): 0.5 + 0.5j,
                    qml.pauli.PauliWord({self.wires[0]: "Z"}): 0.5 - 0.5j,
                }
            )
        return self._pauli_rep

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"S('{wire}')"
        return f"S({wire})"

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.S.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.S.compute_matrix())
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+1.j]]
        """
        return np.array([[1, 0], [0, 1j]])

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.S.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.S.compute_eigvals())
        [1.+0.j 0.+1.j]
        """
        return np.array([1, 1j])

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.S.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.S.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0])]

        """
        return [qml.PhaseShift(np.pi / 2, wires=wires)]

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        z_mod4 = z % 4
        pow_map = {
            0: lambda op: [],
            0.5: lambda op: [T(wires=op.wires)],
            1: lambda op: [copy(op)],
            2: lambda op: [Z(wires=op.wires)],
        }
        return pow_map.get(z_mod4, lambda op: [qml.PhaseShift(np.pi * z_mod4 / 2, wires=op.wires)])(
            self
        )

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # S = RZ(\pi/2) RY(0) RZ(0)
        return [np.pi / 2, 0.0, 0.0]


def _s_phaseshift_resources():
    return {qml.PhaseShift: 1}


@register_resources(_s_phaseshift_resources)
def _s_phaseshift(wires, **__):
    qml.PhaseShift(np.pi / 2, wires=wires)


add_decomps(S, _s_phaseshift)


@register_condition(lambda z, **_: math.allclose(z % 4, 0.5))
@register_resources(lambda **_: {qml.T: 1})
def _pow_s_to_t(wires, **_):
    qml.T(wires=wires)


@register_condition(lambda z, **_: math.allclose(z % 4, 2))
@register_resources(lambda **_: {qml.Z: 1})
def _pow_s_to_z(wires, **_):
    qml.Z(wires=wires)


@register_resources(lambda **_: {qml.PhaseShift: 1})
def _pow_s(wires, z, **_):
    z_mod4 = z % 4
    qml.PhaseShift(np.pi * z_mod4 / 2, wires=wires)


add_decomps("Pow(S)", make_pow_decomp_with_period(4), _pow_s, _pow_s_to_t, _pow_s_to_z)


class T(Operation):
    r"""T(wires)
    The single-qubit T gate

    .. math:: T = \begin{bmatrix}
                1 & 0 \\
                0 & e^{\frac{i\pi}{4}}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None

    resource_keys = set()

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({self.wires[0]: "I"}): (0.5 + INV_SQRT2 * (0.5 + 0.5j)),
                    qml.pauli.PauliWord({self.wires[0]: "Z"}): (0.5 - INV_SQRT2 * (0.5 + 0.5j)),
                }
            )
        return self._pauli_rep

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"T('{wire}')"
        return f"T({wire})"

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.T.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.T.compute_matrix())
        [[1.        +0.j         0.        +0.j        ]
        [0.        +0.j         0.70710678+0.70710678j]]
        """
        return np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]])

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.T.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.T.compute_eigvals())
        [1.        +0.j         0.70710678+0.70710678j]
        """
        return np.array([1, cmath.exp(1j * np.pi / 4)])

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.T.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.T.compute_decomposition(0))
        [PhaseShift(0.7853981633974483, wires=[0])]

        """
        return [qml.PhaseShift(np.pi / 4, wires=wires)]

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        z_mod8 = z % 8
        pow_map = {
            0: lambda op: [],
            1: lambda op: [copy(op)],
            2: lambda op: [S(wires=op.wires)],
            4: lambda op: [Z(wires=op.wires)],
        }
        return pow_map.get(z_mod8, lambda op: [qml.PhaseShift(np.pi * z_mod8 / 4, wires=op.wires)])(
            self
        )

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # T = RZ(\pi/4) RY(0) RZ(0)
        return [np.pi / 4, 0.0, 0.0]


def _t_phaseshift_resources():
    return {qml.PhaseShift: 1}


@register_resources(_t_phaseshift_resources)
def _t_phaseshift(wires, **__):
    qml.PhaseShift(np.pi / 4, wires=wires)


add_decomps(T, _t_phaseshift)


@register_resources(lambda **_: {qml.PhaseShift: 1})
def _pow_t(wires, z, **_):
    z_mod8 = z % 8
    qml.PhaseShift(np.pi * z_mod8 / 4, wires=wires)


add_decomps("Pow(T)", make_pow_decomp_with_period(8), _pow_t)


class SX(Operation):
    r"""SX(wires)
    The single-qubit Square-Root X operator.

    .. math:: SX = \sqrt{X} = \frac{1}{2} \begin{bmatrix}
            1+i &   1-i \\
            1-i &   1+i \\
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "X"

    resource_keys = set()

    @property
    def resource_params(self) -> dict:
        return {}

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({self.wires[0]: "I"}): (0.5 + 0.5j),
                    qml.pauli.PauliWord({self.wires[0]: "X"}): (0.5 - 0.5j),
                }
            )
        return self._pauli_rep

    def __repr__(self) -> str:
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"SX('{wire}')"
        return f"SX({wire})"

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SX.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SX.compute_matrix())
        [[0.5+0.5j 0.5-0.5j]
         [0.5-0.5j 0.5+0.5j]]
        """
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.SX.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.SX.compute_eigvals())
        [1.+0.j 0.+1.j]
        """
        return np.array([1, 1j])

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SX.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SX.compute_decomposition(0))
        [RZ(1.5707963267948966, wires=[0]),
        RY(1.5707963267948966, wires=[0]),
        RZ(-1.5707963267948966, wires=[0]),
        GlobalPhase(-0.7853981633974483, wires=[0])]

        """
        return [
            qml.RZ(np.pi / 2, wires=wires),
            qml.RY(np.pi / 2, wires=wires),
            qml.RZ(-np.pi / 2, wires=wires),
            qml.GlobalPhase(-np.pi / 4, wires=wires),
        ]

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        z_mod4 = z % 4
        if z_mod4 == 2:
            return [X(wires=self.wires)]
        return super().pow(z_mod4)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # SX = RZ(-\pi/2) RY(\pi/2) RZ(\pi/2)
        return [np.pi / 2, np.pi / 2, -np.pi / 2]


def _sx_to_rx_resources():
    return {qml.RX: 1, qml.GlobalPhase: 1}


@register_resources(_sx_to_rx_resources)
def _sx_to_rx(wires: WiresLike, **__):
    qml.RX(np.pi / 2, wires=wires)
    qml.GlobalPhase(-np.pi / 4, wires=wires)


add_decomps(SX, _sx_to_rx)


@register_condition(lambda z, **_: z % 4 == 2)
@register_resources(lambda **_: {qml.X: 1})
def _pow_sx_to_x(wires, **__):
    qml.X(wires)


@register_resources(lambda **_: {qml.RX: 1, qml.GlobalPhase: 1})
def _pow_sx(wires, z, **_):
    z_mod4 = z % 4
    qml.RX(np.pi / 2 * z_mod4, wires=wires)
    qml.GlobalPhase(-np.pi / 4 * z_mod4, wires=wires)


add_decomps("Pow(SX)", make_pow_decomp_with_period(4), _pow_sx_to_x, _pow_sx)


class SWAP(Operation):
    r"""SWAP(wires)
    The swap operator

    .. math:: SWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    resource_keys = set()
    batch_size = None

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({}): 0.5,
                    qml.pauli.PauliWord({self.wires[0]: "X", self.wires[1]: "X"}): 0.5,
                    qml.pauli.PauliWord({self.wires[0]: "Y", self.wires[1]: "Y"}): 0.5,
                    qml.pauli.PauliWord({self.wires[0]: "Z", self.wires[1]: "Z"}): 0.5,
                }
            )
        return self._pauli_rep

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SWAP.compute_matrix())
        [[1 0 0 0]
         [0 0 1 0]
         [0 1 0 0]
         [0 0 0 1]]
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        r"""Sparse Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SWAP.sparse_matrix`

        Returns:
            csr_matrix: matrix

        **Example**

        >>> print(qml.SWAP.compute_sparse_matrix())
        <Compressed Sparse Row sparse matrix of dtype 'int64'
                with 4 stored elements and shape (4, 4)>
          Coords        Values
          (0, 0)        1
          (1, 2)        1
          (2, 1)        1
          (3, 3)        1
        """
        # The same as
        # [[1 0 0 0]
        #  [0 0 1 0]
        #  [0 1 0 0]
        #  [0 0 0 1]]
        data, indices, indptr = [1, 1, 1, 1], [0, 2, 1, 3], [0, 1, 2, 3, 4]
        return sparse.csr_matrix((data, indices, indptr)).asformat(format=format)

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SWAP.compute_decomposition((0,1)))
        [CNOT(wires=[0, 1]), CNOT(wires=[1, 0]), CNOT(wires=[0, 1])]

        """
        return [
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]

    @property
    def resource_params(self) -> dict:
        return {}

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        return super().pow(z % 2)

    def adjoint(self) -> "SWAP":
        return SWAP(wires=self.wires)

    def _controlled(self, wire: WiresLike) -> "qml.CSWAP":
        return qml.CSWAP(wires=wire + self.wires)

    @property
    def is_hermitian(self) -> bool:
        return True


def _swap_to_cnot_resources():
    return {qml.CNOT: 3}


@register_resources(_swap_to_cnot_resources)
def _swap_to_cnot(wires, **__):
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.CNOT(wires=[wires[0], wires[1]])


add_decomps(SWAP, _swap_to_cnot)
add_decomps("Adjoint(SWAP)", self_adjoint)
add_decomps("Pow(SWAP)", pow_involutory)


def _controlled_swap_resources(*_, num_control_wires, num_work_wires, work_wire_type, **__):
    if num_control_wires == 1:
        return {qml.CSWAP: 1}
    return {
        qml.CNOT: 2,
        resource_rep(
            qml.MultiControlledX,
            num_control_wires=num_control_wires + 1,
            num_zero_control_values=0,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1,
    }


@register_resources(_controlled_swap_resources)
def _controlled_swap_decomp(*_, wires, control_wires, work_wires, work_wire_type, **__):

    if len(control_wires) == 1:
        qml.CSWAP(wires=wires)
        return

    qml.CNOT(wires=[wires[-2], wires[-1]])
    qml.MultiControlledX(
        wires=wires[:-2] + [wires[-1], wires[-2]],
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    qml.CNOT(wires=[wires[-2], wires[-1]])


add_decomps("C(SWAP)", flip_zero_control(_controlled_swap_decomp))


class ECR(Operation):
    r""" ECR(wires)

    An echoed RZX(:math:`\pi/2`) gate.

    .. math:: ECR = {\frac{1}{\sqrt{2}}} \begin{bmatrix}
            0 & 0 & 1 & i \\
            0 & 0 & i & 1 \\
            1 & -i & 0 & 0 \\
            -i & 1 & 0 & 0
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 0

    batch_size = None

    resource_keys = set()

    @property
    def resource_params(self) -> dict:
        return {}

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({self.wires[0]: "X"}): INV_SQRT2,
                    qml.pauli.PauliWord({self.wires[0]: "Y", self.wires[1]: "X"}): -INV_SQRT2,
                }
            )
        return self._pauli_rep

    @staticmethod
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ECR.matrix`


        Return type: tensor_like

        **Example**

        >>> from pprint import pprint
        >>> pprint(qml.ECR.compute_matrix())
        array([[ 0.        +0.j        ,  0.        +0.j        ,
                 0.70710678+0.j        ,  0.        +0.70710678j],
               [ 0.        +0.j        ,  0.        +0.j        ,
                 0.        +0.70710678j,  0.70710678+0.j        ],
               [ 0.70710678+0.j        , -0.        -0.70710678j,
                 0.        +0.j        ,  0.        +0.j        ],
               [-0.        -0.70710678j,  0.70710678+0.j        ,
                 0.        +0.j        ,  0.        +0.j        ]])
        """

        return np.array(
            [
                [0, 0, INV_SQRT2, INV_SQRT2 * 1j],
                [0, 0, INV_SQRT2 * 1j, INV_SQRT2],
                [INV_SQRT2, -INV_SQRT2 * 1j, 0, 0],
                [-INV_SQRT2 * 1j, INV_SQRT2, 0, 0],
            ]
        )

    @staticmethod
    def compute_eigvals() -> np.ndarray:
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ECR.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.ECR.compute_eigvals())
        [ 1 -1  1 -1]
        """

        return np.array([1, -1, 1, -1])

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.ECR.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> from pprint import pprint
        >>> pprint(qml.ECR.compute_decomposition((0,1)))
        [Z(0),
        CNOT(wires=[0, 1]),
        SX(1),
        RX(1.5707963267948966, wires=[0]),
        RY(1.5707963267948966, wires=[0]),
        RX(1.5707963267948966, wires=[0])]

        """
        pi = np.pi
        return [
            Z(wires=[wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            SX(wires=[wires[1]]),
            qml.RX(pi / 2, wires=[wires[0]]),
            qml.RY(pi / 2, wires=[wires[0]]),
            qml.RX(pi / 2, wires=[wires[0]]),
        ]

    def adjoint(self) -> "ECR":
        return ECR(wires=self.wires)

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        return super().pow(z % 2)


def _ecr_decomp_resources():
    return {Z: 1, qml.CNOT: 1, SX: 1, qml.RX: 2, qml.RY: 1}


@register_resources(_ecr_decomp_resources)
def _ecr_decomp(wires, **__):
    Z(wires=[wires[0]])
    qml.CNOT(wires=[wires[0], wires[1]])
    SX(wires=[wires[1]])
    qml.RX(np.pi / 2, wires=[wires[0]])
    qml.RY(np.pi / 2, wires=[wires[0]])
    qml.RX(np.pi / 2, wires=[wires[0]])


add_decomps(ECR, _ecr_decomp)
add_decomps("Adjoint(ECR)", self_adjoint)
add_decomps("Pow(ECR)", pow_involutory)


class ISWAP(Operation):
    r"""ISWAP(wires)
    The i-swap operator

    .. math:: ISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0\\
            0 & i & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    batch_size = None
    resource_keys = set()

    @property
    def resource_params(self) -> dict:
        return {}

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({}): 0.5,
                    qml.pauli.PauliWord({self.wires[0]: "X", self.wires[1]: "X"}): 0.5j,
                    qml.pauli.PauliWord({self.wires[0]: "Y", self.wires[1]: "Y"}): 0.5j,
                    qml.pauli.PauliWord({self.wires[0]: "Z", self.wires[1]: "Z"}): 0.5,
                }
            )
        return self._pauli_rep

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ISWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.ISWAP.compute_matrix())
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+1.j 0.+0.j]
         [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ISWAP.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.ISWAP.compute_eigvals())
        [ 0.+1.j -0.-1.j  1.+0.j  1.+0.j]
        """
        return np.array([1j, -1j, 1, 1])

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.ISWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.ISWAP.compute_decomposition((0,1)))
        [S(0),
        S(1),
        H(0),
        CNOT(wires=[0, 1]),
        CNOT(wires=[1, 0]),
        H(1)]

        """
        return [
            S(wires=wires[0]),
            S(wires=wires[1]),
            Hadamard(wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[1], wires[0]]),
            Hadamard(wires=wires[1]),
        ]

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        z_mod4 = z % 4
        if abs(z_mod4 - 0.5) < 1e-6:
            return [SISWAP(wires=self.wires)]
        if abs(z_mod4 - 2) < 1e-6:
            return [qml.Z(wires=self.wires[0]), qml.Z(wires=self.wires[1])]
        return super().pow(z_mod4)


def _iswap_decomp_resources():
    return {qml.S: 2, qml.Hadamard: 2, qml.CNOT: 2}


@register_resources(_iswap_decomp_resources)
def _iswap_decomp(wires, **__):
    S(wires=wires[0])
    S(wires=wires[1])
    Hadamard(wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[0]])
    Hadamard(wires=wires[1])


add_decomps(ISWAP, _iswap_decomp)


@register_condition(lambda z, **_: math.allclose(z % 4, 0.5))
@register_resources(lambda **_: {qml.SISWAP: 1})
def _pow_iswap_to_siswap(wires, **__):
    qml.SISWAP(wires=wires)


@register_condition(lambda z, **_: math.allclose(z % 4, 2))
@register_resources(lambda **_: {qml.Z: 2})
def _pow_iswap_to_zz(wires, **__):
    qml.Z(wires=wires[0])
    qml.Z(wires=wires[1])


add_decomps("Pow(ISWAP)", make_pow_decomp_with_period(4), _pow_iswap_to_zz, _pow_iswap_to_siswap)


class SISWAP(Operation):
    r"""SISWAP(wires)
    The square root of i-swap operator. Can also be accessed as ``qml.SQISW``

    .. math:: SISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1/ \sqrt{2} & i/\sqrt{2} & 0\\
            0 & i/ \sqrt{2} & 1/ \sqrt{2} & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    batch_size = None
    resource_keys = set()

    @property
    def resource_params(self) -> dict:
        return {}

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {
                    qml.pauli.PauliWord({self.wires[0]: "I", self.wires[1]: "I"}): 0.5
                    + 0.5 * INV_SQRT2,
                    qml.pauli.PauliWord({self.wires[0]: "X", self.wires[1]: "X"}): 0.5j * INV_SQRT2,
                    qml.pauli.PauliWord({self.wires[0]: "Y", self.wires[1]: "Y"}): 0.5j * INV_SQRT2,
                    qml.pauli.PauliWord({self.wires[0]: "Z", self.wires[1]: "Z"}): 0.5
                    - 0.5 * INV_SQRT2,
                }
            )
        return self._pauli_rep

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SISWAP.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> from pprint import pprint
        >>> pprint(qml.SISWAP.compute_matrix())
        array([[1.        +0.j        , 0.        +0.j        ,
                0.        +0.j        , 0.        +0.j        ],
            [0.        +0.j        , 0.70710678+0.j        ,
                0.        +0.70710678j, 0.        +0.j        ],
            [0.        +0.j        , 0.        +0.70710678j,
                0.70710678+0.j        , 0.        +0.j        ],
            [0.        +0.j        , 0.        +0.j        ,
                0.        +0.j        , 1.        +0.j        ]])
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, INV_SQRT2, INV_SQRT2 * 1j, 0],
                [0, INV_SQRT2 * 1j, INV_SQRT2, 0],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.SISWAP.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.SISWAP.compute_eigvals())
        [0.70710678+0.70710678j 0.70710678-0.70710678j 1.        +0.j 1.        +0.j        ]
        """
        return np.array([INV_SQRT2 * (1 + 1j), INV_SQRT2 * (1 - 1j), 1, 1])

    @staticmethod
    def compute_decomposition(wires: WiresLike) -> list[qml.operation.Operator]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SISWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SISWAP.compute_decomposition((0,1)))
        [SX(0),
        RZ(1.5707963267948966, wires=[0]),
        CNOT(wires=[0, 1]),
        SX(0),
        RZ(5.497787143782138, wires=[0]),
        SX(0),
        RZ(1.5707963267948966, wires=[0]),
        SX(1),
        RZ(5.497787143782138, wires=[1]),
        CNOT(wires=[0, 1]),
        SX(0),
        SX(1)]

        """
        return [
            SX(wires=wires[0]),
            qml.RZ(np.pi / 2, wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            SX(wires=wires[0]),
            qml.RZ(7 * np.pi / 4, wires=wires[0]),
            SX(wires=wires[0]),
            qml.RZ(np.pi / 2, wires=wires[0]),
            SX(wires=wires[1]),
            qml.RZ(7 * np.pi / 4, wires=wires[1]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            SX(wires=wires[0]),
            SX(wires=wires[1]),
        ]

    def pow(self, z: int | float) -> list[qml.operation.Operator]:
        z_mod8 = z % 8
        if abs(z_mod8 - 2) < 1e-6:
            return [ISWAP(wires=self.wires)]
        if abs(z_mod8 - 4) < 1e-6:
            return [qml.Z(wires=self.wires[0]), qml.Z(wires=self.wires[1])]
        return super().pow(z_mod8)


def _siswap_decomp_resources():
    return {SX: 6, qml.RZ: 4, qml.CNOT: 2}


@register_resources(_siswap_decomp_resources)
def _siswap_decomp(wires, **__):
    SX(wires=wires[0])
    qml.RZ(np.pi / 2, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    SX(wires=wires[0])
    qml.RZ(7 * np.pi / 4, wires=wires[0])
    SX(wires=wires[0])
    qml.RZ(np.pi / 2, wires=wires[0])
    SX(wires=wires[1])
    qml.RZ(7 * np.pi / 4, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    SX(wires=wires[0])
    SX(wires=wires[1])


add_decomps(SISWAP, _siswap_decomp)


@register_condition(lambda z, **_: math.allclose(z % 8, 2))
@register_resources(lambda **_: {qml.ISWAP: 1})
def _pow_siswap_to_iswap(wires, **_):
    qml.ISWAP(wires)


@register_condition(lambda z, **_: math.allclose(z % 8, 4))
@register_resources(lambda **_: {qml.Z: 2})
def _pow_siswap_to_zz(wires, **_):
    qml.Z(wires=wires[0])
    qml.Z(wires=wires[1])


add_decomps("Pow(SISWAP)", make_pow_decomp_with_period(8), _pow_siswap_to_zz, _pow_siswap_to_iswap)


SQISW = SISWAP
