# Copyright 2026 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal list of ops that use the prototype Operator2 base class."""

from functools import lru_cache

import numpy as np
from scipy import sparse

import pennylane as qml
from pennylane import math
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation2 import Operation2, Operator2
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .parametric_ops_single_qubit import stack_last

# FIXME: Ops remaining:

INV_SQRT2 = 1 / math.sqrt(2)


class Hadamard(Operation2):
    r"""Hadamard(wires)
    The Hadamard operator
    """

    is_verified_hermitian = True

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires)

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

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]).asformat(
            format=format
        )

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return [RY(-np.pi / 4, wires=wires)]

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [
            PhaseShift(np.pi / 2, wires=wires),
            RX(np.pi / 2, wires=wires),
            PhaseShift(np.pi / 2, wires=wires),
        ]

    def adjoint(self) -> "Hadamard":
        return Hadamard(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # H = RZ(\pi) RY(\pi/2) RZ(0)
        return [np.pi, np.pi / 2, 0.0]


H = Hadamard


class PauliX(Operation2):
    r"""
    The Pauli X operator
    """

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    basis = "X"

    is_verified_hermitian = True

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {qml.pauli.PauliWord({self.wires[0]: "X"}): 1.0}
            )
        return self._pauli_rep

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires)

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

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[0, 1], [1, 0]]).asformat(format=format)

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return [Hadamard(wires=wires)]

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [RX(np.pi, wires=wires), GlobalPhase(-np.pi / 2, wires=wires)]

    def adjoint(self) -> "PauliX":
        return X(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # X = RZ(-\pi/2) RY(\pi) RZ(\pi/2)
        return [np.pi / 2, np.pi, -np.pi / 2]


X = PauliX


class PauliY(Operation2):
    r"""
    The Pauli Y operator
    """

    is_verified_hermitian = True

    num_wires = 1
    """int: Number of wires that the operator acts on."""

    basis = "Y"

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

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[0, -1j], [1j, 0]]).asformat(format=format)

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return [
            Z(wires=wires),
            S(wires=wires),
            Hadamard(wires=wires),
        ]

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [RY(np.pi, wires=wires), GlobalPhase(-np.pi / 2, wires=wires)]

    def adjoint(self) -> "PauliY":
        return Y(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # Y = RZ(0) RY(\pi) RZ(0)
        return [0.0, np.pi, 0.0]


Y = PauliY


class PauliZ(Operation2):
    r"""
    The Pauli Z operator
    """

    is_verified_hermitian = True
    num_wires = 1

    basis = "Z"

    @property
    def pauli_rep(self):
        if self._pauli_rep is None:
            self._pauli_rep = qml.pauli.PauliSentence(
                {qml.pauli.PauliWord({self.wires[0]: "Z"}): 1.0}
            )
        return self._pauli_rep

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires)

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

    @staticmethod
    @lru_cache
    def compute_matrix() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    @lru_cache
    def compute_sparse_matrix(format="csr") -> sparse.spmatrix:  # pylint: disable=arguments-differ
        return sparse.csr_matrix([[1, 0], [0, -1]]).asformat(format=format)

    @staticmethod
    def compute_eigvals() -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        return qml.pauli.pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return []

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [PhaseShift(np.pi, wires=wires)]

    def adjoint(self) -> "PauliZ":
        return Z(wires=self.wires)

        return [PhaseShift(np.pi * z_mod2, wires=self.wires)]

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # Z = RZ(\pi) RY(0) RZ(0)
        return [np.pi, 0.0, 0.0]


Z = PauliZ


class CNOT(Operation2):
    r"""CNOT(wires)
    The controlled-NOT operator
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    name = "CNOT"

    def __init__(self, wires):
        super().__init__(wires=wires)

    def adjoint(self):
        return CNOT(self.wires)

    def __repr__(self):
        return f"CNOT(wires={self.wires.tolist()})"

    @staticmethod
    @lru_cache
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class RX(Operation2):
    r"""
    The single qubit X rotation
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "X"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, phi: TensorLike, wires: WiresLike):
        super().__init__(phi, wires=wires)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        c = math.cos(phi / 2)
        s = math.sin(phi / 2)

        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = math.cast_like(c, 1j)
            s = math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        js = -1j * s
        return math.stack([stack_last([c, js]), stack_last([js, c])], axis=-2)

    @staticmethod
    def compute_sparse_matrix(phi, format="csr"):
        return sp.sparse.csr_matrix(
            [
                [math.cos(phi / 2), -1j * math.sin(phi / 2)],
                [-1j * math.sin(phi / 2), math.cos(phi / 2)],
            ]
        ).asformat(format)

    def adjoint(self) -> "RX":
        return RX(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [RX(self.phi * z, wires=self.wires)]

    def simplify(self) -> "RX":
        phi = self.phi % (4 * np.pi)
        return RX(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
        pi_half = math.ones_like(self.phi) * (np.pi / 2)
        return [pi_half, self.phi, -pi_half]


class RY(Operation2):
    r"""
    The single qubit Y rotation
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Y"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, phi: TensorLike, wires: WiresLike):
        super().__init__(phi, wires=wires)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""

        c = math.cos(phi / 2)
        s = math.sin(phi / 2)
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            c = math.cast_like(c, 1j)
            s = math.cast_like(s, 1j)
        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        s = (1 + 0j) * s
        return math.stack([stack_last([c, -s]), stack_last([s, c])], axis=-2)

    @staticmethod
    def compute_sparse_matrix(phi, format="csr"):
        return sp.sparse.csr_matrix(
            [
                [math.cos(phi / 2), -math.sin(phi / 2)],
                [math.sin(phi / 2), math.cos(phi / 2)],
            ]
        ).asformat(format)

    def adjoint(self) -> "RY":
        return RY(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [RY(self.phi * z, wires=self.wires)]

    def simplify(self) -> "RY":
        phi = self.phi % (4 * np.pi)
        return RY(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # RY(\theta) = RZ(0) RY(\theta) RZ(0)
        return [0.0, self.phi, 0.0]


class RZ(Operation2):
    r"""
    The single qubit Z rotation
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, phi: TensorLike, wires: WiresLike):
        super().__init__(phi, wires=wires)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            p = math.exp(-0.5j * math.cast_like(phi, 1j))
            z = math.zeros_like(p)

            return math.stack([stack_last([p, z]), stack_last([z, math.conj(p)])], axis=-2)

        signs = math.array([-1, 1], like=phi)
        arg = 0.5j * phi

        if math.ndim(arg) == 0:
            return math.diag(math.exp(arg * signs))

        diags = math.exp(math.outer(arg, signs))
        return diags[:, :, np.newaxis] * math.cast_like(math.eye(2, like=diags), diags)

    @staticmethod
    def compute_sparse_matrix(phi, format="csr"):
        return sp.sparse.csr_matrix(
            [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]]
        ).asformat(format)

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phase = math.exp(-0.5j * math.cast_like(phi, 1j))
            return math.stack([phase, math.conj(phase)], axis=-1)

        prefactors = math.array([-0.5j, 0.5j], like=phi)
        if math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = math.outer(phi, prefactors)
        return math.exp(product)

    def adjoint(self) -> "RZ":
        return RZ(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [RZ(self.phi * z, wires=self.wires)]

    def simplify(self) -> "RZ":
        phi = self.phi % (4 * np.pi)
        return RZ(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # RZ(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.phi, 0.0, 0.0]


class PhaseShift(Operation2):
    r"""
    Arbitrary single qubit local phase shift
    """

    num_wires = 1
    num_params = 1
    ndim_params = (0,)

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, phi: TensorLike, wires: WiresLike):
        super().__init__(phi, wires=wires)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label=base_label or "Rϕ", cache=cache)

    @staticmethod
    def compute_matrix(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            p = math.exp(1j * math.cast_like(phi, 1j))
            ones = math.ones_like(p)
            zeros = math.zeros_like(p)

            return math.stack([stack_last([ones, zeros]), stack_last([zeros, p])], axis=-2)

        signs = math.array([0, 1], like=phi)
        arg = 1j * phi

        if math.ndim(arg) == 0:
            return math.diag(math.exp(arg * signs))

        diags = math.exp(math.outer(arg, signs))
        return diags[:, :, np.newaxis] * math.cast_like(math.eye(2, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        if (
            math.get_interface(phi) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phase = math.exp(1j * math.cast_like(phi, 1j))
            return stack_last([math.ones_like(phase), phase])

        prefactors = math.array([0, 1j], like=phi)
        if math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = math.outer(phi, prefactors)
        return math.exp(product)

    @staticmethod
    def compute_decomposition(phi: TensorLike, wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [RZ(phi, wires=wires), GlobalPhase(-phi / 2)]

    def adjoint(self) -> "PhaseShift":
        return PhaseShift(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [PhaseShift(self.phi * z, wires=self.wires)]

    def simplify(self) -> "PhaseShift":
        phi = self.phi % (2 * np.pi)
        return PhaseShift(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # PhaseShift(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.phi, 0.0, 0.0]


class Rot(Operation2):
    r"""
    Arbitrary single qubit rotation
    """

    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    # pylint: disable=too-many-positional-arguments
    def __init__(self, phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike):
        super().__init__(phi, theta, omega, wires=wires)

    @staticmethod
    def compute_matrix(
        phi: TensorLike,
        theta: TensorLike,
        omega: TensorLike,
    ) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        # It might be that they are in different interfaces, e.g.,
        # Rot(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = math.get_interface(phi, theta, omega)

        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            phi = math.cast_like(math.asarray(phi, like=interface), 1j)
            omega = math.cast_like(math.asarray(omega, like=interface), 1j)
            c = math.cast_like(math.asarray(c, like=interface), 1j)
            s = math.cast_like(math.asarray(s, like=interface), 1j)

        # The following variable is used to assert the all terms to be stacked have same shape
        one = math.ones_like(phi) * math.ones_like(omega)
        c = c * one
        s = s * one

        mat = [
            [
                math.exp(-0.5j * (phi + omega)) * c,
                -math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                math.exp(-0.5j * (phi - omega)) * s,
                math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(
        phi: TensorLike, theta: TensorLike, omega: TensorLike, wires: WiresLike
    ):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [
            RZ(phi, wires=wires),
            RY(theta, wires=wires),
            RZ(omega, wires=wires),
        ]

    def adjoint(self) -> "Rot":
        return Rot(-self.omega, -self.theta, -self.phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        return [self.phi, self.theta, self.omega]

    def simplify(self) -> "Rot":
        """Simplifies into single-rotation gates or a Hadamard if possible."""
        p0, p1, p2 = (p % (4 * np.pi) for p in [self.phi, self.theta, self.omega])

        if _can_replace(p0, np.pi / 2) and _can_replace(p2, 7 * np.pi / 2):
            return RX(p1, wires=self.wires)
        if _can_replace(p0, 0) and _can_replace(p2, 0):
            return RY(p1, wires=self.wires)
        if _can_replace(p1, 0):
            return RZ((p0 + p2) % (4 * np.pi), wires=self.wires)
        if _can_replace(p0, np.pi) and _can_replace(p1, np.pi / 2) and _can_replace(p2, 0):
            return Hadamard(wires=self.wires)

        return Rot(p0, p1, p2, wires=self.wires)


class GlobalPhase(Operation2):
    r"""A global phase operation that multiplies all components of the state by :math:`e^{-i \phi}`."""

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"

    def __init__(self, phi, wires):
        super().__init__(phi, wires=wires)

    @staticmethod
    def compute_matrix(phi):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        n_wires = 1
        interface = math.get_interface(phi)
        eye = math.eye(2**n_wires, like=phi)
        exp = math.exp(-1j * math.cast(phi, complex))
        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            eye = math.cast_like(eye, 1j)
        elif interface == "torch":
            eye = eye.to(exp.device)

        if math.ndim(phi) == 0:
            return exp * eye
        return math.tensordot(exp, eye, axes=0)

    @staticmethod
    def compute_diagonalizing_gates(phi, wires):  # pylint: disable=arguments-differ,unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return []

    @staticmethod
    def compute_decomposition(phi, wires):  # pylint:disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method)."""
        return []

    def adjoint(self):
        return GlobalPhase(-1 * self.phi, self.wires)

    def pow(self, z):
        return [GlobalPhase(z * self.phi, self.wires)]


class QubitUnitary(Operation2):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary unitary matrix with a dimension that is a power of two.
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, U: TensorLike, wires: WiresLike):
        wires = Wires(wires)
        U_shape = math.shape(U)
        dim = 2 ** len(wires)

        # For pure QubitUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if len(U_shape) not in {2} or U_shape[-2:] != (dim, dim):
            raise ValueError(
                f"Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) "
                f"to act on {len(wires)} wires. Got shape {U_shape} instead."
            )

        super().__init__(U, wires=wires)

    @staticmethod
    def compute_matrix(U: TensorLike):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return U

    def adjoint(self) -> "QubitUnitary":
        adjoint_mat = self.U.conj().T
        # Note: it is necessary to explicitly cast back to csr, or it will become csc.
        return QubitUnitary(adjoint_mat, wires=self.wires)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


class PauliRot(Operation2):
    r"""
    Arbitrary Pauli word rotation.
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]
    static_argnames = ("pauli_word",)

    _ALLOWED_CHARACTERS = "IXYZ"

    _PAULI_CONJUGATION_MATRICES = {
        "X": Hadamard.compute_matrix(),
        "Y": RX.compute_matrix(np.pi / 2),
        "Z": np.array([[1, 0], [0, 1]]),
    }

    def __init__(self, theta: TensorLike, pauli_word: str, wires: WiresLike):
        super().__init__(theta, pauli_word=pauli_word, wires=wires)

        if not self.wires:
            raise ValueError(
                f"{self.name}: wrong number of wires. At least one wire has to be provided."
            )

        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed. '
                "Allowed characters are I, X, Y and Z"
            )

        num_wires = 1 if isinstance(wires, int) else len(wires)

        if not len(pauli_word) == num_wires:
            raise ValueError(
                f"The number of wires must be equal to the length of the Pauli word. "
                f"The Pauli word {pauli_word} has length {len(pauli_word)}, and "
                f"{num_wires} wires were given {wires}."
            )

    def __repr__(self) -> str:
        return f"PauliRot({self.theta}, {self.pauli_word}, wires={self.wires.tolist()})"

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        r"""A customizable string representation of the operator."""
        pauli_word = self.hyperparameters["pauli_word"]
        op_label = base_label or ("R" + pauli_word)

        # TODO[dwierichs]: Implement a proper label for parameter-broadcasted operators
        if decimals is not None and self.batch_size is None:
            param_string = f"\n({math.asarray(self.parameters[0]):.{decimals}f})"
            op_label += param_string

        return op_label

    @staticmethod
    def _check_pauli_word(pauli_word) -> bool:
        """Check that the given Pauli word has correct structure.

        Args:
            pauli_word (str): Pauli word to be checked

        Returns:
            bool: Whether the Pauli word has correct structure.
        """
        return all(pauli in PauliRot._ALLOWED_CHARACTERS for pauli in set(pauli_word))

    @staticmethod
    def compute_matrix(theta: TensorLike, pauli_word: str) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        dummy_wires = qml.wires.Wires(list(range(len(pauli_word))))
        return qml.matrix(qml.PauliRot(theta, pauli_word=pauli_word, wires=dummy_wires))

    @staticmethod
    def compute_eigvals(theta: TensorLike, pauli_word: str) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method)."""
        if (
            math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)

        # Identity must be treated specially because its eigenvalues are all the same
        if set(pauli_word) == {"I"}:
            return qml.GlobalPhase.compute_eigvals(0.5 * theta, n_wires=len(pauli_word))

        return qml.MultiRZ.compute_eigvals(theta, len(pauli_word))

    @staticmethod
    def compute_decomposition(theta: TensorLike, wires: WiresLike, pauli_word: str):
        r"""Representation of the operator as a product of other operators (static method)."""
        if isinstance(wires, int):  # Catch cases when the wire is passed as a single int.
            wires = [wires]

        # Check for identity and do nothing
        if set(pauli_word) == {"I"}:
            return [GlobalPhase(phi=theta / 2, wires=wires)]

        active_wires, active_gates = zip(
            *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
        )

        ops = []
        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(RX(np.pi / 2, wires=[wire]))

        ops.append(
            QubitUnitary(
                qml.matrix(qml.MultiRZ(theta, wires=list(active_wires))), wires=list(active_wires)
            )
        )

        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard(wires=[wire]))
            elif gate == "Y":
                ops.append(RX(-np.pi / 2, wires=[wire]))
        return ops

    def adjoint(self):
        return PauliRot(-self.theta, self.pauli_word, wires=self.wires)

    def pow(self, z):
        return [PauliRot(self.theta * z, self.pauli_word, wires=self.wires)]


class MidMeasure(Operator2):
    """Mid-circuit measurement."""

    def __repr__(self):
        return f"MidMeasure(wires={list(self.wires)}, postselect={self.postselect}, reset={self.reset})"

    num_wires = 1
    num_params = 0
    static_argnames = ("reset", "postselect", "meas_uid")

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires: Wires | None = None,
        reset: bool = False,
        postselect: int | None = None,
        meas_uid: str | None = None,
    ):
        super().__init__(wires=Wires(wires), reset=reset, postselect=postselect, meas_uid=meas_uid)
        self._name = "MidMeasureMP"

    @staticmethod
    def compute_diagonalizing_gates(*params, wires, **hyperparams):
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        r"""How the mid-circuit measurement is represented in diagrams and drawings."""
        _label = "┤↗"
        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label

    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        return hash((self.__class__.__name__, tuple(self.wires.tolist()), self.meas_uid))


class OutMultiplier(Operation2):
    r"""Performs the out-place modular multiplication operation."""

    grad_method = None

    static_argnames = ("mod",)
    wire_argnames = ("x_wires", "y_wires", "output_wires", "work_wires")

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        num_work_wires = len(work_wires)

        if mod is None:
            mod = 2 ** len(output_wires)
        if mod != 2 ** len(output_wires) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(output_wires)}, two work wires should be provided."
            )
        if mod > 2 ** (len(output_wires)):
            raise ValueError(
                "OutMultiplier must have enough wires to represent mod. The maximum mod "
                f"with len(output_wires)={len(output_wires)} is {2 ** len(output_wires)}, but received {mod}."
            )

        if len(work_wires) != 0:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")

        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        if any(wire in y_wires for wire in output_wires):
            raise ValueError("None of the wires in y_wires should be included in output_wires.")

        super().__init__(
            x_wires=x_wires,
            y_wires=y_wires,
            output_wires=output_wires,
            mod=mod,
            work_wires=work_wires,
        )
