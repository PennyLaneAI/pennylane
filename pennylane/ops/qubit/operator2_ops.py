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

from functools import lru_cache, reduce

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation2 import Operation2, Operator2
from pennylane.templates.core import AbstractArray
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .parametric_ops_single_qubit import stack_last

INV_SQRT2 = 1 / math.sqrt(2)


class Hadamard2(Operation2):
    r"""Hadamard2(wires)
    The Hadamard2 operator
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
        import jax

        wire = (
            self.wires[0]
            if not qml.math.is_abstract(self.wires)
            and not isinstance(self.wires, jax.core.ShapedArray)
            else self.wires
        )
        if isinstance(wire, str):
            return f"H('{wire}')"
        return f"H({wire})"

    @property
    def name(self) -> str:
        return "Hadamard2"

    @staticmethod
    @lru_cache
    def compute_matrix(wires) -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return [RY2(-np.pi / 4, wires=wires)]

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [
            PhaseShift2(np.pi / 2, wires=wires),
            RX2(np.pi / 2, wires=wires),
            PhaseShift2(np.pi / 2, wires=wires),
        ]

    def adjoint(self) -> "Hadamard2":
        return Hadamard2(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # H = RZ2(\pi) RY2(\pi/2) RZ2(0)
        return [np.pi, np.pi / 2, 0.0]


H2 = Hadamard2


def _hadamard_rz_rx_resources():
    return {RZ2: 2, RX2: 1, GlobalPhase2: 1}


@qml.decomposition.register_resources(_hadamard_rz_rx_resources)
def _hadamard_to_rz_rx(wires: WiresLike, **__):
    RZ2(np.pi / 2, wires=wires)
    RX2(np.pi / 2, wires=wires)
    RZ2(np.pi / 2, wires=wires)
    GlobalPhase2(-np.pi / 2, wires=wires)


def _hadamard_rz_ry_resources():
    return {RZ2: 1, RY2: 1, GlobalPhase2: 1}


@qml.decomposition.register_resources(_hadamard_rz_ry_resources)
def _hadamard_to_rz_ry(wires: WiresLike, **__):
    RZ2(np.pi, wires=wires)
    RY2(np.pi / 2, wires=wires)
    GlobalPhase2(-np.pi / 2)


qml.decomposition.add_decomps(Hadamard2, _hadamard_to_rz_rx, _hadamard_to_rz_ry)


class PauliX2(Operation2):
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
        import jax

        wire = (
            self.wires[0]
            if not qml.math.is_abstract(self.wires)
            and not isinstance(self.wires, jax.core.ShapedArray)
            else self.wires
        )
        if isinstance(wire, str):
            return f"X('{wire}')"
        return f"X({wire})"

    @property
    def name(self) -> str:
        return "PauliX2"

    @staticmethod
    @lru_cache
    def compute_matrix(wires) -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return [Hadamard2(wires=wires)]

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [RX2(np.pi, wires=wires), GlobalPhase2(-np.pi / 2, wires=wires)]

    def adjoint(self) -> "PauliX2":
        return X(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # X = RZ2(-\pi/2) RY2(\pi) RZ2(\pi/2)
        return [np.pi / 2, np.pi, -np.pi / 2]


X2 = PauliX2


def _paulix_to_rx_resources(*_, **__):
    return {GlobalPhase2: 1, RX2: 1}


@qml.decomposition.register_resources(_paulix_to_rx_resources)
def _paulix_to_rx(wires: WiresLike, **__):
    RX2(np.pi, wires=wires)
    GlobalPhase2(-np.pi / 2, wires=wires)


qml.decomposition.add_decomps(PauliX2, _paulix_to_rx)


class PauliY2(Operation2):
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
        import jax

        wire = (
            self.wires[0]
            if not qml.math.is_abstract(self.wires)
            and not isinstance(self.wires, jax.core.ShapedArray)
            else self.wires
        )
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
        return "PauliY2"

    @staticmethod
    @lru_cache
    def compute_matrix(wires) -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [RY2(np.pi, wires=wires), GlobalPhase2(-np.pi / 2, wires=wires)]

    def adjoint(self) -> "PauliY2":
        return Y(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # Y = RZ2(0) RY2(\pi) RZ2(0)
        return [0.0, np.pi, 0.0]


Y2 = PauliY2


def _pauliy_to_ry_gp_resources(*_, **__):
    return {GlobalPhase2: 1, RY2: 1}


@qml.decomposition.register_resources(_pauliy_to_ry_gp_resources)
def _pauliy_to_ry_gp(wires: WiresLike, **__):
    RY2(np.pi, wires=wires)
    GlobalPhase2(-np.pi / 2, wires=wires)


qml.decomposition.add_decomps(PauliY2, _pauliy_to_ry_gp)


class PauliZ2(Operation2):
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
        import jax

        wire = (
            self.wires[0]
            if not qml.math.is_abstract(self.wires)
            and not isinstance(self.wires, jax.core.ShapedArray)
            else self.wires
        )
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
        return "PauliZ2"

    @staticmethod
    @lru_cache
    def compute_matrix(wires) -> np.ndarray:  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method)."""
        return []

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [PhaseShift2(np.pi, wires=wires)]

    def adjoint(self) -> "PauliZ2":
        return Z(wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # Z = RZ2(\pi) RY2(0) RZ2(0)
        return [np.pi, 0.0, 0.0]


Z2 = PauliZ2


def _pauliz_to_ps_resources(*_, **__):
    return {PhaseShift2: 1}


@qml.decomposition.register_resources(_pauliz_to_ps_resources)
def _pauliz_to_ps(wires: WiresLike, **__):
    PhaseShift2(np.pi, wires=wires)


qml.decomposition.add_decomps(PauliZ2, _pauliz_to_ps)


class CNOT2(Operation2):
    r"""CNOT2(wires)
    The controlled-NOT operator
    """

    num_wires = 2
    """int: Number of wires that the operator acts on."""

    name = "CNOT2"

    def __init__(self, wires):
        super().__init__(wires=wires)

    def adjoint(self):
        return CNOT2(self.wires)

    def __repr__(self):
        return f"CNOT2(wires={self.wires.tolist()})"

    @staticmethod
    @lru_cache
    def compute_matrix(wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class RX2(Operation2):
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
    def compute_matrix(phi: TensorLike, wires) -> TensorLike:  # pylint: disable=arguments-differ
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

    def adjoint(self) -> "RX2":
        return RX2(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [RX2(self.phi * z, wires=self.wires)]

    def simplify(self) -> "RX2":
        phi = self.phi % (4 * np.pi)
        return RX2(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # RX2(\theta) = RZ2(-\pi/2) RY2(\theta) RZ2(\pi/2)
        pi_half = math.ones_like(self.phi) * (np.pi / 2)
        return [pi_half, self.phi, -pi_half]


def _rx_to_rot_resources(*_, **__):
    return {Rot2: 1}


@qml.decomposition.register_resources(_rx_to_rot_resources)
def _rx_to_rot(phi, wires: WiresLike, **__):
    Rot2(np.pi / 2, phi, 3.5 * np.pi, wires=wires)


def _rx_to_rz_ry_resources(*_, **__):
    return {RZ2: 2, RY2: 1}


@qml.decomposition.register_resources(_rx_to_rz_ry_resources)
def _rx_to_rz_ry(phi, wires: WiresLike, **__):
    RZ2(np.pi / 2, wires=wires)
    RY2(phi, wires=wires)
    RZ2(-np.pi / 2, wires=wires)


qml.decomposition.add_decomps(RX2, _rx_to_rot, _rx_to_rz_ry)


class RY2(Operation2):
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
    def compute_matrix(phi: TensorLike, wires) -> TensorLike:  # pylint: disable=arguments-differ
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

    def adjoint(self) -> "RY2":
        return RY2(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [RY2(self.phi * z, wires=self.wires)]

    def simplify(self) -> "RY2":
        phi = self.phi % (4 * np.pi)
        return RY2(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # RY2(\theta) = RZ2(0) RY2(\theta) RZ2(0)
        return [0.0, self.phi, 0.0]


def _ry_to_rot_resources(*_, **__):
    return {Rot2: 1}


@qml.decomposition.register_resources(_ry_to_rot_resources)
def _ry_to_rot(phi, wires: WiresLike, **__):
    Rot2(0, phi, 0, wires=wires)


def _ry_to_rz_rx_resources(*_, **__):
    return {RZ2: 2, RX2: 1}


@qml.decomposition.register_resources(_ry_to_rz_rx_resources)
def _ry_to_rz_rx(phi, wires: WiresLike, **__):
    RZ2(-np.pi / 2, wires=wires)
    RX2(phi, wires=wires)
    RZ2(np.pi / 2, wires=wires)


qml.decomposition.add_decomps(RY2, _ry_to_rot, _ry_to_rz_rx)


class RZ2(Operation2):
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
    def compute_matrix(phi: TensorLike, wires) -> TensorLike:  # pylint: disable=arguments-differ
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

    def adjoint(self) -> "RZ2":
        return RZ2(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [RZ2(self.phi * z, wires=self.wires)]

    def simplify(self) -> "RZ2":
        phi = self.phi % (4 * np.pi)
        return RZ2(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # RZ2(\theta) = RZ2(\theta) RY2(0) RZ2(0)
        return [self.phi, 0.0, 0.0]


def _rz_to_ps_resources(*_, **__):
    return {PhaseShift2: 1, GlobalPhase2: 1}


@qml.decomposition.register_resources(_rz_to_ps_resources)
def _rz_to_ps(phi, wires: WiresLike, **_):
    PhaseShift2(phi, wires)
    GlobalPhase2(phi / 2)


def _rz_to_rot_resources(*_, **__):
    return {Rot2: 1}


@qml.decomposition.register_resources(_rz_to_rot_resources)
def _rz_to_rot(phi, wires: WiresLike, **__):
    Rot2(0, 0, phi, wires=wires)


def _rz_to_ry_rx_resources(*_, **__):
    return {RY2: 2, RX2: 1}


@qml.decomposition.register_resources(_rz_to_ry_rx_resources)
def _rz_to_ry_rx(phi, wires: WiresLike, **__):
    RY2(np.pi / 2, wires=wires)
    RX2(phi, wires=wires)
    RY2(-np.pi / 2, wires=wires)


qml.decomposition.add_decomps(RZ2, _rz_to_ps, _rz_to_rot, _rz_to_ry_rx)


class PhaseShift2(Operation2):
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
    def compute_matrix(phi: TensorLike, wires) -> TensorLike:  # pylint: disable=arguments-differ
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
    def compute_decomposition(phi: TensorLike, wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method)."""
        return [RZ2(phi, wires=wires), GlobalPhase2(-phi / 2)]

    def adjoint(self) -> "PhaseShift2":
        return PhaseShift2(-self.phi, wires=self.wires)

    def pow(self, z: int | float):
        return [PhaseShift2(self.phi * z, wires=self.wires)]

    def simplify(self) -> "PhaseShift2":
        phi = self.phi % (2 * np.pi)
        return PhaseShift2(phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        # PhaseShift2(\theta) = RZ2(\theta) RY2(0) RZ2(0)
        return [self.phi, 0.0, 0.0]


def _phaseshift_to_rz_gp_resources(*_, **__):
    return {RZ2: 1, GlobalPhase2: 1}


@qml.decomposition.register_resources(_phaseshift_to_rz_gp_resources)
def _phaseshift_to_rz_gp(phi, wires: WiresLike, **__):
    RZ2(phi, wires=wires)
    GlobalPhase2(-phi / 2)


qml.decomposition.add_decomps(PhaseShift2, _phaseshift_to_rz_gp)


class Rot2(Operation2):
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
    def compute_matrix(phi: TensorLike, theta: TensorLike, omega: TensorLike, wires) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        # It might be that they are in different interfaces, e.g.,
        # Rot2(0.2, 0.3, tf.Variable(0.5), wires=0)
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
            RZ2(phi, wires=wires),
            RY2(theta, wires=wires),
            RZ2(omega, wires=wires),
        ]

    def adjoint(self) -> "Rot2":
        return Rot2(-self.omega, -self.theta, -self.phi, wires=self.wires)

    def single_qubit_rot_angles(self) -> list[TensorLike]:
        return [self.phi, self.theta, self.omega]


def _rot_to_rz_ry_rz_resources(*_, **__):
    return {RZ2: 2, RY2: 1}


@qml.decomposition.register_resources(_rot_to_rz_ry_rz_resources)
def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
    RZ2(phi, wires=wires)
    RY2(theta, wires=wires)
    RZ2(omega, wires=wires)


qml.decomposition.add_decomps(Rot2, _rot_to_rz_ry_rz)


class GlobalPhase2(Operation2):
    r"""A global phase operation that multiplies all components of the state by :math:`e^{-i \phi}`."""

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"

    def __init__(self, phi, wires=()):
        super().__init__(phi, wires=wires)

    @staticmethod
    def compute_matrix(phi, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        n_wires = len(wires)
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
        return GlobalPhase2(-1 * self.phi, self.wires)

    def pow(self, z):
        return [GlobalPhase2(z * self.phi, self.wires)]


class QubitUnitary2(Operation2):
    r"""QubitUnitary2(U, wires)
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

        # For pure QubitUnitary2 operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if len(U_shape) not in {2} or U_shape[-2:] != (dim, dim):
            raise ValueError(
                f"Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) "
                f"to act on {len(wires)} wires. Got shape {U_shape} instead."
            )

        super().__init__(U, wires=wires)

    @staticmethod
    def compute_matrix(U: TensorLike, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        return U

    def adjoint(self) -> "QubitUnitary2":
        adjoint_mat = self.U.conj().T
        # Note: it is necessary to explicitly cast back to csr, or it will become csc.
        return QubitUnitary2(adjoint_mat, wires=self.wires)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


class MultiRZ2(Operation2):
    r"""
    Arbitrary multi Z rotation.
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, theta: TensorLike, wires: WiresLike):
        super().__init__(theta, wires=wires)

    @staticmethod
    def compute_matrix(theta: TensorLike, wires) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        num_wires = len(wires)
        eigs = math.convert_like(qml.pauli.pauli_eigs(num_wires), theta)

        if (
            math.get_interface(theta) == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)
            eigs = math.cast_like(eigs, 1j)

        if math.ndim(theta) == 0:
            return math.diag(math.exp(-0.5j * theta * eigs))

        diags = math.exp(math.outer(-0.5j * theta, eigs))
        return diags[:, :, np.newaxis] * math.cast_like(math.eye(2**num_wires, like=diags), diags)

    @staticmethod
    def compute_decomposition(  # pylint: disable=arguments-differ,unused-argument
        theta: TensorLike, wires: WiresLike
    ):
        r"""Representation of the operator as a product of other operators (static method)."""
        ops = [CNOT2(wires=(w0, w1)) for w0, w1 in zip(wires[~0:0:-1], wires[~1::-1])]
        ops.append(RZ2(theta, wires=wires[0]))
        ops += [CNOT2(wires=(w0, w1)) for w0, w1 in zip(wires[1:], wires[:~0])]

        return ops

    def adjoint(self) -> "MultiRZ2":
        return MultiRZ2(-self.theta, wires=self.wires)

    def pow(self, z: int | float):
        return [MultiRZ2(self.theta * z, wires=self.wires)]

    def simplify(self) -> "MultiRZ2":
        theta = self.theta % (4 * np.pi)
        return MultiRZ2(theta, wires=self.wires)


def _multi_rz_decomposition_resources(theta, wires):
    num_wires = len(wires)
    return {RZ2: 1, CNOT2: 2 * (num_wires - 1)}


@qml.decomposition.register_resources(_multi_rz_decomposition_resources)
def _multi_rz_decomposition(theta: TensorLike, wires: WiresLike, **__):
    @qml.for_loop(len(wires) - 1, 0, -1)
    def _pre_cnot(i):
        CNOT2(wires=(wires[i], wires[i - 1]))

    @qml.for_loop(1, len(wires), 1)
    def _post_cnot(i):
        CNOT2(wires=(wires[i], wires[i - 1]))

    _pre_cnot()  # pylint: disable=no-value-for-parameter
    RZ2(theta, wires=wires[0])
    _post_cnot()  # pylint: disable=no-value-for-parameter


qml.decomposition.add_decomps(MultiRZ2, _multi_rz_decomposition)


class PauliRot2(Operation2):
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
        "X": Hadamard2.compute_matrix(0),
        "Y": RX2.compute_matrix(np.pi / 2, 0),
        "Z": np.array([[1, 0], [0, 1]]),
    }

    def __init__(self, theta: TensorLike, pauli_word: str, wires: WiresLike):
        super().__init__(theta, pauli_word=pauli_word, wires=wires)

        if not self.wires:
            raise ValueError(
                f"{self.name}: wrong number of wires. At least one wire has to be provided."
            )

        if not PauliRot2._check_pauli_word(pauli_word):
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
        return f"PauliRot2({self.theta}, {self.pauli_word}, wires={self.wires.tolist()})"

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
        return all(pauli in PauliRot2._ALLOWED_CHARACTERS for pauli in set(pauli_word))

    @staticmethod
    def compute_matrix(theta: TensorLike, pauli_word: str, wires) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method)."""
        if not PauliRot2._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed. '
                "Allowed characters are I, X, Y and Z"
            )

        interface = math.get_interface(theta)

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            theta = math.cast_like(theta, 1j)

        # Simplest case is if the Pauli is the identity matrix
        if set(pauli_word) == {"I"}:
            return GlobalPhase2.compute_matrix(0.5 * theta, wires=wires)

        # We first generate the matrix excluding the identity parts and expand it afterwards.
        # To this end, we have to store on which wires the non-identity parts act
        non_identity_wires, non_identity_gates = zip(
            *[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != "I"]
        )

        multi_Z_rot_matrix = MultiRZ2.compute_matrix(theta, non_identity_wires)

        # now we conjugate with Hadamard2 and RX2 to create the Pauli string
        conjugation_matrix = reduce(
            math.kron,
            [PauliRot2._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates],
        )
        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            conjugation_matrix = math.cast_like(conjugation_matrix, 1j)
        # Note: we use einsum with reverse arguments here because it is not multi-dispatched
        # and the tensordot containing multi_Z_rot_matrix should decide about the interface
        return math.expand_matrix(
            math.einsum(
                "...jk,ij->...ik",
                math.tensordot(multi_Z_rot_matrix, conjugation_matrix, axes=[[-1], [0]]),
                math.conj(conjugation_matrix),
            ),
            non_identity_wires,
            list(range(len(pauli_word))),
        )

    @staticmethod
    def compute_decomposition(theta: TensorLike, pauli_word: str, wires):
        r"""Representation of the operator as a product of other operators (static method)."""
        if isinstance(wires, int):  # Catch cases when the wire is passed as a single int.
            wires = [wires]

        # Check for identity and do nothing
        if set(pauli_word) == {"I"}:
            return [GlobalPhase2(phi=theta / 2)]

        active_wires, active_gates = zip(
            *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
        )

        ops = []
        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard2(wires=[wire]))
            elif gate == "Y":
                ops.append(RX2(np.pi / 2, wires=[wire]))

        ops.append(MultiRZ2(theta, wires=list(active_wires)))

        for wire, gate in zip(active_wires, active_gates):
            if gate == "X":
                ops.append(Hadamard2(wires=[wire]))
            elif gate == "Y":
                ops.append(RX2(-np.pi / 2, wires=[wire]))
        return ops

    def adjoint(self):
        return PauliRot2(-self.theta, self.pauli_word, wires=self.wires)

    def pow(self, z):
        return [PauliRot2(self.theta * z, self.pauli_word, wires=self.wires)]


def _pauli_rot_resources(theta, pauli_word, wires):
    if set(pauli_word) == {"I"}:
        return {GlobalPhase2: 1}
    num_active_wires = len(pauli_word.replace("I", ""))
    return {
        Hadamard2: 2 * pauli_word.count("X"),
        RX2: 2 * pauli_word.count("Y"),
        MultiRZ2(AbstractArray((), float), AbstractArray((num_active_wires,), int)): 1,
    }


@qml.decomposition.register_resources(_pauli_rot_resources)
def _pauli_rot_decomposition(theta: TensorLike, pauli_word, wires: WiresLike):
    if set(pauli_word) == {"I"}:
        GlobalPhase2(theta / 2, wires=wires[0])
        return

    active_wires, active_gates = zip(
        *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
    )

    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            Hadamard2(wires=[wire])
        elif gate == "Y":
            RX2(np.pi / 2, wires=[wire])

    MultiRZ2(theta, wires=list(active_wires))

    for wire, gate in zip(active_wires, active_gates):
        if gate == "X":
            Hadamard2(wires=[wire])
        elif gate == "Y":
            RX2(-np.pi / 2, wires=[wire])


qml.decomposition.add_decomps(PauliRot2, _pauli_rot_decomposition)


class MidMeasure2(Operator2):
    """Mid-circuit measurement."""

    def __repr__(self):
        return f"MidMeasure2(wires={list(self.wires)}, postselect={self.postselect}, reset={self.reset})"

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


class OutMultiplier2(Operation2):
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
                "OutMultiplier2 must have enough wires to represent mod. The maximum mod "
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
