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
This submodule contains the discrete-variable quantum operations that are the
core parameterized gates.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import functools
import math
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ, Hadamard
from pennylane.utils import expand, pauli_eigs
from pennylane.wires import Wires


INV_SQRT2 = 1 / math.sqrt(2)


class RX(Operation):
    r"""RX(phi, wires)
    The single qubit X rotation

    .. math:: R_x(\phi) = e^{-i\phi\sigma_x/2} = \begin{bmatrix}
                \cos(\phi/2) & -i\sin(\phi/2) \\
                -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_x(\phi)) = \frac{1}{2}\left[f(R_x(\phi+\pi/2)) - f(R_x(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_x(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    basis = "X"
    grad_method = "A"
    generator = [PauliX, -1 / 2]

    @property
    def num_params(self):
        return 1

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        js = -1j * s

        return qml.math.diag([c, c]) + qml.math.stack(
            [qml.math.stack([0, js]), qml.math.stack([js, 0])]
        )

    def adjoint(self):
        return RX(-self.data[0], wires=self.wires)

    def _controlled(self, wire):
        CRX(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        # RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
        return [np.pi / 2, self.data[0], -np.pi / 2]


class RY(Operation):
    r"""RY(phi, wires)
    The single qubit Y rotation

    .. math:: R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{bmatrix}
                \cos(\phi/2) & -\sin(\phi/2) \\
                \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_y(\phi)) = \frac{1}{2}\left[f(R_y(\phi+\pi/2)) - f(R_y(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_y(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    basis = "Y"
    grad_method = "A"
    generator = [PauliY, -1 / 2]

    @property
    def num_params(self):
        return 1

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        return qml.math.diag([c, c]) + qml.math.stack(
            [qml.math.stack([0, -s]), qml.math.stack([s, 0])]
        )

    def adjoint(self):
        return RY(-self.data[0], wires=self.wires)

    def _controlled(self, wire):
        CRY(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        # RY(\theta) = RZ(0) RY(\theta) RZ(0)
        return [0.0, self.data[0], 0.0]


class RZ(Operation):
    r"""RZ(phi, wires)
    The single qubit Z rotation

    .. math:: R_z(\phi) = e^{-i\phi\sigma_z/2} = \begin{bmatrix}
                e^{-i\phi/2} & 0 \\
                0 & e^{i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_z(\phi)) = \frac{1}{2}\left[f(R_z(\phi+\pi/2)) - f(R_z(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_z(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    basis = "Z"
    grad_method = "A"
    generator = [PauliZ, -1 / 2]

    @property
    def num_params(self):
        return 1

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        p = qml.math.exp(-0.5j * theta)

        return qml.math.diag([p, qml.math.conj(p)])

    @classmethod
    def _eigvals(cls, *params):
        theta = params[0]

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        p = qml.math.exp(-0.5j * theta)

        return qml.math.stack([p, qml.math.conj(p)])

    def adjoint(self):
        return RZ(-self.data[0], wires=self.wires)

    def _controlled(self, wire):
        CRZ(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        # RZ(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.data[0], 0.0, 0.0]


class PhaseShift(Operation):
    r"""PhaseShift(phi, wires)
    Arbitrary single qubit local phase shift

    .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_\phi(\phi)) = \frac{1}{2}\left[f(R_\phi(\phi+\pi/2)) - f(R_\phi(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    basis = "Z"
    grad_method = "A"
    generator = [np.array([[0, 0], [0, 1]]), 1]

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "Rϕ")

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        return qml.math.diag([1, exp_part])

    @classmethod
    def _eigvals(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        return qml.math.stack([1, exp_part])

    @staticmethod
    def decomposition(phi, wires):
        decomp_ops = [RZ(phi, wires=wires)]
        return decomp_ops

    def adjoint(self):
        return PhaseShift(-self.data[0], wires=self.wires)

    def _controlled(self, wire):
        ControlledPhaseShift(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        # PhaseShift(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.data[0], 0.0, 0.0]


class ControlledPhaseShift(Operation):
    r"""ControlledPhaseShift(phi, wires)
    A qubit controlled phase shift.

    .. math:: CR_\phi(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\phi}
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_\phi(\phi)) = \frac{1}{2}\left[f(CR_\phi(\phi+\pi/2)) - f(CR_\phi(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`CR_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
    """
    num_wires = 2
    basis = "Z"
    grad_method = "A"
    generator = [np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]), 1]

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "Rϕ")

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        return qml.math.diag([1, 1, 1, exp_part])

    @classmethod
    def _eigvals(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        return qml.math.stack([1, 1, 1, exp_part])

    @staticmethod
    def decomposition(phi, wires):
        decomp_ops = [
            qml.PhaseShift(phi / 2, wires=wires[0]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(phi / 2, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        return ControlledPhaseShift(-self.data[0], wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


CPhase = ControlledPhaseShift


class Rot(Operation):
    r"""Rot(phi, theta, omega, wires)
    Arbitrary single qubit rotation

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R(\phi, \theta, \omega)) = \frac{1}{2}\left[f(R(\phi+\pi/2, \theta, \omega)) - f(R(\phi-\pi/2, \theta, \omega))\right]`
      where :math:`f` is an expectation value depending on :math:`R(\phi, \theta, \omega)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    .. note::

        If the ``Rot`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.RZ` and :class:`~.RY` gates.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    grad_method = "A"

    @property
    def num_params(self):
        return 3

    @classmethod
    def _matrix(cls, *params):
        # There are three input parameters to be dealt with
        phi, theta, omega = params

        # It might be that they are in different interfaces, e.g.,
        # Rot(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math._multi_dispatch(params)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = [
            [
                qml.math.exp(-0.5j * (phi + omega)) * c,
                -qml.math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                qml.math.exp(-0.5j * (phi - omega)) * s,
                qml.math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return qml.math.stack([qml.math.stack(row) for row in mat])

    @staticmethod
    def decomposition(phi, theta, omega, wires):
        decomp_ops = [
            RZ(phi, wires=wires),
            RY(theta, wires=wires),
            RZ(omega, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        phi, theta, omega = self.parameters
        return Rot(-omega, -theta, -phi, wires=self.wires)

    def _controlled(self, wire):
        CRot(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        return self.data


class MultiRZ(Operation):
    r"""MultiRZ(theta, wires)
    Arbitrary multi Z rotation.

    .. math::

        MultiRZ(\theta) = \exp(-i \frac{\theta}{2} Z^{\otimes n})

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\theta}f(MultiRZ(\theta)) = \frac{1}{2}\left[f(MultiRZ(\theta +\pi/2)) - f(MultiRZ(\theta-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`MultiRZ(\theta)`.

    .. note::

        If the ``MultiRZ`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RZ` and :class:`~.CNOT` gates.

    Args:
        theta (float): rotation angle :math:`\theta`
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_wires = AnyWires
    grad_method = "A"

    @property
    def num_params(self):
        return 1

    @classmethod
    def _matrix(cls, theta, n):
        """Matrix representation of a MultiRZ gate.

        Args:
            theta (float): Rotation angle.
            n (int): Number of wires the rotation acts on. This has
                to be given explicitly in the static method as the
                wires object is not available.

        Returns:
            array[complex]: The matrix representation
        """
        multi_Z_rot_eigs = MultiRZ._eigvals(theta, n)
        multi_Z_rot_matrix = qml.math.diag(multi_Z_rot_eigs)

        return multi_Z_rot_matrix

    _generator = None

    @property
    def generator(self):
        if self._generator is None:
            self._generator = [np.diag(pauli_eigs(len(self.wires))), -1 / 2]
        return self._generator

    @property
    def matrix(self):
        # Redefine the property here to pass additionally the number of wires to the ``_matrix`` method
        if self.inverse:
            # The matrix is diagonal, so there is no need to transpose
            return qml.math.conj(self._matrix(*self.parameters, len(self.wires)))

        return self._matrix(*self.parameters, len(self.wires))

    @classmethod
    def _eigvals(cls, theta, n):
        eigs = qml.math.convert_like(pauli_eigs(n), theta)

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
            eigs = qml.math.cast_like(eigs, 1j)

        return qml.math.exp(-1j * theta / 2 * eigs)

    @property
    def eigvals(self):
        # Redefine the property here to pass additionally the number of wires to the ``_eigvals`` method
        if self.inverse:
            return qml.math.conj(self._eigvals(*self.parameters, len(self.wires)))

        return self._eigvals(*self.parameters, len(self.wires))

    @staticmethod
    def decomposition(theta, wires):
        with qml.tape.OperationRecorder() as rec:
            for i in range(len(wires) - 1, 0, -1):
                qml.CNOT(wires=[wires[i], wires[i - 1]])

            RZ(theta, wires=wires[0])

            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i + 1], wires[i]])
        return rec.queue

    def adjoint(self):
        return MultiRZ(-self.parameters[0], wires=self.wires)


class PauliRot(Operation):
    r"""PauliRot(theta, pauli_word, wires)
    Arbitrary Pauli word rotation.

    .. math::

        RP(\theta, P) = \exp(-i \frac{\theta}{2} P)

    **Details:**

    * Number of wires: Any
    * Number of parameters: 2 (1 differentiable parameter)
    * Gradient recipe: :math:`\frac{d}{d\theta}f(RP(\theta)) = \frac{1}{2}\left[f(RP(\theta +\pi/2)) - f(RP(\theta-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`RP(\theta)`.

    .. note::

        If the ``PauliRot`` gate is not supported on the targeted device, PennyLane
        will decompose the gate using :class:`~.RX`, :class:`~.Hadamard`, :class:`~.RZ`
        and :class:`~.CNOT` gates.

    Args:
        theta (float): rotation angle :math:`\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Sequence[int] or int): the wire the operation acts on

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.PauliRot(0.5, 'X',  wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(example_circuit())
    0.8775825618903724
    """
    num_wires = AnyWires
    do_check_domain = False
    grad_method = "A"

    _ALLOWED_CHARACTERS = "IXYZ"

    _PAULI_CONJUGATION_MATRICES = {
        "X": Hadamard._matrix(),
        "Y": RX._matrix(np.pi / 2),
        "Z": np.array([[1, 0], [0, 1]]),
    }

    def __init__(self, theta, pauli_word, wires=None, do_queue=True):
        super().__init__(theta, pauli_word, wires=wires, do_queue=do_queue)

        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed.'
                " Allowed characters are I, X, Y and Z"
            )

        num_wires = 1 if isinstance(wires, int) else len(wires)

        if not len(pauli_word) == num_wires:
            raise ValueError(
                f"The given Pauli word has length {len(pauli_word)}, length {num_wires} was expected for wires {wires}"
            )

    @property
    def num_params(self):
        return 2

    def label(self, decimals=None, base_label=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.PauliRot(0.1, "XYY", wires=(0,1,2))
        >>> op.label()
        'R(XYY)'
        >>> op.label(decimals=2)
        'R(XYY)\n(0.10)'
        >>> op.label(base_label="PauliRot")
        'PauliRot\n(0.10)'

        """
        op_label = base_label or ("R(" + self.parameters[1] + ")")

        if self.inverse:
            op_label += "⁻¹"

        if decimals is not None:
            param_string = f"\n({qml.math.asarray(self.parameters[0]):.{decimals}f})"
            op_label += param_string

        return op_label

    @staticmethod
    def _check_pauli_word(pauli_word):
        """Check that the given Pauli word has correct structure.

        Args:
            pauli_word (str): Pauli word to be checked

        Returns:
            bool: Whether the Pauli word has correct structure.
        """
        return all(pauli in PauliRot._ALLOWED_CHARACTERS for pauli in pauli_word)

    @classmethod
    def _matrix(cls, *params):
        pauli_word = params[1]

        if not PauliRot._check_pauli_word(pauli_word):
            raise ValueError(
                f'The given Pauli word "{pauli_word}" contains characters that are not allowed.'
                " Allowed characters are I, X, Y and Z"
            )

        theta = params[0]

        interface = qml.math.get_interface(theta)

        if interface == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        # Simplest case is if the Pauli is the identity matrix
        if pauli_word == "I" * len(pauli_word):

            exp = qml.math.exp(-1j * theta / 2)
            iden = qml.math.eye(2 ** len(pauli_word))
            if interface == "torch":
                # Use convert_like to ensure that the tensor is put on the correct
                # Torch device
                iden = qml.math.convert_like(iden, theta)
                return exp * iden

            return qml.math.array(exp * iden, like=interface)

        # We first generate the matrix excluding the identity parts and expand it afterwards.
        # To this end, we have to store on which wires the non-identity parts act
        non_identity_wires, non_identity_gates = zip(
            *[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != "I"]
        )

        multi_Z_rot_matrix = MultiRZ._matrix(theta, len(non_identity_gates))

        # now we conjugate with Hadamard and RX to create the Pauli string
        conjugation_matrix = functools.reduce(
            qml.math.kron,
            [PauliRot._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates],
        )

        return expand(
            qml.math.dot(
                qml.math.conj(conjugation_matrix),
                qml.math.dot(multi_Z_rot_matrix, conjugation_matrix),
            ),
            non_identity_wires,
            list(range(len(pauli_word))),
        )

    _generator = None

    @property
    def generator(self):
        if self._generator is None:
            pauli_word = self.parameters[1]

            # Simplest case is if the Pauli is the identity matrix
            if pauli_word == "I" * len(pauli_word):
                self._generator = [np.eye(2 ** len(pauli_word)), -1 / 2]
                return self._generator

            # We first generate the matrix excluding the identity parts and expand it afterwards.
            # To this end, we have to store on which wires the non-identity parts act
            non_identity_wires, non_identity_gates = zip(
                *[(wire, gate) for wire, gate in enumerate(pauli_word) if gate != "I"]
            )

            # get MultiRZ's generator
            multi_Z_rot_generator = qml.math.diag(pauli_eigs(len(non_identity_gates)))

            # now we conjugate with Hadamard and RX to create the Pauli string
            conjugation_matrix = functools.reduce(
                qml.math.kron,
                [PauliRot._PAULI_CONJUGATION_MATRICES[gate] for gate in non_identity_gates],
            )

            self._generator = [
                expand(
                    qml.math.dot(
                        qml.math.conj(qml.math.T(conjugation_matrix)),
                        qml.math.dot(multi_Z_rot_generator, conjugation_matrix),
                    ),
                    non_identity_wires,
                    list(range(len(pauli_word))),
                ),
                -1 / 2,
            ]

        return self._generator

    @classmethod
    def _eigvals(cls, theta, pauli_word):
        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        # Identity must be treated specially because its eigenvalues are all the same
        if pauli_word == "I" * len(pauli_word):
            return qml.math.exp(-1j * theta / 2) * qml.math.ones(2 ** len(pauli_word))

        return MultiRZ._eigvals(theta, len(pauli_word))

    @staticmethod
    def decomposition(theta, pauli_word, wires):
        # Catch cases when the wire is passed as a single int.
        if isinstance(wires, int):
            wires = [wires]
        with qml.tape.OperationRecorder() as rec:
            # Check for identity and do nothing
            if pauli_word == "I" * len(wires):
                return []

            active_wires, active_gates = zip(
                *[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != "I"]
            )

            for wire, gate in zip(active_wires, active_gates):
                if gate == "X":
                    Hadamard(wires=[wire])
                elif gate == "Y":
                    RX(np.pi / 2, wires=[wire])

            MultiRZ(theta, wires=list(active_wires))

            for wire, gate in zip(active_wires, active_gates):
                if gate == "X":
                    Hadamard(wires=[wire])
                elif gate == "Y":
                    RX(-np.pi / 2, wires=[wire])
        return rec.queue

    def adjoint(self):
        return PauliRot(-self.parameters[0], self.parameters[1], wires=self.wires)


# Four term gradient recipe for controlled rotations
c1 = INV_SQRT2 * (np.sqrt(2) + 1) / 4
c2 = INV_SQRT2 * (np.sqrt(2) - 1) / 4
a = np.pi / 2
b = 3 * np.pi / 2
four_term_grad_recipe = ([[c1, 1, a], [-c1, 1, -a], [-c2, 1, b], [c2, 1, -b]],)


class CRX(Operation):
    r"""CRX(phi, wires)
    The controlled-RX operator

    .. math::

        \begin{align}
            CR_x(\phi) &=
            \begin{bmatrix}
            & 1 & 0 & 0 & 0 \\
            & 0 & 1 & 0 & 0\\
            & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\
            & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The controlled-RX operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\phi}f(CR_x(\phi)) = c_+ \left[f(CR_x(\phi+a)) - f(CR_x(\phi-a))\right] - c_- \left[f(CR_x(\phi+b)) - f(CR_x(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_x(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
    """
    num_wires = 2
    basis = "X"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe

    generator = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        -1 / 2,
    ]

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "RX")

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if interface == "torch":
            # Use convert_like to ensure that the tensor is put on the correct
            # Torch device
            z = qml.math.convert_like(qml.math.zeros([4]), theta)
        else:
            z = qml.math.zeros([4], like=interface)

        if interface == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
            z = qml.math.cast_like(z, 1j)

        js = -1j * s

        mat = qml.math.diag([1, 1, c, c])
        return mat + qml.math.stack(
            [z, z, qml.math.stack([0, 0, 0, js]), qml.math.stack([0, 0, js, 0])]
        )

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            RZ(np.pi / 2, wires=wires[1]),
            RY(theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RY(-theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RZ(-np.pi / 2, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        return CRX(-self.data[0], wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CRY(Operation):
    r"""CRY(phi, wires)
    The controlled-RY operator

    .. math::

        \begin{align}
            CR_y(\phi) &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\
                0 & 0 & \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The controlled-RY operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\phi}f(CR_y(\phi)) = c_+ \left[f(CR_y(\phi+a)) - f(CR_y(\phi-a))\right] - c_- \left[f(CR_y(\phi+b)) - f(CR_y(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_y(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
    """
    num_wires = 2
    basis = "Y"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe

    generator = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        -1 / 2,
    ]

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "RY")

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]
        interface = qml.math.get_interface(theta)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if interface == "torch":
            # Use convert_like to ensure that the tensor is put on the correct
            # Torch device
            z = qml.math.convert_like(qml.math.zeros([4]), theta)
        else:
            z = qml.math.zeros([4], like=interface)

        mat = qml.math.diag([1, 1, c, c])
        return mat + qml.math.stack(
            [z, z, qml.math.stack([0, 0, 0, -s]), qml.math.stack([0, 0, s, 0])]
        )

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            RY(theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RY(-theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return CRY(-self.data[0], wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CRZ(Operation):
    r"""CRZ(phi, wires)
    The controlled-RZ operator

    .. math::

        \begin{align}
             CR_z(\phi) &=
             \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & e^{-i\phi/2} & 0\\
                0 & 0 & 0 & e^{i\phi/2}
            \end{bmatrix}.
        \end{align}


    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The controlled-RZ operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\phi}f(CR_z(\phi)) = c_+ \left[f(CR_z(\phi+a)) - f(CR_z(\phi-a))\right] - c_- \left[f(CR_z(\phi+b)) - f(CR_z(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_z(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
    """
    num_wires = 2
    basis = "Z"
    grad_method = "A"
    grad_recipe = four_term_grad_recipe

    generator = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        -1 / 2,
    ]

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "RZ")

    @classmethod
    def _matrix(cls, *params):
        theta = params[0]

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        exp_part = qml.math.exp(-0.5j * theta)

        return qml.math.diag([1, 1, exp_part, qml.math.conj(exp_part)])

    @classmethod
    def _eigvals(cls, *params):
        theta = qml.math.flatten(qml.math.stack([params[0]]))[0]

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        exp_part = qml.math.exp(-0.5j * theta)

        return qml.math.stack([1, 1, exp_part, qml.math.conj(exp_part)])

    @staticmethod
    def decomposition(lam, wires):
        decomp_ops = [
            PhaseShift(lam / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            PhaseShift(-lam / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return CRZ(-self.data[0], wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CRot(Operation):
    r"""CRot(phi, theta, omega, wires)
    The controlled-Rot operator

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 3
    * Gradient recipe: The controlled-Rot operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\mathbf{x}_i}f(CR(\mathbf{x}_i)) = c_+ \left[f(CR(\mathbf{x}_i+a)) - f(CR(\mathbf{x}_i-a))\right] - c_- \left[f(CR(\mathbf{x}_i+b)) - f(CR(\mathbf{x}_i-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR(\mathbf{x}_i)`, and

      - :math:`\mathbf{x} = (\phi, \theta, \omega)` and `i` is an index to :math:`\mathbf{x}`
      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int]): the wire the operation acts on
    """
    num_wires = 2
    grad_method = "A"
    grad_recipe = four_term_grad_recipe * 3

    @property
    def num_params(self):
        return 3

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "Rot")

    @classmethod
    def _matrix(cls, *params):
        phi, theta, omega = params

        # It might be that they are in different interfaces, e.g.,
        # Rot(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math._multi_dispatch(params)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                qml.math.exp(-0.5j * (phi + omega)) * c,
                -qml.math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                0,
                0,
                qml.math.exp(-0.5j * (phi - omega)) * s,
                qml.math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return qml.math.stack([qml.math.stack(row) for row in mat])

    @staticmethod
    def decomposition(phi, theta, omega, wires):
        decomp_ops = [
            RZ((phi - omega) / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RZ(-(phi + omega) / 2, wires=wires[1]),
            RY(-theta / 2, wires=wires[1]),
            qml.CNOT(wires=wires),
            RY(theta / 2, wires=wires[1]),
            RZ(omega, wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        phi, theta, omega = self.parameters
        return CRot(-omega, -theta, -phi, wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class U1(Operation):
    r"""U1(phi)
    U1 gate.

    .. math:: U_1(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.

    .. note::

        The ``U1`` gate is an alias for the phase shift operation :class:`~.PhaseShift`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_1(\phi)) = \frac{1}{2}\left[f(U_1(\phi+\pi/2)) - f(U_1(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_1(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    grad_method = "A"
    generator = [np.array([[0, 0], [0, 1]]), 1]

    @property
    def num_params(self):
        return 1

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        return qml.math.diag([1, exp_part])

    @staticmethod
    def decomposition(phi, wires):
        return [PhaseShift(phi, wires=wires)]

    def adjoint(self):
        return U1(-self.data[0], wires=self.wires)


class U2(Operation):
    r"""U2(phi, lambda, wires)
    U2 gate.

    .. math::

        U_2(\phi, \lambda) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -\exp(i \lambda)
        \\ \exp(i \phi) & \exp(i (\phi + \lambda)) \end{bmatrix}

    The :math:`U_2` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_2(\phi, \lambda) = R_\phi(\phi+\lambda) R(\lambda,\pi/2,-\lambda)

    .. note::

        If the ``U2`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.Rot` and :class:`~.PhaseShift` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_2(\phi, \lambda)) = \frac{1}{2}\left[f(U_2(\phi+\pi/2, \lambda)) - f(U_2(\phi-\pi/2, \lambda))\right]`
      where :math:`f` is an expectation value depending on :math:`U_2(\phi, \lambda)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \lambda\}`.

    Args:
        phi (float): azimuthal angle :math:`\phi`
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_wires = 1
    grad_method = "A"

    @property
    def num_params(self):
        return 2

    @classmethod
    def _matrix(cls, *params):
        phi, lam = params

        interface = qml.math._multi_dispatch(params)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            lam = qml.math.cast_like(qml.math.asarray(lam, like=interface), 1j)

        mat = [
            [1, -qml.math.exp(1j * lam)],
            [qml.math.exp(1j * phi), qml.math.exp(1j * (phi + lam))],
        ]

        return INV_SQRT2 * qml.math.stack([qml.math.stack(row) for row in mat])

    @staticmethod
    def decomposition(phi, lam, wires):
        decomp_ops = [
            Rot(lam, np.pi / 2, -lam, wires=wires),
            PhaseShift(lam, wires=wires),
            PhaseShift(phi, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        phi, lam = self.parameters
        new_lam = (np.pi - phi) % (2 * np.pi)
        new_phi = (np.pi - lam) % (2 * np.pi)
        return U2(new_phi, new_lam, wires=self.wires)


class U3(Operation):
    r"""U3(theta, phi, lambda, wires)
    Arbitrary single qubit unitary.

    .. math::

        U_3(\theta, \phi, \lambda) = \begin{bmatrix} \cos(\theta/2) & -\exp(i \lambda)\sin(\theta/2) \\
        \exp(i \phi)\sin(\theta/2) & \exp(i (\phi + \lambda))\cos(\theta/2) \end{bmatrix}

    The :math:`U_3` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_3(\theta, \phi, \lambda) = R_\phi(\phi+\lambda) R(\lambda,\theta,-\lambda)

    .. note::

        If the ``U3`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.PhaseShift` and :class:`~.Rot` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_3(\theta, \phi, \lambda)) = \frac{1}{2}\left[f(U_3(\theta+\pi/2, \phi, \lambda)) - f(U_3(\theta-\pi/2, \phi, \lambda))\right]`
      where :math:`f` is an expectation value depending on :math:`U_3(\theta, \phi, \lambda)`.
      This gradient recipe applies for each angle argument :math:`\{\theta, \phi, \lambda\}`.

    Args:
        theta (float): polar angle :math:`\theta`
        phi (float): azimuthal angle :math:`\phi`
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_wires = 1
    grad_method = "A"

    @property
    def num_params(self):
        return 3

    @classmethod
    def _matrix(cls, *params):
        theta, phi, lam = params

        # It might be that they are in different interfaces, e.g.,
        # Rot(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math._multi_dispatch(params)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            lam = qml.math.cast_like(qml.math.asarray(lam, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        mat = [
            [c, -s * qml.math.exp(1j * lam)],
            [s * qml.math.exp(1j * phi), c * qml.math.exp(1j * (phi + lam))],
        ]

        return qml.math.stack([qml.math.stack(row) for row in mat])

    @staticmethod
    def decomposition(theta, phi, lam, wires):
        decomp_ops = [
            Rot(lam, theta, -lam, wires=wires),
            PhaseShift(lam, wires=wires),
            PhaseShift(phi, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        theta, phi, lam = self.parameters
        new_lam = (np.pi - phi) % (2 * np.pi)
        new_phi = (np.pi - lam) % (2 * np.pi)
        return U3(theta, new_phi, new_lam, wires=self.wires)


class IsingXX(Operation):
    r"""IsingXX(phi, wires)
    Ising XX coupling gate

    .. math:: XX(\phi) = \begin{bmatrix}
            \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
            0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
            0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(XX(\phi)) = \frac{1}{2}\left[f(XX(\phi +\pi/2)) - f(XX(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`XX(\phi)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_wires = 2
    grad_method = "A"

    generator = [
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),
        -1 / 2,
    ]

    @property
    def num_params(self):
        return 1

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]

        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        Y = qml.math.convert_like(np.eye(4)[::-1].copy(), phi)

        if qml.math.get_interface(phi) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
            Y = qml.math.cast_like(Y, 1j)

        mat = qml.math.diag([c, c, c, c]) - 1j * s * Y
        return mat

    @staticmethod
    def decomposition(phi, wires):
        decomp_ops = [
            qml.CNOT(wires=wires),
            RX(phi, wires=[wires[0]]),
            qml.CNOT(wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        (phi,) = self.parameters
        return IsingXX(-phi, wires=self.wires)


class IsingYY(Operation):
    r"""IsingYY(phi, wires)
    Ising YY coupling gate

    .. math:: \mathtt{YY}(\phi) = \begin{bmatrix}
        \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
        0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
        0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
        i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(YY(\phi)) = \frac{1}{2}\left[f(YY(\phi +\pi/2)) - f(YY(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`YY(\phi)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_wires = 2
    grad_method = "A"
    generator = [
        np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]),
        -1 / 2,
    ]

    @property
    def num_params(self):
        return 1

    @staticmethod
    def decomposition(phi, wires):
        return [
            qml.CY(wires=wires),
            qml.RY(phi, wires=[wires[0]]),
            qml.CY(wires=wires),
        ]

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]

        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        Y = qml.math.convert_like(np.diag([1, -1, -1, 1])[::-1].copy(), phi)

        if qml.math.get_interface(phi) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
            Y = qml.math.cast_like(Y, 1j)

        return qml.math.diag([c, c, c, c]) + 1j * s * Y

    def adjoint(self):
        (phi,) = self.parameters
        return IsingYY(-phi, wires=self.wires)


class IsingZZ(Operation):
    r""" IsingZZ(phi, wires)
    Ising ZZ coupling gate

    .. math:: ZZ(\phi) = \begin{bmatrix}
        e^{-i \phi / 2} & 0 & 0 & 0 \\
        0 & e^{i \phi / 2} & 0 & 0 \\
        0 & 0 & e^{i \phi / 2} & 0 \\
        0 & 0 & 0 & e^{-i \phi / 2}
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(ZZ(\phi)) = \frac{1}{2}\left[f(ZZ(\phi +\pi/2)) - f(ZZ(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`ZZ(\theta)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_wires = 2
    grad_method = "A"
    generator = [
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        -1 / 2,
    ]

    @property
    def num_params(self):
        return 1

    @staticmethod
    def decomposition(phi, wires):
        return [
            qml.CNOT(wires=wires),
            qml.RZ(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        pos_phase = qml.math.exp(1.0j * phi / 2)
        neg_phase = qml.math.exp(-1.0j * phi / 2)

        return qml.math.diag([neg_phase, pos_phase, pos_phase, neg_phase])

    @classmethod
    def _eigvals(cls, *params):
        phi = params[0]

        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        pos_phase = qml.math.exp(1.0j * phi / 2)
        neg_phase = qml.math.exp(-1.0j * phi / 2)

        return qml.math.stack([neg_phase, pos_phase, pos_phase, neg_phase])

    def adjoint(self):
        (phi,) = self.parameters
        return IsingZZ(-phi, wires=self.wires)
