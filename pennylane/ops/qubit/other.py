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
This module submodule contains the discrete-variable quantum operations that
do not fit nicely in any of the other categories.
"""
import functools

# pylint:disable=abstract-method,arguments-differ,protected-access
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.wires import Wires


class Projector(Observable):
    r"""Projector(basis_state, wires)
    Observable corresponding to the computational basis state projector :math:`P=\ket{i}\bra{i}`.

    The expectation of this observable returns the value

    .. math::
        |\langle \psi | i \rangle |^2

    corresponding to the probability of measuring the quantum state in the :math:`i` -th eigenstate of the specified :math:`n` qubits.

    For example, the projector :math:`\ket{11}\bra{11}` , or in integer notation :math:`\ket{3}\bra{3}`, is created by ``basis_state=np.array([1, 1])``.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        basis_state (tensor-like): binary input of shape ``(n, )``
        wires (Iterable): wires that the projector acts on
    """
    num_wires = AnyWires
    num_params = 1
    par_domain = "A"

    def __init__(self, basis_state, wires, do_queue=True):
        wires = Wires(wires)
        shape = qml.math.shape(basis_state)

        if len(shape) != 1:
            raise ValueError(f"Basis state must be one-dimensional; got shape {shape}.")

        n_basis_state = shape[0]
        if n_basis_state != len(wires):
            raise ValueError(
                f"Basis state must be of length {len(wires)}; got length {n_basis_state}."
            )

        basis_state = list(qml.math.toarray(basis_state))

        if not set(basis_state).issubset({0, 1}):
            raise ValueError(f"Basis state must only consist of 0s and 1s; got {basis_state}")

        super().__init__(basis_state, wires=wires, do_queue=do_queue)

    @classmethod
    def _eigvals(cls, *params):
        """Eigenvalues of the specific projector operator.

        Returns:
            array: eigenvalues of the projector observable in the computational basis
        """
        w = np.zeros(2 ** len(params[0]))
        idx = int("".join(str(i) for i in params[0]), 2)
        w[idx] = 1
        return w

    def diagonalizing_gates(self):
        """Return the gate set that diagonalizes a circuit according to the
        specified Projector observable.

        Returns:
            list: list containing the gates diagonalizing the projector observable
        """
        return []


class QFT(Operation):
    r"""QFT(wires)
    Apply a quantum Fourier transform (QFT).

    For the :math:`N`-qubit computational basis state :math:`|m\rangle`, the QFT performs the
    transformation

    .. math::

        |m\rangle \rightarrow \frac{1}{\sqrt{2^{N}}}\sum_{n=0}^{2^{N} - 1}\omega_{N}^{mn} |n\rangle,

    where :math:`\omega_{N} = e^{\frac{2 \pi i}{2^{N}}}` is the :math:`2^{N}`-th root of unity.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        wires (int or Iterable[Number, str]]): the wire(s) the operation acts on

    **Example**

    The quantum Fourier transform is applied by specifying the corresponding wires:

    .. code-block::

        wires = 3

        dev = qml.device('default.qubit',wires=wires)

        @qml.qnode(dev)
        def circuit_qft(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.state()

        circuit_qft([1.0, 0.0, 0.0])
    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None
    grad_method = None

    @property
    def matrix(self):
        # Redefine the property here to allow for a custom _matrix signature
        mat = self._matrix(len(self.wires))
        if self.inverse:
            mat = mat.conj()
        return mat

    @classmethod
    @functools.lru_cache()
    def _matrix(cls, num_wires):
        dimension = 2 ** num_wires

        mat = np.zeros((dimension, dimension), dtype=np.complex128)
        omega = np.exp(2 * np.pi * 1j / dimension)

        for m in range(dimension):
            for n in range(dimension):
                mat[m, n] = omega ** (m * n)

        return mat / np.sqrt(dimension)

    @staticmethod
    def decomposition(wires):
        num_wires = len(wires)
        shifts = [2 * np.pi * 2 ** -i for i in range(2, num_wires + 1)]

        decomp_ops = []
        for i, wire in enumerate(wires):
            decomp_ops.append(qml.Hadamard(wire))

            for shift, control_wire in zip(shifts[: len(shifts) - i], wires[i + 1 :]):
                op = qml.ControlledPhaseShift(shift, wires=[control_wire, wire])
                decomp_ops.append(op)

        first_half_wires = wires[: num_wires // 2]
        last_half_wires = wires[-(num_wires // 2) :]

        for wire1, wire2 in zip(first_half_wires, reversed(last_half_wires)):
            swap = qml.SWAP(wires=[wire1, wire2])
            decomp_ops.append(swap)

        return decomp_ops

    def adjoint(self):
        return QFT(wires=self.wires).inv()
