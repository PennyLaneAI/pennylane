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
This submodule contains the template for QFT.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import functools
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation


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
            qml.templates.QFT(wires=range(wires))
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
