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
            qml.QFT(wires=range(wires))
            return qml.state()

    .. code-block:: pycon

        >>> circuit_qft(np.array([1.0, 0.0, 0.0]))
        [ 0.35355339+0.j -0.35355339+0.j  0.35355339+0.j -0.35355339+0.j
          0.35355339+0.j -0.35355339+0.j  0.35355339+0.j -0.35355339+0.j]

    .. details::
        :title: Semiclassical Quantum Fourier transform

        If the QFT is the last subroutine applied within a circuit, it can be
        replaced by a
        `semiclassical Fourier transform <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.3228>`_.
        It makes use of mid-circuit measurements and dynamic circuit control based
        on the measurement values, allowing to reduce the number of two-qubit gates.

        As an example, consider the following circuit implementing addition between two
        numbers with ``n_wires`` bits (modulo ``2**n_wires``):

        .. code-block:: python

            dev = qml.device("default.qubit", shots=1)

            @qml.qnode(dev)
            def qft_add(m, k, n_wires):
                qml.BasisEmbedding(m, wires=range(n_wires))
                qml.adjoint(qml.QFT)(wires=range(n_wires))
                for j in range(n_wires):
                    qml.RZ(-k * np.pi / (2**j), wires=j)
                qml.QFT(wires=range(n_wires))
                return qml.sample()

        .. code-block:: pycon

            >>> qft_add(7, 3, n_wires=4)
            [1 0 1 0]

        The last building block of this circuit is a QFT, so we may replace it by its
        semiclassical counterpart:

        .. code-block:: python

            def scFT(n_wires):
                '''semiclassical Fourier transform'''
                for w in range(n_wires-1):
                    qml.Hadamard(w)
                    mcm = qml.measure(w)
                    for m in range(w + 1, n_wires):
                        qml.cond(mcm, qml.PhaseShift)(np.pi / 2 ** (m + 1), wires=m)
                qml.Hadamard(n_wires-1)

            @qml.qnode(dev)
            def scFT_add(m, k, n_wires):
                qml.BasisEmbedding(m, wires=range(n_wires))
                qml.adjoint(qml.QFT)(wires=range(n_wires))
                for j in range(n_wires):
                    qml.RZ(-k * np.pi / (2**j), wires=j)
                scFT(n_wires)
                # Revert wire order because of PL's QFT convention
                return qml.sample(wires=list(range(n_wires-1, -1, -1)))

        .. code-block:: pycon

            >>> scFT_add(7, 3, n_wires=4)
            [1 0 1 0]
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, wires=None, id=None):
        wires = qml.wires.Wires(wires)
        self.hyperparameters["n_wires"] = len(wires)
        super().__init__(wires=wires, id=id)

    def _flatten(self):
        return tuple(), (self.wires, tuple())

    @property
    def num_params(self):
        return 0

    @staticmethod
    @functools.lru_cache()
    def compute_matrix(n_wires):  # pylint: disable=arguments-differ
        return np.fft.ifft(np.eye(2**n_wires), norm="ortho")

    @staticmethod
    def compute_decomposition(wires, n_wires):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.QFT.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            n_wires (int): number of wires or ``len(wires)``

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> qml.QFT.compute_decomposition((0,1,2,4))
        [Toffoli(wires=[1, 2, 4]), CNOT(wires=[1, 2]), Toffoli(wires=[0, 2, 4])]

        """
        shifts = [2 * np.pi * 2**-i for i in range(2, n_wires + 1)]

        decomp_ops = []
        for i, wire in enumerate(wires):
            decomp_ops.append(qml.Hadamard(wire))

            for shift, control_wire in zip(shifts[: len(shifts) - i], wires[i + 1 :]):
                op = qml.ControlledPhaseShift(shift, wires=[control_wire, wire])
                decomp_ops.append(op)

        first_half_wires = wires[: n_wires // 2]
        last_half_wires = wires[-(n_wires // 2) :]

        for wire1, wire2 in zip(first_half_wires, reversed(last_half_wires)):
            swap = qml.SWAP(wires=[wire1, wire2])
            decomp_ops.append(swap)

        return decomp_ops
