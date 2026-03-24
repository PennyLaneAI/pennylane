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


import functools

import numpy as np

from pennylane import math
from pennylane.capture import enabled
from pennylane.control_flow import for_loop
from pennylane.ops import SWAP, ControlledPhaseShift, Hadamard
from pennylane.templates.core import Subroutine
from pennylane.wires import WiresLike


def qft_decomp_resources(wires: WiresLike):
    """Calculate the resources for QFT."""
    num_wires = len(wires)
    return {
        Hadamard: num_wires,
        SWAP: num_wires // 2,
        ControlledPhaseShift: num_wires * (num_wires - 1) // 2,
    }


@functools.partial(
    Subroutine,
    static_argnames=[],
    compute_resources=qft_decomp_resources,
)
def QFT(wires: WiresLike):
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

    .. code-block:: python

        wires = 3

        dev = qml.device('default.qubit',wires=wires)

        @qml.qnode(dev)
        def circuit_qft(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.state()

    >>> circuit_qft(np.array([1.0, 0.0, 0.0])) # doctest: +SKIP
    array([ 0.3536+0.j, -0.3536+0.j,  0.3536+0.j, -0.3536+0.j,  0.3536+0.j,
           -0.3536+0.j,  0.3536+0.j, -0.3536+0.j])

    .. details::
        :title: Semiclassical Quantum Fourier transform

        If the QFT is the last subroutine applied within a circuit, it can be
        replaced by a
        `semiclassical Fourier transform <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.3228>`_.
        It makes use of mid-circuit measurements and dynamic circuit control based
        on the measurement values, allowing to reduce the number of two-qubit gates.

        As an example, consider the following circuit implementing addition between two
        numbers with ``num_wires`` bits (modulo ``2**num_wires``):

        .. code-block:: python

            dev = qml.device("default.qubit")

            @qml.qnode(dev, shots=1)
            def qft_add(m, k, num_wires):
                qml.BasisEmbedding(m, wires=range(num_wires))
                qml.adjoint(qml.QFT)(wires=range(num_wires))
                for j in range(num_wires):
                    qml.RZ(-k * np.pi / (2**j), wires=j)
                qml.QFT(wires=range(num_wires))
                return qml.sample()

        >>> qft_add(7, 3, num_wires=4)
        array([[1, 0, 1, 0]])

        The last building block of this circuit is a QFT, so we may replace it by its
        semiclassical counterpart:

        .. code-block:: python

            def scFT(num_wires):
                '''semiclassical Fourier transform'''
                for w in range(num_wires-1):
                    qml.Hadamard(w)
                    mcm = qml.measure(w)
                    for m in range(w + 1, num_wires):
                        qml.cond(mcm, qml.PhaseShift)(np.pi / 2 ** (m + 1), wires=m)
                qml.Hadamard(num_wires-1)

            @qml.qnode(dev)
            def scFT_add(m, k, num_wires):
                qml.BasisEmbedding(m, wires=range(num_wires))
                qml.adjoint(qml.QFT)(wires=range(num_wires))
                for j in range(num_wires):
                    qml.RZ(-k * np.pi / (2**j), wires=j)
                scFT(num_wires)
                # Revert wire order because of PL's QFT convention
                return qml.sample(wires=list(range(num_wires-1, -1, -1)))

        >>> qml.set_shots(scFT_add, 1)(7, 3, num_wires=4) # doctest: +SKIP
        array([[1, 1, 1, 0]])
    """
    num_wires = len(wires)
    shifts = [2 * np.pi * 2**-i for i in range(2, num_wires + 1)]

    if enabled():
        shifts = math.array(shifts, like="jax")
        wires = math.array(wires, like="jax")

    shift_len = len(shifts)

    @for_loop(num_wires)
    def outer_loop(i):
        Hadamard(wires[i])

        if num_wires > 1:

            @for_loop(shift_len - i)
            def cphaseshift_loop(j):
                ControlledPhaseShift(shifts[j], wires=[wires[i + j + 1], wires[i]])

            cphaseshift_loop()  # pylint: disable=no-value-for-parameter

    outer_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(num_wires // 2)
    def swaps(i):
        SWAP(wires=[wires[i], wires[num_wires - i - 1]])

    swaps()  # pylint: disable=no-value-for-parameter
