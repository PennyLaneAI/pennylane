# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the qml.iterative_qpe function.
"""

import numpy as np
import pennylane as qml


def iterative_qpe(base, ancilla, iters):
    r"""Performs the `iterative quantum phase estimation <https://arxiv.org/pdf/quant-ph/0610214.pdf>`_ circuit.

    Given a unitary :math:`U`, this function applies the circuit for iterative quantum phase
    estimation and returns a list of mid-circuit measurements with qubit reset.

    Args:
      base (Operator): the phase estimation unitary, specified as an :class:`~.Operator`
      ancilla (Union[Wires, int, str]): the wire to be used for the estimation
      iters (int): the number of measurements to be performed

    Returns:
      list[MidMeasureMP]: the list of measurements performed

    .. seealso:: :class:`~.QuantumPhaseEstimation`, :func:`~.measure`

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", shots=5)

        @qml.qnode(dev)
        def circuit():

          # Initial state
          qml.X(0)

          # Iterative QPE
          measurements = qml.iterative_qpe(qml.RZ(2.0, wires=[0]), ancilla=1, iters=3)

          return qml.sample(measurements)

    .. code-block:: pycon

        >>> print(circuit())
        [[0 0 1]
         [0 0 1]
         [0 0 1]
         [1 1 1]
         [0 0 1]]

    The output is an array of size ``(number of shots, number of iterations)``.

    .. code-block:: pycon

        >>> print(qml.draw(circuit, max_length=150)())

        0: ──X─╭RZ(2.00)⁴─────────────────╭RZ(2.00)²────────────────────────────╭RZ(2.00)¹────────────────────────────────────┤
        1: ──H─╰●──────────H──┤↗│  │0⟩──H─╰●──────────Rϕ(-1.57)──H──┤↗│  │0⟩──H─╰●──────────Rϕ(-1.57)──Rϕ(-0.79)──H──┤↗│  │0⟩─┤
                               ╚══════════════════════╩══════════════║══════════════════════║══════════╩══════════════║═══════╡ ╭Sample[MCM]
                                                                     ╚══════════════════════╩═════════════════════════║═══════╡ ├Sample[MCM]
                                                                                                                      ╚═══════╡ ╰Sample[MCM]
    """

    measurements = []

    for i in range(iters):
        qml.Hadamard(wires=ancilla)
        qml.ctrl(qml.pow(base, z=2 ** (iters - i - 1)), control=ancilla)

        for ind, meas in enumerate(measurements):
            qml.cond(meas, qml.PhaseShift)(-2.0 * np.pi / 2 ** (ind + 2), wires=ancilla)

        qml.Hadamard(wires=ancilla)
        measurements.insert(0, qml.measure(wires=ancilla, reset=True))

    return measurements
