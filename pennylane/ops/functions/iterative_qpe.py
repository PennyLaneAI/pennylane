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
This file contains the function which generates the iterative QPE subroutine.
"""

import pennylane as qml
import numpy as np


def iterative_qpe(base, estimation_wire, iters):
    r"""Function that is in charge of adding to the queue the necessary gates to implement iterative QPE.
    It returns a list of the mid-circuit measurements performed.

    Args:
      base (Operator): the phase estimation unitary, specified as an :class:`~.Operator`
      estimation_wire (Union[Wire, int, str]): the wire to be used for the estimation.
      iters (int): the number of measurements to be performed.

    Returns:
      list[MidMeasureMP]: the list of measurements performed.

    .. seealso:: :class:`~.QuantumPhaseEstimation`, :func:`~.measure`

    **Example**

    .. code-block:: python

          dev = qml.device("default.qubit", shots = 1)

          @qml.qnode(dev)
          def circuit():

              # Initial state
              qml.PauliX(wires = [0])

              # Iterative QPE
              measurements = qml.iterative_qpe(qml.RZ(2., wires = [0]), estimation_wire = [1], iters = 3)

              return [qml.sample(op = meas) for meas in measurements]

    >>> print(circuit())
    [array(0), array(0), array(1)]
    """

    measurements = []

    for i in range(iters):
        qml.Hadamard(wires=estimation_wire)
        qml.ctrl(qml.pow(base, z=2 ** (iters - i - 1)), control=estimation_wire)

        for ind, meas in enumerate(measurements):
            qml.cond(meas, qml.PhaseShift)(-2 * np.pi / 2 ** (ind + 2), wires=estimation_wire)

        qml.Hadamard(wires=estimation_wire)
        measurements.insert(0, qml.measure(wires=estimation_wire, reset=True))

    return measurements
