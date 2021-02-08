# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the available built-in quantum projection operations
supported by PennyLane.
"""

import numpy as np

from pennylane.operation import AnyWires, Projection


class Measure(Projection):
    r"""Measure(wires)
    A measurement in the computational basis.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        wires (Sequence[int] or int): the wire(s) the operation acts on

    .. UsageDetails::

        It is important to note when using the `Measurement` operation that
        one **applies** it to the wire one wishes to measure, and then returns
        one of the PennyLane measurement functions.

        For instance, consider the following, basic circuit:

        .. code-block:: python3

           dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.Measure(wires=[0, 1])

                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        By running this circuit, we would get an output of the form:

        >>> circuit()
        array([1., 1.])

        which corresponding to a measurement of :math:`|00\rangle`.
    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None

    @classmethod
    def _projectors(cls, wires):
        nr_proj = int(2 ** len(wires))
        projectors = [np.zeros((nr_proj, nr_proj)) for i in range(nr_proj)]

        for i in range(nr_proj):
            projectors[i][i, i] = 1

        return projectors


__qubit_projections__ = {"Measure"}

__all__ = list(__qubit_projections__)
