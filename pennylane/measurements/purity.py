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
This module contains the qml.purity measurement.
"""
from pennylane.math import dm_from_state_vector
from pennylane.math import purity as math_purity
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .measurements import StateMeasurement


class PurityMP(StateMeasurement):
    """Measurement process that computes the purity of the system prior to measurement.

    Please refer to :func:`pennylane.purity` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some
            applications where the instance has to be identified
    """

    def __str__(self):
        return "purity"

    _shortname = "purity"

    def __init__(self, wires: Wires, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()

    def process_state(self, state: TensorLike, wire_order: Wires):
        wire_map = dict(zip(wire_order, list(range(len(wire_order)))))
        indices = [wire_map[w] for w in self.wires]
        state = dm_from_state_vector(state)
        return math_purity(state, indices=indices, c_dtype=state.dtype)

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires):
        wire_map = dict(zip(wire_order, list(range(len(wire_order)))))
        indices = [wire_map[w] for w in self.wires]
        return math_purity(density_matrix, indices=indices, c_dtype=density_matrix.dtype)


def purity(wires) -> PurityMP:
    r"""The purity of the system prior to measurement.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is the density matrix. The purity of a normalized quantum state satisfies
    :math:`\frac{1}{d} \leq \gamma \leq 1`, where :math:`d` is the dimension of the Hilbert space.
    A pure state has a purity of 1.

    Args:
        wires (Sequence[int] or int): The wires of the subsystem

    Returns:
        PurityMP: Measurement process instance

    **Example**

    .. code-block:: python3

        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def circuit_purity(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.purity(wires=[0,1])

    >>> circuit_purity(0.1)
    array(0.7048)

    .. seealso:: :func:`pennylane.math.purity`
    """
    wires = Wires(wires)
    return PurityMP(wires=wires)
