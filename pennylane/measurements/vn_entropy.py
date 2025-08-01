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
This module contains the qml.vn_entropy measurement.
"""

from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .measurements import StateMeasurement


class VnEntropyMP(StateMeasurement):
    """Measurement process that computes the Von Neumann entropy of the system prior to measurement.

    Please refer to :func:`~pennylane.vn_entropy` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): Base for the logarithm.
    """

    def __str__(self):
        return "vnentropy"

    _shortname = "vnentropy"

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("log_base", self.log_base))
        return (None, None), metadata

    def __init__(
        self,
        wires: Wires | None = None,
        id: str | None = None,
        log_base: float | None = None,
    ):
        self.log_base = log_base
        super().__init__(wires=wires, id=id)

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (self.__class__.__name__, tuple(self.wires.tolist()), self.log_base)

        return hash(fingerprint)

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()

    def process_state(self, state: TensorLike, wire_order: Wires):
        state = math.dm_from_state_vector(state)
        return math.vn_entropy(state, indices=self.wires, c_dtype=state.dtype, base=self.log_base)

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires):
        return math.vn_entropy(
            density_matrix, indices=self.wires, c_dtype=density_matrix.dtype, base=self.log_base
        )


def vn_entropy(wires, log_base=None) -> VnEntropyMP:
    r"""Von Neumann entropy of the system prior to measurement.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        wires (Sequence[int] or int): The wires of the subsystem
        log_base (float): Base for the logarithm.

    Returns:
        VnEntropyMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=[0])

    Executing this QNode:

    >>> circuit_entropy(np.pi/2)
    0.6931472

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(circuit_entropy)(param)
    tensor(0.62322524, requires_grad=True)

    .. note::

        Calculating the derivative of :func:`~pennylane.vn_entropy` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. note::

        ``qml.vn_entropy`` can also be used to compute the entropy of entanglement between two
        subsystems by computing the Von Neumann entropy of either of the subsystems.

    .. seealso:: :func:`pennylane.math.vn_entropy`, :func:`pennylane.math.vn_entanglement_entropy`
    """
    wires = Wires(wires)
    return VnEntropyMP(wires=wires, log_base=log_base)
