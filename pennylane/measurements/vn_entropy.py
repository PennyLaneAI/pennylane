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
# pylint: disable=protected-access
"""
This module contains the qml.vn_entropy measurement.
"""
from typing import Sequence, Optional

import pennylane as qml
from pennylane.wires import Wires

from .measurements import StateMeasurement, VnEntropy


def vn_entropy(wires, log_base=None) -> "VnEntropyMP":
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

        Calculating the derivative of :func:`~.vn_entropy` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`pennylane.qinfo.transforms.vn_entropy` and :func:`pennylane.math.vn_entropy`
    """
    wires = Wires(wires)
    return VnEntropyMP(wires=wires, log_base=log_base)


class VnEntropyMP(StateMeasurement):
    """Measurement process that computes the Von Neumann entropy of the system prior to measurement.

    Please refer to :func:`vn_entropy` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): Base for the logarithm.
    """

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("log_base", self.log_base))
        return (None, None), metadata

    # pylint: disable=too-many-arguments, unused-argument
    def __init__(
        self,
        wires: Optional[Wires] = None,
        id: Optional[str] = None,
        log_base: Optional[float] = None,
    ):
        self.log_base = log_base
        super().__init__(wires=wires, id=id)

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (self.__class__.__name__, tuple(self.wires.tolist()), self.log_base)

        return hash(fingerprint)

    @property
    def return_type(self):
        return VnEntropy

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        state = qml.math.dm_from_state_vector(state)
        return qml.math.vn_entropy(
            state, indices=self.wires, c_dtype=state.dtype, base=self.log_base
        )
