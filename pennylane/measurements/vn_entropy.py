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
import copy
from typing import Sequence

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import StateMeasurement, VnEntropy


def vn_entropy(wires, log_base=None):
    r"""Von Neumann entropy of the system prior to measurement.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        wires (Sequence[int] or int): The wires of the subsystem
        log_base (float): Base for the logarithm.

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
    0.6232252401402305

    .. note::

        Calculating the derivative of :func:`~.vn_entropy` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`pennylane.qinfo.transforms.vn_entropy` and :func:`pennylane.math.vn_entropy`
    """
    wires = Wires(wires)
    return _VnEntropy(wires=wires, log_base=log_base)


class _VnEntropy(StateMeasurement):
    """Measurement process that returns the Von Neumann entropy."""

    # pylint: disable=too-many-arguments, unused-argument
    def __init__(
        self,
        obs: Operator = None,
        wires=None,
        eigvals=None,
        id=None,
        log_base=None,
    ):
        self.log_base = log_base
        super().__init__(obs=obs, wires=wires, eigvals=eigvals, id=id)

    @property
    def return_type(self):
        return VnEntropy

    @property
    def numeric_type(self):
        return float

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        return qml.math.vn_entropy(
            state, indices=self.wires, c_dtype=state.dtype, base=self.log_base
        )

    def __copy__(self):
        return self.__class__(
            obs=copy.copy(self.obs),
            wires=self._wires,
            eigvals=self._eigvals,
            log_base=self.log_base,
        )
