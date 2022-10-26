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
from pennylane.wires import Wires

from .measurements import MeasurementProcess, VnEntropy


def vn_entropy(wires, log_base=None):
    r"""Von Neumann entropy of the system prior to measurement.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        wires (Sequence[int] or int): The wires of the subsystem
        log_base (float): Base for the logarithm. If None, the natural logarithm is used.

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
    return MeasurementProcess(VnEntropy, wires=wires, log_base=log_base)
