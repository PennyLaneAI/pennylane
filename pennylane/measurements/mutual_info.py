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
This module contains the qml.mutual_info measurement.
"""
import pennylane as qml

from .measurements import MeasurementProcess, ObservableReturnTypes

MutualInfo = ObservableReturnTypes.MutualInfo
"""Enum: An enumeration which represents returning the mutual information before measurements."""


def mutual_info(wires0, wires1, log_base=None):
    r"""Mutual information between the subsystems prior to measurement:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    Args:
        wires0 (Sequence[int] or int): the wires of the first subsystem
        wires1 (Sequence[int] or int): the wires of the second subsystem
        log_base (float): Base for the logarithm. If None, the natural logarithm is used.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_mutual(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

    Executing this QNode:

    >>> circuit_mutual(np.pi/2)
    1.3862943611198906

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(circuit_mutual)(param)
    1.2464504802804612

    .. note::

        Calculating the derivative of :func:`~.mutual_info` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`~.vn_entropy`, :func:`pennylane.qinfo.transforms.mutual_info` and :func:`pennylane.math.mutual_info`
    """
    # the subsystems cannot overlap
    if [wire for wire in wires0 if wire in wires1]:
        raise qml.QuantumFunctionError(
            "Subsystems for computing mutual information must not overlap."
        )

    wires0 = qml.wires.Wires(wires0)
    wires1 = qml.wires.Wires(wires1)
    return MeasurementProcess(MutualInfo, wires=[wires0, wires1], log_base=log_base)
