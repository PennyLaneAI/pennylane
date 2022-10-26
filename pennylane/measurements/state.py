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
This module contains the qml.state measurement.
"""
from pennylane.wires import Wires

from .measurements import MeasurementProcess, ObservableReturnTypes

State = ObservableReturnTypes.State
"""Enum: An enumeration which represents returning the state in the computational basis."""


def state():
    r"""Quantum state in the computational basis.

    This function accepts no observables and instead instructs the QNode to return its state. A
    ``wires`` argument should *not* be provided since ``state()`` always returns a pure state
    describing all wires in the device.

    Note that the output shape of this measurement process depends on the
    number of wires defined for the device.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.state()

    Executing this QNode:

    >>> circuit()
    array([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])

    The returned array is in lexicographic order. Hence, we have a :math:`1/\sqrt{2}` amplitude
    in both :math:`|00\rangle` and :math:`|01\rangle`.

    .. note::

        Differentiating :func:`~.state` is currently only supported when using the
        classical backpropagation differentiation method (``diff_method="backprop"``) with a
        compatible device.

    .. details::
        :title: Usage Details

        A QNode with the ``qml.state`` output can be used in a cost function with
        is then differentiated:

        >>> dev = qml.device('default.qubit', wires=2)
        >>> qml.qnode(dev, diff_method="backprop")
        ... def test(x):
        ...     qml.RY(x, wires=[0])
        ...     return qml.state()
        >>> def cost(x):
        ...     return np.abs(test(x)[0])
        >>> cost(x)
        tensor(0.98877108, requires_grad=True)
        >>> qml.grad(cost)(x)
        -0.07471906623679961
    """
    # pylint: disable=protected-access
    return MeasurementProcess(State)


def density_matrix(wires):
    r"""Quantum density matrix in the computational basis.

    This function accepts no observables and instead instructs the QNode to return its density
    matrix or reduced density matrix. The ``wires`` argument gives the possibility
    to trace out a part of the system. It can result in obtaining a mixed state, which can be
    only represented by the reduced density matrix.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliY(wires=0)
            qml.Hadamard(wires=1)
            return qml.density_matrix([0])

    Executing this QNode:

    >>> circuit()
    array([[0.+0.j 0.+0.j]
        [0.+0.j 1.+0.j]])

    The returned matrix is the reduced density matrix, where system 1 is traced out.

    Args:
        wires (Sequence[int] or int): the wires of the subsystem

    .. note::

        Calculating the derivative of :func:`~.density_matrix` is currently only supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device.
    """
    # pylint: disable=protected-access
    wires = Wires(wires)
    return MeasurementProcess(State, wires=wires)
