# Copyright 2018 Xanadu Quantum Technologies Inc.

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
The QNode decorator
===================

**Module name:** :mod:`pennylane.decorator`

Decorator for converting a quantum function containing PennyLane quantum
operations to a :mod:`QNode <pennylane.qnode>` that will run on a quantum device.

This decorator is provided for convenience, and allows a qfunc to be
converted to a QNode implicitly, avoiding the need to manually
instantiate a :mod:`QNode <pennylane.qnode>` object.

Note that the decorator completely replaces the Python-defined
function with a QNode of the same name - as such, the original
function is no longer accessible (but is accessible via the
:attr:`~.QNode.func` attribute).

Example
-------

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)

    @qml.qnode(dev1)
    def qfunc1(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval.PauliZ(0)

    result = qfunc1(0.543)

Once defined, the QNode can then be used like any other function in Python.
This includes combining it with other QNodes and classical functions to
build a hybrid computation. For example,

.. code-block:: python

    dev2 = qml.device('default.gaussian', wires=2)

    @qml.qnode(dev2)
    def qfunc2(x, y):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(y, 0, wires=[0, 1])
        return qml.expval.MeanPhoton(0)

    def hybrid_computation(x, y):
        return np.sin(qfunc1(y))*np.exp(-qfunc2(x+y, x)**2)

.. note::

    Applying the :func:`~.decorator.qnode` decorator to a user-defined
    function is equivalent to instantiating the QNode object manually.
    For example, the above example can also be written as follows:

    .. code-block:: python

        def qfunc1(x):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0,1])
            qml.RY(x, wires=1)
            return qml.expval.PauliZ(0)

        qnode1 = qml.QNode(qfunc1, dev1)
        result = qnode1(0.543)

Code details
^^^^^^^^^^^^
"""
# pylint: disable=redefined-outer-name
import logging as log
from functools import wraps, lru_cache

from .qnode import QNode

log.getLogger()


def qnode(device):
    """QNode decorator.

    Args:
        device (~pennylane._device.Device): an PennyLane-compatible device
    """
    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""

        qnode = QNode(func, device)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function"""
            return qnode(*args, **kwargs)

        # bind the jacobian method to the wrapped function
        wrapper.jacobian = qnode.jacobian

        # bind the variance method to the wrapped function
        wrapper.var = qnode.var

        # bind the qnode attributes to the wrapped function
        wrapper.__dict__.update(qnode.__dict__)

        return wrapper
    return qfunc_decorator
