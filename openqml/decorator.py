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

Decorator for converting a quantum function containing OpenQML quantum
operations to a :mod:`QNode <openqml.qnode>` that will run on a quantum device.

This decorator is provided for convenience, and allows a qfunc to be
converted to a QNode implicitly, avoiding the need to manually
instantiate a :mod:`QNode <openqml.qnode>` object.

Note that the decorator completely replaces the Python-defined
function with a QNode of the same name - as such, the original
function is no longer accessible (but is accessible via the
:attr:`~.QNode.func` attribute).

Example
-------

.. code-block:: python

    dev1 = qm.device('default.qubit', wires=2)

    @qm.qnode(dev1)
    def qfunc1(x):
        qm.RZ(x, wires=0)
        qm.CNOT(wires=[0,1])
        qm.RY(x, wires=1)
        return qm.expval.PauliZ(0)

    result = qfunc1(0.543)

Once defined, the QNode can then be used like any other function in Python.
This includes combining it with other QNodes and classical functions to
build a hybrid computation. For example,

.. code-block:: python

    dev2 = qm.device('default.gaussian', wires=2)

    @qm.qnode(dev2)
    def qfunc2(x, y):
        qm.Displacement(x, 0, wires=0)
        qm.Beamsplitter(y, 0, wires=[0, 1])
        return qm.expval.MeanPhoton(0)

    def hybrid_computation(x, y):
        return np.sin(qfunc1(y))*np.exp(-qfunc2(x+y, x)**2)

.. note::

    Applying the :func:`~.decorator.qnode` decorator to a user-defined
    function is equivalent to instantiating the QNode object manually.
    For example, the above example can also be written as follows:

    .. code-block:: python

        def qfunc1(x):
            qm.RZ(x, wires=0)
            qm.CNOT(wires=[0,1])
            qm.RY(x, wires=1)
            return qm.expval.PauliZ(0)

        qnode1 = qm.QNode(qfunc1, dev1)
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
        device (~openqml._device.Device): an OpenQML-compatible device
    """
    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""

        qnode = QNode(func, device)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function"""
            return qnode(*args, **kwargs)
        return wrapper
    return qfunc_decorator
