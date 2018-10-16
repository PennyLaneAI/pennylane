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

    device1 = qm.device('strawberryfields.fock', wires=2)

    @qm.qnode(device1)
    def my_quantum_function(x):
        qm.Zrotation(x, 0)
        qm.CNOT(0,1)
        qm.Yrotation(x**2, 1)
        return qm.expval.Z(0)

    result = my_quantum_function(0.543)

Once defined, the QNode can then be used to construct the loss function,
and processed classically using NumPy. For example,

.. code-block:: python

    def loss(x):
        return np.sin(my_quantum_function(x))

.. note::

    Applying the :func:`~.decorator.qnode` decorator to a user-defined
    function is equivalent to instantiating the QNode object manually.
    For example, the above example can also be written as follows:

    .. code-block:: python

        def my_quantum_function(x):
            qm.Zrotation(x, 0)
            qm.CNOT(0,1)
            qm.Yrotation(x**2, 1)
            return qm.expval.Z(0)

        my_qnode = qm.QNode(my_quantum_function, dev1)
        result = my_qnode(0.543)

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
        device (~openqml._device.Device): an OpenQML-compatible device.
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
