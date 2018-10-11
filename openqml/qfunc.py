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
The QFunc decorator
===================

Decorator factory for converting a Python function containing OpenQML quantum
operations to a QNode that will run on a quantum device.

Example:

.. code-block:: python

    device1 = qm.device('strawberryfields.fock', wires=2)

    @qfunc(device1)
    def my_quantum_function(x):
        qm.Zrotation(x, 0)
        qm.CNOT(0,1)
        qm.Yrotation(x**2, 1)
        return qm.expectation.Z(0)

    result = my_quantum_function(0.543)

To become a valid QNode, the user-defined function must consist of
only OpenQML operators and expectation values, one per line, and must
end with the measurement of a single or tuple of expectation values.

Once defined, the QNode can then be used to construct the loss function,
and processed classically using NumPy. For example,

.. code-block:: python

    def loss(x):
        return np.sin(my_quantum_function(x))

.. note::

    Applying the qfunc decorator to a user-defined function is equivalent
    to instatiating the QNode object manually. For example, the above example
    can also be written as follows:

    .. code-block:: python

        def my_quantum_function(x):
            qm.Zrotation(x, 0)
            qm.CNOT(0,1)
            qm.Yrotation(x**2, 1)
            return qm.expectation.Z(0)

        my_qnode = qm.QNode(my_quantum_function, dev1)
        result = my_qnode(0.543)
"""
import logging as log
log.getLogger()

from functools import wraps, lru_cache

from .qnode import QNode


def qfunc(device):
    """QFunc decorator.

    Args:
        device (openqml.Device): an OpenQML-compatible device.
    """
    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""

        qnode = QNode(func, device)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return qnode(*args, **kwargs)
        return wrapper
    return qfunc_decorator
