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
.. _qnode_decorator:

Quantum circuits
================

:ref:`QNodes <quantum_nodes>` form part of the core structure of PennyLane --- they are used
to encapsulate a quantum function that runs on a quantum hardware device.

By defining QNodes, either via the :mod:`QNode decorator <pennylane.decorator>`
or the :mod:`QNode class <pennylane.qnode>`, dispatching them to devices, and
combining them with classical processing, it is easy to create arbitrary
classical-quantum hybrid computations.


The QNode decorator
-------------------

**Module name:** :mod:`pennylane.decorator`

The standard way for creating 'quantum nodes' or QNodes is the provided
`qnode` decorator. This decorator converts a quantum circuit function containing PennyLane quantum
operations to a :mod:`QNode <pennylane.qnode>` that will run on a quantum device.

This decorator is provided for convenience, and allows a quantum circuit function to be
converted to a :mod:`QNode <pennylane.qnode>` implicitly, avoiding the need to manually
instantiate a :mod:`QNode <pennylane.qnode>` object.

Note that the decorator completely replaces the Python-defined
function with a :mod:`QNode <pennylane.qnode>` of the same name - as such, the original
function is no longer accessible (but is accessible via the
:attr:`~.QNode.func` attribute).

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)

    @qml.qnode(dev1)
    def qfunc1(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval(qml.PauliZ(0))

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
        return qml.expval(qml.NumberOperator(0))

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
            return qml.expval(qml.PauliZ(0))

        qnode1 = qml.QNode(qfunc1, dev1)
        result = qnode1(0.543)


Machine learning interfaces
---------------------------

.. automodule:: pennylane.interfaces
   :members:
   :private-members:
   :inherited-members:

.. raw:: html

    <h2>Code details</h2>

.. autofunction:: pennylane.decorator.qnode
"""
# pylint: disable=redefined-outer-name
from functools import wraps, lru_cache

from .qnode import QNode


def qnode(device, interface='numpy', cache=False):
    """QNode decorator.

    Args:
        device (~pennylane._device.Device): a PennyLane-compatible device
        interface (str): the interface that will be used for automatic
            differentiation and classical processing. This affects
            the types of objects that can be passed to/returned from the QNode:

            * ``interface='numpy'``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: The QNode accepts and returns Torch tensors.

            * ``interface='tfe'``: The QNode accepts and returns eager execution
              TensorFlow ``tfe.Variable`` objects.

        cache (bool): If ``True``, the quantum function used to generate the QNode will
            only be called to construct the quantum circuit once, on first execution,
            and this circuit structure (i.e., the placement of templates, gates, measurements, etc.) will be cached for all further executions. The circuit parameters can still change with every call. Only activate this
            feature if your quantum circuit structure will never change.
    """
    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""

        qnode = QNode(func, device, cache=cache)

        if interface == 'torch':
            return qnode.to_torch()

        if interface == 'tfe':
            return qnode.to_tfe()

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function"""
            return qnode(*args, **kwargs)

        # bind the jacobian method to the wrapped function
        wrapper.jacobian = qnode.jacobian
        wrapper.metric_tensor = qnode.metric_tensor

        # bind the qnode attributes to the wrapped function
        wrapper.__dict__.update(qnode.__dict__)

        return wrapper
    return qfunc_decorator
