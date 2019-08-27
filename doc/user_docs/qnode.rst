.. _qnodes:

Quantum Nodes
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


.. autofunction:: pennylane.decorator.qnode



QNodes with interfaces
---------------------------

.. automodule:: pennylane.interfaces
   :members:
   :private-members:
   :inherited-members:

.. raw:: html

    <h2>Code details</h2>
