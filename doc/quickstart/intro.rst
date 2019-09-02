 .. role:: html(raw)
   :format: html

.. _pl_intro:

Introduction
============

This section is an introduction to how the concept of a :ref:`variational quantum circuit <overview>` is implemented in PennyLane.

It shows new PennyLane users how to:

* Construct quantum circuits via **quantum functions**

* Define **computational devices**

* Combine quantum functions and devices to **quantum nodes**

* Conveniently create quantum nodes using the quantum node **decorator**

* Compute **gradients** of quantum nodes

* **Optimize** hybrid computations that contain quantum nodes

* Save **configurations** for PennyLane

More information about PennyLane's code base can be found in the :ref:`Code Documentation <library_overview>`.

Quantum functions
-----------------

A quantum circuit is constructed as a special Python function, a *quantum circuit function*, or *quantum function* in short.
For example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))


Quantum functions are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the function must consist of only supported PennyLane
  :mod:`operations <pennylane.ops>` or sequences of gates called :mod:`templates <pennylane.templates>`, using one instruction per line.

* The function must always return either a single or a tuple of
  *measured observable values*, by applying a :mod:`measurement function <pennylane.measure>`
  to an :mod:`observable <pennylane.ops>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

.. note::

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operations <pennylane.operation.Operation>` require a QNode in order to perform queuing on initialization.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.


Defining a device
-----------------

To run - and later optimize - a quantum circuit, one needs to first specify a *computational device*.

The device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`~device` loader. 

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

PennyLane comes included with
some basic devices; additional devices can be installed as plugins
(see :ref:`plugins` for more details).

Quantum nodes
-------------

Together, a quantum function and a device are used to create a *quantum node* or
:class:`QNode` object, which wraps the quantum function and binds it to the device.
A quantum node is a subroutine executed by a quantum computer, which is part of a
larger :ref:`hybrid computation <_hybrid_computation>`.

A `QNode` can be explicitly created as follows:

.. code-block:: python

    qnode = qml.QNode(my_quantum_function, dev)

The `QNode` can be used to compute the result of a quantum circuit as if it was a standard Python
function. It takes the same arguments as the original quantum function:

>>> qnode(np.pi/4, 0.7)

One or more :class:`QNodes` can be combined in standard python functions:

.. code-block:: python

    from pennylane import numpy as np

    def my_quantum_function2(x, y):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(y, 0, wires=[0, 1])
        return qml.expval(qml.NumberOperator(0))

    dev2 = qml.device('default.gaussian', wires=2)

    qnode2 = qml.QNode(my_quantum_function2, dev2)

    def hybrid_computation(a, b):
        return np.sin(qnode1(a, b))*np.exp(-qnode2(a+b, a-b)**2)


Here, `hybrid_computation` contains results from two different devices, one being a qubit-based
and the other a continuous-variable device.

.. note::

    The NumPy functions :func:`np.sin` and :func:`np.exp` have to be imported from PennyLane's NumPy library instead of the standard NumPy library. This allows PennyLane to automatically differentiate through these operations.

The QNode decorator
-------------------

A more convenient - and in fact the recommended - way for creating `QNodes` is the provided
quantum node decorator. This decorator converts a quantum function containing PennyLane quantum
operations to a :mod:`QNode <pennylane.qnode>` that will run on a quantum device.

.. note::
    The decorator completely replaces the Python-defined function with
    a :mod:`QNode <pennylane.qnode>` of the same name - as such, the original
    function is no longer accessible (but is accessible via the :attr:`~.QNode.func` attribute).

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def qfunc(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval(qml.PauliZ(0))

    result = qfunc(0.543)


Quantum gradients
-----------------

The gradient of the different `QNodes` defined previously can be computed as follows:

.. code-block:: python

    g1 = qml.grad(qnode, [0, 1])
    g2 = qml.grad(qnode1, [0])
    g3 = qml.grad(qfunc, [1])

The first argument of :func:`grad` is the quantum node, and the second is a list of indices of the parameters we want to derive for. The result is a new function which computes gradients for specific values of the parameters, for example:

>>> x = 1.1
>>> y = -2.2
>>> g1(x, y)
(array(0.56350015), array(0.17825313))
>>> g2(x, y)
(array(0.56350015), array(0.17825313))
>>> g3(x, y)
(array(0.56350015), array(0.17825313))

We can also compute gradients of *functions of qnodes*:

.. code-block:: python

    g4 = qml.grad(hybrid_computation, [0, 1])

To evaluate the gradient at a specific position, use:

>>> g4(1.1, -2.2)
(array(0.56350015), array(0.17825313))

Optimization
------------

PennyLane comes with a collection of optimizers for a basic, NumPy-interfacing `QNode`. They
can be found in the :mod:`pennylane.optimize` module.

For other interfaces such as PyTorch and TensorFlow's Eager mode, read the next section on :ref:`interfaces <interfaces>`.



Reusing configurations
----------------------

The settings for PennyLane's devices, such as the shot number to measure an expectation, can be saved for ease of use.

[EXPLAIN!] 

