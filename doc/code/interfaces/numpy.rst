.. _numpy_qnode:

NumPy interface
***************

.. note:: This interface is the default interface supported by PennyLane's :class:`~.QNodes`.


Using the NumPy interface
-------------------------

Using the NumPy interface is easy in PennyLane, and the default approach ---
designed so it will feel like you are just using standard NumPy, with the
added benefit of automatic differentiation.

All you have to do is make sure to import the wrapped version of NumPy
provided by PennyLane:

>>> from pennylane import numpy as np

This is provided via `autograd <https://github.com/HIPS/autograd>`_, and enables
automatic differentiation and backpropagation in classical nodes using familiar
NumPy functions and modules (such as ``np.sin``, ``np.cos``, ``np.exp``, ``np.linalg``,
``np.fft``), as well as standard Python constructs, such as ``if`` statements, and ``for``
and ``while`` loops.


Via the QNode decorator
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`QNode decorator <qnode_decorator>` is the recommended way for creating QNodes
in PennyLane. By default, all QNodes are by default constructed for the NumPy interface,
but this can also be specified explicitly by passing the ``interface='numpy'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='numpy')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval.PauliZ(0), qml.expval.Hadamard(1)

The QNode ``circuit()`` is a NumPy-interfacing QNode, accepting standard Python
data types such as ints, floats, lists, and tuples, as well as NumPy arrays, and
returning NumPy arrays.

It can now be used like any other Python/NumPy function:

>>> phi = np.array([0.5, 0.1])
>>> theta = 0.2
>>> circuit(phi, theta)
array([ 0.87758256,  0.68803733])

Via the QNode class
^^^^^^^^^^^^^^^^^^^

Sometimes, it is more convenient to instantiate a :class:`~.QNode` object directly, for example,
if you would like to reuse the same quantum function across multiple devices, or even
using different classical interfaces:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('forest.wavefunction', wires=2)

    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval.PauliZ(0), qml.expval.Hadamard(1)

    qnode1 = qml.QNode(circuit, dev1)
    qnode2 = qml.QNode(circuit, dev2, interface='numpy')

As with the QNode decorator, if we want to explicitly create a NumPy interfacing
QNode, we simply pass the ``interface`` keyword argument; this is also the default
if not provided.


Quantum gradients
-----------------

To calculate the gradient of a NumPy-QNode, we can simply use the provided
:func:`~.grad` function.

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval.PauliZ(0)

Using :func:`~.grad` to create a QNode *gradient function*,
with respect to both QNode parameters ``phi`` and ``theta``:

>>> phi = np.array([0.5, 0.1])
>>> theta = 0.2
>>> dcircuit = qml.grad(circuit, argnum=[0, 1])

we can now evaluate this gradient function at specific parameter values:

>>> dcircuit(phi, theta)
(array([ -4.79425539e-01,   1.11022302e-16]), array(0.0))



Optimization
------------

To optimize your hybrid classical-quantum model using the NumPy interface,
you may write your own optimization method, or use the provided :ref:`PennyLane optimizers <optimization_methods>`.

For example, to optimize a NumPy-interfacing QNode (below) such that the weights ``x``
result in an expectation value of 0.5:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RZ(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(x[2], wires=0)
        return qml.expval.PauliZ(0)

    def cost(x):
        return np.abs(circuit(x) - 0.5)**2

    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    steps = 100
    params = np.array([0.011, 0.012, 0.05])

    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)

The final weights and circuit value:

>>> params
array([ 0.19846757,  0.012     ,  1.03559806])
>>> circuit(params)
0.5

For more details on the NumPy optimizers, check out the tutorials, as well as the
:ref:`optimization_methods` documentation.
