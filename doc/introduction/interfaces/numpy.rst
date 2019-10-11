.. _numpy_interf:

NumPy interface
***************

.. note:: This interface is the default interface supported by PennyLane's :class:`~.QNode`.


Using the NumPy interface
-------------------------

Using the NumPy interface is easy in PennyLane, and the default approach ---
designed so it will feel like you are just using standard NumPy, with the
added benefit of automatic differentiation.

All you have to do is make sure to import the wrapped version of NumPy
provided by PennyLane alongside with the PennyLane library:

.. code::

    import pennylane as qml
    from pennylane import numpy as np

This is provided via `autograd <https://github.com/HIPS/autograd>`_, and enables
automatic differentiation and backpropagation of classical computations using familiar
NumPy functions and modules (such as ``np.sin``, ``np.cos``, ``np.exp``, ``np.linalg``,
``np.fft``), as well as standard Python constructs, such as ``if`` statements, and ``for``
and ``while`` loops.


Via the QNode decorator
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating QNodes
in PennyLane. By default, all QNodes are constructed for the NumPy interface,
but this can also be specified explicitly by passing the ``interface='numpy'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='numpy')
    def circuit1(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit1()`` is a NumPy-interfacing QNode, accepting standard Python
data types such as ints, floats, lists, and tuples, as well as NumPy arrays, and
returning NumPy arrays.

It can now be used like any other Python/NumPy function:

>>> phi = np.array([0.5, 0.1])
>>> theta = 0.2
>>> circuit1(phi, theta)
array([ 0.87758256,  0.68803733])

Via the QNode class
^^^^^^^^^^^^^^^^^^^

In the :ref:`introduction <intro_vcirc_qnode>` it was shown how to instantiate a :class:`QNode <pennylane.qnode.QNode>`
object directly, for example, if you would like to reuse the same quantum function across
multiple devices, or even use different classical interfaces:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('forest.wavefunction', wires=2)

    def circuit2(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

    qnode1 = qml.QNode(circuit2, dev1)
    qnode2 = qml.QNode(circuit2, dev2)

By default, all QNodes created this way are NumPy interfacing QNodes.


Quantum gradients
-----------------

To calculate the gradient of a NumPy-QNode, we can simply use the provided
:func:`~.grad` function.

For example, consider the following QNode:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit3(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

We can now use :func:`~.grad` to create a QNode *gradient function*,
with respect to both QNode parameters ``phi`` and ``theta``:

.. code-block:: python

    phi = np.array([0.5, 0.1])
    theta = 0.2
    dcircuit = qml.grad(circuit3, argnum=[0, 1])

Evaluating this gradient function at specific parameter values:

>>> dcircuit(phi, theta)
(array([ -4.79425539e-01,   1.11022302e-16]), array(0.0))



Optimization
------------

To optimize your hybrid classical-quantum model using the NumPy interface,
use the provided :ref:`PennyLane optimizers <intro_ref_opt>`.

For example, we can optimize a NumPy-interfacing QNode (below) such that the weights ``x``
lead to a final expectation value of 0.5:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit4(x):
        qml.RX(x[0], wires=0)
        qml.RZ(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(x[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost(x):
        return np.abs(circuit4(x) - 0.5)**2

    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    steps = 100
    params = np.array([0.011, 0.012, 0.05])

    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)

The final weights and circuit value are:

>>> params
array([ 0.19846757,  0.012     ,  1.03559806])
>>> circuit4(params)
0.5

For more details on the NumPy optimizers, check out the tutorials, as well as the
:mod:`pennylane.optimize` documentation.



Vector-valued QNodes and the Jacobian
-------------------------------------

How does automatic differentiation work in the case where the QNode returns multiple expectation values?

.. code::

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit5(x):
        qml.RX(x, wires=0)
        qml.RX(x, wires=1)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

If we were to naively try computing the gradient of ``circuit5`` using the :func:`~.grad` function,

.. code::

    g1 = qml.grad(circuit5, argnum=0)
    g1(np.pi/2)

we would get an error message. This is because the `gradient <https://en.wikipedia.org/wiki/Gradient>`_ is
only defined for scalar functions, i.e., functions which return a single value. In the case where the QNode
returns multiple expectation values, the correct differential operator to use is
the `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_.
This can be accessed in PennyLane as :func:`~.jacobian`:

>>> j1 = qml.jacobian(circuit5, argnum=0)
>>> j1(np.pi/2)
array([-1. -1.])


The output of :func:`~.jacobian` is a two-dimensional vector, with the first/second element being
the partial derivative of the first/second expectation value with respect to the input parameter.
The Jacobian function has the same signature as the gradient function, requiring the user to specify
which argument should be differentiated.

If you want to compute the Jacobian matrix for a function with multiple input parameters
and multiple expectation values, the recommended way to do this is to combine the parameters
into a single list/array and index into this inside your quantum circuit function.
Consider the following circuit:

.. code-block:: python

    @qml.qnode(dev)
    def circuit6(params):
        qml.RX(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        qml.RX(params[2], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

It has a full Jacobian with two rows and three columns:

>>> j2 = qml.jacobian(circuit6, argnum=0)
>>> params = np.array([np.pi / 3, 0.25, np.pi / 2])
>>> j2(params)
array([[-8.66025404e-01, -5.55111512e-17,  0.00000000e+00],
       [-4.71844785e-16, -1.38777878e-17, -5.00000000e-01]])

.. warning::

    Currently, :func:`pennylane.jacobian` supports only the case where
    ``argnum`` is a single integer. For quantum functions with multiple arguments,
    use the above method of combining arguments in an array to get the full Jacobian matrix.


Advanced Autograd usage
-----------------------

The PennyLane NumPy interface leverages the Python library `autograd <https://github.com/HIPS/autograd>`_ to enable automatic differentiation of NumPy code, and extends it to provide gradients of quantum circuit functions encapsulated in QNodes. In order to make NumPy code differentiable, Autograd provides a wrapped version of NumPy (exposed in PennyLane as :code:`pennylane.numpy`).

As stated in other sections, using this interface, any hybrid computation should be coded using the wrapped version of NumPy provided by PennyLane. **If you accidentally import the vanilla version of NumPy, your code will not be automatically differentiable.**

Because of the way autograd wraps NumPy, the PennyLane NumPy interface does not require users to learn a new mini-language for declaring classical computations, or invoke awkward language-dependent functions which replicate basic python control-flow statements (``if`` statements, loops, etc.). Users can continue using many of the standard numerical programming practices common in Python and NumPy.

That being said, autograd's coverage of NumPy is not complete. It is best to consult the `autograd docs <https://github.com/HIPS/autograd/blob/master/docs/tutorial.md>`_ for a more complete overview of supported and unsupported features. We highlight a few of the major 'gotchas' here.

**Do not use:**

- Assignment to arrays, such as ``A[0, 0] = x``.

..

- Implicit casting of lists to arrays, for example ``A = np.sum([x, y])``.
  Make sure to explicitly cast to a NumPy array first, i.e., ``A = np.sum(np.array([x, y]))`` instead.

..

- ``A.dot(B)`` notation.
  Use ``np.dot(A, B)`` or ``A @ B`` instead.

..

- In-place operations such as ``a += b``.
  Use ``a = a + b`` instead.

..

- Some ``isinstance`` checks, like ``isinstance(x, np.ndarray)`` or ``isinstance(x, tuple)``, without first doing ``from autograd.builtins import isinstance, tuple``.
