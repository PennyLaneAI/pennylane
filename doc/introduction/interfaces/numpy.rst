.. _numpy_interf:

NumPy interface
===============

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

This is powered via `autograd <https://github.com/HIPS/autograd>`_, and enables
automatic differentiation and backpropagation of classical computations using familiar
NumPy functions and modules (such as ``np.sin``, ``np.cos``, ``np.exp``, ``np.linalg``,
``np.fft``), as well as standard Python constructs, such as ``if`` statements, and ``for``
and ``while`` loops.


Via the QNode decorator
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating QNodes
in PennyLane. By default, all QNodes are constructed for the NumPy interface,
but this can also be specified explicitly by passing the ``interface='autograd'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='autograd')
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

Via the QNode constructor
^^^^^^^^^^^^^^^^^^^^^^^^^

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
    dcircuit = qml.grad(circuit3)

Evaluating this gradient function at specific parameter values:

>>> dcircuit(phi, theta)
(array([ -4.79425539e-01,   1.11022302e-16]), array(0.0))


Differentiable and non-differentiable arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How does PennyLane know which arguments of a quantum function to differentiate, and which to ignore?
For example, you may want to pass arguments as positional arguments to a QNode but *not* have
PennyLane consider them when computing gradients.

As a basic rule, **all positional arguments provided to the QNode are assumed to be differentiable
by default**. To accomplish this, all arrays created by the PennyLane NumPy module have a special
flag ``requires_grad`` specifying whether they are trainable or not:

>>> from pennylane import numpy as np
>>> np.array([0.1, 0.2])
tensor([0.1, 0.2], requires_grad=True)

If you would like to provide explicit non-differentiable arguments to the
QNode or gradient function, make sure to use a NumPy array that specifies
``requires_grad=False``:

>>> from pennylane import numpy as np
>>> np.array([0.1, 0.2], requires_grad=False)
tensor([0.1, 0.2], requires_grad=False)

.. note::

    The ``requires_grad`` argument can be passed to any NumPy function provided by PennyLane,
    including NumPy functions that create arrays like ``np.random.random``, ``np.zeros``, etc.

For example, consider the following QNode that accepts one trainable argument ``weights``,
and two non-trainable arguments ``data`` and ``wires``:

.. code-block:: python

    dev = qml.device('default.qubit', wires=5)

    @qml.qnode(dev)
    def circuit(weights, data, wires):
        qml.templates.AmplitudeEmbedding(data, wires=wires, normalize=True)
        qml.RX(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RZ(weights[2], wires=wires[2])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CNOT(wires=[wires[0], wires[2]])
        return qml.expval(qml.PauliZ(wires[0]))

We must specify that ``data`` and ``wires`` are NumPy arrays with ``requires_grad=False``:

>>> weights = np.array([0.1, 0.2, 0.3])
>>> data = np.random.random([2**3], requires_grad=False)
>>> wires = np.array([2, 0, 1], requires_grad=False)
>>> circuit(weights, data, wires)
0.16935626052294817

When we compute the derivative, arguments with ``requires_grad=False`` are explicitly ignored
by :func:`~.grad`:

>>> grad_fn = qml.grad(circuit)
>>> grad_fn(weights, data, wires)
(array([-1.69923049e-02,  0.00000000e+00, -8.32667268e-17]),)


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
    def circuit5(params):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(1))

If we were to naively try computing the gradient of ``circuit5`` using the :func:`~.grad` function,

>>> g1 = qml.grad(circuit5)
>>> params = np.array([np.pi/2, 0.2])
>>> g1(params)

we would get an error message. This is because the `gradient <https://en.wikipedia.org/wiki/Gradient>`_ is
only defined for scalar functions, i.e., functions which return a single value. In the case where the QNode
returns multiple expectation values, the correct differential operator to use is
the `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_.
This can be accessed in PennyLane as :func:`~.jacobian`:

>>> j1 = qml.jacobian(circuit5)
>>> j1(params)
array([[ 0.        , -0.98006658],
       [-0.98006658,  0.        ]])


The output of :func:`~.jacobian` is a two-dimensional vector, with the first/second element being
the partial derivative of the first/second expectation value with respect to the input parameter.


Advanced Autograd usage
-----------------------

The PennyLane NumPy interface leverages the Python library `autograd
<https://github.com/HIPS/autograd>`_ to enable automatic differentiation of NumPy code, and extends
it to provide gradients of quantum circuit functions encapsulated in QNodes. In order to make NumPy
code differentiable, Autograd provides a wrapped version of NumPy (exposed in PennyLane as
:code:`pennylane.numpy`).

As stated in other sections, using this interface, any hybrid computation should be coded using the
wrapped version of NumPy provided by PennyLane. **If you accidentally import the vanilla version of
NumPy, your code will not be automatically differentiable.**

Because of the way autograd wraps NumPy, the PennyLane NumPy interface allows standard NumPy
functions and basic Python control statements (``if`` statements, loops, etc.) for declaring
differentiable classical computations.

That being said, autograd's coverage of NumPy is not complete. It is best to consult the `autograd
docs <https://github.com/HIPS/autograd/blob/master/docs/tutorial.md>`_ for a more complete overview
of supported and unsupported features. We highlight a few of the major 'gotchas' here.

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
