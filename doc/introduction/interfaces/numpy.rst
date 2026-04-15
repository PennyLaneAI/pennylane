.. _numpy_interf:

NumPy interface
===============


Using the NumPy interface
-------------------------

Using the NumPy interface is easy in PennyLane, and the default approach ---
designed so it will feel like you are just using standard NumPy, with the
added benefit of automatic differentiation.

All you have to do is make sure to import the wrapped version of NumPy
provided by PennyLane alongside with the PennyLane library:

.. code::

    import pennylane as qp
    from pennylane import numpy as np

This is powered via `Autograd <https://github.com/HIPS/autograd>`_, and enables
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

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev, interface='autograd')
    def circuit1(phi, theta):
        qp.RX(phi[0], wires=0)
        qp.RY(phi[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.PhaseShift(theta, wires=0)
        return qp.expval(qp.PauliZ(0)), qp.expval(qp.Hadamard(1))

The QNode ``circuit1()`` is a NumPy-interfacing QNode, accepting standard Python
data types such as ints, floats, lists, and tuples, as well as NumPy arrays, and
returning NumPy arrays.

It can now be used like any other Python/NumPy function:

>>> phi = np.array([0.5, 0.1])
>>> theta = 0.2
>>> circuit1(phi, theta)
(tensor(0.87758256, requires_grad=True),
 tensor(0.68803733, requires_grad=True))

The interface can also be automatically determined when the ``QNode`` is called. You do not need to pass the interface
if you provide parameters.

Via the QNode constructor
^^^^^^^^^^^^^^^^^^^^^^^^^

In the :ref:`introduction <intro_vcirc_qnode>` it was shown how to instantiate a :class:`QNode <pennylane.qnode.QNode>`
object directly, for example, if you would like to reuse the same quantum function across
multiple devices, or even use different classical interfaces:

.. code-block:: python

    dev1 = qp.device('default.qubit', wires=2)
    dev2 = qp.device('forest.wavefunction', wires=2)

    def circuit2(phi, theta):
        qp.RX(phi[0], wires=0)
        qp.RY(phi[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.PhaseShift(theta, wires=0)
        return qp.expval(qp.PauliZ(0)), qp.expval(qp.Hadamard(1))

    qnode1 = qp.QNode(circuit2, dev1)
    qnode2 = qp.QNode(circuit2, dev2)

By default, all QNodes created this way are NumPy interfacing QNodes.


Quantum gradients
-----------------

To calculate the gradient of a NumPy-QNode, we can simply use the provided
:func:`~.grad` function.

For example, consider the following QNode:

.. code-block:: python

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev)
    def circuit3(phi, theta):
        qp.RX(phi[0], wires=0)
        qp.RY(phi[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.PhaseShift(theta, wires=0)
        return qp.expval(qp.PauliZ(0))

We can now use :func:`~.grad` to create a QNode *gradient function*,
with respect to both QNode parameters ``phi`` and ``theta``:

.. code-block:: python

    phi = np.array([0.5, 0.1], requires_grad=True)
    theta = np.array(0.2, requires_grad=True)
    dcircuit = qp.grad(circuit3)

Evaluating this gradient function at specific parameter values:

>>> dcircuit(phi, theta)
(array([ -4.79425539e-01,   1.11022302e-16]), array(0.0))


Differentiable and non-differentiable arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How does PennyLane know which arguments of a quantum function to differentiate, and which to ignore?
For example, you may want to pass arguments to a QNode but *not* have
PennyLane consider them when computing gradients.

Regular positional arguments provided to the QNode are not assumed to be differentiable
by default. This includes arguments in the form of built-in Python data types, and arrays from
the original NumPy module. Thus, arguments need to be explicitly marked as trainable or selected
using the ``argnum`` keyword. To mark an argument as trainable, a special flag ``requires_grad``
has been added to arrays from PennyLane's NumPy module:

>>> from pennylane import numpy as np
>>> np.array([0.1, 0.2], requires_grad=True)
tensor([0.1, 0.2], requires_grad=True)

When omitted, the value for this flag is ``True``, so if you would like to provide a
non-differentiable PennyLane NumPy array to the QNode or gradient function, make sure
to specify ``requires_grad=False``:

>>> from pennylane import numpy as np
>>> np.array([0.1, 0.2], requires_grad=False)
tensor([0.1, 0.2], requires_grad=False)

.. note::

    The ``requires_grad`` argument can be passed to any NumPy function provided by PennyLane,
    including NumPy functions that create arrays like ``np.random.random``, ``np.zeros``, etc.

On the other hand, keyword arguments (whether they have a default value or not), are always
considered non-trainable, no matter their data type or flags they may have. For example, consider
the following QNode that accepts two arguments ``data`` and ``weights``:

.. code-block:: python

    dev = qp.device('default.qubit', wires=5)

    @qp.qnode(dev)
    def circuit(data, weights):
        qp.AmplitudeEmbedding(data, wires=[0, 1, 2], normalize=True)
        qp.RX(weights[0], wires=0)
        qp.RY(weights[1], wires=1)
        qp.RZ(weights[2], wires=2)
        qp.CNOT(wires=[0, 1])
        qp.CNOT(wires=[0, 2])
        return qp.expval(qp.PauliZ(0))

    rng = np.random.default_rng(seed=42)  # make the results reproducible
    data = rng.random([2 ** 3], requires_grad=False)
    weights = np.array([0.1, 0.2, 0.3], requires_grad=True)

When we compute the derivative, arguments with ``requires_grad=False`` as well as arguments
passed as keyword arguments are ignored by :func:`~.grad`, which in this case means no gradient
is computed at all:

>>> qp.grad(circuit)(data, weights=weights)
UserWarning: Attempted to differentiate a function with no trainable parameters. If this is unintended, please add trainable parameters via the 'requires_grad' attribute or 'argnum' keyword.
()

Optimization
------------

To optimize your hybrid classical-quantum model using the NumPy interface,
use the provided optimizers:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.AdagradOptimizer
    ~pennylane.AdamOptimizer
    ~pennylane.GradientDescentOptimizer
    ~pennylane.MomentumOptimizer
    ~pennylane.NesterovMomentumOptimizer
    ~pennylane.QNGOptimizer
    ~pennylane.RiemannianGradientOptimizer
    ~pennylane.RMSPropOptimizer
    ~pennylane.RotosolveOptimizer
    ~pennylane.RotoselectOptimizer
    ~pennylane.ShotAdaptiveOptimizer

:html:`</div>`

For example, we can optimize a NumPy-interfacing QNode (below) such that the weights ``x``
lead to a final expectation value of 0.5:

.. code-block:: python

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev)
    def circuit4(x):
        qp.RX(x[0], wires=0)
        qp.RZ(x[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.RX(x[2], wires=0)
        return qp.expval(qp.PauliZ(0))

    def cost(x):
        return np.abs(circuit4(x) - 0.5)**2

    opt = qp.GradientDescentOptimizer(stepsize=0.4)

    steps = 100
    params = np.array([0.011, 0.012, 0.05], requires_grad=True)

    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)

The final weights and circuit value are:

>>> params
tensor([0.19846757, 0.012     , 1.03559806], requires_grad=True)
>>> circuit4(params)
tensor(0.5, requires_grad=True)

For more details on the NumPy optimizers, check out the tutorials, as well as the
:mod:`pennylane.optimize` documentation.



Vector-valued QNodes and the Jacobian
-------------------------------------

How does automatic differentiation work in the case where the QNode returns multiple expectation values?

.. code::

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev)
    def circuit5(params):
        qp.Hadamard(wires=0)
        qp.CNOT(wires=[0, 1])
        qp.RX(params[0], wires=0)
        qp.RY(params[1], wires=1)
        qp.CNOT(wires=[0, 1])
        return qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(1))

If we were to naively try computing the gradient of ``circuit5`` using the :func:`~.grad` function,

>>> g1 = qp.grad(circuit5)
>>> params = np.array([np.pi/2, 0.2], requires_grad=True)
>>> g1(params)
TypeError: Grad only applies to real scalar-output functions. Try jacobian, elementwise_grad or holomorphic_grad.

we would get an error message. This is because the `gradient <https://en.wikipedia.org/wiki/Gradient>`_ is
only defined for scalar functions, i.e., functions which return a single value. In the case where the QNode
returns multiple expectation values, the correct differential operator to use is
the `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_.
This can be accessed in PennyLane as :func:`~pennylane.jacobian`.

As the ``circuit5`` returns a tuple of numpy arrays instead of a single numpy array, the results need
to be stacked into a single array before use with :func:`~pennylane.jacobian`.

>>> j1 = qp.jacobian(lambda x: np.stack(circuit5(x)))
>>> j1(params)
array([[ 0.        , -0.98006658],
       [-0.98006658,  0.        ]])


The output of :func:`~pennylane.jacobian` is a two-dimensional vector, with the first/second element being
the partial derivative of the first/second expectation value with respect to the input parameter.


Advanced Autograd usage
-----------------------

The PennyLane NumPy interface leverages the Python library `Autograd
<https://github.com/HIPS/autograd>`_ to enable automatic differentiation of NumPy code, and extends
it to provide gradients of quantum circuit functions encapsulated in QNodes. In order to make NumPy
code differentiable, Autograd provides a wrapped version of NumPy (exposed in PennyLane as
:code:`pennylane.numpy`).

.. warning::
    As stated in other sections, using this interface, any hybrid computation should be coded using the
    wrapped version of NumPy provided by PennyLane. If you accidentally import the vanilla version of
    NumPy, your code will not be automatically differentiable.

Because of the way Autograd wraps NumPy, the PennyLane NumPy interface allows standard NumPy
functions and basic Python control statements (``if`` statements, loops, etc.) for declaring
differentiable classical computations.

That being said, Autograd's coverage of NumPy is not complete. It is best to consult the `Autograd
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


SciPy Optimization
------------------

In addition to using autodifferentiation provided by Autograd, the NumPy interface
also allows QNodes to be optimized directly using the
`SciPy optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`__ module.

Simply pass the QNode, or your hybrid cost function containing QNodes, directly
to the ``scipy.minimize`` function:

.. code-block:: python

    from scipy.optimize import minimize

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev)
    def circuit(x):
        qp.RX(x[0], wires=0)
        qp.RZ(x[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.RX(x[2], wires=0)
        return qp.expval(qp.PauliZ(0))

    def cost(x):
        return np.abs(circuit(x) - 0.5) ** 2

    params = np.array([0.011, 0.012, 0.05], requires_grad=True)

    minimize(cost, params, method='BFGS')

Some of the SciPy minimization methods require information about the gradient
of the cost function via the ``jac`` keyword argument. This is easy to include; we
can simply create a function that computes the gradient using ``qp.grad``. Since
``minimize`` does not use our wrapped version of numpy, we need to explicitly
specify which arguments are trainable via the ``argnum`` keyword.

>>> minimize(cost, params, method='BFGS', jac=qp.grad(cost, argnum=0))
      fun: 6.3491130264451484e-18
 hess_inv: array([[ 1.85642354e+00, -8.84954187e-22,  3.89539943e+00],
       [-8.84954187e-22,  1.00000000e+00, -4.02571211e-21],
       [ 3.89539943e+00, -4.02571211e-21,  1.87180282e+01]])
      jac: array([5.81636983e-10, 3.23117427e-27, 4.21456861e-09])
  message: 'Optimization terminated successfully.'
     nfev: 8
      nit: 2
     njev: 8
   status: 0
  success: True
        x: array([0.22685818, 0.012     , 1.03194789])
