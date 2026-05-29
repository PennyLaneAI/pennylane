.. _jax_interf:

JAX interface
=================

.. important::

    To use the JAX interface in PennyLane, you must first
    install ``jax`` and ``jaxlib`` with:
    
    .. code-block:: bash

        pip install jax==0.7.1 jaxlib==0.7.1
    
    You can then import PennyLane and JAX as follows:

    .. code::

        import pennylane as qp
        import jax
        import jax.numpy as jnp

Born out of the autograd package, `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ is the
next generation of differentiable functional computation, adding support for powerful hardware
accelerators like GPUs and TPUs via `XLA <https://www.tensorflow.org/xla>`_. To use
PennyLane in combination with JAX, we have to generate JAX-compatible quantum nodes. A basic
``QNode`` can be translated into a quantum node that interfaces with JAX by using the
``interface='jax'`` flag in the QNode decorator.

.. note::

    When using ``diff_method="parameter-shift"``, ``diff_method="finite-diff"``
    or ``diff_method="adjoint"`` with the JAX interface some restrictions apply to
    the measurements in the QNode:

    * Sample and probability measurements cannot be mixed with other measurement
      types in QNodes;
    * Multiple probability measurements need to have the same number of wires
      specified;

    However, when using ``diff_method="backprop"``, all QNode measurement statistics
    are supported.

.. note::

    JAX supports the single-precision numbers by default. To enable
    double-precision, add the following code on startup:

    .. code-block:: python

        jax.config.update("jax_enable_x64", True)


Construction via the decorator
------------------------------

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating
a JAX-capable QNode in PennyLane. Simply specify the ``interface='jax'`` keyword argument:

.. code-block:: python

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev, interface='jax')
    def circuit1(phi, theta):
        qp.RX(phi[0], wires=0)
        qp.RY(phi[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.PhaseShift(theta, wires=0)
        return qp.expval(qp.PauliZ(0)), qp.expval(qp.Hadamard(1))

The QNode ``circuit1()`` is now a JAX-capable QNode, accepting ``jax.Array`` objects
as input, and returning ``jax.Array`` objects. It can now be used like any other JAX function:

>>> phi = jnp.array([0.5, 0.1])
>>> theta = jnp.array(0.2)
>>> circuit1(phi, theta)
(Array(0.87758256, dtype=float64), Array(0.68803733, dtype=float64))

The interface can also be automatically determined when the ``QNode`` is called. You do not need to pass the interface
if you provide parameters.

Quantum gradients using JAX
---------------------------

Since a JAX-interfacing QNode acts like any other JAX interfacing python function,
the standard method used to calculate gradients with JAX can be used.

For example:

.. code-block:: python

    dev = qp.device('default.qubit', wires=2)

    @qp.qnode(dev, interface='jax')
    def circuit3(phi, theta):
        qp.RX(phi[0], wires=0)
        qp.RY(phi[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.PhaseShift(theta, wires=0)
        return qp.expval(qp.PauliZ(0))

    phi = jnp.array([0.5, 0.1])
    theta = jnp.array(0.2)
    grads = jax.grad(circuit3, argnums=(0, 1))
    phi_grad, theta_grad = grads(phi, theta)

This has output:

>>> phi_grad
Array([-0.47942555,  0.        ], dtype=float32)
>>> theta_grad
Array(-3.4332792e-10, dtype=float32)


.. _jax_jit:

Using jax.jit on QNodes
-----------------------

To fully utilize the power and speed of JAX, you'll need to just-in-time compile your functions - a
process called "jitting". If only expectation values or variances are returned,
the ``@jax.jit`` decorator can be directly applied to the QNode.

.. code-block:: python

    dev = qp.device('default.qubit', wires=2)

    @jax.jit  # QNode calls will now be jitted, and should run faster.
    @qp.qnode(dev, interface='jax')
    def circuit4(phi, theta):
        qp.RX(phi[0], wires=0)
        qp.RZ(phi[1], wires=1)
        qp.CNOT(wires=[0, 1])
        qp.RX(theta, wires=0)
        return qp.expval(qp.PauliZ(0))

.. note::

    For differentiation methods other than ``backprop``, when
    ``interface='jax'`` is specified, PennyLane will attempt to determine if
    the computation was just-in-time compiled. This is done by checking if any
    of the input parameters were subject to a JAX transformation. If so, a
    variant of the interface that supports the just-in-time compilation of
    QNodes will be used. This is equivalent to passing ``interface='jax-jit'``.

    Computing the jacobian of vector-valued QNodes is not supported with the
    JAX JIT interface. The output of vector-valued QNodes can, however, be used
    in the definition of scalar-valued cost functions whose gradients can be
    computed.

    Specify ``interface='jax-python'`` to enforce support for computing the
    backward pass of vector-valued QNodes (e.g., QNodes with probability, state
    or multiple expectation value measurements). This option does not support
    just-in-time compilation.


Randomness: Shots and Samples
-----------------------------
In JAX, there is no such thing as statefull randomness, meaning all random number generators must be
explicitly seeded. (See the `JAX random package documentation
<https://jax.readthedocs.io/en/latest/jax.random.html?highlight=random#module-jax.random>`_ for more
details).

When simulations include randomness (i.e., if the device has a finite ``shots`` value, or the qnode
returns ``qp.sample()``), the JAX device requires a ``jax.random.PRNGKey``. Usually, PennyLane
automatically handles this for you. However, if you wish to use jitting with randomness, both the
qnode and the device need to be created in the context of the ``jax.jit`` decorator. This can be
achieved by wrapping device and qnode creation into a function decorated by ``@jax.jit``:

Example:

.. code-block:: python

    import jax
    import pennylane as qp


    @jax.jit
    def sample_circuit(phi, theta, key):

        # Device construction should happen inside a `jax.jit` decorated
        # method when using a PRNGKey.
        dev = qp.device('default.qubit', wires=2, seed=key)

        @qp.set_shots(shots=100)
        @qp.qnode(dev, interface='jax', diff_method=None)
        def circuit(phi, theta):
            qp.RX(phi[0], wires=0)
            qp.RZ(phi[1], wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RX(theta, wires=0)
            return qp.sample() # Here, we take samples instead.

        return circuit(phi, theta)

    # Get the samples from the jitted method.
    samples = sample_circuit([0.2, 1.0], 5.2, jax.random.PRNGKey(0))

.. note::

    If you don't pass a PRNGKey when sampling with a ``jax.jit``, every call to the sample function
    will return the same result.

.. _jax_optimize:

Optimization using JAXopt and Optax
-----------------------------------

To optimize your hybrid classical-quantum model using the JAX interface, you
**must** make use of a package meant for optimizing JAX code (such as `JAXopt
<https://jaxopt.github.io/stable/>`_ or `Optax
<https://optax.readthedocs.io/en/latest/>`_) or your own custom JAX optimizer.
**The** :ref:`PennyLane optimizers <intro_ref_opt>` **cannot be used with the
JAX interface**.

As an example of using ``JAXopt``, the ``GradientDescent`` optimizer may be
used to optimize a QNode that is transformed by ``jax.jit``:

.. code-block:: python

    import pennylane as qp
    import jax
    import jaxopt

    jax.config.update("jax_enable_x64", True)

    dev = qp.device("default.qubit", wires=1)

    @jax.jit
    @qp.set_shots(shots=None)
    @qp.qnode(dev, interface="jax")
    def energy(a):
        qp.RX(a, wires=0)
        return qp.expval(qp.PauliZ(0))

    gd = jaxopt.GradientDescent(energy, maxiter=5)

    res = gd.run(0.5)
    optimized_params = res.params

>>> optimized_params
Array(3.1415861, dtype=float64, weak_type=True)

Alternatively, optimizers from ``Optax`` may also be used to optimize the same
QNode:

.. code-block:: python

    import pennylane as qp
    from jax import numpy as jnp
    import jax
    import optax

    learning_rate = 0.15

    dev = qp.device("default.qubit", wires=1)

    @jax.jit
    @qp.set_shots(shots=None)
    @qp.qnode(dev, interface="jax")
    def energy(a):
        qp.RX(a, wires=0)
        return qp.expval(qp.PauliZ(0))

    optimizer = optax.adam(learning_rate)

    params = jnp.array(0.5)
    opt_state = optimizer.init(params)

    for _ in range(200):
        grads = jax.grad(energy)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

>>> params
Array(3.14159111, dtype=float64)
