.. _jax_interf:

JAX interface
=================

Born out of the autograd package, `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ is the
next generation of differentiable functional computation, adding support for powerful hardware
accelerators like GPUs and TPUs via `XLA <https://www.tensorflow.org/xla>`_. To use
PennyLane in combination with JAX, we have to generate JAX-compatible quantum nodes. A basic
``QNode`` can be translated into a quantum node that interfaces with JAX by using the
``interface='jax'`` flag in the QNode decorator.


.. note::

    When using ``diff_method="parameter-shift"`` with the JAX interface, only QNodes that
    return a single expectation value or variance are supported. Returning more
    than one expectation value, or other statistics such as probabilities, is not supported.

    However, when using ``diff_method="backprop"``, all QNode measurement statistics
    are supported.

.. note::

    To use the JAX interface in PennyLane, you must first
    install ``jax`` and ``jaxlib``. You can then import PennyLane and JAX as follows:

    .. code::

        import pennylane as qml
        import jax
        import jax.numpy as jnp


Construction via the decorator
------------------------------

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating
a JAX-capable QNode in PennyLane. Simply specify the ``interface='jax'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit.jax', wires=2)

    @qml.qnode(dev, interface='jax')
    def circuit1(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit1()`` is now a JAX-capable QNode, accepting ``jax.DeviceArray`` objects
as input, and returning ``jax.DeviceArray`` objects. It can now be used like any other JAX function:

>>> phi = jnp.array([0.5, 0.1])
>>> theta = jnp.array(0.2)
>>> circuit1(phi, theta)
DeviceArray([0.8776, 0.6880], dtype=float64)

Quantum gradients using JAX
---------------------------

Since a JAX-interfacing QNode acts like any other JAX interfacing python function,
the standard method used to calculate gradients with JAX can be used.

For example:

.. code-block:: python

    dev = qml.device('default.qubit.jax', wires=2)

    @qml.qnode(dev, interface='jax')
    def circuit3(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = jnp.array([0.5, 0.1])
    theta = jnp.array(0.2)
    grads = jax.grad(circuit3, argnums=(0, 1))
    phi_grad, theta_grad = grads(phi, theta)

This has output:

>>> phi_grad
DeviceArray([-0.47942555,  0.        ], dtype=float32)
>>> theta_grad
DeviceArray(-3.4332792e-10, dtype=float32)


.. _jax_optimize:

Using jax.jit on QNodes
-----------------------

To fully utilize the power and speed of JAX, you'll need to just-in-time compile your functions - a
process called "jitting". If only expectation values are returned, the ``@jax.jit`` decorator can be
directly applied to the QNode.

.. code-block:: python

    dev = qml.device('default.qubit.jax', wires=2)

    @jax.jit  # QNode calls will now be jitted, and should run faster.
    @qml.qnode(dev, interface='jax')
    def circuit4(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RZ(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))


Randomness: Shots and Samples
-----------------------------
In JAX, there is no such thing as statefull randomness, meaning all random number generators must be
explicitly seeded. (See the `JAX random package documentation
<https://jax.readthedocs.io/en/latest/jax.random.html?highlight=random#module-jax.random>`_ for more
details).

When simulations include randomness (i.e., if the device has a finite ``shots`` value, or the qnode
returns ``qml.samples()``), the JAX device requires a ``jax.random.PRNGKey``. Usually, PennyLane
automatically handles this for you. However, if you wish to use jitting with randomness, both the
qnode and the device need to be created in the context of the ``jax.jit`` decorator. This can be
achieved by wrapping device and qnode creation into a function decorated by ``@jax.jit``:

Example:

.. code-block:: python

    import jax
    import pennylane as qml


    @jax.jit
    def sample_circuit(phi, theta, key)

        # Device construction should happen inside a `jax.jit` decorated
        # method when using a PRNGKey.
        dev = qml.device('default.qubit.jax', wires=2, prng_key=key)


        @qml.qnode(dev, interface='jax')
        def circuit(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RZ(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(theta, wires=0)
            return qml.samples() # Here, we take samples instead.

        return circuit(phi, theta, key)

    # Get the samples from the jitted method.
    samples = sample_circuit([0.0, 1.0], 0.0, jax.random.PRNGKey(0))

.. note::

    If you don't pass a PRNGKey when sampling with a ``jax.jit``, every call to the sample function
    will return the same result.
