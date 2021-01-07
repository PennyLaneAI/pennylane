.. _jax_interf:

JAX interface
=================

In order to use PennyLane in combination with JAX, we have to generate JAX-compatible
quantum nodes. A basic ``QNode`` can be translated into a quantum node that interfaces with JAX by using the ``interface='jax'`` flag in the QNode Decorator.

.. note::
    
    Currently, only the ``default.qubit.jax`` device supports the JAX interface.


.. note::

    To use the JAX interface in PennyLane, you must first
    install ``jax`` and ``jaxlib``. You can then import pennylane and jax as follows:

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

    dev = qml.device('default.qubit', wires=2)

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



.. _jax_optimize:

Using jax.jit on Qnodes
-----------------------

To fully utilize the power and speed of JAX, you'll need to just-in-time compile your functions. If you're only taking expectation values, you can simply add the decorator on your Qnode directly.

.. code-block:: python

    dev = qml.device('default.qubit.jax', wires=2)
    @jax.jit  # Qnode calls will now be jitted, and should run faster.
    @qml.qnode(dev, interface='jax')
    def circuit4(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RZ(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))


However, if you want to do random sampling instead, you'll need to pass a ``jax.random.PRNGKey`` to the device construction. Your jitted function will also need to include the device construction.

.. code-block:: python

    import jax
    import pennylane as qml


    @jax.jit
    def sample_circuit(phi, theta, key)
        
        # Device construction should happen inside a `jax.jit` function
        # when using a PRNGKey.
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
    
    If you don't pass a PRNGKey when sampling with a ``jax.jit``, every call to the sample function will return the same result. 