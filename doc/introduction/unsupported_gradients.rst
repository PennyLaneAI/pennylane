.. _unsupported_gradients:

Unsupported gradient configurations
===================================

.. _Device jacobian:

Device jacobian
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use ``diff_method="device"`` for a QNode, the device passed into
the constructor of a QNode must have its ``"provides_jacobian"`` capability set to ``True``
and must contain a method ``jacobian(circuits, **kwargs)`` that returns the gradients for
each quantum circuit. This is not implemented for the ``default.qubit`` device because
the device doesn't provide such a `jacobian` method (instead allows backpropagation to work).

See the :ref:`custom plugins <plugin_overview>` page for more detail.

An exception is raised if this configuration is used:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def print_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, interface="jax", diff_method='device')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = jnp.array([0.1])
        print(qp.grad(circuit)(x))

>>> print_grad()
Traceback (most recent call last):
    ...
pennylane.exceptions.QuantumFunctionError: Device ... does not support device with requested circuit.

.. _Analytic backpropagation:

Backpropagation
~~~~~~~~~~~~~~~~~~~~~~~~

The backpropagation algorithm is analytic by nature, and hence passing ``shots=None``
is the only supported configuration when ``diff_method="backprop"`` is used.

Currently an exception is raised if this invalid configuration is used:

.. code-block:: python

    def print_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=100)
        @qp.qnode(dev, interface="jax", diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = jnp.array([0.1])
        print(qp.grad(circuit)(x))

>>> print_grad()
Traceback (most recent call last):
    ...
pennylane.exceptions.QuantumFunctionError: Device ... does not support backprop with requested circuit.
...

Changing to ``shots=None`` allows computing the analytic gradient:

.. code-block:: python

    def print_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, interface="jax", diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = jnp.array([0.1])
        print(qp.grad(circuit)(x))

>>> print_grad()
[-0.09983342]

.. _Adjoint differentiation:

Adjoint differentiation
~~~~~~~~~~~~~~~~~~~~~~~

PennyLane implements the adjoint differentiation method from
`2009.02823 <https://arxiv.org/pdf/2009.02823.pdf>`__, which only discusses
the gradient of expectation values of observables.

In particular, the following code works as expected:

.. code-block:: python

    def print_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, interface="jax", diff_method='adjoint')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = jnp.array([0.1])
        print(qp.grad(circuit)(x))

>>> print_grad()
[-0.09983342]

For a scalar cost function, ``device_vjp=True`` allows the device to compute
vector-Jacobian products without constructing the full Jacobian. The following
example obtains probabilities as expectation values of computational-basis projectors.
Adjoint differentiation on ``lightning.qubit`` also supports expectation values.

.. code-block:: python

    @qp.qnode(
        qp.device('default.qubit'), interface="jax", diff_method="adjoint", device_vjp=True
    )
    def circuit(x):
        qp.IsingXX(x, wires=(0, 1))
        return tuple(
            qp.expval(qp.Projector(bits, wires=(0, 1)))
            for bits in [[0, 0], [0, 1], [1, 0], [1, 1]]
        )

    def cost(x):
        probs = jnp.stack(circuit(x))
        target = jnp.array([0, 0, 0, 1])
        return jnp.linalg.norm(probs - target)

>>> float(qp.grad(cost)(jnp.array(0.1)))
-0.07059288589999413

Furthermore, the adjoint differentiation algorithm is analytic by nature. If an execution
has ``shots>0``, an error is raised:

.. code-block:: python

    def print_grad_ok():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=100)
        @qp.qnode(dev, interface="jax", diff_method='adjoint')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = jnp.array([0.1])
        print(qp.grad(circuit)(x))

>>> print_grad_ok()
Traceback (most recent call last):
    ...
pennylane.exceptions.QuantumFunctionError: Device ... does not support adjoint with requested circuit.
...

.. _State gradients:

State gradients
~~~~~~~~~~~~~~~~

Quantum states are complex-valued. A real scalar cost function can be differentiated
with respect to real circuit parameters using backpropagation:

.. code-block:: python

    def state_scalar_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.qnode(dev, interface="jax", diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        def cost_fn(x):
            out = circuit(x)
            return jnp.abs(out[0])

        x = jnp.array([0.1])
        print(qp.grad(cost_fn)(x))

>>> state_scalar_grad()
[-0.02498958]

To obtain derivatives of every state amplitude with respect to real parameters,
differentiate its real and imaginary parts separately:

.. code-block:: python

    def state_vector_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.qnode(dev, interface="jax", diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        def real_imag_state(x):
            psi = circuit(x)
            return jnp.stack((jnp.real(psi), jnp.imag(psi)))

        x = jnp.array([0.1])
        jacobian = jax.jacobian(real_imag_state)(x)
        print(np.round(np.asarray(jacobian), 8) + 0.0)

>>> state_vector_grad()
[[[-0.02498958]
  [ 0.        ]]
<BLANKLINE>
 [[ 0.        ]
  [-0.49937513]]]

The first axis selects the real or imaginary part, the second selects the state
amplitude, and the final axis selects the input parameter.

.. _Sample gradients:

Sample gradients
~~~~~~~~~~~~~~~~~~~~~~~

In PennyLane, samples are drawn from the eigenvalues of an observable, or from the
computational basis states if no observable is provided. This process is not differentiable
in general, so no gradient flow backwards through the sampling is allowed.

Attempting to differentiate integer-valued samples with JAX raises an error:

.. code-block:: python

    def sample_backward():
        dev = qp.device('default.qubit', wires=1, seed=123)

        @qp.set_shots(shots=5)
        @qp.qnode(dev, interface="jax")
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.sample(wires=0)

        x = jnp.array([np.pi / 2])
        print(qp.jacobian(circuit)(x))

>>> sample_backward()
Traceback (most recent call last):
    ...
TypeError: Custom JVP rule must produce primal and tangent outputs with corresponding shapes and dtypes...
...

The forward pass is supported and will work as expected:

.. code-block:: python

    def sample_forward():
        dev = qp.device('default.qubit', wires=1, seed=123)

        @qp.set_shots(shots=20)
        @qp.qnode(dev, interface="numpy")
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.sample(wires=0)

        x = np.array([np.pi / 2])
        print(circuit(x))

>>> sample_forward()
[[1]
 [0]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]]
