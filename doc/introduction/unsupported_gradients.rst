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

    def print_grad():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, diff_method='device')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(circuit)(x))

>>> print_grad()
Traceback (most recent call last):
  ...
  File "C:\pennylane\pennylane\qnode.py", line 448, in _validate_device_method
    raise qml.QuantumFunctionError(
pennylane.QuantumFunctionError: The default.qubit device does not provide a native method for computing the jacobian.

.. _Analytic backpropagation:

Backpropagation
~~~~~~~~~~~~~~~~~~~~~~~~

The backpropagation algorithm is analytic by nature, and hence passing ``shots=None``
is the only supported configuration when ``diff_method="backprop"`` is used. Though
it is possible to always use the analytic gradient even when ``shots>0`` (as is the case
with adjoint differentiation, see next section), in the current state of the code this would
break other things.

Currently an exception is raised if this invalid configuration is used:

.. code-block:: python

    def print_grad():
        dev = qml.device('default.qubit', wires=1, shots=100)

        @qml.qnode(dev, diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(circuit)(x))

>>> print_grad()
Traceback (most recent call last):
  ...
  File "C:\pennylane\pennylane\qnode.py", line 375, in _validate_backprop_method
    raise qml.QuantumFunctionError("Backpropagation is only supported when shots=None.")
pennylane.QuantumFunctionError: Backpropagation is only supported when shots=None.

Changing to ``shots=None`` allows computing the analytic gradient:

.. code-block:: python

    def print_grad():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(circuit)(x))

>>> print_grad()
[-0.09983342]

.. _Adjoint differentation:

Adjoint differentiation
~~~~~~~~~~~~~~~~~~~~~~~

PennyLane implements the adjoint differentiation method from
`2009.02823 <https://arxiv.org/pdf/2009.02823.pdf>`__, which only discusses
the gradient of expectation of observables. The implementation is specific to the paper, hence the return statement
of the quantum function wrapped in ``qml.qnode`` can only contain :func:`~.pennylane.expval` as a measurement.

In particular, the following code works as expected:

.. code-block:: python

    def print_grad():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, diff_method='adjoint')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(circuit)(x))

>>> print_grad()
[-0.09983342]

But the following code raises an error:

.. code-block:: python

    def print_grad_bad():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, diff_method='adjoint')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.state()

        def cost_fn(x):
            out = circuit(x)
            return np.abs(out[0])

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(cost_fn)(x))

>>> print_grad_bad()
Traceback (most recent call last):
  ...
  File "C:\pennylane\pennylane\_qubit_device.py", line 951, in adjoint_jacobian
    raise qml.QuantumFunctionError(
pennylane.QuantumFunctionError: Adjoint differentiation method does not support measurement state

Furthermore, the adjoint differentiation algorithm is analytic by nature. If the user creates a device
with ``shots>0``, a warning is raised and gradients are computed analytically:

.. code-block:: python

    def print_grad_ok():
        dev = qml.device('default.qubit', wires=1, shots=100)

        @qml.qnode(dev, diff_method='adjoint')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(circuit)(x))

>>> print_grad_ok()
C:\pennylane\pennylane\qnode.py:434: UserWarning: Requested adjoint differentiation to be computed with finite shots. Adjoint differentiation always calculated exactly.
  warnings.warn(
C:\pennylane\pennylane\_qubit_device.py:965: UserWarning: Requested adjoint differentiation to be computed with finite shots. The derivative is always exact when using the adjoint differentiation method.
  warnings.warn(
[-0.09983342]

.. _State gradients:

State gradients
~~~~~~~~~~~~~~~~

In general, the state of a quantum circuit will be complex-valued, so differentiating
the state directly is not possible without the use of
`complex analysis <https://en.wikipedia.org/wiki/Holomorphic_function>`__. Though complex
gradients can be implemented for most "simple" functions, this is not supported in Autograd
but is done in the other three interfaces.

Instead, in Autograd, real scalar-valued post-processing should be performed on the output state to allow
the auto-differentiation frameworks to backpropagate through them. For example, the following
code uses a scalar cost function dependent on the output state:

.. code-block:: python

    def state_scalar_grad():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.state()

        def cost_fn(x):
            out = circuit(x)
            return np.abs(out[0])

        x = np.array([0.1], requires_grad=True)
        print(qml.grad(cost_fn)(x))

>>> state_scalar_grad()
[-0.02498958]

However, changing from differentiating the scalar cost to differentiating the state
directly will fail with an error:

.. code-block:: python

    def state_vector_grad():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.state()

        x = np.array([0.1], requires_grad=True)
        print(qml.jacobian(circuit)(x))

>>> state_vector_grad()
Traceback (most recent call last):
  ...
  File "C:\Python38\lib\site-packages\numpy\core\fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: cannot reshape array of size 4 into shape (2,1)

Using a different interface that supports complex differentiation will fix this error:

.. code-block:: python

    def state_vector_grad_jax():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, interface='jax', diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.state()

        x = jnp.array([0.1], dtype=np.complex64)
        print(jax.jacrev(circuit, holomorphic=True)(x))

    def state_vector_grad_tf():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, interface='tf', diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.state()

        x = tf.Variable([0.1], trainable=True, dtype=np.complex64)
        with tf.GradientTape() as tape:
            out = circuit(x)

        print(tape.jacobian(out, [x]))

    def state_vector_grad_torch():
        dev = qml.device('default.qubit', wires=1, shots=None)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.state()

        x = torch.tensor([0.1], requires_grad=True, dtype=torch.complex64)
        print(F.jacobian(circuit, (x,)))

>>> state_vector_grad_jax()
[[-0.02498958+0.j        ]
 [ 0.        -0.49937513j]]
>>> state_vector_grad_tf()
[<tf.Tensor: shape=(2, 1), dtype=complex64, numpy=
array([[-0.02498958+0.j        ],
       [-0.        +0.49937513j]], dtype=complex64)>]
>>> state_vector_grad_torch()
(tensor([[-0.0250+0.0000j],
        [ 0.0000+0.4994j]]),)

.. _Sample gradients:

Sample gradients
~~~~~~~~~~~~~~~~~~~~~~~

In PennyLane, samples are drawn from the eigenvalues of an observable, or from the
computational basis states if no observable is provided. This process is not differentiable
in general, so no gradient flow backwards through the sampling is allowed.

Currently, attempting to compute the gradient in this scenario will not raise an
error, but the results will be incorrect:

.. code-block:: python

    def sample_backward():
        dev = qml.device('default.qubit', wires=1, shots=20)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.sample(wires=0)

        x = np.array([np.pi / 2])
        print(qml.jacobian(circuit)(x))

>>> sample_backward()
[[0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]
 [0.5]]

The forward pass is supported and will work as expected:

.. code-block:: python

    def sample_forward():
        dev = qml.device('default.qubit', wires=1, shots=20)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0)
            return qml.sample(wires=0)

        x = np.array([np.pi / 2])
        print(circuit(x))

>>> sample_forward()
[0 1 0 0 0 1 1 0 0 1 1 1 0 0 0 1 1 0 0 0]
