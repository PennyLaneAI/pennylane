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
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, diff_method='device')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qp.grad(circuit)(x))

>>> print_grad()
QuantumFunctionError: Device <default.qubit device (wires=1) at 0x11ad22bf0> does not support device with requested circuit.

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
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=100)
        @qp.qnode(dev, diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qp.grad(circuit)(x))

>>> print_grad()
QuantumFunctionError: Device <default.qubit device (wires=1, shots=100) at 0x119d6e8c0> does not support backprop with requested circuit.

Changing to ``shots=None`` allows computing the analytic gradient:

.. code-block:: python

    def print_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qp.grad(circuit)(x))

>>> print_grad()
[-0.09983342]

.. _Adjoint differentation:

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
        @qp.qnode(dev, diff_method='adjoint')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qp.grad(circuit)(x))

>>> print_grad()
[-0.09983342]

``default.qubit`` can differentiate any other measurement process as long as it
is in the Z measurement basis. In this case, we recommend using the device-provided vjp
(``device_vjp=True``) for improved performance scaling. This algorithm works
best when the final cost function only has a scalar value.

``lightning.qubit`` only supports expectation values.

.. code-block:: python 

    @qp.qnode(qp.device('default.qubit'), diff_method="adjoint", device_vjp=True)
    def circuit(x):
        qp.IsingXX(x, wires=(0,1))
        return qp.probs(wires=(0,1))

    def cost(x):
        probs = circuit(x)
        target = np.array([0, 0, 0, 1])
        return qp.math.norm(probs-target)

>>> qp.grad(cost)(qp.numpy.array(0.1))
-0.07059288589999416

Furthermore, the adjoint differentiation algorithm is analytic by nature. If the an execution
has ``shots>0``, an error is raised:

.. code-block:: python

    def print_grad_ok():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=100)
        @qp.qnode(dev, diff_method='adjoint')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.expval(qp.Z(wires=0))

        x = np.array([0.1], requires_grad=True)
        print(qp.grad(circuit)(x))

>>> print_grad_ok()
DeviceError: Finite shots are not supported with adjoint + default.qubit

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
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        def cost_fn(x):
            out = circuit(x)
            return np.abs(out[0])

        x = np.array([0.1], requires_grad=True)
        print(qp.grad(cost_fn)(x))

>>> state_scalar_grad()
[-0.02498958]

However, changing from differentiating the scalar cost to differentiating the state
directly will fail with an error:

.. code-block:: python

    def state_vector_grad():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        x = np.array([0.1], requires_grad=True)
        print(qp.jacobian(circuit)(x))

>>> state_vector_grad()
Traceback (most recent call last):
  ...
  File "C:\Python38\lib\site-packages\numpy\core\fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: cannot reshape array of size 4 into shape (2,1)

Using a different interface that supports complex differentiation will fix this error:

.. code-block:: python

    def state_vector_grad_jax():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, interface='jax', diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        x = jnp.array([0.1], dtype=np.complex64)
        print(jax.jacrev(circuit, holomorphic=True)(x))

    def state_vector_grad_tf():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, interface='tf', diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        x = tf.Variable([0.1], trainable=True, dtype=np.complex64)
        with tf.GradientTape() as tape:
            out = circuit(x)

        print(tape.jacobian(out, [x]))

    def state_vector_grad_torch():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=None)
        @qp.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.state()

        x = torch.tensor([0.1], requires_grad=True, dtype=torch.complex64)
        print(torch.autograd.functional.jacobian(circuit, (x,)))

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
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=5)
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.sample(wires=0)

        x = np.array([np.pi / 2])
        print(qp.jacobian(circuit)(x))

>>> sample_backward()
[[[0.5]]
<BLANKLINE>
 [[0.5]]
<BLANKLINE>
 [[0.5]]
<BLANKLINE>
 [[0.5]]
<BLANKLINE>
 [[0.5]]]

The forward pass is supported and will work as expected:

.. code-block:: python

    def sample_forward():
        dev = qp.device('default.qubit', wires=1)

        @qp.set_shots(shots=20)
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x[0], wires=0)
            return qp.sample(wires=0)

        x = np.array([np.pi / 2])
        print(circuit(x))

>>> sample_forward()
[[0]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [1]
 [1]
 [0]
 [1]
 [1]
 [0]
 [1]
 [0]
 [0]
 [1]]
