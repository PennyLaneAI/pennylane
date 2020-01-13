.. _tf_interf:

TensorFlow interface
====================

To use a quantum node in combination with TensorFlow's Eager mode, we have to make it
compatible with TensorFlow. A TensorFlow-compatible quantum node can be created
either by using the ``interface='tf'`` flag in the qnode decorator, or
by calling the :meth:`QNode.to_tf <pennylane.QNode.to_tf>` method. Internally, the translation is executed by
the :func:`TFQNode <pennylane.interfaces.tf.TFQNode>` function that returns the new quantum node object.

.. note::
    To use the TensorFlow eager execution interface in PennyLane, you must first install TensorFlow.
    Note that this interface only supports TensorFlow versions >=1.12 (including version 2.0)
    in eager execution mode!

Tensorflow is imported as follows:

.. code::

    import pennylane as qml
    import tensorflow as tf

Using the TensorFlow interface is easy in PennyLane --- let's consider a few ways
it can be done.

Construction via the decorator
------------------------------

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating QNodes
in PennyLane. The only change required to construct a TensorFlow-capable QNode is to
specify the ``interface='tf'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev, interface='tf')
    def circuit1(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit1()`` is now a TensorFlow-capable QNode, accepting ``tf.Variable`` objects
as input, and returning ``tf.Tensor`` objects.

>>> phi = tf.Variable([0.5, 0.1])
>>> theta = tf.Variable(0.2)
>>> circuit1(phi, theta)
<tf.Tensor: id=22, shape=(2,), dtype=float64, numpy=array([ 0.87758256,  0.68803733])>

Construction from a basic QNode
-------------------------------

Let us first create two basic, NumPy-interfacing QNodes.

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

We can convert the default NumPy-interfacing QNodes to TensorFlow-interfacing QNodes by
using the :meth:`~.QNode.to_tf` method:

>>> qnode1 = qnode1.to_tf()
>>> qnode1
<QNode: device='default.qubit', func=circuit, wires=2, interface=TensorFlow>

Internally, the :meth:`~.QNode.to_tf` method uses the :func:`~.TFQNode` function
to do the conversion.

Quantum gradients using TensorFlow
----------------------------------

Since a TensorFlow-interfacing QNode acts like any other TensorFlow function,
the standard method used to calculate gradients in eager mode with TensorFlow can be used.

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='tf')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = Variable([0.5, 0.1])
    theta = Variable(0.2)

    with tf.GradientTape() as tape:
        # Use the circuit to calculate the loss value
        loss = circuit(phi, theta)
        phi_grad, theta_grad = tape.gradient(loss, [phi, theta])

Now, printing the gradients, we get:

>>> phi_grad
array([-0.47942549,  0.        ])
>>> theta_grad
-5.5511151231257827e-17

Optimization using TensorFlow
-----------------------------

To optimize your hybrid classical-quantum model using the TensorFlow eager interface,
you **must** make use of the TensorFlow optimizers provided in the ``tf.train`` module,
or your own custom TensorFlow optimizer. **The** :ref:`PennyLane optimizers <intro_ref_opt>`
**cannot be used with the TensorFlow interface**.

For example, to optimize a TensorFlow-interfacing QNode (below) such that the weights ``x``
result in an expectation value of 0.5, we can do the following:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='tf')
    def circuit4(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = tf.Variable([0.5, 0.1], dtype=tf.float64)
    theta = tf.Variable(0.2, dtype=tf.float64)

    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    steps = 200

    for i in range(steps):
        with tf.GradientTape() as tape:
            loss = tf.abs(circuit4(phi, theta) - 0.5)**2

        gradients = tape.gradient(loss, [phi, theta])
        opt.apply_gradients(zip(gradients, [phi, theta]))


The final weights and circuit value are:

>>> phi
<tf.Variable 'Variable:0' shape=(2,) dtype=float64, numpy=array([ 1.04719755,  0.1       ])>
>>> theta
<tf.Variable 'Variable:0' shape=() dtype=float64, numpy=0.20000000000000001>
>>> circuit(phi, theta)
<tf.Tensor: id=106269, shape=(), dtype=float64, numpy=0.5000000000000091>
