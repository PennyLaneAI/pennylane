.. _tf_interf:

TensorFlow interface
=====================

In order to use PennyLane in combination with TensorFlow, we have to generate TensorFlow-compatible
quantum nodes. Such a QNode can be created explicitly using the ``interface='tf'`` keyword in the
QNode decorator or QNode class constructor. Alternatively, an existing, basic QNode can be
translated into a quantum node that interfaces with TensorFlow by calling the
:meth:`QNode.to_tf() <pennylane.qnode.QNode.to_tf>` method.

.. note::
    To use the TensorFlow interface in PennyLane, you must first install TensorFlow.
    Note that this interface only supports TensorFlow versions >= 2.0!

Tensorflow is imported as follows:

.. code::

    import pennylane as qml
    import tensorflow as tf

Using the TensorFlow interface is easy in PennyLane --- let's consider a few ways
it can be done.


.. _tf_interf_keyword:

Construction via keyword
------------------------

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating
:class:`~.QNode` objects in PennyLane. The only change required to construct a TensorFlow-capable
QNode is to specify the ``interface='tf'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='tf')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit()`` is now a TensorFlow-capable QNode, accepting ``tf.Variable`` and
``tf.Tensor`` objects as input, and returning ``tf.Tensor`` objects.

>>> phi = tf.Variable([0.5, 0.1])
>>> theta = tf.Variable(0.2)
>>> circuit(phi, theta)
<tf.Tensor: id=22, shape=(2,), dtype=float64, numpy=array([ 0.87758256,  0.68803733])>

TensorFlow-capable QNodes can also be created using the
:ref:`QNode class constructor <intro_vcirc_qnode>`:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('default.mixed', wires=2)

    def circuit1(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

    qnode1 = qml.QNode(circuit1, dev1)
    qnode2 = qml.QNode(circuit1, dev2, interface='tf')

``qnode1()`` is a default NumPy-interfacing QNode, while ``qnode2()`` is a TensorFlow-capable
QNode:

>>> qnode2(phi, theta)
<tf.Tensor: id=22, shape=(2,), dtype=float64, numpy=array([ 0.87758256,  0.68803733])>


.. _tf_interf_convert:

Construction from an existing QNode
-----------------------------------

Let's say we change our mind and now want ``qnode1()`` to interface with TensorFlow. We can easily
perform the conversion by using the :meth:`~.QNode.to_tf` method:

>>> qnode1.to_tf()
>>> qnode1
<QNode: device='default.mixed', func=circuit1, wires=2, interface=TensorFlow>

``qnode1()`` is now a TensorFlow-capable QNode, as well.


.. _tf_qgrad:

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

    phi = tf.Variable([0.5, 0.1])
    theta = tf.Variable(0.2)

    with tf.GradientTape() as tape:
        # Use the circuit to calculate the loss value
        loss = circuit(phi, theta)

    phi_grad, theta_grad = tape.gradient(loss, [phi, theta])

Now, printing the gradients, we get:

>>> phi_grad
array([-0.47942549,  0.        ])
>>> theta_grad
-5.5511151231257827e-17

To include non-differentiable data arguments, simply use ``tf.constant``:

.. code-block:: python

    @qml.qnode(dev, interface='tf')
    def circuit3(weights, data):
        qml.templates.AmplitudeEmbedding(data, normalize=True, wires=[0, 1])
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(weights[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    weights = tf.Variable([0.1, 0.2, 0.3])
    data = tf.constant(np.random.random([4]))

    with tf.GradientTape() as tape:
        result = circuit3(weights, data)

Calculating the gradient:

>>> grad = tape.gradient(result, weights)
>>> grad
<tf.Tensor: shape=(3,), dtype=float64, numpy=array([-2.26641213e-02,  8.32667268e-17,  5.55111512e-17])>


.. _tf_optimize:

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
>>> circuit4(phi, theta)
<tf.Tensor: id=106269, shape=(), dtype=float64, numpy=0.5000000000000091>

Keras integration
-----------------

Once you have a TensorFlow-compaible QNode, it is easy to convert this into a Keras layer. To
help automate this process, PennyLane also provides a :class:`~.qnn.KerasLayer` class to easily
convert a QNode to a Keras layer. Please see the corresponding :class:`~.qnn.KerasLayer`
documentation for more details and examples.
