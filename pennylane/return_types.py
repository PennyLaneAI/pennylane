# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Class and functions for activating, deactivating and checking the new return types system
"""
# pylint: disable=global-statement
__activated = False


def enable_return():
    """Function that turns on the experimental return type system that prefers the use of sequences over arrays.

    The new system guarantees that a sequence (e.g., list or tuple) is returned based on the ``return`` statement of the
    quantum function. This system avoids the creation of ragged arrays, where multiple measurements are stacked
    together.

    **Example**

    The following example shows that for multiple measurements the current PennyLane system is creating a ragged tensor.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
              qml.Hadamard(wires=[0])
              qml.CRX(x, wires=[0, 1])
              return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)

    >>> qnode(0.5)
    tensor([0.5       , 0.5       , 0.08014815, 0.96939564, 0.03060436,
        0.93879128], requires_grad=True)

    when you activate the new return type the result is simply a tuple containing each measurement.

    .. code-block:: python

        qml.enable_return()

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
              qml.Hadamard(wires=[0])
              qml.CRX(x, wires=[0, 1])
              return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)

    >>> qnode(0.5)
    (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.96939564, 0.03060436], requires_grad=True), tensor(0.93879128, requires_grad=True))

    .. note::

        This is an experimental feature and may not support every feature in PennyLane. The list of supported features
        from PennyLane include:

        * :func:`~.pennylane.execute`
        * Gradient transforms

          #. :func:`~.pennylane.gradients.param_shift`;
          #. :func:`~.pennylane.gradients.finite_diff`;
          #. :class:`~.pennylane.gradients.hessian_transform`;
          #. :func:`~.pennylane.gradients.param_shift_hessian`.

        * Interfaces

          #. Autograd;
          #. TensorFlow;
          #. JAX (without jitting);

        * PennyLane optimizers
        * :meth:`~.pennylane.tape.QuantumTape.shape`

    Note that this is an experimental feature and may not support every feature in PennyLane. See the ``Usage Details``
    section for more details.

    .. details::
        :title: Usage Details

        **Gotcha: Autograd and TensorFlow can only compute gradients of tensor-valued functions**

        Autograd and TensorFlow only allow differentiating functions that have array or tensor outputs. QNodes that
        have multiple measurements may output other sequences with the new return types may cause errors with Autograd
        or TensorFlow.

        This issue can be overcome by stacking the QNode results before computing derivatives:

        .. code-block:: python

            qml.enable_return()

            a = np.array(0.1, requires_grad=True)
            b = np.array(0.2, requires_grad=True)

            dev = qml.device("lightning.qubit", wires=2)

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(a, b):
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

            def cost(x, y):
                return qml.numpy.hstack(circuit(x, y))

        >>> qml.jacobian(cost)(a, b)
        (array([-0.09983342,  0.01983384]), array([-5.54649074e-19, -9.75170327e-01]))

        If no stacking is performed, Autograd raises the following error:

        .. code-block:: python

            TypeError: 'ArrayVSpace' object cannot be interpreted as an integer

        The solution with TensorFlow is similar with the difference that stacking happens within the
        ``tf.GradientTape()`` context:

        .. code-block:: python

            a = tf.Variable(0.1, dtype=tf.float64)
            b = tf.Variable(0.2, dtype=tf.float64)

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
            def circuit(a, b):
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

            with tf.GradientTape() as tape:
                res = circuit(a, b)
                res = tf.stack(res)

            assert circuit.qtape.trainable_params == [0, 1]

            tape.jacobian(res, [a, b])

        If the measurements do not have the same shape then you need to use concatenation:

        .. code-block:: python

            a = tf.Variable(0.1, dtype=tf.float64)
            b = tf.Variable(0.2, dtype=tf.float64)

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
            def circuit(a, b):
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

            with tf.GradientTape() as tape:
                res = circuit(a, b)
                res = tf.concat([tf.reshape(i, [-1]) for i in res], 0)

            assert circuit.qtape.trainable_params == [0, 1]

            tape.jacobian(res, [a, b])

        If no stacking is performed, TensorFlow raises the following error:

        .. code-block:: python

            AttributeError: 'tuple' object has no attribute 'shape'

        **JAX interface upgrades: higher-order derivatives and mixing measurements**

        Higher-order derivatives can now be computed with the JAX interface:

        .. code-block:: python

            import jax

            qml.enable_return()

            dev = qml.device("lightning.qubit", wires=2)

            par_0 = jax.numpy.array(0.1)
            par_1 = jax.numpy.array(0.2)

            @qml.qnode(dev, interface="jax", diff_method="parameter-shift", max_diff=2)
            def circuit(x, y):
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        >>> jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)
        ((DeviceArray(-0.19767681, dtype=float32, weak_type=True),
          DeviceArray(-0.09784342, dtype=float32, weak_type=True)),
         (DeviceArray(-0.09784339, dtype=float32, weak_type=True),
          DeviceArray(-0.19767687, dtype=float32, weak_type=True)))

        The new return types system also unlocks the use of ``probs`` mixed with different measurements with JAX:

        .. code-block:: python

            import jax

            qml.enable_return()

            dev = qml.device("default.qubit", wires=2)
            qml.enable_return()

            @qml.qnode(dev, interface="jax")
            def circuit(a):
              qml.RX(a[0], wires=0)
              qml.CNOT(wires=(0, 1))
              qml.RY(a[1], wires=1)
              qml.RZ(a[2], wires=1)
              return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1])

            x = jax.numpy.array([0.1, 0.2, 0.3])

        >>> jax.jacobian(circuit)(x)
        (DeviceArray([-9.9833414e-02, -7.4505806e-09,  6.9285655e-10], dtype=float32),
         DeviceArray([[-4.9419206e-02, -9.9086545e-02,  3.4938008e-09],
                      [-4.9750542e-04,  9.9086538e-02,  1.2768372e-10],
                      [ 4.9750548e-04,  2.4812977e-04,  4.8371929e-13],
                      [ 4.9419202e-02, -2.4812980e-04,  2.6696912e-11]],            dtype=float32))

        where before the following error was raised:

        .. code-block:: python

            ValueError: All input arrays must have the same shape.

        The new return types system also unlocks the use of shot vectors with all the previous features. For example you
        can take the second derivative and multiple measurement with with JAX:

        .. code-block:: python

            import jax

            qml.enable_return()

            dev = qml.device("default.qubit", wires=2, shots=(1, 10000))

            params = jax.numpy.array([0.1, 0.2])

            @qml.qnode(dev, interface="jax", diff_method="parameter-shift", max_diff=2)
            def circuit(x):
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        >>> jax.hessian(circuit)(params)
        ((DeviceArray([[ 0.,  0.],
                        [ 2., -3.]], dtype=float32),
          DeviceArray([[[-0.5,  0. ],
                       [ 0. ,  0. ]],
                      [[ 0.5,  0. ],
                       [ 0. ,  0. ]]], dtype=float32)),
         (DeviceArray([[ 0.07677898,  0.0563341 ],
                       [ 0.07238522, -1.830669  ]], dtype=float32),
          DeviceArray([[[-4.9707499e-01,  2.9999996e-04],
                        [-6.2500127e-04,  1.2500001e-04]],
                       [[ 4.9707499e-01, -2.9999996e-04],
                        [ 6.2500127e-04, -1.2500001e-04]]], dtype=float32)))

    """

    global __activated
    __activated = True


def disable_return():
    """Function that turns off the new return type system.

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
              qml.Hadamard(wires=[0])
              qml.CRX(x, wires=[0, 1])
              return qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=1), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)


    >>> qml.enable_return()
    >>> res = qnode(0.5)
    >>> res
    (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.96939564, 0.03060436], requires_grad=True), tensor(0.93879128, requires_grad=True))
    >>> qml.disable_return()
    >>> res = qnode(0.5)
    >>> res
    tensor([0.5       , 0.5       , 0.08014815, 0.96939564, 0.03060436, 0.93879128], requires_grad=True)

    """
    global __activated
    __activated = False  # pragma: no cover


def active_return():
    """Function that checks if the new return types system is activated.

    Returns:
        bool: Returns ``True`` if the new return types system is activated.

    **Example**

    By default, the new return types system is turned off:

    >>> active_return()
    False

    It can be activated:

    >>> enable_return()
    >>> active_return()
    True

    And it can also be deactivated:

    >>> enable_return()
    >>> active_return()
    True
    >>> disable_return()
    >>> active_return()
    False
    """
    return __activated
