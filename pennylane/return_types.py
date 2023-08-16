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
import warnings

__activated = True


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

    .. _return-autograd-tf-gotcha:

    .. details::
        :title: Usage Details
        :href: return-autograd-tf-gotcha

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

        If the measurements do not have the same shape then you need to use TF hstack:

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
                res = tf.experimental.numpy.hstack(res)

            assert circuit.qtape.trainable_params == [0, 1]

            tape.jacobian(res, [a, b])

        If no stacking is performed, TensorFlow raises the following error:

        .. code-block:: python

            AttributeError: 'tuple' object has no attribute 'shape'

    .. details::
        :title: JAX Usage Details

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
        ((Array(-0.19767681, dtype=float32, weak_type=True),
          Array(-0.09784342, dtype=float32, weak_type=True)),
         (Array(-0.09784339, dtype=float32, weak_type=True),
          Array(-0.19767687, dtype=float32, weak_type=True)))

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
        (Array([-9.9833414e-02, -7.4505806e-09,  6.9285655e-10], dtype=float32),
         Array([[-4.9419206e-02, -9.9086545e-02,  3.4938008e-09],
                      [-4.9750542e-04,  9.9086538e-02,  1.2768372e-10],
                      [ 4.9750548e-04,  2.4812977e-04,  4.8371929e-13],
                      [ 4.9419202e-02, -2.4812980e-04,  2.6696912e-11]],            dtype=float32))

        where before the following error was raised:

        .. code-block:: python

            ValueError: All input arrays must have the same shape.

        The new return types system also unlocks the use of shot vectors with all the previous features. For example you
        can take the second derivative and multiple measurement with JAX:

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
        ((Array([[ 0.,  0.],
                        [ 2., -3.]], dtype=float32),
          Array([[[-0.5,  0. ],
                       [ 0. ,  0. ]],
                      [[ 0.5,  0. ],
                       [ 0. ,  0. ]]], dtype=float32)),
         (Array([[ 0.07677898,  0.0563341 ],
                       [ 0.07238522, -1.830669  ]], dtype=float32),
          Array([[[-4.9707499e-01,  2.9999996e-04],
                        [-6.2500127e-04,  1.2500001e-04]],
                       [[ 4.9707499e-01, -2.9999996e-04],
                        [ 6.2500127e-04, -1.2500001e-04]]], dtype=float32)))

        Note that failing to set the ``max-diff`` with jitting will raise a somewhat unrelated error:

        .. code-block:: python

            import jax
            import pennylane as qml

            jax.config.update("jax_enable_x64", True)
            qml.enable_return()

            dev = qml.device("default.qubit", wires=2, shots=10000)

            params = jax.numpy.array([0.1, 0.2])

            @jax.jit
            @qml.qnode(dev, interface="jax", diff_method="parameter-shift") # Note: max_diff=2 should be passed here
            def circuit(x):
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        .. code-block:: python

            >>> jax.hessian(circuit)(params)
            ~/anaconda3/lib/python3.9/site-packages/jax/_src/callback.py in pure_callback_jvp_rule(***failed resolving arguments***)
                 53 def pure_callback_jvp_rule(*args, **kwargs):
                 54   del args, kwargs
            ---> 55   raise ValueError(
                 56       "Pure callbacks do not support JVP. "
                 57       "Please use `jax.custom_jvp` to use callbacks while taking gradients.")
            ValueError: Pure callbacks do not support JVP. Please use `jax.custom_jvp` to use callbacks while taking gradients.

        Correctly specifying ``max_diff=2`` as a QNode argument helps compute the Hessian:

        .. code-block:: python

            >>> jax.hessian(circuit)(params)
            (Array([[ 0.06617428,  0.07165382],
                    [ 0.07165382, -1.8348092 ]], dtype=float64),
             Array([[[-4.974e-01, -2.025e-03],
                     [-2.025e-03,  1.000e-04]],
                    [[ 4.974e-01,  2.025e-03],
                     [ 2.025e-03, -1.000e-04]]], dtype=float64))

    .. details::
        :title: Autograd Usage Details

        Autograd only allows differentiating functions that have array or tensor outputs. QNodes that have
        multiple measurements may output tuples and cause errors with Autograd. Similarly, the outputs of
        ``qml.grad`` or ``qml.jacobian`` may be tuples if there are multiple trainable arguments and
        cause errors with higher-order derivatives.

        These issues can be overcome by stacking the QNode results or gradients before computing derivatives.

        **Examples for shot vectors**

        Example for first-order derivatives with a single return:

        .. code-block:: python

            def func(a, b):
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])

            dev = qml.device("default.qubit", wires=2, shots=[100, 200, 300])

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0))

            def cost(a, b):
                res = circuit(a, b)
                return qml.math.stack(res)

        >>> a = np.array(0.4, requires_grad=True)
        >>> b = np.array(0.6, requires_grad=True)
        >>> cost(a, b)
        [0.94       0.93       0.89333333]
        >>> qml.jacobian(cost)(a, b)
        (array([-0.44, -0.38, -0.38]), array([-0.01      , -0.01      , -0.00666667]))

        Example for first-order derivatives with multiple returns:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])

            def cost(a, b):
                res = circuit(a, b)
                return qml.math.stack([qml.math.hstack(r) for r in res])

        >>> cost(a, b)
        [[0.92       0.91       0.05       0.03       0.01      ]
         [0.95       0.9        0.075      0.         0.025     ]
         [0.95333333 0.89666667 0.08       0.         0.02333333]]
        >>> qml.jacobian(cost)(a, b)
        (array([[-0.41  , -0.16  , -0.045 ,  0.01  ,  0.195 ],
                [-0.325 , -0.1325, -0.03  ,  0.01  ,  0.1525],
                [-0.39  , -0.175 , -0.02  ,  0.02  ,  0.175 ]]),
         array([[-0.01      , -0.26      ,  0.255     ,  0.02      , -0.015     ],
                [ 0.01      , -0.2525    ,  0.2575    ,  0.0075    , -0.0125    ],
                [-0.02      , -0.26666667,  0.25666667,  0.01666667, -0.00666667]]))

        Example for second-order derivatives with a single return:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

            def cost(a, b):
                def cost2(a, b):
                    res = circuit(a, b)
                    return qml.math.stack(res)

                return qml.math.stack(qml.jacobian(cost2)(a, b))

        >>> cost(a, b)
        [[ 0.05       -0.01        0.00666667]
         [-0.51       -0.48       -0.62      ]]
        >>> qml.jacobian(cost)(a, b)
        (array([[-0.03      , -0.02      , -0.00333333],
                [ 0.01      ,  0.025     , -0.02      ]]),
         array([[ 0.005     ,  0.03      , -0.02166667],
                [-0.85      , -0.83      , -0.81333333]]))

        Example for second-order derivatives with multiple returns:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.probs([0, 1])

            def cost(a, b):
                def cost2(a, b):
                    res = circuit(a, b)
                    return qml.math.stack([qml.math.hstack(r) for r in res])

                return qml.math.stack(qml.jacobian(cost2)(a, b))

        >>> cost(a, b)
        [[[-0.05       -0.2         0.          0.025       0.175     ]
          [ 0.02       -0.17       -0.0275      0.0175      0.18      ]
          [-0.00666667 -0.19166667 -0.01333333  0.01666667  0.18833333]]
         [[-0.52       -0.25        0.265      -0.005      -0.01      ]
          [-0.54       -0.25        0.2625      0.0075     -0.02      ]
          [-0.57666667 -0.275       0.28166667  0.00666667 -0.01333333]]]
        >>> qml.jacobian(cost)(a, b)
        (array([[[ 0.02      , -0.38      , -0.055     ,  0.045     ,  0.39      ],
                 [ 0.03      , -0.4275    , -0.04      ,  0.025     ,  0.4425    ],
                 [-0.00666667, -0.42333333, -0.025     ,  0.02833333,  0.42      ]],
                [[ 0.        ,  0.0625    , -0.065     ,  0.065     , -0.0625    ],
                 [ 0.015     ,  0.06875   , -0.05      ,  0.0425    , -0.06125   ],
                 [-0.03      ,  0.03833333, -0.04583333,  0.06083333, -0.05333333]]]),
         array([[[-0.06      ,  0.0325    , -0.0525    ,  0.0825    , -0.0625    ],
                 [ 0.02      ,  0.03      , -0.035     ,  0.025     , -0.02      ],
                 [ 0.00833333,  0.06166667, -0.05666667,  0.0525    , -0.0575    ]],
                [[-0.85      , -0.415     ,  0.425     ,  0.        , -0.01      ],
                 [-0.78      , -0.3775    ,  0.375     ,  0.015     , -0.0125    ],
                 [-0.80666667, -0.38      ,  0.38666667,  0.01666667, -0.02333333]]]))

    .. details::
        :title: TensorFlow Usage Details

        TensorFlow only allows differentiating functions that have array or tensor outputs. QNodes that have
        multiple measurements may output tuples and cause errors with TensorFlow. Similarly, the outputs of
        ``tape.gradient`` or ``tape.jacobian`` may be tuples if there are multiple trainable arguments and
        cause errors with higher-order derivatives.

        These issues can be overcome by stacking the QNode results or gradients before computing derivatives.

        **Examples for shot vectors**

        Example for first-order derivatives with a single return:

        .. code-block:: python

            def func(a, b):
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])

            dev = qml.device("default.qubit", wires=2, shots=[100, 200, 300])

            @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0))

        >>> a = tf.Variable(0.4, dtype=tf.float64, trainable=True)
        >>> b = tf.Variable(0.6, dtype=tf.float64, trainable=True)
        >>> with tf.GradientTape() as tape:
        ...     res = circuit(a, b)
        ...     res = tf.stack(res)
        ...
        >>> res
        tf.Tensor([0.92 0.92 0.94], shape=(3,), dtype=float64)
        >>> tape.jacobian(res, (a, b))
        (<tf.Tensor: shape=(3,), dtype=float64, numpy=array([-0.31      , -0.385     , -0.32666667])>,
         <tf.Tensor: shape=(3,), dtype=float64, numpy=array([-0.02      ,  0.02      , -0.00333333])>)

        Example for first-order derivatives with multiple returns:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])

        >>> with tf.GradientTape() as tape:
        ...     res = circuit(a, b)
        ...     res = tf.stack([tf.experimental.numpy.hstack(r) for r in res])
        ...
        >>> res
        tf.Tensor(
        [[0.96       0.93       0.05       0.         0.02      ]
         [0.9        0.87       0.08       0.01       0.04      ]
         [0.96       0.86666667 0.11333333 0.         0.02      ]], shape=(3, 5), dtype=float64)
        >>> tape.jacobian(res, (a, b))
        (<tf.Tensor: shape=(3, 5), dtype=float64, numpy=
         array([[-0.3       , -0.145     , -0.005     ,  0.03      ,  0.12      ],
                [-0.395     , -0.19      , -0.0075    ,  0.01      ,  0.1875    ],
                [-0.39333333, -0.17833333, -0.01833333,  0.00666667,  0.19      ]])>,
         <tf.Tensor: shape=(3, 5), dtype=float64, numpy=
         array([[-0.03      , -0.21      ,  0.195     ,  0.025     , -0.01      ],
                [ 0.005     , -0.285     ,  0.2875    ,  0.0075    , -0.01      ],
                [-0.01333333, -0.27      ,  0.26333333,  0.015     , -0.00833333]])>)

        Example for second-order derivatives with a single return:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift", max_diff=2, interface="tf")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        >>> with tf.GradientTape() as tape1:
        ...     with tf.GradientTape(persistent=True) as tape2:
        ...         res = circuit(a, b)
        ...         res = tf.stack(res)
        ...
        ...    jac = tape2.jacobian(res, (a, b), experimental_use_pfor=False)
        ...    jac = tf.stack(jac)
        ...
        >>> jac
        tf.Tensor(
        [[-0.02       -0.005       0.00333333]
         [-0.47       -0.64       -0.58      ]], shape=(2, 3), dtype=float64)
        >>> tape1.jacobian(jac, (a, b))
        (<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
         array([[ 0.00000000e+00,  2.50000000e-02, -1.66666667e-02],
                [-1.50000000e-02, -4.00000000e-02,  2.77555756e-17]])>,
         <tf.Tensor: shape=(2, 3), dtype=float64, numpy=
         array([[-0.015, -0.04 ,  0.   ],
                [-0.84 , -0.825, -0.84 ]])>)

        Example for second-order derivatives with multiple returns:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift", max_diff=2, interface="tf")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.probs([0, 1])

        >>> with tf.GradientTape() as tape1:
        ...     with tf.GradientTape(persistent=True) as tape2:
        ...         res = circuit(a, b)
        ...         res = tf.stack([tf.experimental.numpy.hstack(r) for r in res])
        ...
        ...     jac = tape2.jacobian(res, (a, b), experimental_use_pfor=False)
        ...     jac = tf.stack(jac)
        ...
        >>> jac
        tf.Tensor(
        [[[ 0.01       -0.13       -0.03        0.025       0.135     ]
          [-0.045      -0.195      -0.0075      0.03        0.1725    ]
          [-0.00666667 -0.17       -0.01833333  0.02166667  0.16666667]]
         [[-0.61       -0.29        0.315      -0.01       -0.015     ]
          [-0.55       -0.255       0.265       0.01       -0.02      ]
          [-0.65       -0.305       0.31166667  0.01333333 -0.02      ]]], shape=(2, 3, 5), dtype=float64)
        >>> tape1.jacobian(jac, (a, b))
        (<tf.Tensor: shape=(2, 3, 5), dtype=float64, numpy=
         array([[[ 0.07      , -0.415     , -0.055     ,  0.02      ,  0.45      ],
                 [ 0.05      , -0.4       , -0.06      ,  0.035     ,  0.425     ],
                 [ 0.01666667, -0.42166667, -0.04      ,  0.03166667,  0.43      ]],
                [[-0.03      ,  0.0375    , -0.075     ,  0.09      , -0.0525    ],
                 [-0.005     ,  0.0375    , -0.06625   ,  0.06875   , -0.04      ],
                 [-0.04      ,  0.0375    , -0.05166667,  0.07166667, -0.0575    ]]])>,
         <tf.Tensor: shape=(2, 3, 5), dtype=float64, numpy=
         array([[[-0.03      ,  0.0375    , -0.075     ,  0.09      , -0.0525    ],
                 [-0.005     ,  0.0375    , -0.06625   ,  0.06875   , -0.04      ],
                 [-0.04      ,  0.0375    , -0.05166667,  0.07166667, -0.0575    ]],
                [[-0.81      , -0.39      ,  0.385     ,  0.02      , -0.015     ],
                 [-0.77      , -0.375     ,  0.355     ,  0.03      , -0.01      ],
                 [-0.82666667, -0.40166667,  0.4       ,  0.01333333, -0.01166667]]])>)

    .. details::
        :title: PyTorch Usage Details

        PyTorch supports differentiation of Torch tensors or tuple of Torch tensors. It makes it easy to get the
        Jacobian of functions returning any mix of measurements.

        .. code-block:: python

            def func(a, b):
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])

        >>> a = torch.tensor(0.1, requires_grad=True)
        >>> b = torch.tensor(0.2, requires_grad=True)
        >>> torch.autograd.functional.jacobian(circuit, (a, b))
        ((tensor(-0.0998), tensor(0.)), (tensor([-0.0494, -0.0005,  0.0005,  0.0494]), tensor([-0.0991,  0.0991,  0.0002, -0.0002])))

        An issue arises when one requires higher order differentation with multiple measurements, the Jacobian returns
        a tuple of tuple which is not consider as differentiable by PyTorch. This issue can be overcome by stacking
        the original results:

        .. code-block:: python

            @qml.qnode(dev, diff_method="parameter-shift", interface="torch", max_diff=2)
            def circuit(a, b):
                func(a, b)
                return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])

            a = torch.tensor(0.1, requires_grad=True)
            b = torch.tensor(0.2, requires_grad=True)

            def circuit_stack(a, b):
                return torch.hstack(circuit(a, b))

        >>> jac_fn = lambda a, b: torch.autograd.functional.jacobian(circuit_stack, (a, b), create_graph=True)
        >>> torch.autograd.functional.jacobian(jac_fn, (a, b))
        ((tensor([-0.9950, -0.4925, -0.0050,  0.0050,  0.4925]), tensor([ 0.0000,  0.0050, -0.0050,  0.0050, -0.0050])),
        (tensor([ 0.0000,  0.0050, -0.0050,  0.0050, -0.0050]), tensor([ 0.0000, -0.4888,  0.4888,  0.0012, -0.0012])))
    """

    warnings.warn(
        "The old return system is deprecated, and will be removed along with "
        "`qml.enable_return()` in v0.33. Please consult the QNode returns page for guidance on "
        "switching to the new return system: https://docs.pennylane.ai/en/stable/introduction/returns.html",
        UserWarning,
    )

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

    warnings.warn(
        "The old return system is deprecated, and will be removed along with "
        "`qml.disable_return()` in v0.33. Please consult the QNode returns page for guidance on "
        "switching to the new return system: https://docs.pennylane.ai/en/stable/introduction/returns.html",
        UserWarning,
    )

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
