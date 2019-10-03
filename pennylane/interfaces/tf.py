# Copyright 2018 Xanadu Quantum Technologies Inc.

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
.. _tf_qnode:

TensorFlow interface
********************

**Module name:** :mod:`pennylane.interfaces.tf`

.. currentmodule:: pennylane.interfaces.tf

Using the TensorFlow interface
------------------------------

.. note::

    To use the TensorFlow interface in PennyLane, you must first install TensorFlow.

    Note that this interface **only** supports TensorFlow versions >=1.12
    (including version 2.0) in eager execution mode!

Using the TensorFlow interface is easy in PennyLane---let's consider a few ways
it can be done.


Via the QNode decorator
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`QNode decorator <qnode_decorator>` is the recommended way for creating QNodes
in PennyLane. The only change required to construct a TensorFlow-capable QNode is to
specify the ``interface='tf'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='tf')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit()`` is now a TensorFlow-capable QNode, accepting ``Variable`` objects
as input, and returning ``tf.Tensor`` objects.

>>> phi = Variable([0.5, 0.1])
>>> theta = Variable(0.2)
>>> circuit(phi, theta)
<tf.Tensor: id=22, shape=(2,), dtype=float64, numpy=array([ 0.87758256,  0.68803733])>


Via the QNode class
^^^^^^^^^^^^^^^^^^^

Sometimes, it is more convenient to instantiate a :class:`~.QNode` object directly, for example,
if you would like to reuse the same quantum function across multiple devices, or even
using different classical interfaces:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('forest.wavefunction', wires=2)

    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

    qnode1 = qml.QNode(circuit, dev1)
    qnode2 = qml.QNode(circuit, dev2)

We can convert the default NumPy-interfacing QNode to a TensorFlow-interfacing QNode by
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

.. _pytf_optimize:

Optimization using TensorFlow
-----------------------------

To optimize your hybrid classical-quantum model using the TensorFlow eager interface,
you **must** make use of the TensorFlow optimizers provided in the ``tf.train`` module,
or your own custom TensorFlow optimizer. **The** :ref:`PennyLane optimizers <optimization_methods>`
**cannot be used with the TensorFlow interface, only the** :ref:`numpy_qnode`.

For example, to optimize a TF-interfacing QNode (below) such that the weights ``x``
result in an expectation value of 0.5, we can do the following:

.. code-block:: python

    import tensorflow as tf

    # For TensorFlow 2.X:
    from tensorflow import Variable

    # if using TensorFlow 1.X, you will
    # instead require the following three lines:
    #
    # import tensorflow.contrib.eager as tfe
    # Variable = tfe.Variable
    # tf.enable_eager_execution()

    import pennylane as qml

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='tf')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = Variable([0.5, 0.1], dtype=tf.float64)
    theta = Variable(0.2, dtype=tf.float64)

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    steps = 200

    for i in range(steps):
        with tf.GradientTape() as tape:
            loss = tf.abs(circuit(phi, theta) - 0.5) ** 2
            grads = tape.gradient(loss, [phi, theta])

        opt.apply_gradients(zip(grads, [phi, theta]), global_step=tf.train.get_or_create_global_step())


The final weights and circuit value are:

>>> phi
<tf.Variable 'Variable:0' shape=(2,) dtype=float64, numpy=array([ 1.04719755,  0.1       ])>
>>> theta
<tf.Variable 'Variable:0' shape=() dtype=float64, numpy=0.20000000000000001>
>>> circuit(phi, theta)
<tf.Tensor: id=106269, shape=(), dtype=float64, numpy=0.5000000000000091>


Code details
^^^^^^^^^^^^
"""
# pylint: disable=redefined-outer-name
from functools import partial

import numpy as np
import tensorflow as tf

from pennylane.utils import unflatten


if tf.__version__[0] == "1":
    import tensorflow.contrib.eager as tfe # pylint: disable=unused-import,ungrouped-imports
    Variable = tfe.Variable
else:
    from tensorflow import Variable # pylint: disable=unused-import,ungrouped-imports


def TFQNode(qnode):
    """Function that accepts a :class:`~.QNode`, and returns a TensorFlow eager-execution-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode

    Returns:
        function: the QNode as a TensorFlow function
    """
    class qnode_str(partial):
        """TensorFlow QNode"""
        # pylint: disable=too-few-public-methods

        def __str__(self):
            """String representation"""
            detail = "<QNode: device='{}', func={}, wires={}, interface=TensorFlow>"
            return detail.format(qnode.device.short_name, qnode.func.__name__, qnode.num_wires)

        def __repr__(self):
            """REPL representation"""
            return self.__str__()

    @qnode_str
    @tf.custom_gradient
    def _TFQNode(*input_, **input_kwargs):
        # detach all input Tensors, convert to NumPy array
        args = [i.numpy() if isinstance(i, (Variable, tf.Tensor)) else i for i in input_]
        kwargs = {k:v.numpy() if isinstance(v, (Variable, tf.Tensor)) else v for k, v in input_kwargs.items()}

        # if NumPy array is scalar, convert to a Python float
        args = [i.tolist() if (isinstance(i, np.ndarray) and not i.shape) else i for i in args]
        kwargs = {k:v.tolist() if (isinstance(v, np.ndarray) and not v.shape) else v for k, v in kwargs.items()}

        # evaluate the QNode
        res = qnode(*args, **kwargs)

        if not isinstance(res, np.ndarray):
            # scalar result, cast to NumPy scalar
            res = np.array(res)

        def grad(grad_output):
            """Returns the vector-Jacobian product"""
            # evaluate the Jacobian matrix of the QNode
            jacobian = qnode.jacobian(args, **kwargs)

            grad_output_np = grad_output.numpy()

            # perform the vector-Jacobian product
            if not grad_output_np.shape:
                temp = grad_output_np * jacobian
            else:
                temp = grad_output_np.T @ jacobian

            # restore the nested structure of the input args
            grad_input = unflatten(temp.flat, args)
            return tuple(grad_input)

        return res, grad

    return _TFQNode
