# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes and functions for integrating QNodes with the Keras Layer API."""
import functools
import inspect
from collections.abc import Iterable
from typing import Optional

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from pennylane.interfaces.tf import to_tf

    CORRECT_TF_VERSION = int(tf.__version__.split(".")[0]) > 1
except ImportError:
    # The following allows this module to be imported even if TensorFlow is not installed. Users
    # will instead see an ImportError when instantiating the KerasLayer.
    from abc import ABC

    Layer = ABC
    CORRECT_TF_VERSION = False


class KerasLayer(Layer):
    """Converts a :func:`~.QNode` to a Keras
    `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__.

    The result can be used within the Keras
    `Sequential <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`__ or
    `Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ classes for
    creating quantum and hybrid models.

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Keras Layer_
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding shapes
        output_dim (int): the output dimension of the QNode
        weight_specs (dict[str, dict]): An optional dictionary for users to provide additional
            specifications for weights used in the QNode, such as the method of parameter
            initialization. This specification is provided as a dictionary with keys given by the
            arguments of the `add_weight()
            <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight>`__.
            method and values being the corresponding specification.
        **kwargs: additional keyword arguments passed to the Layer_ base class

    **Example**

    .. code-block:: python

        qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
        clayer = tf.keras.layers.Dense(2)
        model = tf.keras.models.Sequential([qlayer, clayer])

    The signature of the QNode **must** contain an ``inputs`` named argument for input data,
    with all other arguments to be treated as internal weights. A valid ``qnode`` for the example
    above would be:

    .. code-block:: python

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights_0, weight_1):
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)
            qml.Rot(*weights_0, wires=0)
            qml.RY(weight_1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    The internal weights of the QNode are automatically initialized within the
    :class:`~.KerasLayer` and must have their shapes specified in a ``weight_shapes`` dictionary.
    For example:

    .. code-block::

        weight_shapes = {"weights_0": 3, "weight_1": 1}

    .. UsageDetails::

        The QNode must have a signature that satisfies the following conditions:

        - Contain an ``inputs`` named argument for input data.
        - All other arguments must accept an array or tensor and are treated as internal
          weights of the QNode.
        - All other arguments must have no default value.
        - The ``inputs`` argument is permitted to have a default value provided the gradient with
          respect to ``inputs`` is not required.
        - There cannot be a variable number of positional or keyword arguments, e.g., no ``*args``
          or ``**kwargs`` present in the signature.

        The optional ``weight_specs`` argument allows for a more fine-grained
        specification of the QNode weights, such as the method of initialization and any
        regularization or constraints. For example, the initialization method of the ``weights``
        argument in the example above could be specified by:

        .. code-block::

            weight_specs = {"weights": {"initializer": "random_uniform"}}

        The values of ``weight_specs`` are dictionaries with keys given by arguments of
        the Keras
        `add_weight() <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight>`__
        method. For the ``"initializer"`` argument, one can specify a string such as
        ``"random_uniform"`` or an instance of an `Initializer
        <https://www.tensorflow.org/api_docs/python/tf/keras/initializers>`__ class, such as
        `tf.keras.initializers.RandomUniform <https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer>`__.

        If ``weight_specs`` is not specified, weights will be added using the Keras default
        initialization and without any regularization or constraints.

        **Additional example**

        The code block below shows how a circuit composed of templates from the
        :doc:`/code/qml_templates` module can be combined with classical
        `Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`__ layers to learn
        the two-dimensional `moons <https://scikit-learn.org/stable/modules/generated/sklearn
        .datasets.make_moons.html>`__ dataset.

        .. code-block:: python

            import pennylane as qml
            import tensorflow as tf
            import sklearn.datasets

            n_qubits = 2
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def qnode(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

            weight_shapes = {"weights": (3, n_qubits, 3)}

            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
            clayer1 = tf.keras.layers.Dense(2)
            clayer2 = tf.keras.layers.Dense(2, activation="softmax")
            model = tf.keras.models.Sequential([clayer1, qlayer, clayer2])

            data = sklearn.datasets.make_moons()
            X = tf.constant(data[0])
            Y = tf.one_hot(data[1], depth=2)

            opt = tf.keras.optimizers.SGD(learning_rate=0.5)
            model.compile(opt, loss='mae')

        The model can be trained using:

        >>> model.fit(X, Y, epochs=8, batch_size=5)
        Train on 100 samples
        Epoch 1/8
        100/100 [==============================] - 9s 90ms/sample - loss: 0.3524
        Epoch 2/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.2441
        Epoch 3/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1908
        Epoch 4/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1832
        Epoch 5/8
        100/100 [==============================] - 9s 88ms/sample - loss: 0.1596
        Epoch 6/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1637
        Epoch 7/8
        100/100 [==============================] - 9s 86ms/sample - loss: 0.1613
        Epoch 8/8
        100/100 [==============================] - 9s 87ms/sample - loss: 0.1474

    .. _Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
    """

    def __init__(
        self, qnode, weight_shapes: dict, output_dim, weight_specs: Optional[dict] = None, **kwargs
    ):
        if not CORRECT_TF_VERSION:
            raise ImportError("KerasLayer requires TensorFlow version 2 and above")

        self.sig = qnode.func.sig

        if self.input_arg not in self.sig:
            raise TypeError(
                "QNode must include an argument with name {} for inputting data".format(
                    self.input_arg
                )
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                "{} argument should not have its dimension specified in "
                "weight_shapes".format(self.input_arg)
            )

        if set(weight_shapes.keys()) | {self.input_arg} != set(self.sig.keys()):
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")

        if qnode.func.var_pos:
            raise TypeError("Cannot have a variable number of positional arguments")

        if qnode.func.var_keyword:
            raise TypeError("Cannot have a variable number of keyword arguments")

        self.qnode = to_tf(qnode, dtype=tf.keras.backend.floatx())
        self.weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        # Allows output_dim to be specified as an int, e.g., 5, or as a length-1 tuple, e.g., (5,)
        self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        defaults = {
            name for name, sig in self.sig.items() if sig.par.default != inspect.Parameter.empty
        }
        self.input_is_default = self.input_arg in defaults
        if defaults - {self.input_arg} != set():
            raise TypeError(
                "Only the argument {} is permitted to have a default".format(self.input_arg)
            )

        self.weight_specs = weight_specs if weight_specs is not None else {}

        self.qnode_weights = {}

        super().__init__(dynamic=True, **kwargs)

    def build(self, input_shape):
        """Initializes the QNode weights.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data
        """
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(name=weight, shape=size, **spec)

        super().build(input_shape)

    def call(self, inputs):
        """Evaluates the QNode on input data using the initialized weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        outputs = []
        for x in inputs:  # iterate over batch

            # The QNode can require some passed arguments to be positional and others to be keyword.
            # The following loops through input arguments in order and uses functools.partial to
            # bind the argument to the QNode.
            qnode = self.qnode

            for arg in self.sig:
                if arg is not self.input_arg:  # Non-input arguments must always be positional
                    w = self.qnode_weights[arg]
                    qnode = functools.partial(qnode, w)
                else:
                    if self.input_is_default:  # The input argument can be positional or keyword
                        qnode = functools.partial(qnode, **{self.input_arg: x})
                    else:
                        qnode = functools.partial(qnode, x)
            outputs.append(qnode())

        return tf.stack(outputs)

    def compute_output_shape(self, input_shape):
        """Computes the output shape after passing data of shape ``input_shape`` through the
        QNode.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data

        Returns:
            tf.TensorShape: shape of output data
        """
        return tf.TensorShape([input_shape[0], self.output_dim])

    def __str__(self):
        detail = "<Quantum Keras Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _input_arg = "inputs"

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Keras
        `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__. Set to
        ``"inputs"``."""
        return self._input_arg
