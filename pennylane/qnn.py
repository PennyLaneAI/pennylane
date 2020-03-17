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
"""This module contains the :class:`~.KerasLayer` class for integrating QNodes with the Keras
layer API."""
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
    from abc import ABC

    Layer = ABC
    CORRECT_TF_VERSION = False

from pennylane.qnodes import QNode

INPUT_ARG = "inputs"


class KerasLayer(Layer):
    """A Keras layer for integrating PennyLane QNodes with the Keras API.

    This class converts a :func:`~.QNode` to a Keras layer. The QNode must have a signature that
    satisfies the following conditions:

    - Contain an ``inputs`` named argument for input data. All other arguments are treated as
      weights within the QNode.
    - All arguments must accept an array or tensor, e.g., arguments should not use nested lists
      of different lengths.
    - All arguments, except ``inputs``, must have no default value.
    - The ``inputs`` argument is permitted to have a default value provided the gradient with
      respect to ``inputs`` is not required.
    - There cannot be a variable number of positional or keyword arguments, e.g., no ``*args`` or
      ``**kwargs`` present in the signature.

    The QNode weights are initialized within the :class:`~.KerasLayer`. Upon instantiation,
    a ``weight_shapes`` dictionary must be passed which describes the shapes of all
    weights in the QNode. The optional ``weight_specs`` argument allows for a more fine-grained
    specification of the QNode weights, such as the method of initialization and any
    regularization or constraints. If not specified, weights will be added using the Keras
    default initialization and without any regularization or constraints.

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Keras layer
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding sizes
        output_dim (int): the dimension of data output from the QNode
        weight_specs (dict[str, dict]): An optional dictionary for users to provide additional
            specifications for weights used in the QNode, such as the method of parameter
            initialization. This specification is provided as a dictionary with keys given by the
            arguments of the `add_weight()
            <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight>`__
            method and values being the corresponding specification.
        **kwargs: additional keyword arguments passed to the `Layer
            <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__ base class

    **Example**

    The following shows how a circuit composed of templates from the :doc:`/code/qml_templates`
    module can be converted into a Keras layer.

    .. code-block:: python

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
            qml.templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        weight_shapes = {"weights": (3, n_qubits, 3)}
        weight_specs = {"weights": {"initializer": "random_uniform"}}

        keras_layer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=2,
                                         weight_specs=weight_specs)

    The resulting ``keras_layer`` can be combined with other layers using the `Sequential
    <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`__ or
    `Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ Keras APIs.
    """

    def __init__(
        self,
        qnode: QNode,
        weight_shapes: dict,
        output_dim: int,
        weight_specs: Optional[dict] = None,
        **kwargs
    ):
        if not CORRECT_TF_VERSION:
            raise ImportError("KerasLayer requires TensorFlow version 2 and above")

        self.sig = qnode.func.sig

        if INPUT_ARG not in self.sig:
            raise TypeError(
                "QNode must include an argument with name {} for inputting data".format(INPUT_ARG)
            )

        if INPUT_ARG in set(weight_shapes.keys()):
            raise ValueError(
                "{} argument should not have its dimension specified in "
                "weight_shapes".format(INPUT_ARG)
            )

        if set(weight_shapes.keys()) | {INPUT_ARG} != set(self.sig.keys()):
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
        self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        defaults = {
            name for name, sig in self.sig.items() if sig.par.default != inspect.Parameter.empty
        }
        self.input_is_default = INPUT_ARG in defaults
        if defaults - {INPUT_ARG} != set():
            raise TypeError("Only the argument {} is permitted to have a default".format(INPUT_ARG))

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
        for x in inputs:
            qnode = self.qnode
            for arg in self.sig:
                if arg is not INPUT_ARG:
                    w = self.qnode_weights[arg]
                    qnode = functools.partial(qnode, w)
                else:
                    if self.input_is_default:
                        qnode = functools.partial(qnode, **{INPUT_ARG: x})
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
        detail = "<Quantum Keras layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__
