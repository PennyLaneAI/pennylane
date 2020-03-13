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
"""Test"""
import functools
import inspect
from collections.abc import Iterable
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from pennylane.interfaces.tf import to_tf
from pennylane.qnodes import QNode

INPUT_ARG = "inputs"


class KerasLayer(Layer):
    def __init__(
        self,
        qnode: QNode,
        weight_shapes: dict,
        output_dim: int,
        weight_specs: Optional[dict] = None,
        **kwargs
    ):
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
            weight: (tuple(size) if isinstance(size, Iterable) else (size,))
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

        super(KerasLayer, self).__init__(dynamic=True, **kwargs)

    def build(self, input_shape):
        """Initializes the :class:`~.KerasLayer` weights.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data
        """
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(name=weight, shape=size, **spec)

        super(KerasLayer, self).build(input_shape)

    def call(self, inputs):
        """Evaluates the :class:`~.KerasLayer` on input data using the
        :attr:`~.KerasLayer.qnode_weights`.

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
                    if w.shape == (1,):
                        qnode = functools.partial(qnode, w[0])
                    else:
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
        :class:`~.KerasLayer`.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data

        Returns:
            tf.TensorShape: shape of output data
        """
        return tf.TensorShape([input_shape[0], self.output_dim])

    def __str__(self):
        detail = "<Quantum Keras layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    def __repr__(self):
        return self.__str__()
