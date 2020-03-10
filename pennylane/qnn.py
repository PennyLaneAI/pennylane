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
import functools
import inspect
from typing import Optional

import tensorflow as tf

import pennylane as qml

if int(tf.__version__.split(".")[0]) < 2:
    raise ImportError("TensorFlow version 2 or above is required for this module")
else:
    from tensorflow.keras.layers import Layer


class KerasLayer(Layer):
    def __init__(
        self,
        qnode: qml.QNode,
        weight_shapes: dict,
        output_dim: int,
        input_dim: Optional[int] = None,
        weight_specs: Optional[dict] = None,
        **kwargs
    ):

        self.sig = qnode.func.sig
        defaults = [
            name for name, sig in self.sig.items() if sig.par.default != inspect.Parameter.empty
        ]
        if len(defaults) != 1:
            raise TypeError("Conversion to a Keras layer requires a QNode with a single "
                            "default argument")
        self.input_arg = defaults[0]

        if self.input_arg in {weight_shapes.keys()}:
            raise ValueError("Input argument dimension should not be specified in weight_shapes")
        if {weight_shapes.keys()} | {self.input_arg} != {self.sig.keys()}:
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")
        if qnode.func.var_pos:
            raise TypeError("Cannot have a variable number of positional arguments")
        if qnode.func.var_keyword:
            raise TypeError("Cannot have a variable number of keyword arguments")
        if len(weight_shapes.keys()) != len({weight_shapes.keys()}):
            raise ValueError("A shape is specified multiple times in weight_shapes")

        self.qnode = qnode
        self.input_dim = input_dim if isinstance(input_dim, (int, type(None))) else input_dim[0]
        self.weight_shapes = {
            weight: ((size,) if isinstance(size, int) else tuple(size))
            for weight, size in weight_shapes.items()
        }
        self.output_dim = output_dim if isinstance(output_dim, int) else output_dim[0]

        if weight_specs:
            self.weight_specs = weight_specs
        else:
            self.weight_specs = {}
        self.qnode_weights = {}

        super(KerasLayer, self).__init__(dynamic=True, **kwargs)

    def build(self, input_shape):
        if self.input_dim and input_shape[-1] != self.input_dim:
            raise ValueError("QNode can only accept inputs of size {}".format(self.input_dim))

        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(name=weight, shape=size, **spec)

        super(KerasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        qnode = self.qnode
        for arg in self.sig:
            if arg is not self.input_arg:
                w = self.qnode_weights[arg]
                qnode = functools.partial(qnode, w)

        outputs = tf.stack([qnode(**{self.input_arg: x}) for x in inputs])
        input_shape = tf.shape(inputs)

        return tf.reshape(outputs, self.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.output_dim])

    def __str__(self):
        detail = "<Quantum Keras layer: device='{}', func={}, wires={}, interface=\"tf\">"
        return detail.format(
            self.qnode.device.short_name,
            self.qnode.func.__name__,
            self.qnode.num_wires,
        )

    def __repr__(self):
        return self.__str__()
