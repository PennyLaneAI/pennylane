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
"""
This module contains the :func:`to_keras` function to convert quantum nodes into Keras layers.
"""
import tensorflow as tf
import pennylane as qml

major, minor, patch = tf.__version__.split(".")

if (int(major) == 1 and int(minor) < 4) or int(major) == 0:
    raise ImportError("Must use TensorFlow v1.4.0 or above")
else:
    from tensorflow import keras


class QuantumLayers(keras.layers.Layer):
    """TODO - This is a layer for a specific architecture given here:
    https://github.com/XanaduAI/qml/issues/28#issuecomment-574889614.
    """
    def __init__(
        self,
        units,
        device="default.qubit",
        n_layers=1,
        rotations_initializer="random_uniform",
        rotations_regularizer=None,
        rotations_constraint=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        if "dynamic" in kwargs:
            del kwargs["dynamic"]
        super(QuantumLayers, self).__init__(dynamic=True, **kwargs)

        self.units = units
        self.device = device
        self.n_layers = n_layers
        self.rotations_initializer = keras.initializers.get(rotations_initializer)
        self.rotations_regularizer = keras.regularizers.get(rotations_regularizer)
        self.rotations_constraint = keras.constraints.get(rotations_constraint)

        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: units})
        self.supports_masking = False

        def circuit(inputs, parameters):
            qml.templates.embeddings.AngleEmbedding(inputs, wires=list(range(self.units)))
            qml.templates.layers.StronglyEntanglingLayers(parameters, wires=list(range(self.units)))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.units)]

        self.dev = qml.device(device, wires=units)
        self.layer = qml.QNode(circuit, self.dev, interface="tf")

    def apply_layer(self, *args):
        return tf.keras.backend.cast_to_floatx(self.layer(*args))

    def build(self, input_shape):
        # assert len(input_shape) == 2
        input_dim = input_shape[-1]
        assert input_dim == self.units

        self.rotations = self.add_weight(
            shape=(self.n_layers, input_dim, 3),
            initializer=self.rotations_initializer,
            name="rotations",
            regularizer=self.rotations_regularizer,
            constraint=self.rotations_constraint,
        )
        self.built = True

    def call(self, inputs):
        return tf.stack([self.apply_layer(i, self.rotations) for i in inputs])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "units": self.units,
            "device": self.device,
            "n_layers": self.n_layers,
            "rotations_initializer": keras.initializers.serialize(self.rotations_initializer),
            "rotations_regularizer": keras.regularizers.serialize(self.rotations_regularizer),
            "rotations_constraint": keras.constraints.serialize(self.rotations_constraint),
        }
        base_config = super(QuantumLayers, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def to_keras(qnode: qml.QNode):
    """TODO
    """
    return qnode
