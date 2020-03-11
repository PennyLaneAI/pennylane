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
Tests for the pennylane.qnn module.
"""
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal

import pennylane as qml
from pennylane.qnn import KerasLayer


@pytest.fixture()
def get_circuit(n_qubits, output_dim):
    """Fixture for getting a sample quantum circuit with a controllable qubit number and output
    dimension. Returns both the circuit and the shape of the weights"""

    dev = qml.device("default.qubit", wires=n_qubits)
    weight_shapes = {"w1": (3, n_qubits, 3), "w2": (2, n_qubits, 3), "w3": (1,), "w4": 1,
                     "w5": [3]}

    @qml.qnode(dev, interface='tf')
    def circuit(w1, w2, w3, w4, w5, x=None):
        """A circuit that embeds data using the AngleEmbedding and then performs a variety of
        operations. The output is a PauliZ measurement on the first output_dim qubits."""
        qml.templates.AngleEmbedding(x, wires=list(range(n_qubits)))
        # qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
        # qml.templates.StronglyEntanglingLayers(w2, wires=list(range(n_qubits)))
        # qml.RX(w3, wires=0)
        qml.RX(w4, wires=0)
        qml.Rot(*w5, wires=0)
        return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

    return circuit, weight_shapes


def indices(n_max):
    a, b = np.tril_indices(n_max)
    return zip(*[a + 1, b + 1])


@pytest.mark.usefixtures("get_circuit")
class TestKerasLayer:
    """Unit tests for the pennylane.qnn.KerasLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_too_many_defaults(self, get_circuit, output_dim):
        """Test if a TypeError is raised when instantiated with a QNode that has two defaults"""
        c, w = get_circuit
        c.func.sig['x2'] = c.func.sig['x']
        with pytest.raises(TypeError, match="Conversion to a Keras layer requires"):
            KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_no_defaults(self, get_circuit, output_dim):
        """Test if a TypeError is raised when instantiated with a QNode that has no defaults"""
        c, w = get_circuit
        del c.func.sig['x']
        with pytest.raises(TypeError, match="Conversion to a Keras layer requires"):
            KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_input_in_weight_shapes(self, get_circuit, n_qubits, output_dim):
        """Test if a ValueError is raised when instantiated with a weight_shapes dictionary that
        contains the shape of the input"""
        c, w = get_circuit
        w['x'] = n_qubits
        with pytest.raises(ValueError, match="Input argument dimension should not"):
            KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_weight_shape_unspecified(self, get_circuit, output_dim):
        """Test if a ValueError is raised when instantiated with a weight missing from the
        weight_shapes dictionary"""
        c, w = get_circuit
        del w['w1']
        with pytest.raises(ValueError, match="Must specify a shape for every non-input parameter"):
            KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_var_pos(self, get_circuit, monkeypatch, output_dim):
        """Test if a TypeError is raised when instantiated with a variable number of positional
        arguments"""
        c, w = get_circuit

        class FuncPatch:
            """Patch for variable number of keyword arguments"""
            sig = c.func.sig
            var_pos = True
            var_keyword = False

        with monkeypatch.context() as m:
            m.setattr(c, 'func', FuncPatch)

            with pytest.raises(TypeError, match="Cannot have a variable number of positional"):
                KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_var_keyword(self, get_circuit, monkeypatch, output_dim):
        """Test if a TypeError is raised when instantiated with a variable number of keyword
        arguments"""
        c, w = get_circuit

        class FuncPatch:
            """Patch for variable number of keyword arguments"""
            sig = c.func.sig
            var_pos = False
            var_keyword = True

        with monkeypatch.context() as m:
            m.setattr(c, 'func', FuncPatch)

            with pytest.raises(TypeError, match="Cannot have a variable number of keyword"):
                KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    @pytest.mark.parametrize("input_dim", zip(*[[None, [1], (1,), 1], [None, 1, 1, 1]]))
    def test_input_dim(self, get_circuit, input_dim, output_dim):
        """Test if the input_dim is correctly processed, i.e., that an iterable is mapped to
        its first element while an int or None is left unchanged."""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim, input_dim[0])
        assert layer.input_dim == input_dim[1]

    @pytest.mark.parametrize("n_qubits", [1])
    @pytest.mark.parametrize("output_dim", zip(*[[[1], (1,), 1], [1, 1, 1]]))
    def test_output_dim(self, get_circuit, output_dim):
        """Test if the output_dim is correctly processed, i.e., that an iterable is mapped to
        its first element while an int is left unchanged."""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim[0])
        assert layer.output_dim == output_dim[1]

    @pytest.mark.parametrize("n_qubits, output_dim", indices(2))
    def test_weight_shapes(self, get_circuit, output_dim, n_qubits):
        """Test if the weight_shapes input argument is correctly processed to be a dictionary
        with values that are tuples."""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)
        assert layer.weight_shapes == {'w1': (3, n_qubits, 3), 'w2': (2, n_qubits, 3), 'w3': (1,), 'w4': (1,),
                                       'w5': (3,)}

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    @pytest.mark.parametrize("weight_specs", zip(*[[None, {"w1": {}}], [{}, {"w1": {}}]]))
    def test_weight_specs_initialize(self, get_circuit, output_dim, weight_specs):
        """Test if the weight_specs input argument is correctly processed, so that it
        initializes to an empty dictionary if not specified but is left unchanged if already a
        dictionary"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim, weight_specs=weight_specs[0])
        assert layer.weight_specs == weight_specs[1]

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_build_wrong_input_shape(self, get_circuit, output_dim):
        """Test if the build() method raises a ValueError if the user has specified an input
        dimension but build() is called with a different dimension. Note that the input_shape
        passed to build is a tuple to include a batch dimension"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim, input_dim=4)
        with pytest.raises(ValueError, match="QNode can only accept inputs of size"):
            layer.build(input_shape=(10, 3))

    @pytest.mark.parametrize("n_qubits, output_dim", indices(2))
    def test_qnode_weights(self, get_circuit, n_qubits, output_dim):
        """Test if the build() method correctly initializes the weights in the qnode_weights
        dictionary, i.e., that each value of the dictionary has correct shape and name."""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)
        layer.build(input_shape=(10, n_qubits))

        for weight, shape in layer.weight_shapes.items():
            assert layer.qnode_weights[weight].shape == shape
            assert layer.qnode_weights[weight].name[:-2] == weight

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_qnode_weights_with_spec(self, get_circuit, monkeypatch, output_dim, n_qubits):
        """Test if the build() method correctly passes on user specified weight_specs to the
        inherited add_weight() method. This is done by monkeypatching add_weight() so that it
        simply returns its input keyword arguments. The qnode_weights dictionary should then have
        values that are the input keyword arguments, and we check that the specified weight_specs
        keywords are there."""

        def add_weight_dummy(*args, **kwargs):
            return kwargs

        weight_specs = {
            "w1": {"initializer": "random_uniform", "trainable": False},
            "w2": {"initializer": RandomNormal(mean=0, stddev=0.5)},
            "w3": {},
            "w4": {},
            "w5": {},
        }

        with monkeypatch.context() as m:
            m.setattr(Layer, 'add_weight', add_weight_dummy)
            c, w = get_circuit
            layer = KerasLayer(c, w, output_dim, weight_specs=weight_specs)
            layer.build(input_shape=(10, n_qubits))

            for weight in layer.weight_shapes:
                assert all(item in layer.qnode_weights[weight].items() for item in weight_specs[
                    weight].items())

    @pytest.mark.parametrize("n_qubits, output_dim", indices(3))
    @pytest.mark.parametrize("input_shape", [(10, 4), (8, 3)])
    def test_compute_output_shape(self, get_circuit, output_dim, input_shape):
        """Test if the compute_output_shape() method performs correctly, i.e., that it replaces
        the last element in the input_shape tuple with the specified output_dim and that the
        output shape is of type tf.TensorShape"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)

        assert layer.compute_output_shape(input_shape) == (input_shape[0], output_dim)
        assert isinstance(layer.compute_output_shape(input_shape), tf.TensorShape)

    @pytest.mark.parametrize("n_qubits, output_dim", indices(4))
    @pytest.mark.parametrize("batch_size", [5, 10, 15])
    def test_call(self, get_circuit, output_dim, batch_size, n_qubits):
        """Test if the call() method performs correctly, i.e., that it outputs with shape
        (batch_size, output_dim) with results that agree with directly calling the QNode"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w[0] if w.shape == (1,) else w for w in layer.qnode_weights.values()]

        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(*weights, x=x[0]))

    @pytest.mark.parametrize("n_qubits, output_dim", indices(1))
    def test_str_repr(self, get_circuit, output_dim):
        """Tests the __str__ and __repr__ representations"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)

        assert layer.__str__() == "<Quantum Keras layer: func=circuit>"
        assert layer.__repr__() == "<Quantum Keras layer: func=circuit>"
