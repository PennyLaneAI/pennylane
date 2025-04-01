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
Tests for the pennylane.qnn.keras module.
"""
from collections import defaultdict

import numpy as np
import pytest
from packaging.version import Version

import pennylane as qml

KerasLayer = qml.qnn.keras.KerasLayer

tf = pytest.importorskip("tensorflow", minversion="2")
# Check for Keras 3
try:
    # Check TensorFlow version - 2.16+ uses Keras 3 by default
    USING_KERAS3 = Version(tf.__version__) >= Version("2.16.0")

    # Alternative check using keras.version if available
    try:
        import keras

        if hasattr(keras, "version"):
            USING_KERAS3 = Version(keras.version()) >= Version("3.0.0")
    except (ImportError, AttributeError):
        pass

except ImportError:
    USING_KERAS3 = False

# Skip marker for Keras 3
KERAS3_XFAIL_INFO = "This test requires Keras 2. Skipping for Keras 3 (TF >= 2.16) until proper support is implemented."
# pylint: disable=unnecessary-dunder-call


@pytest.fixture
def model(get_circuit, n_qubits, output_dim):
    """Fixture for creating a hybrid Keras model. The model is composed of KerasLayers sandwiched
    between Dense layers."""
    c, w = get_circuit
    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        layer1 = KerasLayer(c, w, output_dim)
        layer2 = KerasLayer(c, w, output_dim)

    m = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(n_qubits, input_shape=(n_qubits,)),
            layer1,
            tf.keras.layers.Dense(n_qubits),
            layer2,
            tf.keras.layers.Dense(output_dim),
        ]
    )

    return m


@pytest.fixture
def model_dm(get_circuit_dm, n_qubits, output_dim):
    """The Keras NN model."""
    c, w = get_circuit_dm
    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        layer1 = KerasLayer(c, w, output_dim)
        layer2 = KerasLayer(c, w, output_dim)

    m = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(n_qubits, input_shape=(n_qubits,)),
            layer1,
            # Adding a lambda layer to take only the real values from density matrix
            tf.keras.layers.Lambda(lambda x: tf.abs(x)),  # pylint: disable=unnecessary-lambda
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_qubits),
            layer2,
            # Adding a lambda layer to take only the real values from density matrix
            tf.keras.layers.Lambda(lambda x: tf.abs(x)),  # pylint: disable=unnecessary-lambda
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_dim[0] * output_dim[1]),
        ]
    )

    return m


@pytest.fixture(autouse=True)
def reset_keraslayer_input_arg():
    # Reset for every test so they don't interfere
    KerasLayer.set_input_argument("inputs")
    yield
    KerasLayer.set_input_argument("inputs")


def indices_up_to(n_max):
    """Returns an iterator over the number of qubits and output dimension, up to value n_max.
    The output dimension never exceeds the number of qubits."""

    a, b = np.tril_indices(n_max)
    return zip(*[a + 1, b + 1])


def indices_up_to_dm(n_max):
    """Returns an iterator over the number of qubits and output dimension, up to value n_max.
    The output dimension values never exceeds 2 ** (n_max). This is to test for density_matrix
    qnodes."""

    # If the output_dim is to be used as a tuple. First element is for n_qubits and
    # the second is for output_dim. For example, for n_max = 3 it will return,
    # [(1, (2, 2)), (2, (2, 2)), (2, (4, 4)), (3, (2, 2)), (3, (4, 4)), (3, (8, 8))]

    a, b = np.tril_indices(n_max)
    return zip(*[a + 1], zip(*[2 ** (b + 1), 2 ** (b + 1)]))


@pytest.mark.tf
def test_no_attribute():
    """Test that the qnn module raises an AttributeError if accessing an unavailable attribute"""
    with pytest.raises(AttributeError, match="module 'pennylane.qnn' has no attribute 'random'"):
        qml.qnn.random  # pylint: disable=pointless-statement


@pytest.mark.tf
@pytest.mark.parametrize("interface", ["tf"])  # required for the get_circuit fixture
@pytest.mark.usefixtures("get_circuit")
@pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
def test_bad_tf_version(get_circuit, output_dim, monkeypatch):  # pylint: disable=no-self-use
    """Test if an ImportError is raised when instantiated with an incorrect version of
    TensorFlow"""
    c, w = get_circuit
    with monkeypatch.context() as m:
        m.setattr(qml.qnn.keras, "CORRECT_TF_VERSION", False)

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            with pytest.raises(ImportError, match="KerasLayer requires TensorFlow version 2"):
                KerasLayer(c, w, output_dim)


# pylint: disable=too-many-public-methods
@pytest.mark.tf
@pytest.mark.parametrize("interface", ["tf"])  # required for the get_circuit fixture
@pytest.mark.usefixtures("get_circuit")
class TestKerasLayer:
    """Unit tests for the pennylane.qnn.keras.KerasLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_input(self):  # pylint: disable=no-self-use
        """Test if a TypeError is raised when instantiated with a QNode that does not have an
        argument with name equal to the input_arg class attribute of KerasLayer"""
        dev = qml.device("default.qubit", wires=1)
        weight_shapes = {"w1": (3, 3), "w2": 1}

        @qml.qnode(dev, interface="tf")
        def circuit(w1, w2):  # pylint: disable=unused-argument
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            with pytest.raises(TypeError, match="QNode must include an argument with name"):
                KerasLayer(circuit, weight_shapes, output_dim=1)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_input_in_weight_shapes(
        self, get_circuit, n_qubits, output_dim
    ):  # pylint: disable=no-self-use
        """Test if a ValueError is raised when instantiated with a weight_shapes dictionary that
        contains the shape of the input argument given by the input_arg class attribute of
        KerasLayer"""
        c, w = get_circuit
        w[qml.qnn.keras.KerasLayer._input_arg] = n_qubits  # pylint: disable=protected-access
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            with pytest.raises(
                ValueError,
                match=f"{qml.qnn.keras.KerasLayer._input_arg} argument should not have its dimension",  # pylint: disable=protected-access
            ):
                KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_weight_shape_unspecified(self, get_circuit, output_dim):  # pylint: disable=no-self-use
        """Test if a ValueError is raised when instantiated with a weight missing from the
        weight_shapes dictionary"""
        c, w = get_circuit
        del w["w1"]
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            with pytest.raises(
                ValueError, match="Must specify a shape for every non-input parameter"
            ):
                KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_pos(self):  # pylint: disable=no-self-use
        """Test if a TypeError is raised when instantiated with a variable number of positional
        arguments"""
        dev = qml.device("default.qubit", wires=1)
        weight_shapes = {"w1": (3, 3), "w2": 1}

        @qml.qnode(dev, interface="tf")
        def circuit(inputs, w1, w2, *args):  # pylint: disable=unused-argument
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            with pytest.raises(TypeError, match="Cannot have a variable number of positional"):
                KerasLayer(circuit, weight_shapes, output_dim=1)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_keyword(self):  # pylint: disable=no-self-use
        """Test that variable number of keyword arguments works"""
        n_qubits = 2
        output_dim = 2

        dev = qml.device("default.qubit", wires=n_qubits)
        w = {
            "w1": (3, n_qubits, 3),
            "w2": (1,),
            "w3": 1,
            "w4": [3],
            "w5": (2, n_qubits, 3),
            "w6": 3,
            "w7": 0,
        }

        @qml.qnode(dev, interface="tf")
        def c(inputs, **kwargs):
            """A circuit that embeds data using the AngleEmbedding and then performs a variety of
            operations. The output is a PauliZ measurement on the first output_dim qubits. One set of
            parameters, w5, are specified as non-trainable."""
            qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
            qml.templates.StronglyEntanglingLayers(kwargs["w1"], wires=list(range(n_qubits)))
            qml.RX(kwargs["w2"][0], wires=0 % n_qubits)
            qml.RX(kwargs["w3"], wires=1 % n_qubits)
            qml.Rot(*kwargs["w4"], wires=2 % n_qubits)
            qml.templates.StronglyEntanglingLayers(kwargs["w5"], wires=list(range(n_qubits)))
            qml.Rot(*kwargs["w6"], wires=3 % n_qubits)
            qml.RX(kwargs["w7"], wires=4 % n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim=output_dim)
        x = tf.ones((2, n_qubits))

        layer_out = layer(x)

        circ_weights = layer.qnode_weights.copy()
        circ_weights["w4"] = tf.convert_to_tensor(circ_weights["w4"])  # To allow for slicing
        circ_weights["w6"] = tf.convert_to_tensor(circ_weights["w6"])
        circuit_out = tf.stack(c(x[0], **circ_weights))

        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits", [1])
    @pytest.mark.parametrize("output_dim", zip(*[[[1], (1,), 1], [1, 1, 1]]))
    def test_output_dim(self, get_circuit, output_dim):  # pylint: disable=no-self-use
        """Test if the output_dim is correctly processed, i.e., that an iterable is mapped to
        its first element while an int is left unchanged."""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim[0])
        assert layer.output_dim == output_dim[1]

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_weight_shapes(self, get_circuit, output_dim, n_qubits):  # pylint: disable=no-self-use
        """Test if the weight_shapes input argument is correctly processed to be a dictionary
        with values that are tuples."""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)
        assert layer.weight_shapes == {
            "w1": (3, n_qubits, 3),
            "w2": (1,),
            "w3": (),
            "w4": (3,),
            "w5": (2, n_qubits, 3),
            "w6": (3,),
            "w7": (),
        }

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_non_input_defaults(self):  # pylint: disable=no-self-use
        """Test that everything works when default arguments that are not the input argument are
        present in the QNode"""

        n_qubits = 2
        output_dim = 2

        dev = qml.device("default.qubit", wires=n_qubits)
        w = {
            "w1": (3, n_qubits, 3),
            "w2": (1,),
            "w3": 1,
            "w4": [3],
            "w5": (2, n_qubits, 3),
            "w6": 3,
            "w7": 0,
        }

        @qml.qnode(dev, interface="tf")
        def c(inputs, w1, w2, w4, w5, w6, w7, w3=0.5):  # pylint: disable=too-many-arguments
            """A circuit that embeds data using the AngleEmbedding and then performs a variety of
            operations. The output is a PauliZ measurement on the first output_dim qubits. One set of
            parameters, w5, are specified as non-trainable."""
            qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
            qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
            qml.RX(w2[0], wires=0 % n_qubits)
            qml.RX(w3, wires=1 % n_qubits)
            qml.Rot(*w4, wires=2 % n_qubits)
            qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
            qml.Rot(*w6, wires=3 % n_qubits)
            qml.RX(w7, wires=4 % n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim=output_dim)
        x = tf.ones((2, n_qubits))

        layer_out = layer(x)
        circ_weights = layer.qnode_weights.copy()
        circ_weights["w4"] = tf.convert_to_tensor(circ_weights["w4"])  # To allow for slicing
        circ_weights["w6"] = tf.convert_to_tensor(circ_weights["w6"])
        circuit_out = c(x[0], **circ_weights)

        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights(self, get_circuit, n_qubits, output_dim):  # pylint: disable=no-self-use
        """Test if the build() method correctly initializes the weights in the qnode_weights
        dictionary, i.e., that each value of the dictionary has correct shape and name."""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)
        layer.build(input_shape=(10, n_qubits))

        for weight, shape in layer.weight_shapes.items():
            assert layer.qnode_weights[weight].shape == shape
            assert layer.qnode_weights[weight].name[:-2] == weight

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_qnode_weights_with_spec(
        self, get_circuit, monkeypatch, output_dim, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the build() method correctly passes on user specified weight_specs to the
        inherited add_weight() method. This is done by monkeypatching add_weight() so that it
        simply returns its input keyword arguments. The qnode_weights dictionary should then have
        values that are the input keyword arguments, and we check that the specified weight_specs
        keywords are there."""

        add_weight = KerasLayer.add_weight

        specs = {}

        def add_weight_dummy(self, **kwargs):
            """Dummy function for mocking out the add_weight method to store the kwargs in a dict"""
            specs[kwargs["name"]] = kwargs
            return add_weight(self, **kwargs)

        weight_specs = {
            "w1": {"initializer": "random_uniform", "trainable": False},
            "w2": {"initializer": tf.keras.initializers.RandomNormal(mean=0, stddev=0.5)},
            "w3": {},
            "w4": {},
            "w5": {},
            "w6": {},
            "w7": {},
        }

        with monkeypatch.context() as m:
            m.setattr(tf.keras.layers.Layer, "add_weight", add_weight_dummy)
            c, w = get_circuit
            with pytest.warns(
                qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
            ):
                layer = KerasLayer(c, w, output_dim, weight_specs=weight_specs)
            layer.build(input_shape=(10, n_qubits))

            for weight in layer.weight_shapes:
                assert all(item in specs[weight].items() for item in weight_specs[weight].items())

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(3))
    @pytest.mark.parametrize("input_shape", [(10, 4), (8, 3)])
    def test_compute_output_shape(
        self, get_circuit, output_dim, input_shape
    ):  # pylint: disable=no-self-use
        """Test if the compute_output_shape() method performs correctly, i.e., that it replaces
        the last element in the input_shape tuple with the specified output_dim and that the
        output shape is of type tf.TensorShape"""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)

        assert layer.compute_output_shape(input_shape) == (input_shape[0], output_dim)
        assert isinstance(layer.compute_output_shape(input_shape), tf.TensorShape)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2])
    def test_call(
        self, get_circuit, output_dim, batch_size, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the call() method performs correctly, i.e., that it outputs with shape
        (batch_size, output_dim) with results that agree with directly calling the QNode"""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w.numpy() for w in layer.qnode_weights.values()]
        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(x[0], *weights), atol=1e-7)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    @pytest.mark.parametrize("batch_size", [2])
    def test_call_shuffled_args(
        self, get_circuit, output_dim, batch_size, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the call() method performs correctly when the inputs argument is not the first
        positional argument, i.e., that it outputs with shape (batch_size, output_dim) with
        results that agree with directly calling the QNode"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="tf")
        def c_shuffled(w1, inputs, w2, w3, w4, w5, w6, w7):  # pylint: disable=too-many-arguments
            """Version of the circuit with a shuffled signature"""
            qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
            qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
            qml.RX(w2[0], wires=0)
            qml.RX(w3, wires=0)
            qml.Rot(*w4, wires=0)
            qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
            qml.Rot(*w6, wires=0)
            qml.RX(w7, wires=0)
            return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c_shuffled, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w.numpy() for w in layer.qnode_weights.values()]

        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(x[0], *weights), atol=1e-7)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    @pytest.mark.parametrize("batch_size", [2])
    def test_call_default_input(
        self, get_circuit, output_dim, batch_size, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the call() method performs correctly when the inputs argument is a default
        argument, i.e., that it outputs with shape (batch_size, output_dim) with results that
        agree with directly calling the QNode"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="tf")
        def c_default(
            w1, w2, w3, w4, w5, w6, w7, inputs=None
        ):  # pylint: disable=too-many-arguments
            """Version of the circuit with inputs as a default argument"""
            qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
            qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
            qml.RX(w2[0], wires=0)
            qml.RX(w3, wires=0)
            qml.Rot(*w4, wires=0)
            qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
            qml.Rot(*w6, wires=0)
            qml.RX(w7, wires=0)
            return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c_default, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w.numpy() for w in layer.qnode_weights.values()]

        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(x[0], *weights), atol=1e-7)

    @pytest.mark.slow
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2, 4, 6])
    @pytest.mark.parametrize("middle_dim", [2, 5, 8])
    def test_call_broadcast(
        self, get_circuit, output_dim, middle_dim, batch_size, n_qubits
    ):  # pylint: disable=no-self-use,too-many-arguments
        """Test if the call() method performs correctly when the inputs argument has an arbitrary shape (that can
        correctly be broadcast over), i.e., for input of shape (batch_size, dn, ... , d0) it outputs with shape
        (batch_size, dn, ... , d1, output_dim). Also tests if gradients are still backpropagated correctly.
        """
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)
        x = tf.ones((batch_size, middle_dim, n_qubits))

        with tf.GradientTape() as tape:
            layer_out = layer(x)

        g_layer = tape.gradient(layer_out, layer.trainable_variables)

        # test gradients are at least calculated
        assert g_layer is not None
        assert layer_out.shape == (batch_size, middle_dim, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_str_repr(self, get_circuit, output_dim):  # pylint: disable=no-self-use
        """Test the __str__ and __repr__ representations"""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)

        assert layer.__str__() == "<Quantum Keras Layer: func=circuit>"
        assert layer.__repr__() == "<Quantum Keras Layer: func=circuit>"

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_gradients(self, get_circuit, output_dim, n_qubits):  # pylint: disable=no-self-use
        """Test if the gradients of the KerasLayer are equal to the gradients of the circuit when
        taken with respect to the trainable variables"""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)
        x = tf.ones((1, n_qubits))

        with tf.GradientTape() as tape:
            out_layer = layer(x)

        g_layer = tape.gradient(out_layer, layer.trainable_variables)

        circuit_weights = layer.trainable_variables.copy()
        circuit_weights[3] = tf.convert_to_tensor(circuit_weights[3])  # To allow for slicing
        circuit_weights[5] = tf.convert_to_tensor(circuit_weights[5])

        with tf.GradientTape() as tape:
            out_circuit = c(x[0], *circuit_weights)

        g_circuit = tape.gradient(out_circuit, layer.trainable_variables)

        for i in range(len(out_layer)):
            assert np.allclose(g_layer[i], g_circuit[i])

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_backprop_gradients(self, mocker):  # pylint: disable=no-self-use
        """Test if KerasLayer is compatible with the backprop diff method."""

        dev = qml.device("default.qubit")

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def f(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(2))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        weight_shapes = {"weights": (3, 2, 3)}

        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            qlayer = qml.qnn.KerasLayer(f, weight_shapes, output_dim=2)

        inputs = tf.ones((4, 2))

        with tf.GradientTape() as tape:
            out = tf.reduce_sum(qlayer(inputs))

        spy = mocker.spy(qml.gradients, "param_shift")

        grad = tape.gradient(out, qlayer.trainable_weights)
        assert grad is not None
        spy.assert_not_called()

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_compute_output_shape_2(self, get_circuit, output_dim):  # pylint: disable=no-self-use
        """Test that the compute_output_shape method returns the expected shape"""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)

        inputs = tf.keras.Input(shape=(2,))
        inputs_shape = inputs.shape

        output_shape = layer.compute_output_shape(inputs_shape)
        assert output_shape.as_list() == [None, 1]

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(3))
    def test_construct(self, get_circuit, n_qubits, output_dim):
        """Test that the construct method builds the correct tape with correct differentiability"""
        c, w = get_circuit
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            layer = KerasLayer(c, w, output_dim)

        x = tf.ones((1, n_qubits))

        tape = qml.workflow.construct_tape(layer)(x)

        assert tape is not None
        assert (
            len(tape.get_parameters(trainable_only=False))
            == len(tape.get_parameters(trainable_only=True)) + 1
        )


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
def test_invalid_interface_error(interface):
    """Test an error gets raised if input QNode has the wrong interface"""
    dev = qml.device("default.qubit", wires=3)
    weight_shapes = {"w1": 1}

    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        with pytest.raises(ValueError, match="Invalid interface"):
            _ = KerasLayer(circuit, weight_shapes, output_dim=2)


@pytest.mark.tf
@pytest.mark.parametrize(
    "interface", ("auto", "tf", "tensorflow", "tensorflow-autograph", "tf-autograph")
)
def test_qnode_interface_not_mutated(interface):
    """Test that the input QNode's interface is not mutated by KerasLayer"""
    dev = qml.device("default.qubit", wires=3)
    weight_shapes = {"w1": 1}

    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        qlayer = KerasLayer(circuit, weight_shapes, output_dim=2)
    assert (
        qlayer.qnode.interface
        == circuit.interface
        == qml.math.get_canonical_interface_name(interface).value
    )


@pytest.mark.tf
@pytest.mark.parametrize("interface", ["tf"])
@pytest.mark.usefixtures("get_circuit", "model")
class TestKerasLayerIntegration:
    """Integration tests for the pennylane.qnn.keras.KerasLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2])
    def test_train_model(
        self, model, batch_size, n_qubits, output_dim
    ):  # pylint: disable=no-self-use,redefined-outer-name
        """Test if a model can train using the KerasLayer. The model is composed of two
        KerasLayers sandwiched between Dense neural network layers, and the dataset is simply
        input and output vectors of zeros."""

        x = np.zeros((batch_size, n_qubits))
        y = np.zeros((batch_size, output_dim))

        model.compile(optimizer="sgd", loss="mse")

        model.fit(x, y, batch_size=batch_size, verbose=0)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_model_gradients(
        self, model, output_dim, n_qubits
    ):  # pylint: disable=no-self-use,redefined-outer-name
        """Test if a gradient can be calculated with respect to all of the trainable variables in
        the model"""
        x = tf.zeros((2, n_qubits))
        y = tf.zeros((2, output_dim))

        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.keras.losses.mean_squared_error(out, y)

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g.dtype == tf.keras.backend.floatx() for g in gradients)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_save_model_weights(
        self, get_circuit, n_qubits, output_dim, tmpdir
    ):  # pylint: disable=no-self-use,redefined-outer-name
        """Test if the model can be successfully saved and reloaded using the get_weights()
        method"""
        clayer = tf.keras.layers.Dense(n_qubits, use_bias=False, input_shape=(n_qubits,))
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            qlayer = KerasLayer(*get_circuit, output_dim)
        model = tf.keras.models.Sequential([clayer, qlayer])
        weights = model.get_weights()

        file = str(tmpdir) + "/model"
        model.save_weights(file)

        new_clayer = tf.keras.layers.Dense(n_qubits, use_bias=False, input_shape=(n_qubits,))
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            new_qlayer = KerasLayer(*get_circuit, output_dim)
        new_model = tf.keras.models.Sequential([new_clayer, new_qlayer])
        new_weights = new_model.get_weights()

        assert len(weights) == len(new_weights)
        assert all(w.shape == nw.shape for w, nw in zip(weights, new_weights))

        # assert that the new model's weights are different
        assert all(tf.math.reduce_any(w != nw) for w, nw in zip(weights, new_weights))

        new_model.load_weights(file)
        new_weights = new_model.get_weights()

        assert len(weights) == len(new_weights)
        assert all(w.shape == nw.shape for w, nw in zip(weights, new_weights))

        # assert that the new model's weights are now the same
        assert all(tf.math.reduce_all(w == nw) for w, nw in zip(weights, new_weights))

        # assert that the results are the same
        x = tf.constant(np.arange(5 * n_qubits).reshape(5, n_qubits))
        res = model(x)
        new_res = new_model(x)
        assert tf.math.reduce_all(res == new_res)

    # the test is slow since TensorFlow needs to compile the execution graph
    # in order to save the model
    @pytest.mark.slow
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_save_whole_model(
        self, model, n_qubits, tmpdir
    ):  # pylint: disable=redefined-outer-name
        """Test if the entire model can be successfully saved and reloaded
        using the .save() method"""
        weights = model.get_weights()

        file = str(tmpdir) + "/model"
        model.save(file)

        new_model = tf.keras.models.load_model(file)
        new_weights = new_model.get_weights()

        assert len(weights) == len(new_weights)
        assert all(w.shape == nw.shape for w, nw in zip(weights, new_weights))
        assert all(tf.math.reduce_all(w == nw) for w, nw in zip(weights, new_weights))

        # assert that the results are the same
        x = tf.constant(np.arange(5 * n_qubits).reshape(5, n_qubits))
        res = model(x)
        new_res = new_model(x)
        assert tf.math.reduce_all(res == new_res)


@pytest.mark.tf
@pytest.mark.parametrize("interface", ["tf"])
@pytest.mark.usefixtures("get_circuit_dm", "model_dm")
class TestKerasLayerIntegrationDM:
    """Integration tests for the pennylane.qnn.keras.KerasLayer class for
    density_matrix() returning circuits."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(3))
    @pytest.mark.parametrize("batch_size", [2])
    def test_train_model_dm(
        self, model_dm, batch_size, n_qubits, output_dim
    ):  # pylint: disable=no-self-use,redefined-outer-name
        """Test if a model can train using the KerasLayer when QNode returns a density_matrix().
        The model is composed of two KerasLayers sandwiched between Dense neural network layers,
        and the dataset is simply input and output vectors of zeros."""

        x = np.zeros((batch_size, n_qubits))
        y = np.zeros((batch_size, output_dim[0] * output_dim[1]))

        model_dm.compile(optimizer="sgd", loss="mse")

        model_dm.fit(x, y, batch_size=batch_size, verbose=0)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(2))
    def test_model_gradients_dm(
        self, model_dm, output_dim, n_qubits
    ):  # pylint: disable=no-self-use,redefined-outer-name
        """Test if a gradient can be calculated with respect to all of the trainable variables in
        the model."""
        x = tf.zeros((2, n_qubits))
        y = tf.zeros((2, output_dim[0] * output_dim[1]))

        with tf.GradientTape() as tape:
            out = model_dm(x)
            loss = tf.keras.losses.mean_squared_error(out, y)

        gradients = tape.gradient(loss, model_dm.trainable_variables)
        assert all(g.dtype == tf.keras.backend.floatx() for g in gradients)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(2))
    def test_save_model_weights_dm(
        self, get_circuit_dm, n_qubits, output_dim, tmpdir
    ):  # pylint: disable=no-self-use,redefined-outer-name
        """Test if the model_dm can be successfully saved and reloaded using the get_weights()
        method"""
        clayer = tf.keras.layers.Dense(n_qubits, use_bias=False, input_shape=(n_qubits,))
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            qlayer = KerasLayer(*get_circuit_dm, output_dim)
        model = tf.keras.models.Sequential([clayer, qlayer])
        weights = model.get_weights()

        file = str(tmpdir) + "/model"
        model.save_weights(file)

        new_clayer = tf.keras.layers.Dense(n_qubits, use_bias=False, input_shape=(n_qubits,))
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
        ):
            new_qlayer = KerasLayer(*get_circuit_dm, output_dim)
        new_model = tf.keras.models.Sequential([new_clayer, new_qlayer])
        new_weights = new_model.get_weights()

        assert len(weights) == len(new_weights)
        assert all(w.shape == nw.shape for w, nw in zip(weights, new_weights))

        # assert that the new model's weights are different
        assert all(tf.math.reduce_any(w != nw) for w, nw in zip(weights, new_weights))

        new_model.load_weights(file)
        new_weights = new_model.get_weights()

        assert len(weights) == len(new_weights)
        assert all(w.shape == nw.shape for w, nw in zip(weights, new_weights))

        # assert that the new model's weights are now the same
        assert all(tf.math.reduce_all(w == nw) for w, nw in zip(weights, new_weights))

        # assert that the results are the same
        x = tf.constant(np.arange(5 * n_qubits).reshape(5, n_qubits))
        res = model(x)
        new_res = new_model(x)
        assert tf.math.reduce_all(res == new_res)

    # the test is slow since TensorFlow needs to compile the execution graph
    # in order to save the model
    @pytest.mark.slow
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(2))
    def test_save_whole_model_dm(
        self, model_dm, n_qubits, tmpdir
    ):  # pylint: disable=redefined-outer-name
        """Test if the entire model can be successfully saved and reloaded
        using the .save() method"""
        weights = model_dm.get_weights()

        file = str(tmpdir) + "/model"
        model_dm.save(file)

        new_model_dm = tf.keras.models.load_model(file)
        new_weights = new_model_dm.get_weights()

        for w, nw in zip(weights, new_weights):
            assert np.allclose(w, nw)

        # assert that the results are the same
        x = tf.constant(np.arange(5 * n_qubits).reshape(5, n_qubits))
        res = model_dm(x)
        new_res = new_model_dm(x)
        assert tf.math.reduce_all(res == new_res)


@pytest.mark.tf
def test_batch_input_single_measure(tol):
    """Test input batching in keras"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev, interface="tf", diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.probs(op=qml.PauliZ(1))

    KerasLayer.set_input_argument("x")
    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        layer = KerasLayer(circuit, weight_shapes={"weights": (2,)}, output_dim=(2,))
    layer.build((None, 2))

    x = tf.constant(np.random.uniform(0, 1, (10, 4)))
    res = layer(x)

    assert res.shape == (10, 2)

    for x_, r in zip(x, res):
        assert qml.math.allclose(r, circuit(x_, layer.qnode_weights["weights"]), atol=tol)


@pytest.mark.tf
def test_batch_input_multi_measure(tol):
    """Test input batching in keras for multiple measurements"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev, interface="tf", diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return [qml.expval(qml.PauliZ(1)), qml.probs(wires=range(2))]

    KerasLayer.set_input_argument("x")
    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        layer = KerasLayer(circuit, weight_shapes={"weights": (2,)}, output_dim=(5,))
    layer.build((None, 4))

    x = tf.constant(np.random.uniform(0, 1, (10, 4)))
    res = layer(x)

    assert res.shape == (10, 5)

    for x_, r in zip(x, res):
        exp = tf.experimental.numpy.hstack(circuit(x_, layer.qnode_weights["weights"]))
        assert qml.math.allclose(r, exp, atol=tol)


@pytest.mark.tf
def test_draw():
    """Test that a KerasLayer can be drawn using qml.draw"""

    dev = qml.device("default.qubit", wires=2)
    weight_shapes = {"w1": 1, "w2": (3, 2, 3)}

    @qml.qnode(dev, interface="tensorflow")
    def circuit(inputs, w1, w2):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        qml.templates.StronglyEntanglingLayers(w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        qlayer = KerasLayer(circuit, weight_shapes, output_dim=2)

    # make the rotation angle positive to prevent the extra minus sign messing up the alignment
    qlayer.qnode_weights["w1"] = tf.abs(qlayer.qnode_weights["w1"])

    batch_size = 5
    x = tf.constant(np.random.uniform(0, 1, (batch_size, 2)))

    actual = qml.draw(qlayer)(x)

    w1 = f"{qlayer.qnode_weights['w1'].numpy():.2f}"
    m1 = f"{qlayer.qnode_weights['w2'].numpy()}"
    expected = (
        f"0: ─╭AngleEmbedding(M0)──RX({w1})─╭StronglyEntanglingLayers(M1)─┤  <Z>\n"
        f"1: ─╰AngleEmbedding(M0)───────────╰StronglyEntanglingLayers(M1)─┤  <Z>\n"
        f"\n"
        f"M0 = \n{x}\n"
        f"M1 = \n{m1}"
    )

    assert actual == expected


@pytest.mark.tf
def test_draw_mpl():
    """Test that a KerasLayer can be drawn using qml.draw_mpl"""

    dev = qml.device("default.qubit", wires=2)
    weight_shapes = {"w1": 1, "w2": (3, 2, 3)}

    @qml.qnode(dev, interface="tensorflow")
    def circuit(inputs, w1, w2):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        qml.templates.StronglyEntanglingLayers(w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        qlayer = KerasLayer(circuit, weight_shapes, output_dim=2)

    # make the rotation angle positive to prevent the extra minus sign messing up the alignment
    qlayer.qnode_weights["w1"] = tf.abs(qlayer.qnode_weights["w1"])

    batch_size = 5
    x = tf.constant(np.random.uniform(0, 1, (batch_size, 2)))

    _, ax = qml.draw_mpl(qlayer)(x)

    assert len(ax.patches) == 9  # 3 boxes, 3 patches for each measure
    assert len(ax.lines) == 2  # 2 wires
    assert len(ax.texts) == 5  # 2 wire labels, 3 box labels

    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "1"
    assert ax.texts[2].get_text() == "AngleEmbedding"
    assert ax.texts[3].get_text() == "RX"
    assert ax.texts[4].get_text() == "StronglyEntanglingLayers"


@pytest.mark.tf
def test_specs():
    """Test that the qml.specs transform works for KerasLayer"""

    dev = qml.device("default.qubit", wires=3)
    weight_shapes = {"w1": 1, "w2": (3, 2, 3)}

    @qml.qnode(dev, interface="tf")
    def circuit(inputs, w1, w2):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        qml.templates.StronglyEntanglingLayers(w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        qlayer = KerasLayer(circuit, weight_shapes, output_dim=2)

    batch_size = 5
    x = tf.constant(np.random.uniform(0, 1, (batch_size, 2)))

    info = qml.specs(qlayer)(x)

    gate_sizes = defaultdict(int, {1: 1, 2: 2})
    gate_types = defaultdict(int, {"AngleEmbedding": 1, "RX": 1, "StronglyEntanglingLayers": 1})
    expected_resources = qml.resource.Resources(
        num_wires=2, num_gates=3, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
    )
    assert info["resources"] == expected_resources

    assert info["num_observables"] == 2
    assert info["num_device_wires"] == 3
    assert info["num_tape_wires"] == 2
    assert info["num_trainable_params"] == 2
    assert info["interface"] == "tf"
    assert info["device_name"] == "default.qubit"


@pytest.mark.xfail(
    condition=USING_KERAS3,
    reason=KERAS3_XFAIL_INFO,
)
@pytest.mark.slow
@pytest.mark.tf
def test_save_and_load_preserves_weights(tmpdir):
    """
    Test that saving and loading a model doesn't lose the weights. This particular
    test is important aside from `test_save_whole_model` because a bug caused issues
    when the name of (at least one of) your weights is 'weights'.
    """

    dev = qml.device("default.qubit")
    n_qubits = 2

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.RX(weights[0], 0)
        qml.RX(weights[1], 1)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    weight_shapes = {"weights": (n_qubits,)}
    with pytest.warns(
        qml.PennyLaneDeprecationWarning, match="The 'KerasLayer' class is deprecated"
    ):
        quantum_layer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=2)

    model0 = tf.keras.models.Sequential()
    model0.add(quantum_layer)
    model0.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax, input_shape=(2,)))

    model0.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    num_points = 5
    dummy_input_data = np.random.uniform(0, np.pi, size=(num_points, 2))
    dummy_output_data = np.random.randint(2, size=(num_points, 2))

    model0.fit(dummy_input_data, dummy_output_data, epochs=1, batch_size=0)
    file = str(tmpdir) + "/model"
    model0.save(file)
    loaded_model = tf.keras.models.load_model(file)
    assert np.array_equal(model0.layers[0].weights, loaded_model.layers[0].weights)
