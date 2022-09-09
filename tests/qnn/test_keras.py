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
import numpy as np
import pytest

import pennylane as qml

KerasLayer = qml.qnn.keras.KerasLayer

tf = pytest.importorskip("tensorflow", minversion="2")


@pytest.fixture
def model(get_circuit, n_qubits, output_dim):
    """Fixture for creating a hybrid Keras model. The model is composed of KerasLayers sandwiched
    between Dense layers."""
    c, w = get_circuit
    layer1 = KerasLayer(c, w, output_dim)
    layer2 = KerasLayer(c, w, output_dim)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(n_qubits),
            layer1,
            tf.keras.layers.Dense(n_qubits),
            layer2,
            tf.keras.layers.Dense(output_dim),
        ]
    )

    return model


@pytest.fixture
def model_dm(get_circuit_dm, n_qubits, output_dim):
    c, w = get_circuit_dm
    layer1 = KerasLayer(c, w, output_dim)
    layer2 = KerasLayer(c, w, output_dim)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(n_qubits),
            layer1,
            # Adding a lambda layer to take only the real values from density matrix
            tf.keras.layers.Lambda(lambda x: tf.abs(x)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_qubits),
            layer2,
            # Adding a lambda layer to take only the real values from density matrix
            tf.keras.layers.Lambda(lambda x: tf.abs(x)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_dim[0] * output_dim[1]),
        ]
    )

    return model


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
@pytest.mark.parametrize("interface", ["tf"])  # required for the get_circuit fixture
@pytest.mark.usefixtures("get_circuit")
class TestKerasLayer:
    """Unit tests for the pennylane.qnn.keras.KerasLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_bad_tf_version(self, get_circuit, output_dim, monkeypatch):
        """Test if an ImportError is raised when instantiated with an incorrect version of
        TensorFlow"""
        c, w = get_circuit
        with monkeypatch.context() as m:
            m.setattr(qml.qnn.keras, "CORRECT_TF_VERSION", False)
            with pytest.raises(ImportError, match="KerasLayer requires TensorFlow version 2"):
                KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_input(self):
        """Test if a TypeError is raised when instantiated with a QNode that does not have an
        argument with name equal to the input_arg class attribute of KerasLayer"""
        dev = qml.device("default.qubit", wires=1)
        weight_shapes = {"w1": (3, 3), "w2": 1}

        @qml.qnode(dev, interface="tf")
        def circuit(w1, w2):
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(TypeError, match="QNode must include an argument with name"):
            KerasLayer(circuit, weight_shapes, output_dim=1)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_input_in_weight_shapes(self, get_circuit, n_qubits, output_dim):
        """Test if a ValueError is raised when instantiated with a weight_shapes dictionary that
        contains the shape of the input argument given by the input_arg class attribute of
        KerasLayer"""
        c, w = get_circuit
        w[qml.qnn.keras.KerasLayer._input_arg] = n_qubits
        with pytest.raises(
            ValueError,
            match=f"{qml.qnn.keras.KerasLayer._input_arg} argument should not have its dimension",
        ):
            KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_weight_shape_unspecified(self, get_circuit, output_dim):
        """Test if a ValueError is raised when instantiated with a weight missing from the
        weight_shapes dictionary"""
        c, w = get_circuit
        del w["w1"]
        with pytest.raises(ValueError, match="Must specify a shape for every non-input parameter"):
            KerasLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_pos(self):
        """Test if a TypeError is raised when instantiated with a variable number of positional
        arguments"""
        dev = qml.device("default.qubit", wires=1)
        weight_shapes = {"w1": (3, 3), "w2": 1}

        @qml.qnode(dev, interface="tf")
        def circuit(inputs, w1, w2, *args):
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(TypeError, match="Cannot have a variable number of positional"):
            KerasLayer(circuit, weight_shapes, output_dim=1)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_keyword(self):
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

        layer = KerasLayer(c, w, output_dim=output_dim)
        x = tf.ones((2, n_qubits))

        layer_out = layer(x)
        circ_weights = layer.qnode_weights.copy()
        circ_weights["w4"] = tf.convert_to_tensor(circ_weights["w4"])  # To allow for slicing
        circ_weights["w6"] = tf.convert_to_tensor(circ_weights["w6"])
        circuit_out = c(x[0], **circ_weights)

        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits", [1])
    @pytest.mark.parametrize("output_dim", zip(*[[[1], (1,), 1], [1, 1, 1]]))
    def test_output_dim(self, get_circuit, output_dim):
        """Test if the output_dim is correctly processed, i.e., that an iterable is mapped to
        its first element while an int is left unchanged."""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim[0])
        assert layer.output_dim == output_dim[1]

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_weight_shapes(self, get_circuit, output_dim, n_qubits):
        """Test if the weight_shapes input argument is correctly processed to be a dictionary
        with values that are tuples."""
        c, w = get_circuit
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
    def test_non_input_defaults(self):
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
        def c(inputs, w1, w2, w4, w5, w6, w7, w3=0.5):
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

        layer = KerasLayer(c, w, output_dim=output_dim)
        x = tf.ones((2, n_qubits))

        layer_out = layer(x)
        circ_weights = layer.qnode_weights.copy()
        circ_weights["w4"] = tf.convert_to_tensor(circ_weights["w4"])  # To allow for slicing
        circ_weights["w6"] = tf.convert_to_tensor(circ_weights["w6"])
        circuit_out = c(x[0], **circ_weights)

        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights(self, get_circuit, n_qubits, output_dim):
        """Test if the build() method correctly initializes the weights in the qnode_weights
        dictionary, i.e., that each value of the dictionary has correct shape and name."""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)
        layer.build(input_shape=(10, n_qubits))

        for weight, shape in layer.weight_shapes.items():
            assert layer.qnode_weights[weight].shape == shape
            assert layer.qnode_weights[weight].name[:-2] == weight

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_qnode_weights_with_spec(self, get_circuit, monkeypatch, output_dim, n_qubits):
        """Test if the build() method correctly passes on user specified weight_specs to the
        inherited add_weight() method. This is done by monkeypatching add_weight() so that it
        simply returns its input keyword arguments. The qnode_weights dictionary should then have
        values that are the input keyword arguments, and we check that the specified weight_specs
        keywords are there."""

        def add_weight_dummy(*args, **kwargs):
            """Dummy function for mocking out the add_weight method to simply return the input
            keyword arguments"""
            return kwargs

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
            layer = KerasLayer(c, w, output_dim, weight_specs=weight_specs)
            layer.build(input_shape=(10, n_qubits))

            for weight in layer.weight_shapes:
                assert all(
                    item in layer.qnode_weights[weight].items()
                    for item in weight_specs[weight].items()
                )

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(3))
    @pytest.mark.parametrize("input_shape", [(10, 4), (8, 3)])
    def test_compute_output_shape(self, get_circuit, output_dim, input_shape):
        """Test if the compute_output_shape() method performs correctly, i.e., that it replaces
        the last element in the input_shape tuple with the specified output_dim and that the
        output shape is of type tf.TensorShape"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)

        assert layer.compute_output_shape(input_shape) == (input_shape[0], output_dim)
        assert isinstance(layer.compute_output_shape(input_shape), tf.TensorShape)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2])
    def test_call(self, get_circuit, output_dim, batch_size, n_qubits):
        """Test if the call() method performs correctly, i.e., that it outputs with shape
        (batch_size, output_dim) with results that agree with directly calling the QNode"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w.numpy() for w in layer.qnode_weights.values()]
        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(x[0], *weights))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    @pytest.mark.parametrize("batch_size", [2])
    def test_call_shuffled_args(self, get_circuit, output_dim, batch_size, n_qubits):
        """Test if the call() method performs correctly when the inputs argument is not the first
        positional argument, i.e., that it outputs with shape (batch_size, output_dim) with
        results that agree with directly calling the QNode"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="tf")
        def c_shuffled(w1, inputs, w2, w3, w4, w5, w6, w7):
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

        layer = KerasLayer(c_shuffled, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w.numpy() for w in layer.qnode_weights.values()]

        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(x[0], *weights))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    @pytest.mark.parametrize("batch_size", [2])
    def test_call_default_input(self, get_circuit, output_dim, batch_size, n_qubits):
        """Test if the call() method performs correctly when the inputs argument is a default
        argument, i.e., that it outputs with shape (batch_size, output_dim) with results that
        agree with directly calling the QNode"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="tf")
        def c_default(w1, w2, w3, w4, w5, w6, w7, inputs=None):
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

        layer = KerasLayer(c_default, w, output_dim)
        x = tf.ones((batch_size, n_qubits))

        layer_out = layer(x)
        weights = [w.numpy() for w in layer.qnode_weights.values()]

        assert layer_out.shape == (batch_size, output_dim)
        assert np.allclose(layer_out[0], c(x[0], *weights))

    @pytest.mark.slow
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2, 4, 6])
    @pytest.mark.parametrize("middle_dim", [2, 5, 8])
    def test_call_broadcast(self, get_circuit, output_dim, middle_dim, batch_size, n_qubits):
        """Test if the call() method performs correctly when the inputs argument has an arbitrary shape (that can
        correctly be broadcast over), i.e., for input of shape (batch_size, dn, ... , d0) it outputs with shape
        (batch_size, dn, ... , d1, output_dim). Also tests if gradients are still backpropagated correctly."""
        c, w = get_circuit

        layer = KerasLayer(c, w, output_dim)
        x = tf.ones((batch_size, middle_dim, n_qubits))

        with tf.GradientTape() as tape:
            layer_out = layer(x)

        g_layer = tape.gradient(layer_out, layer.trainable_variables)

        # test gradients are at least calculated
        assert g_layer is not None
        assert layer_out.shape == (batch_size, middle_dim, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_str_repr(self, get_circuit, output_dim):
        """Test the __str__ and __repr__ representations"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)

        assert layer.__str__() == "<Quantum Keras Layer: func=circuit>"
        assert layer.__repr__() == "<Quantum Keras Layer: func=circuit>"

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_gradients(self, get_circuit, output_dim, n_qubits):
        """Test if the gradients of the KerasLayer are equal to the gradients of the circuit when
        taken with respect to the trainable variables"""
        c, w = get_circuit
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
    def test_backprop_gradients(self, mocker):
        """Test if KerasLayer is compatible with the backprop diff method."""

        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def f(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(2))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        weight_shapes = {"weights": (3, 2, 3)}

        qlayer = qml.qnn.KerasLayer(f, weight_shapes, output_dim=2)

        inputs = tf.ones((4, 2))

        with tf.GradientTape() as tape:
            out = tf.reduce_sum(qlayer(inputs))

        spy = mocker.spy(qml.gradients, "param_shift")

        grad = tape.gradient(out, qlayer.trainable_weights)
        assert grad is not None
        spy.assert_not_called()

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_compute_output_shape(self, get_circuit, output_dim):
        """Test that the compute_output_shape method returns the expected shape"""
        c, w = get_circuit
        layer = KerasLayer(c, w, output_dim)

        inputs = tf.keras.Input(shape=(2,))
        inputs_shape = inputs.shape

        output_shape = layer.compute_output_shape(inputs_shape)
        assert output_shape.as_list() == [None, 1]


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "torch", "tf"])
@pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
@pytest.mark.usefixtures("get_circuit")
def test_interface_conversion(get_circuit, output_dim):
    """Test if input QNodes with all types of interface are converted internally to the TensorFlow
    interface"""
    c, w = get_circuit
    layer = KerasLayer(c, w, output_dim)
    assert layer.qnode.interface == "tf"


@pytest.mark.tf
@pytest.mark.parametrize("interface", ["tf"])
@pytest.mark.usefixtures("get_circuit", "model")
class TestKerasLayerIntegration:
    """Integration tests for the pennylane.qnn.keras.KerasLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2])
    def test_train_model(self, model, batch_size, n_qubits, output_dim):
        """Test if a model can train using the KerasLayer. The model is composed of two
        KerasLayers sandwiched between Dense neural network layers, and the dataset is simply
        input and output vectors of zeros."""

        x = np.zeros((batch_size, n_qubits))
        y = np.zeros((batch_size, output_dim))

        model.compile(optimizer="sgd", loss="mse")

        model.fit(x, y, batch_size=batch_size, verbose=0)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_model_gradients(self, model, output_dim, n_qubits):
        """Test if a gradient can be calculated with respect to all of the trainable variables in
        the model"""
        x = tf.zeros((2, n_qubits))
        y = tf.zeros((2, output_dim))

        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.keras.losses.mean_squared_error(out, y)

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all([g.dtype == tf.keras.backend.floatx() for g in gradients])

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_model_save_weights(self, model, n_qubits, tmpdir):
        """Test if the model can be successfully saved and reloaded using the get_weights()
        method"""
        prediction = model.predict(np.ones((1, n_qubits)))
        weights = model.get_weights()
        file = str(tmpdir) + "/model"
        model.save_weights(file)
        model.load_weights(file)
        prediction_loaded = model.predict(np.ones((1, n_qubits)))
        weights_loaded = model.get_weights()

        assert np.allclose(prediction, prediction_loaded)
        for i, w in enumerate(weights):
            assert np.allclose(w, weights_loaded[i])


@pytest.mark.tf
@pytest.mark.parametrize("interface", ["tf"])
@pytest.mark.usefixtures("get_circuit_dm", "model_dm")
class TestKerasLayerIntegrationDM:
    """Integration tests for the pennylane.qnn.keras.KerasLayer class for
    density_matrix() returning circuits."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(3))
    @pytest.mark.parametrize("batch_size", [2])
    def test_train_model_dm(self, model_dm, batch_size, n_qubits, output_dim):
        """Test if a model can train using the KerasLayer when QNode returns a density_matrix().
        The model is composed of two KerasLayers sandwiched between Dense neural network layers,
        and the dataset is simply input and output vectors of zeros."""
        x = np.zeros((batch_size, n_qubits))
        y = np.zeros((batch_size, output_dim[0] * output_dim[1]))

        model_dm.compile(optimizer="sgd", loss="mse")

        model_dm.fit(x, y, batch_size=batch_size, verbose=0)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(2))
    def test_model_gradients_dm(self, model_dm, output_dim, n_qubits):
        """Test if a gradient can be calculated with respect to all of the trainable variables in
        the model."""
        x = tf.zeros((2, n_qubits))
        y = tf.zeros((2, output_dim[0] * output_dim[1]))

        with tf.GradientTape() as tape:
            out = model_dm(x)
            loss = tf.keras.losses.mean_squared_error(out, y)

        gradients = tape.gradient(loss, model_dm.trainable_variables)
        assert all([g.dtype == tf.keras.backend.floatx() for g in gradients])

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to_dm(2))
    def test_model_save_weights_dm(self, model_dm, n_qubits, tmpdir):
        """Test if the model_dm can be successfully saved and reloaded using the get_weights()
        method"""

        prediction = model_dm.predict(np.ones((1, n_qubits)))
        weights = model_dm.get_weights()
        file = str(tmpdir) + "/model"
        model_dm.save_weights(file)
        model_dm.load_weights(file)
        prediction_loaded = model_dm.predict(np.ones((1, n_qubits)))
        weights_loaded = model_dm.get_weights()

        assert np.allclose(prediction, prediction_loaded)
        for i, w in enumerate(weights):
            assert np.allclose(w, weights_loaded[i])


@pytest.mark.tf
def test_no_attribute():
    """Test that the qnn module raises an AttributeError if accessing an unavailable attribute"""
    with pytest.raises(AttributeError, match="module 'pennylane.qnn' has no attribute 'random'"):
        qml.qnn.random


@pytest.mark.tf
def test_batch_input():
    """Test input batching in keras"""
    dev = qml.device("default.qubit.tf", wires=4)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.probs(op=qml.PauliZ(1))

    KerasLayer.set_input_argument("x")
    layer = KerasLayer(circuit, weight_shapes={"weights": (2,)}, output_dim=(2,), batch_idx=0)
    conf = layer.get_config()
    layer.build((None, 2))
    assert layer(np.random.uniform(0, 1, (10, 4))).shape == (10, 2)
