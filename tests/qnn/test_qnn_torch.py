# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the pennylane.qnn.torch module.
"""
import math
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest

import pennylane as qml
from pennylane.qnn.torch import TorchLayer

torch = pytest.importorskip("torch")

# pylint: disable=unnecessary-dunder-call


def indices_up_to(n_max):
    """Returns an iterator over the number of qubits and output dimension, up to value n_max.
    The output dimension never exceeds the number of qubits."""
    a, b = np.tril_indices(n_max)
    return zip(*[a + 1, b + 1])


@pytest.mark.usefixtures("get_circuit")  # this fixture is in tests/qnn/conftest.py
@pytest.fixture
def module(get_circuit, n_qubits, output_dim):
    """Fixture for creating a hybrid Torch module. The module is composed of quantum TorchLayers
    sandwiched between Linear layers."""
    c, w = get_circuit

    class Net(torch.nn.Module):  # pylint: disable=too-few-public-methods
        """Neural network."""

        def __init__(self):
            super().__init__()
            self.clayer1 = torch.nn.Linear(n_qubits, n_qubits)
            self.clayer2 = torch.nn.Linear(output_dim, n_qubits)
            self.clayer3 = torch.nn.Linear(output_dim, output_dim)
            self.qlayer1 = TorchLayer(c, w)
            self.qlayer2 = TorchLayer(c, w)

        def forward(self, x):
            """Forward pass."""
            x = self.clayer1(x)
            x = self.qlayer1(x)
            x = self.clayer2(x)
            x = self.qlayer2(x)
            x = self.clayer3(x)
            return x

    return Net()


@pytest.mark.torch
@pytest.mark.parametrize("interface", ["torch"])  # required for the get_circuit fixture
@pytest.mark.usefixtures("get_circuit")  # this fixture is in tests/qnn/conftest.py
class TestTorchLayer:  # pylint: disable=too-many-public-methods
    """Unit tests for the pennylane.qnn.torch.TorchLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_torch(self, get_circuit, monkeypatch):  # pylint: disable=no-self-use
        """Test if an ImportError is raised when instantiated without PyTorch"""
        c, w = get_circuit
        with monkeypatch.context() as m:
            m.setattr(qml.qnn.torch, "TORCH_IMPORTED", False)
            with pytest.raises(ImportError, match="TorchLayer requires PyTorch"):
                TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_input(self):  # pylint: disable=no-self-use
        """Test if a TypeError is raised when instantiated with a QNode that does not have an
        argument with name equal to the input_arg class attribute of TorchLayer"""
        dev = qml.device("default.qubit", wires=1)
        weight_shapes = {"w1": (3, 3), "w2": 1}

        @qml.qnode(dev, interface="torch")
        def circuit(w1, w2):  # pylint: disable=unused-argument
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(TypeError, match="QNode must include an argument with name"):
            TorchLayer(circuit, weight_shapes)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_input_in_weight_shapes(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test if a ValueError is raised when instantiated with a weight_shapes dictionary that
        contains the shape of the input argument given by the input_arg class attribute of
        TorchLayer"""
        c, w = get_circuit
        w[qml.qnn.torch.TorchLayer._input_arg] = n_qubits  # pylint: disable=protected-access
        with pytest.raises(
            ValueError,
            match=f"{qml.qnn.torch.TorchLayer._input_arg} argument should not have its dimension",  # pylint: disable=protected-access
        ):
            TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_weight_shape_unspecified(self, get_circuit):  # pylint: disable=no-self-use
        """Test if a ValueError is raised when instantiated with a weight missing from the
        weight_shapes dictionary"""
        c, w = get_circuit
        del w["w1"]
        with pytest.raises(ValueError, match="Must specify a shape for every non-input parameter"):
            TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_pos(self):  # pylint: disable=no-self-use
        """Test if a TypeError is raised when instantiated with a variable number of positional
        arguments"""
        dev = qml.device("default.qubit", wires=1)
        weight_shapes = {"w1": (3, 3), "w2": 1}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, w1, w2, *args):  # pylint: disable=unused-argument
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(TypeError, match="Cannot have a variable number of positional"):
            TorchLayer(circuit, weight_shapes)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_keyword(self, n_qubits, output_dim):  # pylint: disable=no-self-use
        """Test that variable number of keyword arguments works"""
        dev = qml.device("default.qubit", wires=n_qubits)
        w = {
            "w1": (3, n_qubits, 3),
            "w2": (1,),
            "w3": 1,
            "w4": [3],
            "w5": (2, n_qubits, 3),
            "w6": 3,
            "w7": 1,
        }

        @qml.qnode(dev, interface="torch")
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

        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        layer_out = layer._evaluate_qnode(x)  # pylint: disable=protected-access
        circuit_out = torch.hstack(c(x, **layer.qnode_weights)).type(x.dtype)

        assert torch.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_nonspecified_init(self, get_circuit, monkeypatch):  # pylint: disable=no-self-use
        """Test if weights are initialized according to the uniform distribution in [0, 2 pi]"""
        c, w = get_circuit

        uniform_ = mock.MagicMock(return_value=torch.Tensor(1))
        with monkeypatch.context() as m:
            m.setattr(torch.nn.init, "uniform_", uniform_)
            TorchLayer(c, w)
            kwargs = uniform_.call_args[1]
            assert kwargs["b"] == 2 * math.pi

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_callable_init(self, get_circuit, monkeypatch):  # pylint: disable=no-self-use
        """Test if weights are initialized according to the callable function specified in the
        init_method argument."""
        c, w = get_circuit

        normal_ = mock.MagicMock(return_value=torch.normal(mean=0, std=1, size=[1]))
        with monkeypatch.context() as m:
            m.setattr(torch.nn.init, "uniform_", normal_)
            layer = TorchLayer(qnode=c, weight_shapes=w, init_method=normal_)
        normal_.assert_called()
        for _, weight in layer.qnode_weights.items():
            data = weight.data.tolist()
            if not isinstance(data, list):
                data = [data]
            assert data == normal_().tolist()

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_fixed_init(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test if weights are initialized according to the value specified in the
        init_method argument."""
        c, w = get_circuit

        init_method = {
            "w1": torch.normal(mean=0, std=1, size=(3, n_qubits, 3)),
            "w2": torch.normal(mean=0, std=1, size=(1,)),
            "w3": torch.normal(mean=0, std=1, size=[]),
            "w4": torch.normal(mean=0, std=1, size=(3,)),
            "w5": torch.normal(mean=0, std=1, size=(2, n_qubits, 3)),
            "w6": torch.normal(mean=0, std=1, size=(3,)),
            "w7": torch.normal(mean=0, std=1, size=[]),
        }

        layer = TorchLayer(qnode=c, weight_shapes=w, init_method=init_method)
        for name, weight in layer.qnode_weights.items():
            assert weight.data.tolist() == init_method[name].tolist()

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_fixed_init_raises_error(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test that a ValueError is raised when using a Tensor with the wrong shape."""
        c, w = get_circuit

        init_method = {
            "w1": torch.normal(mean=0, std=1, size=(1,)),
            "w2": torch.normal(mean=0, std=1, size=(1,)),
            "w3": torch.normal(mean=0, std=1, size=[]),
            "w4": torch.normal(mean=0, std=1, size=(3,)),
            "w5": torch.normal(mean=0, std=1, size=(2, n_qubits, 3)),
            "w6": torch.normal(mean=0, std=1, size=(3,)),
            "w7": torch.normal(mean=0, std=1, size=[]),
        }

        with pytest.raises(ValueError):
            TorchLayer(qnode=c, weight_shapes=w, init_method=init_method)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_init_missing_weights(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test that a KeyError is raised when using an init_method with missing weights."""
        c, w = get_circuit

        init_method = {
            "w1": torch.normal(mean=0, std=1, size=(3, n_qubits, 3)),
            "w2": torch.normal(mean=0, std=1, size=(1,)),
            "w3": torch.normal(mean=0, std=1, size=[]),
            "w5": torch.normal(mean=0, std=1, size=(2, n_qubits, 3)),
            "w7": torch.normal(mean=0, std=1, size=[]),
        }

        with pytest.raises(KeyError):
            TorchLayer(qnode=c, weight_shapes=w, init_method=init_method)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_fixed_and_callable_init(self, get_circuit):  # pylint: disable=no-self-use
        """Test if weights are initialized according to the callables and values specified in the
        init_method argument."""
        c, w = get_circuit

        normal_ = mock.MagicMock(return_value=torch.normal(mean=0, std=1, size=[1]))

        init_method = {
            "w1": normal_,
            "w2": torch.normal(mean=0, std=1, size=(1,)),
            "w3": normal_,
            "w4": torch.normal(mean=0, std=1, size=(3,)),
            "w5": normal_,
            "w6": torch.normal(mean=0, std=1, size=(3,)),
            "w7": normal_,
        }

        layer = TorchLayer(qnode=c, weight_shapes=w, init_method=init_method)
        normal_.assert_called()
        for name, weight in layer.qnode_weights.items():
            data = weight.data.tolist()
            if not isinstance(data, list):
                data = [data]
            if isinstance(init_method[name], mock.MagicMock):
                assert data == init_method[name]().tolist()
            else:
                assert data == init_method[name].tolist()

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_non_input_defaults(self, n_qubits, output_dim):  # pylint: disable=no-self-use
        """Test that everything works when default arguments that are not the input argument are
        present in the QNode"""
        dev = qml.device("default.qubit", wires=n_qubits)
        w = {
            "w1": (3, n_qubits, 3),
            "w2": (1,),
            "w3": 1,
            "w4": [3],
            "w5": (2, n_qubits, 3),
            "w6": 3,
            "w7": 1,
        }

        @qml.qnode(dev, interface="torch")
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

        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        layer_out = layer._evaluate_qnode(x)  # pylint: disable=protected-access
        circuit_out = torch.hstack(c(x, **layer.qnode_weights)).type(x.dtype)

        assert torch.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights_shapes(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test if the weights in qnode_weights have the correct shape"""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        ideal_shapes = {
            "w1": torch.Size((3, n_qubits, 3)),
            "w2": torch.Size((1,)),
            "w3": torch.Size([]),
            "w4": torch.Size((3,)),
            "w5": torch.Size((2, n_qubits, 3)),
            "w6": torch.Size((3,)),
            "w7": torch.Size([]),
        }

        for name, weight in layer.qnode_weights.items():
            assert weight.shape == ideal_shapes[name]

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights_registered(self, get_circuit):  # pylint: disable=no-self-use
        """Test if the weights in qnode_weights are registered to the internal _parameters
        dictionary and that requires_grad == True"""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        for name, weight in layer.qnode_weights.items():
            assert torch.allclose(
                weight, layer._parameters[name]  # pylint: disable=protected-access
            )
            assert weight.requires_grad

        assert layer.w1 is layer._parameters["w1"]  # pylint: disable=protected-access
        assert layer.w2 is layer._parameters["w2"]  # pylint: disable=protected-access

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_evaluate_qnode(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test if the _evaluate_qnode() method works correctly, i.e., that it gives the same
        result as calling the QNode directly"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        layer_out = layer._evaluate_qnode(x)  # pylint: disable=protected-access
        weights = layer.qnode_weights.values()
        circuit_out = torch.hstack(c(x, *weights)).type(x.dtype)

        assert torch.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("shots", [5, [5, 5]])
    def test_evaluate_qnode_shots(
        self, get_circuit_shots, n_qubits, shots
    ):  # pylint: disable=no-self-use
        """Tests _evaluate_qnode() works with different type of shots"""

        c, w = get_circuit_shots
        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        layer_out = layer._evaluate_qnode(x)  # pylint: disable=protected-access
        weights = layer.qnode_weights.values()
        circuit_out_raw = c(x, *weights)

        if isinstance(shots, list):

            assert isinstance(layer_out, tuple)
            for out, exp in zip(layer_out, circuit_out_raw):
                circuit_out = torch.hstack(exp)
                assert out.shape == circuit_out.shape

        else:
            circuit_out = torch.hstack(circuit_out_raw)
            assert layer_out.shape == circuit_out.shape

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_evaluate_qnode_shuffled_args(
        self, get_circuit, output_dim, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the _evaluate_qnode() method works correctly when the inputs argument is not the
        first positional argument, i.e., that it gives the same result as calling the QNode
        directly"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch")
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

        layer = TorchLayer(c_shuffled, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer._evaluate_qnode(x)  # pylint: disable=protected-access
        weights = layer.qnode_weights.values()
        circuit_out = torch.hstack(c(x, *weights)).type(x.dtype)

        assert torch.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_evaluate_qnode_default_input(
        self, get_circuit, output_dim, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the _evaluate_qnode() method works correctly when the inputs argument is a
        default argument, i.e., that it gives the same result as calling the QNode directly"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch")
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

        layer = TorchLayer(c_default, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer._evaluate_qnode(x)  # pylint: disable=protected-access
        weights = layer.qnode_weights.values()
        circuit_out = torch.hstack(c(x, *weights)).type(x.dtype)

        assert torch.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_forward_single_input(
        self, get_circuit, output_dim, n_qubits
    ):  # pylint: disable=no-self-use
        """Test if the forward() method accepts a single input (i.e., not with an extra batch
        dimension) and returns a tensor of the right shape"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer.forward(x)
        assert layer_out.shape == torch.Size((output_dim,))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_forward(self, get_circuit, output_dim, n_qubits):  # pylint: disable=no-self-use
        """Test if the forward() method accepts a batched input and returns a tensor of the right
        shape"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.Tensor(np.ones((2, n_qubits)))

        layer_out = layer.forward(x)

        assert layer_out.shape == torch.Size((2, output_dim))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("shots", [[10, 10]])
    def test_forward_shot_batching(self, get_circuit_shots, output_dim, n_qubits):
        """Tests forward() works with shot batching"""

        c, w = get_circuit_shots
        layer = TorchLayer(c, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer.forward(x)
        assert layer_out.shape == torch.Size((2, output_dim))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [4, 6])
    @pytest.mark.parametrize("shots", [[10, 10]])
    def test_forward_shot_batching_broadcasting(
        self, get_circuit_shots, batch_size, output_dim, n_qubits
    ):
        """Tests forward() works with shot batching and broadcasting"""

        c, w = get_circuit_shots
        layer = TorchLayer(c, w)
        x = torch.Tensor(np.ones((batch_size, n_qubits)))

        layer_out = layer.forward(x)
        assert layer_out.shape == torch.Size((2, batch_size, output_dim))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2, 4, 6])
    @pytest.mark.parametrize("middle_dim", [2, 5, 8])
    def test_forward_broadcasting(
        self, get_circuit, output_dim, middle_dim, batch_size, n_qubits
    ):  # pylint: disable=too-many-arguments,no-self-use
        """Test if the forward() method accepts a batched input with multiple dimensions and returns a tensor of the
        right shape by broadcasting. Also tests if gradients are still backpropagated correctly."""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        x = torch.Tensor(np.ones((batch_size, middle_dim, n_qubits)))

        weights = layer.qnode_weights.values()

        layer_out = layer.forward(x)
        layer_out.backward(torch.ones_like(layer_out))

        g_layer = [w.grad for w in weights]

        assert g_layer.count(None) == 0
        assert layer_out.shape == torch.Size((batch_size, middle_dim, output_dim))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_str_repr(self, get_circuit):  # pylint: disable=no-self-use
        """Test the __str__ and __repr__ representations"""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        assert layer.__str__() == "<Quantum Torch Layer: func=circuit>"
        assert layer.__repr__() == "<Quantum Torch Layer: func=circuit>"

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_gradients(self, get_circuit, n_qubits):  # pylint: disable=no-self-use
        """Test if the gradients of the TorchLayer are equal to the gradients of the circuit when
        taken with respect to the trainable variables"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        weights = layer.qnode_weights.values()

        out_layer = layer(x)
        out_layer.backward()

        g_layer = [w.grad for w in weights]

        out_circuit = torch.hstack(c(x, *weights)).type(x.dtype)
        out_circuit.backward()

        g_circuit = [w.grad for w in weights]

        for g1, g2 in zip(g_layer, g_circuit):
            assert torch.allclose(g1, g2)
        assert len(weights) == len(list(layer.parameters()))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(3))
    def test_construct(self, get_circuit, n_qubits):
        """Test that the construct method builds the correct tape with correct differentiability"""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        x = torch.ones(n_qubits)

        tape = qml.workflow.construct_tape(layer)(x)

        assert tape is not None
        assert (
            len(tape.get_parameters(trainable_only=False))
            == len(tape.get_parameters(trainable_only=True)) + 1
        )


@pytest.mark.parametrize(
    "num_qubits, weight_shapes",
    [(2, {"weights": [3, 2, 3]}), (3, {"weights": [7, 3, 3]}), (4, {"weights": [3, 4, 3]})],
)
def test_forward_tuple(num_qubits, weight_shapes):
    """Test that the forward method accepts a tuple input and returns a torch tensor."""

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return qml.expval(qml.Z(0)), qml.var(qml.Y(1))

    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    x = torch.tensor(np.random.random((5, num_qubits)), dtype=torch.float32)
    assert isinstance(qlayer.forward(x), torch.Tensor)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax"])
def test_invalid_interface_error(interface):
    """Test an error gets raised if input QNode has the wrong interface"""
    dev = qml.device("default.qubit", wires=3)
    weight_shapes = {"w1": 1}

    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    with pytest.raises(ValueError, match="Invalid interface"):
        _ = TorchLayer(circuit, weight_shapes)


@pytest.mark.torch
@pytest.mark.parametrize("interface", ("auto", "torch"))
def test_qnode_interface_not_mutated(interface):
    """Test that the input QNode's interface is not mutated by TorchLayer"""
    dev = qml.device("default.qubit", wires=3)
    weight_shapes = {"w1": 1}

    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    qlayer = TorchLayer(circuit, weight_shapes)
    assert qlayer.qnode.interface == circuit.interface == qml.math.Interface(interface).value


@pytest.mark.torch
@pytest.mark.parametrize("interface", ["torch"])
@pytest.mark.usefixtures("get_circuit", "module")
class TestTorchLayerIntegration:
    """Integration tests for the pennylane.qnn.torch.TorchLayer class."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2])
    def test_step_module(
        self, module, batch_size, n_qubits, output_dim, dtype
    ):  # pylint: disable=redefined-outer-name,too-many-arguments,no-self-use
        """Test if a module that includes TorchLayers can perform one optimization step. This
        test checks that some of the parameters in the module are different after one step.
        The module is composed of two TorchLayers sandwiched between Linear neural network layers,
        and the dataset is simply input and output vectors of zeros."""
        if dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float64

        module = module.type(dtype)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.5)

        x = torch.zeros((batch_size, n_qubits)).type(dtype)
        y = torch.zeros((batch_size, output_dim)).type(dtype)

        params_before = [w.detach().clone() for w in module.parameters()]

        module_out = module(x)
        optimizer.zero_grad()
        loss = loss_func(module_out, y)
        loss.backward()
        optimizer.step()

        params_after = [w.detach().clone() for w in module.parameters()]

        params_similar = [torch.allclose(p1, p2) for p1, p2 in zip(params_before, params_after)]
        assert not all(params_similar)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_module_gradients(
        self, module, output_dim, n_qubits, get_circuit
    ):  # pylint: disable=redefined-outer-name,no-self-use
        """Test if a gradient can be calculated with respect to all of the trainable variables in
        the module"""
        _, w = get_circuit

        x = torch.zeros((2, n_qubits))
        y = torch.zeros((2, output_dim))

        module_out = module(x)
        loss_func = torch.nn.MSELoss()
        loss = loss_func(module_out, y)
        loss.backward()

        gradients = [w.grad for w in module.parameters()]
        assert all(g.is_floating_point() for g in gradients)
        assert len(gradients) == 2 * len(w) + 6  # six parameters come from classical layers

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_module_state_dict(
        self, module, get_circuit
    ):  # pylint: disable=redefined-outer-name,no-self-use
        """Test if the state dictionary output by the module contains all the expected trainable
        parameters"""
        _, w = get_circuit

        state_dict = module.state_dict()
        dict_keys = set(state_dict.keys())

        clayer_weights = {f"clayer{i + 1}.weight" for i in range(3)}
        clayer_biases = {f"clayer{i + 1}.bias" for i in range(3)}
        qlayer_params = {f"qlayer{i + 1}.w{j + 1}" for i in range(2) for j in range(len(w))}

        all_params = clayer_weights | clayer_biases | qlayer_params

        assert dict_keys == all_params
        assert len(dict_keys) == len(all_params)

    # pylint: disable=unused-argument
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_save_model(self, get_circuit, n_qubits, output_dim, tmpdir):
        """Test if the model can be saved and loaded"""
        qlayer = TorchLayer(*get_circuit)
        clayer = torch.nn.Linear(output_dim, 2)
        model = torch.nn.Sequential(qlayer, clayer)

        saved_state_dict = model.state_dict()
        path = str(tmpdir) + "/dummy.pt"
        torch.save(saved_state_dict, path)

        new_qlayer = TorchLayer(*get_circuit)
        new_clayer = torch.nn.Linear(output_dim, 2)
        new_model = torch.nn.Sequential(new_qlayer, new_clayer)
        new_state_dict = new_model.state_dict()

        assert list(saved_state_dict.keys()) == list(new_state_dict.keys())

        # test that the new model's weights are different
        assert all(
            torch.any(saved_state_dict[k] != new_state_dict[k]) for k in saved_state_dict.keys()
        )

        new_model.load_state_dict(torch.load(path))
        new_state_dict = new_model.state_dict()

        assert list(saved_state_dict.keys()) == list(new_state_dict.keys())

        # test that the new model's weights are now the same
        assert all(
            torch.all(saved_state_dict[k] == new_state_dict[k]) for k in saved_state_dict.keys()
        )


@pytest.mark.torch
def test_vjp_is_unwrapped_for_param_shift():
    """Test that the intermediate vjps used by the batch Torch interface
    are unwrapped and no error is raised for a custom operation.

    Note: the execution flow of the operation resembles the implementation
    of the Kerr gate in Strawberry Fields as a similar example was failing
    for qml.Kerr and the strawberryfields.fock device.
    """
    nqubits = 2

    device = qml.device("default.qubit", wires=nqubits)

    class DummyOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
        """Dummy operation."""

        num_wires = 1
        num_params = 1

        @staticmethod
        def compute_matrix(
            *params,
        ):  # pylint: disable=arguments-differ
            z = params[0]
            return np.diag([z, z])

    @qml.qnode(device=device, interface="torch", diff_method="parameter-shift")
    def circ(inputs, w0):  # pylint: disable=unused-argument
        DummyOp(inputs[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"w0": (1)}

    qlayer = qml.qnn.TorchLayer(circ, weight_shapes)
    qlayers = torch.nn.Sequential(qlayer)

    x = torch.tensor([0.1], dtype=float, requires_grad=True)
    u = qlayers(x)

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u, requires_grad=True),
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]

    assert isinstance(u_x, torch.Tensor)


@pytest.mark.torch
def test_batch_input_single_measure(tol):
    """Test input batching in torch"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return qml.probs(op=qml.PauliZ(1))

    TorchLayer.set_input_argument("x")
    layer = TorchLayer(circuit, weight_shapes={"weights": (2,)})
    x = torch.Tensor(np.random.uniform(0, 1, (10, 4)))
    res = layer(x)

    assert res.shape == (10, 2)

    for x_, r in zip(x, res):
        assert qml.math.allclose(r, circuit(x_, layer.qnode_weights["weights"]), atol=tol)

    # reset back to the old name
    TorchLayer.set_input_argument("inputs")


@pytest.mark.torch
def test_batch_input_multi_measure(tol):
    """Test input batching in torch for multiple measurements"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return [qml.expval(qml.PauliZ(1)), qml.probs(wires=range(2))]

    TorchLayer.set_input_argument("x")
    layer = TorchLayer(circuit, weight_shapes={"weights": (2,)})
    x = torch.Tensor(np.random.uniform(0, 1, (10, 4)))
    res = layer(x)

    assert res.shape == (10, 5)

    for x_, r in zip(x, res):
        exp = torch.hstack(circuit(x_, layer.qnode_weights["weights"]))
        assert qml.math.allclose(r, exp, atol=tol)

    # reset back to the old name
    TorchLayer.set_input_argument("inputs")


@pytest.mark.torch
def test_draw():
    """Test that a TorchLayer can be drawn using qml.draw"""

    dev = qml.device("default.qubit", wires=2)
    weight_shapes = {"w1": 1, "w2": (3, 2, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w1, w2):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        qml.templates.StronglyEntanglingLayers(w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    qlayer = TorchLayer(circuit, weight_shapes)

    # make the rotation angle positive to prevent the extra minus sign messing up the alignment
    qlayer.qnode_weights["w1"] = torch.abs(qlayer.qnode_weights["w1"])

    batch_size = 5
    x = torch.Tensor(np.random.uniform(0, 1, (batch_size, 2)))

    actual = qml.draw(qlayer)(x)

    w1 = f"{qlayer.qnode_weights['w1'].detach().numpy():.2f}"
    m1 = f"{qlayer.qnode_weights['w2'].detach()}"
    expected = (
        f"0: ─╭AngleEmbedding(M0)──RX({w1})─╭StronglyEntanglingLayers(M1)─┤  <Z>\n"
        f"1: ─╰AngleEmbedding(M0)───────────╰StronglyEntanglingLayers(M1)─┤  <Z>\n"
        f"\n"
        f"M0 = \n{x}\n"
        f"M1 = \n{m1}"
    )

    assert actual == expected


@pytest.mark.torch
def test_draw_mpl():
    """Test that a TorchLayer can be drawn using qml.draw_mpl"""

    dev = qml.device("default.qubit", wires=2)
    weight_shapes = {"w1": 1, "w2": (3, 2, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w1, w2):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        qml.templates.StronglyEntanglingLayers(w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    qlayer = TorchLayer(circuit, weight_shapes)

    # make the rotation angle positive to prevent the extra minus sign messing up the alignment
    qlayer.qnode_weights["w1"] = torch.abs(qlayer.qnode_weights["w1"])

    batch_size = 5
    x = torch.Tensor(np.random.uniform(0, 1, (batch_size, 2)))

    _, ax = qml.draw_mpl(qlayer)(x)

    assert len(ax.patches) == 9  # 3 boxes, 3 patches for each measure
    assert len(ax.lines) == 2  # 2 wires
    assert len(ax.texts) == 5  # 2 wire labels, 3 box labels

    assert ax.texts[0].get_text() == "0"
    assert ax.texts[1].get_text() == "1"
    assert ax.texts[2].get_text() == "AngleEmbedding"
    assert ax.texts[3].get_text() == "RX"
    assert ax.texts[4].get_text() == "StronglyEntanglingLayers"


@pytest.mark.torch
def test_specs():
    """Test that the qml.specs transform works for TorchLayer"""

    dev = qml.device("default.qubit", wires=3)
    weight_shapes = {"w1": 1, "w2": (3, 2, 3)}

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w1, w2):
        qml.templates.AngleEmbedding(inputs, wires=[0, 1])
        qml.RX(w1, wires=0)
        qml.templates.StronglyEntanglingLayers(w2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    qlayer = TorchLayer(circuit, weight_shapes)

    batch_size = 5
    x = torch.Tensor(np.random.uniform(0, 1, (batch_size, 2)))

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
    assert info["interface"] == "torch"
    assert info["device_name"] == "default.qubit"
