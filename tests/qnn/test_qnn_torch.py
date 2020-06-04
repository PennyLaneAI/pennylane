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
Tests for the pennylane.qnn.torch module.
"""
from unittest import mock
import math
import numpy as np
import pytest

import pennylane as qml
from pennylane.qnn.torch import TorchLayer

torch = pytest.importorskip("torch")


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

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clayer1 = torch.nn.Linear(n_qubits, n_qubits)
            self.clayer2 = torch.nn.Linear(output_dim, n_qubits)
            self.clayer3 = torch.nn.Linear(output_dim, output_dim)
            self.qlayer1 = TorchLayer(c, w)
            self.qlayer2 = TorchLayer(c, w)

        def forward(self, x):
            x = self.clayer1(x)
            x = self.qlayer1(x)
            x = self.clayer2(x)
            x = self.qlayer2(x)
            x = self.clayer3(x)
            return x

    return Net()


ordered_weights = ["w{}".format(i) for i in range(1, 8)]  # we do this for Python 3.5


@pytest.mark.parametrize("interface", ["torch"])  # required for the get_circuit fixture
@pytest.mark.usefixtures("get_circuit")  # this fixture is in tests/qnn/conftest.py
class TestTorchLayer:
    """Unit tests for the pennylane.qnn.torch.TorchLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_torch(self, get_circuit, monkeypatch):
        """Test if an ImportError is raised when instantiated without PyTorch"""
        c, w = get_circuit
        with monkeypatch.context() as m:
            m.setattr(qml.qnn.torch, "TORCH_IMPORTED", False)
            with pytest.raises(ImportError, match="TorchLayer requires PyTorch"):
                TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_input(self, get_circuit):
        """Test if a TypeError is raised when instantiated with a QNode that does not have an
        argument with name equal to the input_arg class attribute of TorchLayer"""
        c, w = get_circuit
        del c.func.sig[qml.qnn.torch.TorchLayer._input_arg]
        with pytest.raises(TypeError, match="QNode must include an argument with name"):
            TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_input_in_weight_shapes(self, get_circuit, n_qubits):
        """Test if a ValueError is raised when instantiated with a weight_shapes dictionary that
        contains the shape of the input argument given by the input_arg class attribute of
        TorchLayer"""
        c, w = get_circuit
        w[qml.qnn.torch.TorchLayer._input_arg] = n_qubits
        with pytest.raises(
            ValueError,
            match="{} argument should not have its dimension".format(
                qml.qnn.torch.TorchLayer._input_arg
            ),
        ):
            TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_weight_shape_unspecified(self, get_circuit):
        """Test if a ValueError is raised when instantiated with a weight missing from the
        weight_shapes dictionary"""
        c, w = get_circuit
        del w["w1"]
        with pytest.raises(ValueError, match="Must specify a shape for every non-input parameter"):
            TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_pos(self, get_circuit, monkeypatch):
        """Test if a TypeError is raised when instantiated with a variable number of positional
        arguments"""
        c, w = get_circuit

        class FuncPatch:
            """Patch for variable number of keyword arguments"""

            sig = c.func.sig
            var_pos = True
            var_keyword = False

        with monkeypatch.context() as m:
            m.setattr(c, "func", FuncPatch)

            with pytest.raises(TypeError, match="Cannot have a variable number of positional"):
                TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_var_keyword(self, get_circuit, monkeypatch):
        """Test if a TypeError is raised when instantiated with a variable number of keyword
        arguments"""
        c, w = get_circuit

        class FuncPatch:
            """Patch for variable number of keyword arguments"""

            sig = c.func.sig
            var_pos = False
            var_keyword = True

        with monkeypatch.context() as m:
            m.setattr(c, "func", FuncPatch)

            with pytest.raises(TypeError, match="Cannot have a variable number of keyword"):
                TorchLayer(c, w)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_nonspecified_init(self, get_circuit, n_qubits, monkeypatch):
        """Test if weights are initialized according to the uniform distribution in [0, 2 pi]"""
        c, w = get_circuit

        uniform_ = mock.MagicMock(return_value=torch.Tensor(1))
        with monkeypatch.context() as m:
            m.setattr(torch.nn.init, "uniform_", uniform_)
            TorchLayer(c, w)
            kwargs = uniform_.call_args[1]
            assert kwargs["b"] == 2 * math.pi

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_non_input_defaults(self, get_circuit, n_qubits):
        """Test if a TypeError is raised when default arguments that are not the input argument are
        present in the QNode"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch")
        def c_dummy(inputs, w1, w2, w3, w4, w5, w6, w7, w8=None):
            """Dummy version of the circuit with a default argument"""
            return c(inputs, w1, w2, w3, w4, w5, w6, w7)

        with pytest.raises(
            TypeError,
            match="Only the argument {} is permitted".format(qml.qnn.torch.TorchLayer._input_arg),
        ):
            TorchLayer(c_dummy, {**w, **{"w8": 1}})

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights_shapes(self, get_circuit, n_qubits):
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
    def test_qnode_weights_registered(self, get_circuit, n_qubits):
        """Test if the weights in qnode_weights are registered to the internal _parameters
        dictionary and that requires_grad == True"""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        for name, weight in layer.qnode_weights.items():
            assert torch.allclose(weight, layer._parameters[name])
            assert weight.requires_grad

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_evaluate_qnode(self, get_circuit, n_qubits):
        """Test if the _evaluate_qnode() method works correctly, i.e., that it gives the same
        result as calling the QNode directly"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        layer_out = layer._evaluate_qnode(x).detach().numpy()

        weights = [layer.qnode_weights[weight].detach().numpy() for weight in ordered_weights]

        circuit_out = c(x, *weights)
        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_evaluate_qnode_shuffled_args(self, get_circuit, output_dim, n_qubits):
        """Test if the _evaluate_qnode() method works correctly when the inputs argument is not the
        first positional argument, i.e., that it gives the same result as calling the QNode
        directly"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch")
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

        layer = TorchLayer(c_shuffled, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer._evaluate_qnode(x).detach().numpy()

        weights = [layer.qnode_weights[weight].detach().numpy() for weight in ordered_weights]

        circuit_out = c(x, *weights)
        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_evaluate_qnode_default_input(self, get_circuit, output_dim, n_qubits):
        """Test if the _evaluate_qnode() method works correctly when the inputs argument is a
        default argument, i.e., that it gives the same result as calling the QNode directly"""
        c, w = get_circuit

        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch")
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

        layer = TorchLayer(c_default, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer._evaluate_qnode(x).detach().numpy()

        weights = [layer.qnode_weights[weight].detach().numpy() for weight in ordered_weights]

        circuit_out = c(x, *weights)
        assert np.allclose(layer_out, circuit_out)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_forward_single_input(self, get_circuit, output_dim, n_qubits):
        """Test if the forward() method accepts a single input (i.e., not with an extra batch
        dimension) and returns a tensor of the right shape"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer.forward(x)
        assert layer_out.shape == torch.Size((output_dim,))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_forward(self, get_circuit, output_dim, n_qubits):
        """Test if the forward() method accepts a batched input and returns a tensor of the right
        shape"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.Tensor(np.ones((2, n_qubits)))

        layer_out = layer.forward(x)
        assert layer_out.shape == torch.Size((2, output_dim))

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_str_repr(self, get_circuit):
        """Test the __str__ and __repr__ representations"""
        c, w = get_circuit
        layer = TorchLayer(c, w)

        assert layer.__str__() == "<Quantum Torch Layer: func=circuit>"
        assert layer.__repr__() == "<Quantum Torch Layer: func=circuit>"

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_gradients(self, get_circuit, n_qubits):
        """Test if the gradients of the TorchLayer are equal to the gradients of the circuit when
        taken with respect to the trainable variables"""
        c, w = get_circuit
        layer = TorchLayer(c, w)
        x = torch.ones(n_qubits)

        weights = [layer.qnode_weights[weight] for weight in ordered_weights]

        out_layer = layer(x)
        out_layer.backward()

        g_layer = [w.grad.numpy() for w in weights]

        out_circuit = c(x, *weights)
        out_circuit.backward()

        g_circuit = [w.grad.numpy() for w in weights]

        for g1, g2 in zip(g_layer, g_circuit):
            assert np.allclose(g1, g2)
        assert len(weights) == len(list(layer.parameters()))


@pytest.mark.parametrize("interface", qml.qnodes.decorator.ALLOWED_INTERFACES)
@pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
@pytest.mark.usefixtures("get_circuit")  # this fixture is in tests/qnn/conftest.py
@pytest.mark.usefixtures("skip_if_no_tf_support")
def test_interface_conversion(get_circuit, skip_if_no_tf_support):
    """Test if input QNodes with all types of interface are converted internally to the PyTorch
    interface"""
    c, w = get_circuit
    layer = TorchLayer(c, w)
    assert layer.qnode.interface == "torch"


@pytest.mark.parametrize("interface", ["torch"])
@pytest.mark.usefixtures("get_circuit", "module")
class TestTorchLayerIntegration:
    """Integration tests for the pennylane.qnn.torch.TorchLayer class."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    @pytest.mark.parametrize("batch_size", [2])
    def test_step_module(self, module, batch_size, n_qubits, output_dim, dtype):
        """Test if a module that includes TorchLayers can perform one optimization step. This
        test checks that some of the parameters in the module are different after one step.
        The module is composed of two TorchLayers sandwiched between Linear neural network layers,
        and the dataset is simply input and output vectors of zeros."""
        module = module.type(dtype)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.5)

        x = torch.zeros((batch_size, n_qubits)).type(dtype)
        y = torch.zeros((batch_size, output_dim)).type(dtype)

        params_before = [w.detach().numpy().copy() for w in list(module.parameters())]

        module_out = module(x)
        optimizer.zero_grad()
        loss = loss_func(module_out, y)
        loss.backward()
        optimizer.step()

        params_after = [w.detach().numpy().copy() for w in list(module.parameters())]

        params_similar = [np.allclose(p1, p2) for p1, p2 in zip(params_before, params_after)]
        assert not all(params_similar)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_module_gradients(self, module, output_dim, n_qubits, get_circuit):
        """Test if a gradient can be calculated with respect to all of the trainable variables in
        the module"""
        c, w = get_circuit

        x = torch.zeros((2, n_qubits))
        y = torch.zeros((2, output_dim))

        module_out = module(x)
        loss_func = torch.nn.MSELoss()
        loss = loss_func(module_out, y)
        loss.backward()

        gradients = [w.grad for w in module.parameters()]
        assert all([g.is_floating_point() for g in gradients])
        assert len(gradients) == 2 * len(w) + 6  # six parameters come from classical layers

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_module_state_dict(self, module, n_qubits, get_circuit):
        """Test if the state dictionary output by the module contains all the expected trainable
        parameters"""
        c, w = get_circuit

        state_dict = module.state_dict()
        dict_keys = set(state_dict.keys())

        clayer_weights = set("clayer{}.weight".format(i + 1) for i in range(3))
        clayer_biases = set("clayer{}.bias".format(i + 1) for i in range(3))
        qlayer_params = set(
            "qlayer{}.w{}".format(i + 1, j + 1) for i in range(2) for j in range(len(w))
        )

        all_params = clayer_weights | clayer_biases | qlayer_params

        assert dict_keys == all_params
        assert len(dict_keys) == len(all_params)
