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


@pytest.mark.parametrize("interface", ["torch"])  # required for the get_circuit fixture
@pytest.mark.usefixtures("get_circuit")
class TestTorchLayer:
    """Unit tests for the pennylane.qnn.torch.TorchLayer class."""

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_torch(self, get_circuit, output_dim, monkeypatch):
        """Test if an ImportError is raised when instantiated with an incorrect version of
        TensorFlow"""
        c, w = get_circuit
        with monkeypatch.context() as m:
            m.setattr(qml.qnn.torch, "TORCH_IMPORTED", False)
            with pytest.raises(ImportError, match="TorchLayer requires PyTorch"):
                TorchLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_no_input(self, get_circuit, output_dim):
        """Test if a TypeError is raised when instantiated with a QNode that does not have an
        argument with name equal to the input_arg class attribute of TorchLayer"""
        c, w = get_circuit
        del c.func.sig[qml.qnn.torch.TorchLayer._input_arg]
        with pytest.raises(TypeError, match="QNode must include an argument with name"):
            TorchLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_input_in_weight_shapes(self, get_circuit, n_qubits, output_dim):
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
            TorchLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_weight_shape_unspecified(self, get_circuit, output_dim):
        """Test if a ValueError is raised when instantiated with a weight missing from the
        weight_shapes dictionary"""
        c, w = get_circuit
        del w["w1"]
        with pytest.raises(ValueError, match="Must specify a shape for every non-input parameter"):
            TorchLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
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
            m.setattr(c, "func", FuncPatch)

            with pytest.raises(TypeError, match="Cannot have a variable number of positional"):
                TorchLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
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
            m.setattr(c, "func", FuncPatch)

            with pytest.raises(TypeError, match="Cannot have a variable number of keyword"):
                TorchLayer(c, w, output_dim)

    @pytest.mark.parametrize("n_qubits", [1])
    @pytest.mark.parametrize("output_dim", zip(*[[[1], (1,), 1], [1, 1, 1]]))
    def test_output_dim(self, get_circuit, output_dim):
        """Test if the output_dim is correctly processed, i.e., that an iterable is mapped to
        its first element while an int is left unchanged"""
        c, w = get_circuit
        layer = TorchLayer(c, w, output_dim[0])
        assert layer.output_dim == output_dim[1]

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_nonspecified_init(self, get_circuit, output_dim, n_qubits, monkeypatch):
        """Test if weights are initialized according to the uniform distribution in [0, 2 pi]"""
        c, w = get_circuit

        uniform_ = mock.MagicMock(return_value=torch.Tensor(1))
        with monkeypatch.context() as m:
            m.setattr(torch.nn.init, "uniform_", uniform_)
            TorchLayer(c, w, output_dim)
            kwargs = uniform_.call_args[1]
            assert kwargs["b"] == 2 * math.pi

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    def test_non_input_defaults(self, get_circuit, output_dim, n_qubits):
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
            TorchLayer(c_dummy, {**w, **{"w8": 1}}, output_dim)

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights_shapes(self, get_circuit, output_dim, n_qubits):
        """Test if the weights in qnode_weights have the correct shape"""
        c, w = get_circuit
        layer = TorchLayer(c, w, output_dim)

        ideal_shapes = {
            "w1": torch.Size((3, n_qubits, 3)),
            "w2": torch.Size((1,)),
            "w3": torch.Size((0,)),
            "w4": torch.Size((3,)),
            "w5": torch.Size((2, n_qubits, 3)),
            "w6": torch.Size((3,)),
            "w7": torch.Size((0,)),
        }

        for name, weight in layer.qnode_weights.items():
            assert weight.shape == ideal_shapes[name]

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_qnode_weights_registered(self, get_circuit, output_dim, n_qubits):
        """Test if the weights in qnode_weights are registered to the internal _parameters
        dictionary and that requires_grad == True"""
        c, w = get_circuit
        layer = TorchLayer(c, w, output_dim)

        for name, weight in layer.qnode_weights.items():
            assert torch.allclose(weight, layer._parameters[name])
            assert weight.requires_grad

    @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    def test_evaluate_qnode(self, get_circuit, output_dim, n_qubits):
        """Test if the _evaluate_qnode() method works correctly, i.e., that it gives the same
        result as calling the QNode directly"""
        c, w = get_circuit
        layer = TorchLayer(c, w, output_dim)
        x = torch.Tensor(np.ones(n_qubits))

        layer_out = layer._evaluate_qnode(x).detach().numpy()

        weights = [w.detach().numpy() for w in layer.qnode_weights.values()]
        circuit_out = c(x, *weights)
        assert np.allclose(layer_out, circuit_out)

    # @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(2))
    # def test_forward_single_input(self, get_circuit, output_dim, n_qubits):
    #     """Test if the forward() method accepts a single input (i.e., not with an extra batch
    #     dimension) and returns a tensor of the right shape"""
    #     c, w = get_circuit
    #     layer = TorchLayer(c, w, output_dim)
    #     x = torch.Tensor(np.ones(n_qubits))
    #
    #     layer_out = layer.forward(x)
    #     assert layer_out.shape == torch.Size((output_dim,))

    # @pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
    # def test_forward(self, get_circuit, output_dim, n_qubits):
    #     c, w = get_circuit
    #     layer = TorchLayer(c, w, output_dim)
    #     x = torch.Tensor(np.ones((5, n_qubits)))
    #
    #     layer_out = layer.forward(x)
    #     # print(layer_out)
    #     # assert layer_out.shape == torch.Size((2, output_dim))

