# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import pennylane as qml
from pennylane import numpy as np

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("cuda not available")

def test_torch_device_cuda_if_tensors_on_cuda():
    """Test that if any tensor passed to operators is on the GPU then CUDA
    is set internally as a device option for 'default.qubit.torch'."""

    n_qubits = 3
    n_layers = 1

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    x = torch.rand((5, n_qubits), dtype=torch.float64).to(torch.device("cuda"))
    res = qlayer(x)
    assert circuit.device.short_name == "default.qubit.torch"
    assert circuit.device._torch_device == "cuda"
    assert res.is_cuda

    loss = torch.sum(res).squeeze()
    loss.backward()
    assert loss.is_cuda

def indices_up_to(n_max):
    """Returns an iterator over the number of qubits and output dimension, up to value n_max.
    The output dimension never exceeds the number of qubits."""
    a, b = np.tril_indices(n_max)
    return zip(*[a + 1, b + 1])

@pytest.mark.parametrize("n_qubits, output_dim", indices_up_to(1))
def test_cuda_backward(n_qubits, output_dim):
    """Test if TorchLayer can be run on GPU"""

    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    n_layers = 1
    weight_shapes = {"weights": (n_layers, n_qubits)}

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    x = torch.rand((5, n_qubits), dtype=torch.float64).to(torch.device("cuda"))
    res = qlayer(x)
    assert res.is_cuda

    loss = torch.sum(res).squeeze()
    loss.backward()
    assert loss.is_cuda
