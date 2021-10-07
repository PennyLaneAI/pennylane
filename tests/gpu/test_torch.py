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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda support")
class TestTorchDevice:
    def test_device_to_cuda(self):
        """Checks device executes with cuda is input data is cuda"""

        dev = qml.device("default.qubit.torch", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.expval(qml.PauliX(0))

        res = dev.execute(tape)

        assert res.is_cuda
        assert dev._torch_device == "cuda"

        res.backward()
        assert x.grad.is_cuda

    def test_mixed_devices(self):
        """Asserts works with both cuda and cpu input data"""

        dev = qml.device("default.qubit.torch", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))
        y = torch.tensor(0.2, requires_grad=True, device=torch.device("cpu"))

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.expval(qml.PauliX(0))

        res = dev.execute(tape)

        assert res.is_cuda
        assert dev._torch_device == "cuda"

        res.backward()
        assert x.grad.is_cuda
        # check that this works
        ygrad = y.grad

    def test_matrix_input(self):
        """Test goes to GPU for matrix valued inputs."""

        dev = qml.device("default.qubit.torch", wires=1)

        U = torch.eye(2, requires_grad=False, device=torch.device("cuda"))

        with qml.tape.QuantumTape() as tape:
            qml.QubitUnitary(U, wires=0)
            qml.expval(qml.PauliZ(0))

        res = dev.execute(tape)
        assert res.is_cuda
        assert dev._torch_device == "cuda"

    def test_resets(self):
        """Asserts reverts to cpu after execution on gpu"""

        dev = qml.device("default.qubit.torch", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))
        y = torch.tensor(0.2, requires_grad=True, device=torch.device("cpu"))

        with qml.tape.QuantumTape() as tape1:
            qml.RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        res1 = dev.execute(tape1)
        assert dev._torch_device == "cuda"
        assert res1.is_cuda

        with qml.tape.QuantumTape() as tape2:
            qml.RY(y, wires=0)
            qml.expval(qml.PauliZ(0))

        res2 = dev.execute(tape2)
        assert dev._torch_device == "cpu"
        assert not res2.is_cuda

    def test_integration(self):
        """Test cuda supported when device created in qnode creation."""

        dev = qml.device("default.qubit", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))
        y = torch.tensor(0.2, requires_grad=True)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circ(x, y)
        assert res.is_cuda

        assert circ.device._torch_device == "cuda"
        res.backward()
        assert x.grad.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda support")
class TestqnnTorchLayer:
    def test_torch_device_cuda_if_tensors_on_cuda(self):
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

    def test_qnn_torchlayer(self):
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
