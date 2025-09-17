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
"""
Unit tests for PyTorch GPU support.
"""
# pylint: disable=protected-access
import pytest

import pennylane as qml

pytestmark_gpu = pytest.mark.gpu
pytestmark_torch = pytest.mark.torch

torch = pytest.importorskip("torch")
torch_cuda = pytest.importorskip("torch.cuda")


@pytest.mark.skipif(not torch_cuda.is_available(), reason="no cuda support")
class TestTorchGPUDevice:
    """Test GPU with cuda for Torch device."""

    def test_device_to_cuda(self):
        """Checks device executes with cuda is input data is cuda"""

        dev = qml.device("default.qubit", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)

        assert res.is_cuda

        res.backward()
        assert x.grad.is_cuda

    def test_mixed_devices(self):
        """Asserts works with both cuda and cpu input data"""

        dev = qml.device("default.qubit", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))
        y = torch.tensor(0.2, requires_grad=True, device=torch.device("cpu"))

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)

        assert res.is_cuda

        res.backward()
        assert x.grad.is_cuda
        # check that computing the gradient with respect to y works
        _ = y.grad

    def test_matrix_input(self):
        """Test goes to GPU for matrix valued inputs."""

        dev = qml.device("default.qubit", wires=1)

        U = torch.eye(2, requires_grad=False, device=torch.device("cuda"))

        with qml.queuing.AnnotatedQueue() as q:
            qml.QubitUnitary(U, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)
        assert res.is_cuda

    def test_resets(self):
        """Asserts reverts to cpu after execution on gpu"""

        dev = qml.device("default.qubit", wires=1)

        x = torch.tensor(0.1, requires_grad=True, device=torch.device("cuda"))
        y = torch.tensor(0.2, requires_grad=True, device=torch.device("cpu"))

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        res1 = dev.execute(tape1)
        assert res1.is_cuda

        with qml.queuing.AnnotatedQueue() as q2:
            qml.RY(y, wires=0)
            qml.expval(qml.PauliZ(0))

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        res2 = dev.execute(tape2)
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

        res.backward()
        assert x.grad.is_cuda

    @pytest.mark.parametrize("par_device", ["cuda", "cpu"])
    def test_matrix_conversion(self, par_device):
        """Test that the matrix conversion functionality of the QNode works with
        data on the host or the device.
        """
        dev = qml.device("default.qubit")
        p = torch.tensor(0.543, dtype=torch.float64, device=par_device)
        U_in = qml.matrix(qml.RZ(0.9, 0))

        @qml.qnode(dev)
        def circuit(x):
            qml.QubitUnitary(U_in, wires=0)
            qml.Hadamard(0)
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        U = qml.matrix(circuit, wire_order=[0])(p)
        assert U.device.type == par_device

    def test_amplitude_embedding(self):
        """Test that the padding capability of amplitude embedding works with
        GPUs."""
        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="torch")
        def circuit_cuda(inputs):
            qml.AmplitudeEmbedding(inputs, wires=range(n_wires), pad_with=0, normalize=True)
            return qml.expval(qml.PauliZ(0))

        inputs = torch.rand(n_wires)  # embedding shorter than 2 ** n_wires
        res1 = circuit_cuda(inputs)
        assert not res1.is_cuda

        inputs = inputs.to(torch.device("cuda"))  # move to GPU
        res2 = circuit_cuda(inputs)

        assert res2.is_cuda


@pytest.mark.skipif(not torch_cuda.is_available(), reason="no cuda support")
def test_qnn_torchlayer():
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


@pytest.mark.skipif(
    not torch_cuda.is_available() or torch.cuda.device_count() < 2,
    reason="a multi-gpu device is required",
)
class TestTorchMultiGPUDevice:
    """Test Multi-GPU with cuda for Torch device."""

    def test_multi_gpu_outval(self):
        """Test that the output value of a multi-GPU QNode is on the correct device."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return qml.state()

        params0 = torch.randn(2, dtype=torch.float64, device="cuda:0")
        res0 = circuit(params0)
        assert res0.device == torch.device("cuda:0")

        param1 = torch.randn(2, dtype=torch.float64, device="cuda:1")
        res1 = circuit(param1)
        assert res1.device == torch.device("cuda:1")

    def test_multi_gpu_concat(self):
        """Test that the concatenation of multi-GPU QNode outputs works correctly."""

        x = torch.tensor([1, 2, 3], device="cuda:1")
        y = torch.tensor([4, 5], device="cuda:1")

        res = qml.math.concatenate([x, y], axis=0)
        assert res.device == torch.device("cuda:1")

    def test_dist_tensors_concat(self):
        """Test that concatenation of distributed tensors works correctly."""

        x = torch.tensor([1, 2, 3], device="cuda:0")
        y = torch.tensor([4, 5], device="cuda:1")

        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            qml.math.concatenate([x, y], axis=0)
