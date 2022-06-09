# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for differentiable quantum entropies.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


def expected_fidelity_rx_pauliz(param):
    """
    Return the analytical fidelity for the RX and PauliZ.
    """
    return (np.cos(param / 2)) ** 2


def expected_grad_fidelity_rx_pauliz(param):
    """
    Return the analytical fidelity for the RX and PauliZ.
    """
    return -np.sin(param) / 2


class TestFidelityQnode:
    """Tests for Fidelity function between two QNodes ."""

    devices = ["default.qubit", "default.mixed"]

    @pytest.mark.parametrize("device", devices)
    def test_not_same_number_wires(self, device):
        """Test that wires must have the same length"""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0():
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        with pytest.raises(
            qml.QuantumFunctionError, match="The two states must have the same number of wires"
        ):
            qml.qinfo.fidelity(circuit0, circuit1, wires0=[0, 1], wires1=[0])()

    @pytest.mark.parametrize("device", devices)
    def test_fidelity_qnodes_rx(self, device):
        """Test the fidelity between two Qnodes."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1(y):
            qml.RX(y, wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((0.1), (0.1))
        assert qml.math.allclose(fid, 1.0)

    @pytest.mark.parametrize("device", devices)
    def test_fidelity_qnodes_rxrz_ry(self, device):
        """Test the fidelity between two Qnodes."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1(y):
            qml.RY(y, wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((0.0, 0.2), (0.2))
        assert qml.math.allclose(fid, 1.0)

    @pytest.mark.parametrize("device", devices)
    def test_fidelity_qnodes_rx_empty(self, device):
        """Test the fidelity between two Qnodes."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((np.pi))
        assert qml.math.allclose(fid, 0.0)

    @pytest.mark.parametrize("device", devices)
    def test_fidelity_qnodes_rx_empty(self, device):
        """Test the fidelity between two Qnodes."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((np.pi))
        assert qml.math.allclose(fid, 0.0)

    @pytest.mark.parametrize("device", devices)
    def test_fidelity_qnodes_rx_empty(self, device):
        """Test the fidelity between two Qnodes."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0():
            return qml.state()

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])(signature1=(np.pi))
        assert qml.math.allclose(fid, 0.0)

    @pytest.mark.parametrize("device", devices)
    def test_fidelity_qnodes_rxrz_rxry(self, device):
        """Test the fidelity between two Qnodes."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])(
            (0.0, np.pi), (0.0, 0.0)
        )

        assert qml.math.allclose(fid, 1.0)

    parameters = np.linspace(0, 2 * np.pi, 50)
    wires = [1, 2]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rx_pauliz(self, device, param, wire):
        """Test the fidelity between Rx and PauliZ circuits."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((param))
        expected_fid = expected_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid, expected_fid)

    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rxrz_pauliz_grad(self, param, wire):
        """Test the gradient of the fidelity between Rx and PauliZ circuits."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid_grad = qml.grad(qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (qml.numpy.array(param, requires_grad=True))
        )
        expected_fid = expected_grad_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid_grad, expected_fid)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rx_pauliz_torch(self, device, param, wire):
        """Test the fidelity between Rx and PauliZ circuits with Torch."""
        import torch

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface="torch")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="torch")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((torch.tensor(param)))
        expected_fid = expected_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid, expected_fid)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rx_pauliz_torch_grad(self, device, param, wire):
        """Test the gradient of fidelity between Rx and PauliZ circuits with Torch."""
        import torch

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        expected_fid_grad = expected_grad_fidelity_rx_pauliz(param)
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])((param))
        fid.backward()
        fid_grad = param.grad

        assert qml.math.allclose(fid_grad, expected_fid_grad)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rx_pauliz_jax(self, device, param, wire):
        """Test the fidelity between Rx and PauliZ circuits with Jax."""
        import jax

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface="jax")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid = qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0])(
            (jax.numpy.array(param))
        )
        expected_fid = expected_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid, expected_fid, rtol=1e-03, atol=1e-04)

    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rx_pauliz_jax_jit(self, param, wire):
        """Test the fidelity between Rx and PauliZ circuits with Jax jit."""
        import jax

        # TODO: add default mixed
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="jax-jit")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid = jax.jit(qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (jax.numpy.array(param))
        )
        expected_fid = expected_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid, expected_fid, rtol=1e-03, atol=1e-04)

    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_fidelity_qnodes_rx_pauliz_jax_grad(self, param, wire):
        """Test the gradient of the fidelity between Rx and PauliZ circuits."""
        import jax

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="jax")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid_grad = jax.grad(qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (jax.numpy.array(param))
        )
        expected_fid = expected_grad_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid_grad, expected_fid, rtol=1e-04, atol=1e-03)

    @pytest.mark.parametrize("param", parameters)
    def test_fidelity_qnodes_rx_pauliz_jax_grad_jit(self, param):
        """Test the gradient of the fidelity between Rx and PauliZ circuits."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        fid_grad = jax.jit(
            jax.grad(qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0]))
        )((jax.numpy.array(param)))
        expected_fid = expected_grad_fidelity_rx_pauliz(param)
        assert qml.math.allclose(fid_grad, expected_fid, rtol=1e-04, atol=1e-03)
