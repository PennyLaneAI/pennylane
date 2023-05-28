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
"""Unit tests for differentiable trace distance transform."""

import pytest

import pennylane as qml
from pennylane import numpy as np


def expected_trace_distance_rx_pauliz(param):
    """Return the analytical trace distance for the RX and PauliZ."""
    return np.abs(np.sin(param / 2))


def expected_grad_trace_distance_rx_pauliz(param):
    """Return the analytical gradient of the trace distance for the RX and PauliZ."""
    return np.sign(np.sin(param / 2)) * np.cos(param / 2) / 2


class TestTraceDistanceQnode:
    """Tests for Fidelity function between two QNodes."""

    devices = ["default.qubit", "lightning.qubit", "default.mixed"]

    @pytest.mark.parametrize("device", devices)
    def test_not_same_number_wires(self, device):
        """Test that wires must have the same length."""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit0():
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        with pytest.raises(
            qml.QuantumFunctionError, match="The two states must have the same number of wires"
        ):
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0, 1], wires1=[0])()

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_qnodes_rxs(self, device):
        """Test the trace distance between two Rx circuits"""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1(y):
            qml.RX(y, wires=0)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((0.1), (0.1))
        assert qml.math.allclose(td, 0)

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_qnodes_rxrz_ry(self, device):
        """Test the trace distance between two circuit Rx Rz and Ry."""
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

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((0.0, 0.2), (0.2))
        assert qml.math.allclose(td, 0)

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_qnodes_rx_state(self, device):
        """Test the trace distance between RX and state(1,0)."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((np.pi))
        assert qml.math.allclose(td, 1.0)

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_qnodes_state_rx(self, device):
        """Test the trace distance between state (1,0) and RX circuit."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0():
            return qml.state()

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            all_args1=(np.pi,)
        )
        assert qml.math.allclose(td, 1.0)

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_qnodes_rxrz_rxry(self, device):
        """Test the trace_distance between two circuit with multiple arguments."""
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

        td_args = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            (0.0, np.pi), (0.0, 0.0)
        )
        td_arg_kwarg = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            (0.0, {"y": np.pi}), (0.0, {"y": 0})
        )
        td_kwargs = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            ({"x": 0, "y": np.pi}), ({"x": 0, "y": 0})
        )

        assert qml.math.allclose(td_args, 0.0)
        assert qml.math.allclose(td_arg_kwarg, 0.0)
        assert qml.math.allclose(td_kwargs, 0.0)

    # 0 and 2 * pi are removed to avoid nan values in the gradient
    parameters = np.linspace(0, 2 * np.pi, 20)[1:-1]
    wires = [1, 2]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz(self, device, param, wire):
        """Test the trace distance between Rx and PauliZ circuits."""
        dev = qml.device(device, wires=wire)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((param))
        expected_td = expected_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td, expected_td)

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz_grad(self, param, wire):
        """Test the gradient of the trace distance between Rx and PauliZ circuits."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td_grad = qml.grad(qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (qml.numpy.array(param, requires_grad=True))
        )
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td)

    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_pauliz_rx_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between PauliZ and Rx circuits."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit0():
            qml.PauliZ(wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        td_grad = qml.grad(qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]))(
            None, (qml.numpy.array(param, requires_grad=True))
        )
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td)

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_tworx_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between two trainable circuits."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        td_grad = qml.grad(qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0]))(
            (qml.numpy.array(param, requires_grad=True)),
            (qml.numpy.array(2 * param, requires_grad=True)),
        )
        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        assert qml.math.allclose(td_grad, expected_td)

    interfaces = ["torch"]

    @pytest.mark.torch
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_torch(self, device, param, wire, interface):
        """Test the trace distance between Rx and PauliZ circuits with Torch."""
        import torch

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="torch")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            (torch.tensor(param))
        )
        expected_td = expected_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td, expected_td)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_torch_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Torch."""
        import torch

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((param))
        td.backward()
        td_grad = param.grad

        assert qml.math.allclose(td_grad, expected_td_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_pauliz_rx_torch_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between PauliZ and Rx circuits with Torch."""
        import torch

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit0():
            qml.PauliZ(wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(None, (param))
        td.backward()
        td_grad = param.grad

        assert qml.math.allclose(td_grad, expected_td)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_tworx_torch_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between two trainable circuits with Torch."""
        import torch

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        params = (
            torch.tensor(param, dtype=torch.float64, requires_grad=True),
            torch.tensor(2 * param, dtype=torch.float64, requires_grad=True),
        )
        td = qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0])(*params)
        td.backward()
        td_grad = [p.grad for p in params]
        assert qml.math.allclose(td_grad, expected_td)

    interfaces = ["tf"]

    @pytest.mark.tf
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_tf(self, device, param, wire, interface):
        """Test the trace distance between Rx and PauliZ circuits with Tensorflow."""
        import tensorflow as tf

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="tf")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            (tf.Variable(param))
        )
        expected_td = expected_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td, expected_td)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_tf_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Tensorflow."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="tf")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        expected_grad_td = expected_grad_trace_distance_rx_pauliz(param)
        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((param))

        td_grad = tape.gradient(entropy, param)
        assert qml.math.allclose(td_grad, expected_grad_td)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_pauliz_rx_tf_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between PauliZ and Rx circuits with Tensorflow."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="tf")
        def circuit0():
            qml.PauliZ(wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
                None, (param)
            )

        td_grad = tape.gradient(entropy, param)
        assert qml.math.allclose(td_grad, expected_td)

    interfaces = ["jax"]

    @pytest.mark.jax
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_jax(self, device, param, wire, interface):
        """Test the trace distance between Rx and PauliZ circuits with Jax."""
        import jax

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
            (jax.numpy.array(param))
        )
        expected_td = expected_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td, expected_td, rtol=1e-03, atol=1e-04)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_jax_jit(self, param, wire):
        """Test the trace distance between Rx and PauliZ circuits with Jax jit."""
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

        td = jax.jit(qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (jax.numpy.array(param))
        )
        expected_td = expected_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td, expected_td, rtol=1e-03, atol=1e-04)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_jax_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits."""
        import jax

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td_grad = jax.grad(qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (jax.numpy.array(param))
        )
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_jax_grad_jit(self, param, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

        td_grad = jax.jit(
            jax.grad(qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]))
        )((jax.numpy.array(param)))
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_pauliz_rx_jax_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between PauliZ and Rx circuits with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="jax")
        def circuit0():
            qml.PauliZ(wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]), argnums=1
        )(None, (jax.numpy.array(param)))
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_pauliz_rx_jax_grad_jit(self, param, interface):
        """Test the gradient of the trace distance between PauliZ and Rx circuits with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit0():
            qml.PauliZ(wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.state()

        td_grad = jax.jit(
            jax.grad(
                qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]), argnums=1
            )
        )(None, (jax.numpy.array(param)))
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_tworx_jax_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between two trainable circuits with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0]), argnums=[0, 1]
        )(
            (jax.numpy.array(param)),
            (jax.numpy.array(2 * param)),
        )
        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_tworx_jax_grad_jit(self, param, interface):
        """Test the gradient of the trace distance between two trainable circuits with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        td_grad = jax.jit(
            jax.grad(
                qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0]), argnums=[0, 1]
            )
        )((jax.numpy.array(param)), (jax.numpy.array(2 * param)))
        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_grad_two_params(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with two parameters."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.PauliZ(wires=0)
            return qml.state()

        td_grad = qml.grad(qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]))(
            (qml.numpy.array(param, requires_grad=True)), (qml.numpy.array(2.0, requires_grad=True))
        )
        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0))

    interfaces = ["torch"]

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_torch_grad_two_params(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Torch and two parameters."""
        import torch

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.PauliZ(wires=0)
            return qml.state()

        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        param2 = torch.tensor(0, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])((param), (param2))
        td.backward()
        td_grad = (param.grad, param2.grad)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0))

    interfaces = ["tf"]

    @pytest.mark.tf
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_tf_grad_two_params(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Tensorflow and two parameters."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.PauliZ(wires=0)
            return qml.state()

        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)

        param1 = tf.Variable(param)
        params2 = tf.Variable(0.0)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(
                (param1), (params2)
            )

        td_grad = tape.gradient(entropy, [param1, params2])
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0))

    interfaces = ["jax"]

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_jax_grad_two_params(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Jax and two params."""
        import jax

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.PauliZ(wires=0)
            return qml.state()

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]), argnums=[0, 1]
        )((jax.numpy.array(param)), (jax.numpy.array(2.0)))
        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0), rtol=1e-03, atol=1e-04)

    interfaces = ["jax"]

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_jax_jit_grad_two_params(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Jax Jit and two params."""
        import jax

        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.PauliZ(wires=0)
            return qml.state()

        td_grad = jax.jit(
            jax.grad(
                qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0]), argnums=[0, 1]
            )
        )((jax.numpy.array(param)), (jax.numpy.array(2.0)))
        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0), rtol=1e-03, atol=1e-04)
