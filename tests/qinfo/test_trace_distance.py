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

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def expected_trace_distance_rx_pauliz(param):
    """Return the analytical trace distance for the RX and PauliZ."""
    return np.abs(np.sin(param / 2))


def expected_grad_trace_distance_rx_pauliz(param):
    """Return the analytical gradient of the trace distance for the RX and PauliZ."""
    return np.sign(np.sin(param / 2)) * np.cos(param / 2) / 2


class TestTraceDistanceQnode:
    """Tests for the Trace Distance function between two QNodes."""

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
    def test_trace_distance_qnodes_rx_state(self, device):
        """Test the trace distance between RX and state (1, 0)."""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[0], wires1=[0])(np.pi, None)
        assert qml.math.allclose(td, 1.0)

        td = qml.qinfo.trace_distance(circuit1, circuit0, wires0=[0], wires1=[0])(None, np.pi)
        assert qml.math.allclose(td, 1.0)

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_qnodes_rxrz_rxry(self, device):
        """Test the trace_distance between two circuits with multiple arguments."""
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

    @pytest.mark.parametrize("device", devices)
    def test_trace_distance_wire_labels(self, device, tol):
        """Test that trace_distance is correct with custom wire labels"""
        param = np.array([0.678, 1.234])
        wires = ["a", 8]
        dev = qml.device(device, wires=wires)

        @qml.qnode(dev)
        def circuit(x):
            qml.PauliX(wires=wires[0])
            qml.IsingXX(x, wires=wires)
            return qml.state()

        td_circuit = qml.qinfo.trace_distance(circuit, circuit, [wires[0]], [wires[1]])
        actual = td_circuit((param[0],), (param[1],))

        expected = 0.5 * (
            np.abs(np.cos(param[0] / 2) ** 2 - np.sin(param[1] / 2) ** 2)
            + np.abs(np.cos(param[1] / 2) ** 2 - np.sin(param[0] / 2) ** 2)
        )
        assert np.allclose(actual, expected, atol=tol)

    # 0 and 2 * pi are removed to avoid nan values in the gradient
    parameters = np.linspace(0, 2 * np.pi, 20)[1:-1]
    wires = [1, 2]

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz_grad(self, param, wire):
        """Test the gradient of the trace distance between Rx and PauliZ circuits."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit1():
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        td_grad = qml.grad(
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1])
        )((qml.numpy.array(param, requires_grad=True)))
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td)

        td_grad = qml.grad(
            qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1])
        )(None, qml.numpy.array(param, requires_grad=True))
        assert qml.math.allclose(td_grad, expected_td)

    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_tworx_grad(self, param, wire, interface):
        """Test the gradient of the trace distance between two trainable circuits."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        td_grad = qml.grad(
            qml.qinfo.trace_distance(circuit, circuit, wires0=[wire - 1], wires1=[wire - 1])
        )(
            (qml.numpy.array(param, requires_grad=True)),
            (qml.numpy.array(2 * param, requires_grad=True)),
        )
        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        assert qml.math.allclose(td_grad, expected_td)

    interfaces_func = [
        ("auto", lambda x: x),
        ("torch", torch.tensor),
        ("tf", tf.Variable),
        ("jax", jax.numpy.array),
        ("jax-jit", jax.numpy.array),
    ]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface_func", interfaces_func)
    def test_trace_distance_qnodes_rx_pauliz(self, device, param, wire, interface_func):
        """Test the trace distance between Rx and PauliZ circuits with multiple interfaces."""
        interface, func = interface_func

        if device == "lightning.qubit" and interface == "jax-jit":
            pytest.skip("Can't use lightning.qubit device with JAX JIT.")

        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface=interface)
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit1():
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1])(
            func(param)
        )
        expected_td = expected_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td, expected_td)

        td = qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1])(
            None, func(param)
        )
        assert qml.math.allclose(td, expected_td)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz_torch_grad(self, param, wire):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Torch."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit1():
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1])(
            param
        )
        td.backward()
        td_grad = param.grad
        assert qml.math.allclose(td_grad, expected_td_grad)

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1])(
            None, param
        )
        td.backward()
        td_grad = param.grad
        assert qml.math.allclose(td_grad, expected_td_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_tworx_torch_grad(self, param, wire):
        """Test the gradient of the trace distance between two trainable circuits with Torch."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        params = (
            torch.tensor(param, dtype=torch.float64, requires_grad=True),
            torch.tensor(2 * param, dtype=torch.float64, requires_grad=True),
        )
        td = qml.qinfo.trace_distance(circuit, circuit, wires0=[wire - 1], wires1=[wire - 1])(
            *params
        )
        td.backward()
        td_grad = [p.grad for p in params]
        assert qml.math.allclose(td_grad, expected_td)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz_tf_grad(self, param, wire):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Tensorflow."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="tf")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="tf")
        def circuit1():
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        expected_grad_td = expected_grad_trace_distance_rx_pauliz(param)
        param = tf.Variable(param)

        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(
                circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1]
            )(param)

        td_grad = tape.gradient(entropy, param)
        assert qml.math.allclose(td_grad, expected_grad_td)

        param = tf.Variable(param)

        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(
                circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1]
            )(None, param)

        td_grad = tape.gradient(entropy, param)
        assert qml.math.allclose(td_grad, expected_grad_td)

    devices_jax_jit = ["default.qubit", "default.mixed"]

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("device", devices_jax_jit)
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_trace_distance_qnodes_rx_pauliz_jax_grad(self, param, wire, device, use_jit):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with JAX."""
        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface="jax" if not use_jit else "jax-jit")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="jax" if not use_jit else "jax-jit")
        def circuit1():
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1])
        )

        if use_jit:
            td_grad = jax.jit(td_grad)

        td_grad = td_grad(jax.numpy.array(param))
        expected_td = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, expected_td)

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1]),
            argnums=1,
        )

        if use_jit:
            td_grad = jax.jit(td_grad)

        td_grad = td_grad(None, jax.numpy.array(param))
        assert qml.math.allclose(td_grad, expected_td)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("device", devices_jax_jit)
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_trace_distance_qnodes_rx_tworx_jax_grad(self, param, wire, device, use_jit):
        """Test the gradient of the trace distance between two trainable circuits with JAX."""
        dev = qml.device(device, wires=wire)

        @qml.qnode(dev, interface="jax" if not use_jit else "jax-jit")
        def circuit(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit, circuit, wires0=[wire - 1], wires1=[wire - 1]),
            argnums=[0, 1],
        )

        if use_jit:
            td_grad = jax.jit(td_grad)

        td_grad = td_grad(
            (jax.numpy.array(param)),
            (jax.numpy.array(2 * param)),
        )
        expected = expected_grad_trace_distance_rx_pauliz(param)
        expected_td = [-expected, expected]
        assert qml.math.allclose(td_grad, expected_td, rtol=1e-04, atol=1e-03)

    interfaces = ["auto", "autograd"]

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("interface", interfaces)
    def test_trace_distance_qnodes_rx_pauliz_grad_two_params(self, param, wire, interface):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with
        two parameters."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit1(x):
            qml.RX(x, wires=wire - 1)
            qml.RX(-x, wires=wire - 1)
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        td_grad = qml.grad(
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1])
        )(qml.numpy.array(param, requires_grad=True), qml.numpy.array(2.0, requires_grad=True))
        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0))

        td_grad = qml.grad(
            qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1])
        )(qml.numpy.array(2.0, requires_grad=True), qml.numpy.array(param, requires_grad=True))
        assert qml.math.allclose(td_grad, (0.0, expected_td_grad))

    @pytest.mark.torch
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz_torch_grad_two_params(self, param, wire):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Torch and
        two parameters."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit1(x):
            qml.RX(x, wires=wire - 1)
            qml.RX(-x, wires=wire - 1)
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        param2 = torch.tensor(0, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1])(
            param, param2
        )
        td.backward()
        td_grad = (param.grad, param2.grad)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0))

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        param2 = torch.tensor(0, dtype=torch.float64, requires_grad=True)
        td = qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1])(
            param2, param
        )
        td.backward()
        td_grad = (param2.grad, param.grad)
        assert qml.math.allclose(td_grad, (0.0, expected_td_grad))

    @pytest.mark.tf
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    def test_trace_distance_qnodes_rx_pauliz_tf_grad_two_params(self, param, wire):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Tensorflow
        and two parameters."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="tf")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="tf")
        def circuit1(x):
            qml.RX(x, wires=wire - 1)
            qml.RX(-x, wires=wire - 1)
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)

        param1 = tf.Variable(param)
        param2 = tf.Variable(0.0)

        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(
                circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1]
            )(param1, param2)

        td_grad = tape.gradient(entropy, [param1, param2])
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0))

        param1 = tf.Variable(param)
        param2 = tf.Variable(0.0)

        with tf.GradientTape() as tape:
            entropy = qml.qinfo.trace_distance(
                circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1]
            )(param2, param1)

        td_grad = tape.gradient(entropy, [param2, param1])
        assert qml.math.allclose(td_grad, (0.0, expected_td_grad))

    @pytest.mark.jax
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wire", wires)
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_trace_distance_qnodes_rx_pauliz_jax_grad_two_params(self, param, wire, use_jit):
        """Test the gradient of the trace distance between Rx and PauliZ circuits with Jax and
        two params."""
        dev = qml.device("default.qubit", wires=wire)

        @qml.qnode(dev, interface="jax" if not use_jit else "jax-jit")
        def circuit0(x):
            qml.RX(x, wires=wire - 1)
            return qml.state()

        @qml.qnode(dev, interface="jax" if not use_jit else "jax-jit")
        def circuit1(x):
            qml.RX(x, wires=wire - 1)
            qml.RX(-x, wires=wire - 1)
            qml.PauliZ(wires=wire - 1)
            return qml.state()

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit0, circuit1, wires0=[wire - 1], wires1=[wire - 1]),
            argnums=[0, 1],
        )

        if use_jit:
            td_grad = jax.jit(td_grad)

        td_grad = td_grad(jax.numpy.array(param), jax.numpy.array(2.0))
        expected_td_grad = expected_grad_trace_distance_rx_pauliz(param)
        assert qml.math.allclose(td_grad, (expected_td_grad, 0.0), rtol=1e-03, atol=1e-04)

        td_grad = jax.grad(
            qml.qinfo.trace_distance(circuit1, circuit0, wires0=[wire - 1], wires1=[wire - 1]),
            argnums=[0, 1],
        )

        if use_jit:
            td_grad = jax.jit(td_grad)

        td_grad = td_grad(jax.numpy.array(2.0), jax.numpy.array(param))
        assert qml.math.allclose(td_grad, (0.0, expected_td_grad), rtol=1e-03, atol=1e-04)


@pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
def test_broadcasting(device):
    """Test that the trace_distance transform supports broadcasting"""
    dev = qml.device(device, wires=2)

    @qml.qnode(dev)
    def circuit_state(x):
        qml.IsingXX(x, wires=[0, 1])
        return qml.state()

    x = np.array([0.4, 0.6, 0.8])
    y = np.array([0.6, 0.8, 1.0])
    dist = qml.qinfo.trace_distance(circuit_state, circuit_state, wires0=[0], wires1=[1])(x, y)

    expected = 0.5 * (
        np.abs(np.cos(x / 2) ** 2 - np.cos(y / 2) ** 2)
        + np.abs(np.sin(x / 2) ** 2 - np.sin(y / 2) ** 2)
    )
    assert qml.math.allclose(dist, expected)
