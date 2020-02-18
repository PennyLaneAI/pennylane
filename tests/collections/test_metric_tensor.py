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
Unit tests for the :mod:`pennylane.collections.metric_tensor` submodule.
"""
import pytest

import pennylane as qml
import numpy as np
from scipy.linalg import block_diag


from conftest import torch, tf, Variable


# ===================================================================
# Fixtures
# ===================================================================


def non_parametrized_layer(a, b, c):
    """A non-parametrized layer"""
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(c, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


@pytest.fixture
def sample_circuit(interface, tf_support, torch_support):
    """Sample variational circuit fixture used in the
    next couple of tests"""
    if interface == "torch" and not torch_support:
        pytest.skip("Skipped, no torch support")

    if interface == "tf" and not tf_support:
        pytest.skip("Skipped, no tf support")

    def _sample_circuit(params, non_params=[0.5, 0.1, 0.5]):
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def final(params):
            non_parametrized_layer(*non_params)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.RZ(params[2], wires=2)
            non_parametrized_layer(*non_params)
            qml.RY(params[5], wires=1)
            qml.RZ(params[4], wires=2)
            qml.RX(params[3], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))

        return final

    return _sample_circuit


def sample_circuit_tensor(params, non_params=[0.5, 0.1, 0.5]):
    """Metric tensor solution for the above circuit fixture"""

    a, b, c = non_params
    x, y, z, h, g, f = params

    # First layer includes the non_parametrized_layer,
    # followed by observables corresponding to generators of:
    #   qml.RX(x, wires=0)
    #   qml.RY(y, wires=1)
    #   qml.RZ(z, wires=2)

    G1 = np.zeros([3, 3])

    # diag elements
    G1[0, 0] = np.sin(a) ** 2 / 4
    G1[1, 1] = (
        16 * np.cos(a) ** 2 * np.sin(b) ** 3 * np.cos(b) * np.sin(2 * c)
        + np.cos(2 * b) * (2 - 8 * np.cos(a) ** 2 * np.sin(b) ** 2 * np.cos(2 * c))
        + np.cos(2 * (a - b))
        + np.cos(2 * (a + b))
        - 2 * np.cos(2 * a)
        + 14
    ) / 64
    G1[2, 2] = (3 - np.cos(2 * a) - 2 * np.cos(a) ** 2 * np.cos(2 * (b + c))) / 16

    # off diag elements
    G1[0, 1] = np.sin(a) ** 2 * np.sin(b) * np.cos(b + c) / 4
    G1[0, 2] = np.sin(a) ** 2 * np.cos(b + c) / 4
    G1[1, 2] = (
        -np.sin(b)
        * (
            np.cos(2 * (a - b - c))
            + np.cos(2 * (a + b + c))
            + 2 * np.cos(2 * a)
            + 2 * np.cos(2 * (b + c))
            - 6
        )
        / 32
    )

    G1[1, 0] = G1[0, 1]
    G1[2, 0] = G1[0, 2]
    G1[2, 1] = G1[1, 2]

    # =============================================
    # The second layer includes the non_parametrized_layer,
    # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
    # followed by the qml.RY(f, wires=2) operation.
    #
    # Observable is simply generator of:
    #   qml.RY(f, wires=2)
    #
    # Note: since this layer only consists of a single parameter,
    # only need to compute a single diagonal element.

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def layer2_diag(x, y, z, h, g, f):
        non_parametrized_layer(a, b, c)
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(z, wires=2)
        non_parametrized_layer(a, b, c)
        qml.RY(f, wires=2)
        return qml.var(qml.PauliX(1))

    G2 = layer2_diag(x, y, z, h, g, f) / 4

    # =============================================
    # The second layer includes the non_parametrized_layer,
    # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
    #
    # Observables are the generators of:
    #   qml.RY(f, wires=1)
    #   qml.RZ(g, wires=2)
    G3 = np.zeros([2, 2])

    @qml.qnode(dev)
    def layer3_diag(x, y, z, h, g, f):
        non_parametrized_layer(a, b, c)
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(z, wires=2)
        non_parametrized_layer(a, b, c)
        return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

    @qml.qnode(dev)
    def layer3_off_diag_first_order(x, y, z, h, g, f):
        non_parametrized_layer(a, b, c)
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(z, wires=2)
        non_parametrized_layer(a, b, c)
        return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliY(1))

    @qml.qnode(dev)
    def layer3_off_diag_second_order(x, y, z, h, g, f):
        non_parametrized_layer(a, b, c)
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(z, wires=2)
        non_parametrized_layer(a, b, c)
        return qml.expval(qml.PauliZ(2) @ qml.PauliY(1))

    # calculate the diagonal terms
    varK0, varK1 = layer3_diag(x, y, z, h, g, f)
    G3[0, 0] = varK0 / 4
    G3[1, 1] = varK1 / 4

    # calculate the off-diagonal terms
    exK0, exK1 = layer3_off_diag_first_order(x, y, z, h, g, f)
    exK01 = layer3_off_diag_second_order(x, y, z, h, g, f)

    G3[0, 1] = (exK01 - exK0 * exK1) / 4
    G3[1, 0] = (exK01 - exK0 * exK1) / 4

    return block_diag(G1, G2, G3)


# ===================================================================
# Tests
# ===================================================================


class TestMetricTensorEvaluation:
    """Tests for metric tensor subcircuit construction and evaluation"""

    def test_no_generator(self,):
        """Test exception is raised if subcircuit contains an
        operation with no generator"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.Rot(a, 0, 0, wires=0)
            return qml.expval(qml.PauliX(0))

        g = qml.MetricTensor(circuit, dev)

        with pytest.raises(qml.QuantumFunctionError, match="has no defined generator"):
            g._make_qnodes([1], {})

    def test_generator_no_expval(self, monkeypatch):
        """Test exception is raised if subcircuit contains an
        operation with generator object that is not an observable"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliX(0))

        g = qml.MetricTensor(circuit, dev)

        with monkeypatch.context() as m:
            m.setattr("pennylane.RX.generator", [qml.RX, 1])

            with pytest.raises(qml.QuantumFunctionError, match="no corresponding observable"):
                g._make_qnodes([1], {})

    def test_construct_qnodes(self):
        """Test correct qnodes constructed"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        g = qml.MetricTensor(circuit, dev)
        qnodes, obs, coeffs, params = g._make_qnodes([1, 1, 1], {})

        # first parameter subcircuit
        # First parameter subcircuit consists of no operations,
        # and a PauliX observable. This corresponds to a Hadamard rotation.
        qnodes[0]([1, 1, 1])
        assert len(qnodes[0].ops) == 2
        assert isinstance(obs[0][0], qml.PauliX)
        assert isinstance(qnodes[0].ops[0], qml.Hadamard)
        assert coeffs[0] == [-0.5]

        # second parameter subcircuit
        # second parameter subcircuit consists of RX,
        # and a PauliY observable.
        qnodes[1]([1, 1, 1])
        assert len(qnodes[1].ops) == 5
        assert isinstance(obs[1][0], qml.PauliY)
        assert isinstance(qnodes[1].ops[0], qml.RX)
        assert isinstance(qnodes[1].ops[1], qml.PauliZ)
        assert isinstance(qnodes[1].ops[2], qml.S)
        assert isinstance(qnodes[1].ops[3], qml.Hadamard)
        assert coeffs[1] == [-0.5]

        # third parameter subcircuit
        # second parameter subcircuit consists of RX, RY, CNOT
        # and a Hermitian observable.
        qnodes[2]([1, 1, 1])
        assert len(qnodes[2].ops) == 5
        assert isinstance(obs[2][0], qml.Hermitian)
        assert isinstance(qnodes[2].ops[0], qml.RX)
        assert isinstance(qnodes[2].ops[1], qml.RY)
        assert isinstance(qnodes[2].ops[2], qml.CNOT)
        assert isinstance(qnodes[2].ops[3], qml.QubitUnitary)
        assert coeffs[2] == [1]

    def test_evaluate_subcircuits(self, tol):
        """Test subcircuits evaluate correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        g = qml.MetricTensor(circuit, dev)

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate subcircuits
        qnodes, obs, coeffs, _ = g._make_qnodes([a, b, c], {})

        # first parameter subcircuit
        probs = dev.marginal_prob(qnodes[0]([a, b, c]), wires=obs[0][0].wires)
        eigvals = obs[0][0].eigvals
        res = (eigvals**2 @ probs - (eigvals @ probs)**2) * coeffs[0][0]**2
        expected = 0.25
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # second parameter subcircuit
        probs = dev.marginal_prob(qnodes[1]([a, b, c]), wires=obs[1][0].wires)
        eigvals = obs[1][0].eigvals
        res = (eigvals**2 @ probs - (eigvals @ probs)**2) * coeffs[1][0]**2
        expected = np.cos(a) ** 2 / 4
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # third parameter subcircuit
        probs = dev.marginal_prob(qnodes[2]([a, b, c]), wires=obs[2][0].wires)
        eigvals = obs[2][0].eigvals
        res = (eigvals**2 @ probs - (eigvals @ probs)**2) * coeffs[2][0]**2
        expected = (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 16
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_evaluate_diag_metric_tensor(self, tol):
        """Test that a diagonal metric tensor evaluates correctly"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        g = qml.MetricTensor(circuit, dev)

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate metric tensor
        res = g(a, b, c)

        # check that the metric tensor is correct
        expected = (
            np.array(
                [1, np.cos(a) ** 2, (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 4]
            )
            / 4
        )
        assert np.allclose(res, np.diag(expected), atol=tol, rtol=0)


class TestMetricTensorInterfaces:
    """Tests for interfaces of the metric tensor"""

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_autograd_metric_tensor(self, sample_circuit, tol):
        """Test that a block diagonal metric tensor evaluates correctly,
        by comparing it to a known analytic result as well as numerical
        computation."""
        dev = qml.device("default.qubit", wires=3)
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        circuit = sample_circuit(params)

        G = qml.MetricTensor(circuit, dev)(params)
        expected = sample_circuit_tensor(params)
        assert np.allclose(G, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_autograd_diag_approx_metric_tensor(self, sample_circuit, tol):
        """Test that a metric tensor under the
        diagonal approximation evaluates correctly."""
        dev = qml.device("default.qubit", wires=3)
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        circuit = sample_circuit(params)

        G = qml.MetricTensor(circuit, dev)(params, diag_approx=True)
        expected = np.diag(np.diag(sample_circuit_tensor(params)))
        assert np.allclose(G, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_torch_metric_tensor(self, sample_circuit, tol):
        """Test that a block diagonal metric tensor evaluates correctly
        using PyTorch"""

        dev = qml.device("default.qubit", wires=3)
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        params_torch = torch.autograd.Variable(torch.tensor(params))
        circuit = sample_circuit(params)

        G = qml.MetricTensor(circuit, dev)(params_torch)
        expected = sample_circuit_tensor(params)

        assert isinstance(G, torch.Tensor)
        assert np.allclose(G, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_tf_metric_tensor(self, sample_circuit, tol):
        """Test that a block diagonal metric tensor evaluates correctly
        using tensorflow"""
        dev = qml.device("default.qubit", wires=3)

        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        params_tf = Variable(params)

        circuit = sample_circuit(params)

        G = qml.MetricTensor(circuit, dev)(params_tf)
        expected = sample_circuit_tensor(params)

        assert isinstance(G, tf.Tensor)
        assert np.allclose(G, expected, atol=tol, rtol=0)


class TestMetricTensorAutodiff:
    """Tests for differentiation of the metric tensor"""

    @pytest.fixture
    def finite_diff(self):
        """Fixture to compute the numeric gradient of the metric tensor
        for some reduction function fn"""
        def _finite_diff(fn, params):
            # Compare the analytic gradient against finite diff
            cost_exp = lambda params: fn(sample_circuit_tensor(params))
            h = 0.0001
            shift = np.zeros([6])
            grad = []

            for i in range(6):
                shift_tmp = shift.copy()
                shift_tmp[i] += h
                pp = np.array(params) + shift_tmp
                pm = np.array(params) - shift_tmp
                grad.append((cost_exp(pp) - cost_exp(pm))/(2*h))

            return grad

        return _finite_diff

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_autograd_metric_tensor(self, sample_circuit, finite_diff, tol):
        """Test that a block diagonal metric tensor gradient evaluates has the correct
        gradient when using the autograd interface"""
        dev = qml.device("default.qubit", wires=3)
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        circuit = sample_circuit(params)

        cost = qml.sum(qml.MetricTensor(circuit, dev))
        cost(params)
        dcost = qml.grad(cost, argnum=0)

        res = dcost(params)
        expected = finite_diff(np.sum, params)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_torch_metric_tensor(self, sample_circuit, finite_diff, tol):
        """Test that a block diagonal metric tensor gradient evaluates correctly
        using PyTorch"""
        dev = qml.device("default.qubit", wires=3)
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        params_torch = torch.autograd.Variable(torch.tensor(params), requires_grad=True)
        circuit = sample_circuit(params)

        cost = qml.sum(qml.MetricTensor(circuit, dev))
        loss = cost(params_torch)
        loss.backward()

        res = params_torch.grad
        expected = finite_diff(np.sum, params)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_tf_metric_tensor(self, sample_circuit, finite_diff, tol):
        """Test that a block diagonal metric tensor gradient evaluates correctly
        using tensorflow"""
        dev = qml.device("default.qubit", wires=3)
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        params_tf = Variable(params)
        circuit = sample_circuit(params)

        cost = qml.sum(qml.MetricTensor(circuit, dev))

        with tf.GradientTape() as tape:
            tape.watch(params_tf)
            loss = cost(params_tf)
            res = tape.gradient(loss, params_tf)

        expected = finite_diff(np.sum, params)

        assert np.allclose(res, expected, atol=tol, rtol=0)
