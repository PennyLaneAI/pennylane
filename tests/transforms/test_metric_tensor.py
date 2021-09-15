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
Unit tests for the metric tensor transform.
"""
import pytest
from pennylane import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from gate_data import Y, Z


class TestMetricTensor:
    """Tests for metric tensor subcircuit construction and evaluation"""

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_rot_decomposition(self, diff_method):
        """Test that the rotation gate is correctly decomposed"""
        dev = qml.device("default.qubit", wires=1)
        params = np.array([1.0, 2.0, 3.0], requires_grad=True)

        with qml.tape.QuantumTape() as circuit:
            qml.Rot(params[0], params[1], params[2], wires=0)
            qml.expval(qml.PauliX(0))

        tapes, _ = qml.metric_tensor(circuit)
        assert len(tapes) == 3

        # first parameter subcircuit
        assert len(tapes[0].operations) == 0

        # Second parameter subcircuit
        assert len(tapes[1].operations) == 4
        assert isinstance(tapes[1].operations[0], qml.RZ)
        assert tapes[1].operations[0].data == [1]
        # PauliY decomp
        assert isinstance(tapes[1].operations[1], qml.PauliZ)
        assert isinstance(tapes[1].operations[2], qml.S)
        assert isinstance(tapes[1].operations[3], qml.Hadamard)

        # Third parameter subcircuit
        assert len(tapes[2].operations) == 2
        assert isinstance(tapes[2].operations[0], qml.RZ)
        assert isinstance(tapes[2].operations[1], qml.RY)
        assert tapes[2].operations[0].data == [1]
        assert tapes[2].operations[1].data == [2]

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_multirz_decomposition(self, diff_method):
        """Test that the MultiRZ gate is correctly decomposed"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.MultiRZ(b, wires=[0, 1, 2])
            return qml.expval(qml.PauliX(0))

        circuit = qml.QNode(circuit, dev, diff_method=diff_method)
        params = [0.1, 0.2]
        result = qml.metric_tensor(circuit)(*params)
        assert result.shape == (2, 2)

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_parameter_fan_out(self, diff_method):
        """The metric tensor is always with respect to the quantum circuit. Any
        classical processing is not taken into account. As a result, if there is
        parameter fan-out, the returned metric tensor will be larger than
        expected.
        """
        dev = qml.device("default.qubit", wires=2)

        def circuit(a):
            qml.RX(a, wires=0)
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliX(0))

        circuit = qml.QNode(circuit, dev, diff_method=diff_method)
        params = [0.1]
        result = qml.metric_tensor(circuit, hybrid=False)(*params)
        assert result.shape == (2, 2)

    def test_generator_no_expval(self, monkeypatch):
        """Test exception is raised if subcircuit contains an
        operation with generator object that is not an observable"""
        with monkeypatch.context() as m:
            m.setattr("pennylane.RX.generator", [qml.RX, 1])

            with qml.tape.QuantumTape() as tape:
                qml.RX(np.array(0.5, requires_grad=True), wires=0)
                qml.expval(qml.PauliX(0))

            with pytest.raises(qml.QuantumFunctionError, match="no corresponding observable"):
                qml.metric_tensor(tape)

    def test_construct_subcircuit(self):
        """Test correct subcircuits constructed"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RX(np.array(1.0, requires_grad=True), wires=0)
            qml.RY(np.array(1.0, requires_grad=True), wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(np.array(1.0, requires_grad=True), wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        tapes, _ = qml.metric_tensor(tape)
        assert len(tapes) == 3

        # first parameter subcircuit
        assert len(tapes[0].operations) == 1
        assert isinstance(tapes[0].operations[0], qml.Hadamard)  # PauliX decomp

        # second parameter subcircuit
        assert len(tapes[1].operations) == 4
        assert isinstance(tapes[1].operations[0], qml.RX)
        # PauliY decomp
        assert isinstance(tapes[1].operations[1], qml.PauliZ)
        assert isinstance(tapes[1].operations[2], qml.S)
        assert isinstance(tapes[1].operations[3], qml.Hadamard)

        # third parameter subcircuit
        assert len(tapes[2].operations) == 4
        assert isinstance(tapes[2].operations[0], qml.RX)
        assert isinstance(tapes[2].operations[1], qml.RY)
        assert isinstance(tapes[2].operations[2], qml.CNOT)
        # Phase shift generator
        assert isinstance(tapes[2].operations[3], qml.QubitUnitary)

    def test_construct_subcircuit_layers(self):
        """Test correct subcircuits constructed
        when a layer structure exists"""
        dev = qml.device("default.qubit", wires=3)
        params = np.ones([8])

        with qml.tape.QuantumTape() as tape:
            # section 1
            qml.RX(params[0], wires=0)
            # section 2
            qml.RY(params[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            # section 3
            qml.RX(params[2], wires=0)
            qml.RY(params[3], wires=1)
            qml.RZ(params[4], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            # section 4
            qml.RX(params[5], wires=0)
            qml.RY(params[6], wires=1)
            qml.RZ(params[7], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))

        tapes, _ = qml.metric_tensor(tape)

        # this circuit should split into 4 independent
        # sections or layers when constructing subcircuits
        assert len(tapes) == 4

        # first layer subcircuit
        assert len(tapes[0].operations) == 1
        assert isinstance(tapes[0].operations[0], qml.Hadamard)  # PauliX decomp

        # second layer subcircuit
        assert len(tapes[1].operations) == 4
        assert isinstance(tapes[1].operations[0], qml.RX)
        # PauliY decomp
        assert isinstance(tapes[1].operations[1], qml.PauliZ)
        assert isinstance(tapes[1].operations[2], qml.S)
        assert isinstance(tapes[1].operations[3], qml.Hadamard)

        # # third layer subcircuit
        assert len(tapes[2].operations) == 8
        assert isinstance(tapes[2].operations[0], qml.RX)
        assert isinstance(tapes[2].operations[1], qml.RY)
        assert isinstance(tapes[2].operations[2], qml.CNOT)
        assert isinstance(tapes[2].operations[3], qml.CNOT)
        # PauliX decomp
        assert isinstance(tapes[2].operations[4], qml.Hadamard)
        # PauliY decomp
        assert isinstance(tapes[2].operations[5], qml.PauliZ)
        assert isinstance(tapes[2].operations[6], qml.S)
        assert isinstance(tapes[2].operations[7], qml.Hadamard)

        # # fourth layer subcircuit
        assert len(tapes[3].operations) == 13
        assert isinstance(tapes[3].operations[0], qml.RX)
        assert isinstance(tapes[3].operations[1], qml.RY)
        assert isinstance(tapes[3].operations[2], qml.CNOT)
        assert isinstance(tapes[3].operations[3], qml.CNOT)
        assert isinstance(tapes[3].operations[4], qml.RX)
        assert isinstance(tapes[3].operations[5], qml.RY)
        assert isinstance(tapes[3].operations[6], qml.RZ)
        assert isinstance(tapes[3].operations[7], qml.CNOT)
        assert isinstance(tapes[3].operations[8], qml.CNOT)
        # PauliX decomp
        assert isinstance(tapes[3].operations[9], qml.Hadamard)
        # PauliY decomp
        assert isinstance(tapes[3].operations[10], qml.PauliZ)
        assert isinstance(tapes[3].operations[11], qml.S)
        assert isinstance(tapes[3].operations[12], qml.Hadamard)

    def test_evaluate_diag_metric_tensor(self, tol):
        """Test that a diagonal metric tensor evaluates correctly"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = qml.QNode(circuit, dev)

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate metric tensor
        g = qml.metric_tensor(circuit)(a, b, c)

        # check that the metric tensor is correct
        expected = (
            np.array(
                [1, np.cos(a) ** 2, (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 4]
            )
            / 4
        )
        assert np.allclose(g, np.diag(expected), atol=tol, rtol=0)

    def test_evaluate_diag_metric_tensor_classical_processing(self, tol):
        """Test that a diagonal metric tensor evaluates correctly
        when the QNode includes classical processing."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(a, b):
            # The classical processing function is
            #     f: ([a0, a1], b) -> (a1, a0, b)
            # So the classical Jacobians will be a permutation matrix and an identity matrix:
            #     classical_jacobian(circuit)(a, b) == ([[0, 1], [1, 0]], [[1]])
            qml.RX(a[1], wires=0)
            qml.RY(a[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(b, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = qml.QNode(circuit, dev)

        a = np.array([0.432, 0.1])
        b = 0.12

        # evaluate metric tensor
        g = qml.metric_tensor(circuit)(a, b)
        assert isinstance(g, tuple)
        assert len(g) == 2
        assert g[0].shape == (len(a), len(a))
        assert g[1].shape == tuple()

        # check that the metric tensor is correct
        expected = np.array([np.cos(a[1]) ** 2, 1]) / 4
        assert np.allclose(g[0], np.diag(expected), atol=tol, rtol=0)

        expected = (3 - 2 * np.cos(a[1]) ** 2 * np.cos(2 * a[0]) - np.cos(2 * a[1])) / 16
        assert np.allclose(g[1], expected, atol=tol, rtol=0)

    @pytest.fixture(params=["parameter-shift", "backprop"])
    def sample_circuit(self, request):
        """Sample variational circuit fixture used in the
        next couple of tests"""
        dev = qml.device("default.qubit", wires=3)

        def non_parametrized_layer(a, b, c):
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

        a = 0.5
        b = 0.1
        c = 0.5

        def final(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=1)
            qml.RZ(g, wires=2)
            qml.RX(h, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))

        final = qml.QNode(final, dev, diff_method=request.param)

        return dev, final, non_parametrized_layer, a, b, c

    def test_evaluate_block_diag_metric_tensor(self, sample_circuit, tol):
        """Test that a block diagonal metric tensor evaluates correctly,
        by comparing it to a known analytic result as well as numerical
        computation."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit

        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        x, y, z, h, g, f = params

        G = qml.metric_tensor(circuit)(*params)

        # ============================================
        # Test block diag metric tensor of first layer is correct.
        # We do this by comparing against the known analytic result.
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

        assert np.allclose(G[:3, :3], G1, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G2 = np.zeros([2, 2])

        def layer2_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        layer2_diag = qml.QNode(layer2_diag, dev)

        def layer2_off_diag_first_order(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliY(1))

        layer2_off_diag_first_order = qml.QNode(layer2_off_diag_first_order, dev)

        def layer2_off_diag_second_order(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.Hermitian(np.kron(Z, Y), wires=[2, 1]))

        layer2_off_diag_second_order = qml.QNode(layer2_off_diag_second_order, dev)

        # calculate the diagonal terms
        varK0, varK1 = layer2_diag(x, y, z, h, g, f)
        G2[0, 0] = varK0 / 4
        G2[1, 1] = varK1 / 4

        # calculate the off-diagonal terms
        exK0, exK1 = layer2_off_diag_first_order(x, y, z, h, g, f)
        exK01 = layer2_off_diag_second_order(x, y, z, h, g, f)

        G2[0, 1] = (exK01 - exK0 * exK1) / 4
        G2[1, 0] = (exK01 - exK0 * exK1) / 4

        assert np.allclose(G[4:6, 4:6], G2, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of third layer is correct.
        # We do this by computing the required expectation values
        # numerically.
        # The third layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
        # followed by the qml.RY(f, wires=2) operation.
        #
        # Observable is simply generator of:
        #   qml.RY(f, wires=2)
        #
        # Note: since this layer only consists of a single parameter,
        # only need to compute a single diagonal element.

        def layer3_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        layer3_diag = qml.QNode(layer3_diag, dev)
        G3 = layer3_diag(x, y, z, h, g, f) / 4
        assert np.allclose(G[3:4, 3:4], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G3, G2)
        assert np.allclose(G, G_expected, atol=tol, rtol=0)

    def test_evaluate_diag_approx_metric_tensor(self, sample_circuit, tol):
        """Test that a metric tensor under the
        diagonal approximation evaluates correctly."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        x, y, z, h, g, f = params

        G = qml.metric_tensor(circuit, diag_approx=True)(*params)

        # ============================================
        # Test block diag metric tensor of first layer is correct.
        # We do this by comparing against the known analytic result.
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

        assert np.allclose(G[:3, :3], G1, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G2 = np.zeros([2, 2])

        def layer2_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        layer2_diag = qml.QNode(layer2_diag, dev)

        # calculate the diagonal terms
        varK0, varK1 = layer2_diag(x, y, z, h, g, f)
        G2[0, 0] = varK0 / 4
        G2[1, 1] = varK1 / 4

        assert np.allclose(G[4:6, 4:6], G2, atol=tol, rtol=0)

        # =============================================
        # Test metric tensor of third layer is correct.
        # We do this by computing the required expectation values
        # numerically.
        # The third layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
        # followed by the qml.RY(f, wires=2) operation.
        #
        # Observable is simply generator of:
        #   qml.RY(f, wires=2)
        #
        # Note: since this layer only consists of a single parameter,
        # only need to compute a single diagonal element.

        def layer3_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        layer3_diag = qml.QNode(layer3_diag, dev)
        G3 = layer3_diag(x, y, z, h, g, f) / 4
        assert np.allclose(G[3:4, 3:4], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G3, G2)
        assert np.allclose(G, G_expected, atol=tol, rtol=0)


@pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
class TestDifferentiability:
    """Test for metric tensor differentiability"""

    def test_autograd(self, diff_method, tol):
        """Test metric tensor differentiability in the autograd interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(weights[2], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        def cost(weights):
            return qml.metric_tensor(circuit)(weights)[2, 2]

        weights = np.array([0.432, 0.12, -0.432], requires_grad=True)
        a, b, c = weights

        grad = qml.grad(cost)(weights)
        expected = np.array(
            [np.cos(a) * np.cos(b) ** 2 * np.sin(a) / 2, np.cos(a) ** 2 * np.sin(2 * b) / 4, 0]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_jax(self, diff_method, tol):
        """Test metric tensor differentiability in the JAX interface"""
        if diff_method == "parameter-shift":
            pytest.skip("Does not support parameter-shift")

        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(weights[2], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit.interface = "jax"

        def cost(weights):
            return qml.metric_tensor(circuit)(weights)[2, 2]

        weights = jnp.array([0.432, 0.12, -0.432])
        a, b, c = weights

        grad = jax.grad(cost)(weights)
        expected = np.array(
            [np.cos(a) * np.cos(b) ** 2 * np.sin(a) / 2, np.cos(a) ** 2 * np.sin(2 * b) / 4, 0]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_tf(self, diff_method, tol):
        """Test metric tensor differentiability in the TF interface"""
        tf = pytest.importorskip("tensorflow", minversion="2.0")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(weights[2], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        weights = np.array([0.432, 0.12, -0.432])
        weights_t = tf.Variable(weights)
        a, b, c = weights

        with tf.GradientTape() as tape:
            loss = qml.metric_tensor(circuit)(weights_t)[2, 2]

        grad = tape.gradient(loss, weights_t)
        expected = np.array(
            [np.cos(a) * np.cos(b) ** 2 * np.sin(a) / 2, np.cos(a) ** 2 * np.sin(2 * b) / 4, 0]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_torch(self, diff_method, tol):
        """Test metric tensor differentiability in the torch interface"""
        torch = pytest.importorskip("torch")

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(weights[2], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        weights = np.array([0.432, 0.12, -0.432])
        a, b, c = weights

        weights_t = torch.tensor(weights, requires_grad=True)
        loss = qml.metric_tensor(circuit)(weights_t)[2, 2]
        loss.backward()

        grad = weights_t.grad
        expected = np.array(
            [np.cos(a) * np.cos(b) ** 2 * np.sin(a) / 2, np.cos(a) ** 2 * np.sin(2 * b) / 4, 0]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)


class TestDeprecatedQNodeMethod:
    """The QNode.metric_tensor method has been deprecated.
    These tests ensure it still works, but raises a deprecation
    warning. These tests can be deleted when the method is removed."""

    def test_warning(self, tol):
        """Test that a warning is emitted"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate metric tensor
        with pytest.warns(UserWarning, match="has been deprecated"):
            g = circuit.metric_tensor(a, b, c)

        # check that the metric tensor is correct
        expected = (
            np.array(
                [1, np.cos(a) ** 2, (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 4]
            )
            / 4
        )
        assert np.allclose(g, np.diag(expected), atol=tol, rtol=0)

    def test_tapes_returned(self, tol):
        """Test that a warning is emitted"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate metric tensor
        with pytest.warns(UserWarning, match="has been deprecated"):
            tapes, fn = circuit.metric_tensor(a, b, c, only_construct=True)

        assert len(tapes) == 3
