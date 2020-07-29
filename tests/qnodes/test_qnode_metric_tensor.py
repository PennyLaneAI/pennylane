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
Unit tests for the :mod:`pennylane` :class:`QubitQNode` metric tensor methods.
"""
import pytest
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.qnodes.qubit import QubitQNode
from pennylane.qnodes.base import QuantumFunctionError
from gate_data import Y, Z


class TestMetricTensor:
    """Tests for metric tensor subcircuit construction and evaluation"""

    def test_no_generator(self):
        """Test exception is raised if subcircuit contains an
        operation with no generator"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(a):
            qml.Rot(a, 0, 0, wires=0)
            return qml.expval(qml.PauliX(0))

        circuit = QubitQNode(circuit, dev)

        with pytest.raises(QuantumFunctionError, match="has no defined generator"):
            circuit.metric_tensor([1], only_construct=True)

    def test_generator_no_expval(self, monkeypatch):
        """Test exception is raised if subcircuit contains an
        operation with generator object that is not an observable"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliX(0))

        circuit = QubitQNode(circuit, dev)

        with monkeypatch.context() as m:
            m.setattr("pennylane.RX.generator", [qml.RX, 1])

            with pytest.raises(QuantumFunctionError, match="no corresponding observable"):
                circuit.metric_tensor([1], only_construct=True)

    def test_construct_subcircuit(self):
        """Test correct subcircuits constructed"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = QubitQNode(circuit, dev)
        circuit.metric_tensor([1, 1, 1], only_construct=True)
        res = circuit._metric_tensor_subcircuits

        # first parameter subcircuit
        assert len(res[(0,)]["queue"]) == 0
        assert res[(0,)]["scale"] == [-0.5]
        assert isinstance(res[(0,)]["observable"][0], qml.PauliX)

        # second parameter subcircuit
        assert len(res[(1,)]["queue"]) == 1
        assert res[(1,)]["scale"] == [-0.5]
        assert isinstance(res[(1,)]["queue"][0], qml.RX)
        assert isinstance(res[(1,)]["observable"][0], qml.PauliY)

        # third parameter subcircuit
        assert len(res[(2,)]["queue"]) == 3
        assert res[(2,)]["scale"] == [1]
        assert isinstance(res[(2,)]["queue"][0], qml.RX)
        assert isinstance(res[(2,)]["queue"][1], qml.RY)
        assert isinstance(res[(2,)]["queue"][2], qml.CNOT)
        assert isinstance(res[(2,)]["observable"][0], qml.Hermitian)
        assert np.all(res[(2,)]["observable"][0].data[0] == qml.PhaseShift.generator[0])

    def test_construct_subcircuit_layers(self):
        """Test correct subcircuits constructed
        when a layer structure exists"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(params):
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

        circuit = QubitQNode(circuit, dev)

        params = np.ones([8])
        circuit.metric_tensor([params], only_construct=True)
        res = circuit._metric_tensor_subcircuits

        # this circuit should split into 4 independent
        # sections or layers when constructing subcircuits
        assert len(res) == 4

        # first layer subcircuit
        layer = res[(0,)]
        assert len(layer["queue"]) == 0
        assert len(layer["observable"]) == 1
        assert isinstance(layer["observable"][0], qml.PauliX)

        # second layer subcircuit
        layer = res[(1,)]
        assert len(layer["queue"]) == 1
        assert len(layer["observable"]) == 1
        assert isinstance(layer["queue"][0], qml.RX)
        assert isinstance(layer["observable"][0], qml.PauliY)

        # third layer subcircuit
        layer = res[(2, 3, 4)]
        assert len(layer["queue"]) == 4
        assert len(layer["observable"]) == 3
        assert isinstance(layer["queue"][0], qml.RX)
        assert isinstance(layer["queue"][1], qml.RY)
        assert isinstance(layer["queue"][2], qml.CNOT)
        assert isinstance(layer["queue"][3], qml.CNOT)
        assert isinstance(layer["observable"][0], qml.PauliX)
        assert isinstance(layer["observable"][1], qml.PauliY)
        assert isinstance(layer["observable"][2], qml.PauliZ)

        # fourth layer subcircuit
        layer = res[(5, 6, 7)]
        assert len(layer["queue"]) == 9
        assert len(layer["observable"]) == 3
        assert isinstance(layer["queue"][0], qml.RX)
        assert isinstance(layer["queue"][1], qml.RY)
        assert isinstance(layer["queue"][2], qml.CNOT)
        assert isinstance(layer["queue"][3], qml.CNOT)
        assert isinstance(layer["queue"][4], qml.RX)
        assert isinstance(layer["queue"][5], qml.RY)
        assert isinstance(layer["queue"][6], qml.RZ)
        assert isinstance(layer["queue"][7], qml.CNOT)
        assert isinstance(layer["queue"][8], qml.CNOT)
        assert isinstance(layer["observable"][0], qml.PauliX)
        assert isinstance(layer["observable"][1], qml.PauliY)
        assert isinstance(layer["observable"][2], qml.PauliZ)

    def test_evaluate_subcircuits(self, tol):
        """Test subcircuits evaluate correctly"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = QubitQNode(circuit, dev)

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate subcircuits
        circuit.metric_tensor((a, b, c))

        # first parameter subcircuit
        res = circuit._metric_tensor_subcircuits[(0,)]["result"]
        expected = 0.25
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # second parameter subcircuit
        res = circuit._metric_tensor_subcircuits[(1,)]["result"]
        expected = np.cos(a) ** 2 / 4
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # third parameter subcircuit
        res = circuit._metric_tensor_subcircuits[(2,)]["result"]
        expected = (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 16
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_evaluate_diag_metric_tensor(self, tol):
        """Test that a diagonal metric tensor evaluates correctly"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = QubitQNode(circuit, dev)

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate metric tensor
        g = circuit.metric_tensor((a, b, c))

        # check that the metric tensor is correct
        expected = (
            np.array(
                [1, np.cos(a) ** 2, (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 4]
            )
            / 4
        )
        assert np.allclose(g, np.diag(expected), atol=tol, rtol=0)

    @pytest.fixture
    def sample_circuit(self):
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

        final = QubitQNode(final, dev)

        return dev, final, non_parametrized_layer, a, b, c

    def test_evaluate_block_diag_metric_tensor(self, sample_circuit, tol):
        """Test that a block diagonal metric tensor evaluates correctly,
        by comparing it to a known analytic result as well as numerical
        computation."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit

        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        x, y, z, h, g, f = params

        G = circuit.metric_tensor(params)

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
        # numerically.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
        # followed by the qml.RY(f, wires=2) operation.
        #
        # Observable is simply generator of:
        #   qml.RY(f, wires=2)
        #
        # Note: since this layer only consists of a single parameter,
        # only need to compute a single diagonal element.

        def layer2_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        layer2_diag = QubitQNode(layer2_diag, dev)
        G2 = layer2_diag(x, y, z, h, g, f) / 4
        assert np.allclose(G[3:4, 3:4], G2, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of third layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G3 = np.zeros([2, 2])

        def layer3_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        layer3_diag = QubitQNode(layer3_diag, dev)

        def layer3_off_diag_first_order(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliY(1))

        layer3_off_diag_first_order = QubitQNode(layer3_off_diag_first_order, dev)

        def layer3_off_diag_second_order(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.Hermitian(np.kron(Z, Y), wires=[2, 1]))

        layer3_off_diag_second_order = QubitQNode(layer3_off_diag_second_order, dev)

        # calculate the diagonal terms
        varK0, varK1 = layer3_diag(x, y, z, h, g, f)
        G3[0, 0] = varK0 / 4
        G3[1, 1] = varK1 / 4

        # calculate the off-diagonal terms
        exK0, exK1 = layer3_off_diag_first_order(x, y, z, h, g, f)
        exK01 = layer3_off_diag_second_order(x, y, z, h, g, f)

        G3[0, 1] = (exK01 - exK0 * exK1) / 4
        G3[1, 0] = (exK01 - exK0 * exK1) / 4

        assert np.allclose(G[4:6, 4:6], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G2, G3)
        assert np.allclose(G, G_expected, atol=tol, rtol=0)

    def test_evaluate_diag_approx_metric_tensor(self, sample_circuit, tol):
        """Test that a metric tensor under the
        diagonal approximation evaluates correctly."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        x, y, z, h, g, f = params

        G = circuit.metric_tensor(params, diag_approx=True)

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
        # Test metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
        # followed by the qml.RY(f, wires=2) operation.
        #
        # Observable is simply generator of:
        #   qml.RY(f, wires=2)
        #
        # Note: since this layer only consists of a single parameter,
        # only need to compute a single diagonal element.

        def layer2_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        layer2_diag = QubitQNode(layer2_diag, dev)
        G2 = layer2_diag(x, y, z, h, g, f) / 4
        assert np.allclose(G[3:4, 3:4], G2, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of third layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G3 = np.zeros([2, 2])

        def layer3_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        layer3_diag = QubitQNode(layer3_diag, dev)

        # calculate the diagonal terms
        varK0, varK1 = layer3_diag(x, y, z, h, g, f)
        G3[0, 0] = varK0 / 4
        G3[1, 1] = varK1 / 4

        assert np.allclose(G[4:6, 4:6], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G2, G3)
        assert np.allclose(G, G_expected, atol=tol, rtol=0)
