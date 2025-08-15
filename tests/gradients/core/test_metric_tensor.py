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
Unit tests for the metric tensor transform.
"""
import importlib

# pylint: disable=too-many-arguments,too-many-public-methods,too-few-public-methods
# pylint: disable=not-callable,too-many-statements, too-many-positional-arguments
import pytest
from scipy.linalg import block_diag

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.gradients.metric_tensor import _get_aux_wire


class TestMetricTensor:
    """Tests for metric tensor subcircuit construction and evaluation"""

    def assert_Y_decomp(self, ops):
        assert isinstance(ops[0], qml.PauliZ)
        assert isinstance(ops[1], qml.S)
        assert isinstance(ops[2], qml.Hadamard)

    def test_rot_decomposition(self):
        """Test that the rotation gate is correctly decomposed"""
        params = np.array([1.0, 2.0, 3.0], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q_circuit:
            qml.Rot(params[0], params[1], params[2], wires=0)
            qml.expval(qml.PauliX(0))

        circuit = qml.tape.QuantumScript.from_queue(q_circuit)
        tapes, _ = qml.metric_tensor(circuit, approx="block-diag")
        assert len(tapes) == 3

        # first parameter subcircuit
        assert len(tapes[0].operations) == 0

        # Second parameter subcircuit
        assert len(tapes[1].operations) == 4
        assert isinstance(tapes[1].operations[0], qml.RZ)
        assert tapes[1].operations[0].data == (1,)
        # PauliY decomp
        self.assert_Y_decomp(tapes[1].operations[1:4])

        # Third parameter subcircuit
        assert len(tapes[2].operations) == 2
        assert isinstance(tapes[2].operations[0], qml.RZ)
        assert isinstance(tapes[2].operations[1], qml.RY)
        assert tapes[2].operations[0].data == (1,)
        assert tapes[2].operations[1].data == (2,)

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_multirz_decomposition(self, diff_method):
        """Test that the MultiRZ gate is correctly decomposed"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.MultiRZ(b, wires=[0, 1, 2])
            return qml.expval(qml.PauliX(0))

        params = np.array([0.1, 0.2], requires_grad=True)
        result = qml.metric_tensor(circuit, approx="block-diag")(*params)
        assert isinstance(result, tuple) and len(result) == 2
        assert qml.math.shape(result[0]) == ()
        assert qml.math.shape(result[1]) == ()

    def test_construct_subcircuit(self):
        """Test correct subcircuits constructed"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(np.array(1.0, requires_grad=True), wires=0)
            qml.RY(np.array(1.0, requires_grad=True), wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(np.array(1.0, requires_grad=True), wires=1)
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.metric_tensor(tape, approx="block-diag")
        assert len(tapes) == 3

        # first parameter subcircuit
        assert len(tapes[0].operations) == 1
        assert isinstance(tapes[0].operations[0], qml.Hadamard)  # PauliX decomp

        # second parameter subcircuit
        assert len(tapes[1].operations) == 4
        assert isinstance(tapes[1].operations[0], qml.RX)
        # PauliY decomp
        self.assert_Y_decomp(tapes[1].operations[1:4])

        # third parameter subcircuit
        assert len(tapes[2].operations) == 3
        assert isinstance(tapes[2].operations[0], qml.RX)
        assert isinstance(tapes[2].operations[1], qml.RY)
        assert isinstance(tapes[2].operations[2], qml.CNOT)
        # No decomposition for operator that is diagonal in computational basis

    def test_construct_subcircuit_layers(self):
        """Test correct subcircuits constructed
        when a layer structure exists"""
        params = np.ones([8])

        with qml.queuing.AnnotatedQueue() as q:
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
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliX(1))
            qml.expval(qml.PauliX(2))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.metric_tensor(tape, approx="block-diag")

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
        self.assert_Y_decomp(tapes[1].operations[1:4])
        # # third layer subcircuit
        assert len(tapes[2].operations) == 8
        assert isinstance(tapes[2].operations[0], qml.RX)
        assert isinstance(tapes[2].operations[1], qml.RY)
        assert isinstance(tapes[2].operations[2], qml.CNOT)
        assert isinstance(tapes[2].operations[3], qml.CNOT)
        # PauliX decomp
        assert isinstance(tapes[2].operations[4], qml.Hadamard)
        # PauliY decomp
        self.assert_Y_decomp(tapes[2].operations[5:8])

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
        self.assert_Y_decomp(tapes[3].operations[10:13])

    def test_evaluate_diag_metric_tensor(self, tol):
        """Test that a diagonal metric tensor evaluates correctly for
        block-diagonal and diagonal setting."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(abc):
            a, b, c = abc
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = qml.QNode(circuit, dev)

        abc = np.array([0.432, 0.12, -0.432], requires_grad=True)
        a, b, _ = abc

        # evaluate metric tensor
        g_diag = qml.metric_tensor(circuit, approx="diag")(abc)
        g_blockdiag = qml.metric_tensor(circuit, approx="block-diag")(abc)

        # check that the metric tensor is correct
        expected = (
            np.array(
                [1, np.cos(a) ** 2, (3 - 2 * np.cos(a) ** 2 * np.cos(2 * b) - np.cos(2 * a)) / 4]
            )
            / 4
        )
        assert qml.math.allclose(g_diag, np.diag(expected), atol=tol, rtol=0)
        assert qml.math.allclose(g_blockdiag, np.diag(expected), atol=tol, rtol=0)

    def test_template_integration(self):
        """Test that the metric tensor transform acts on QNodes
        correctly when the QNode contains a template"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.probs(wires=[0, 1])

        weights = np.ones([2, 3, 3], dtype=np.float64, requires_grad=True)
        res = qml.metric_tensor(circuit, approx="block-diag")(weights)
        assert res.shape == (2, 3, 3, 2, 3, 3)

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
            qml.U1(b, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        circuit = qml.QNode(circuit, dev)

        a = np.array([0.432, 0.1], requires_grad=True)
        b = np.array(0.12, requires_grad=True)

        # evaluate metric tensor
        g = qml.metric_tensor(circuit, approx="block-diag")(a, b)
        assert isinstance(g, tuple)
        assert len(g) == 2
        assert g[0].shape == (len(a), len(a))
        assert g[1].shape == tuple()

        # check that the metric tensor is correct
        expected = np.array([np.cos(a[1]) ** 2, 1]) / 4
        assert qml.math.allclose(g[0], np.diag(expected), atol=tol, rtol=0)

        expected = (3 - 2 * np.cos(a[1]) ** 2 * np.cos(2 * a[0]) - np.cos(2 * a[1])) / 16
        assert qml.math.allclose(g[1], expected, atol=tol, rtol=0)

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

        def final(params):
            x, y, z, h, g, f = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.adjoint(qml.RY)(-y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=1)
            qml.RZ(g, wires=2)
            qml.RX(h, wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))

        final = qml.QNode(final, dev, diff_method=request.param)

        return dev, final, non_parametrized_layer, a, b, c

    def test_evaluate_block_diag_metric_tensor(self, sample_circuit, tol):
        """Test that a block-diagonal metric tensor evaluates correctly,
        by comparing it to a known analytic result as well as numerical
        computation."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit

        params = np.array(
            [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272],
            requires_grad=True,
        )

        G = qml.metric_tensor(circuit, approx="block-diag")(params)

        # ============================================
        # Test block-diag metric tensor of first layer is correct.
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
        G1[0, 1] = G1[1, 0] = np.sin(a) ** 2 * np.sin(b) * np.cos(b + c) / 4
        G1[0, 2] = G1[2, 0] = np.sin(a) ** 2 * np.cos(b + c) / 4
        G1[1, 2] = G1[2, 1] = (
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
        assert qml.math.allclose(G[:3, :3], G1, atol=tol, rtol=0)

        # =============================================
        # Test block-diag metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)

        @qml.qnode(dev)
        def layer2_diag(params):
            x, y, z, *_ = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        @qml.qnode(dev)
        def layer2_off_diag_first_order(params):
            x, y, z, *_ = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliY(1))

        @qml.qnode(dev)
        def layer2_off_diag_second_order(params):
            x, y, z, *_ = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.PauliY(1) @ qml.PauliZ(2))

        # calculate the diagonal terms
        varK0, varK1 = layer2_diag(params)
        G2 = np.diag([varK0 / 4, varK1 / 4])

        # calculate the off-diagonal terms
        exK0, exK1 = layer2_off_diag_first_order(params)
        exK01 = layer2_off_diag_second_order(params)

        G2[0, 1] = G2[1, 0] = (exK01 - exK0 * exK1) / 4

        assert qml.math.allclose(G[4:6, 4:6], G2, atol=tol, rtol=0)

        # =============================================
        # Test block-diag metric tensor of third layer is correct.
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

        @qml.qnode(dev)
        def layer3_diag(params):
            x, y, z, *_, f = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        G3 = layer3_diag(params) / 4
        assert qml.math.allclose(G[3:4, 3:4], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        assert qml.math.allclose(G, block_diag(G1, G3, G2), atol=tol, rtol=0)

    def test_evaluate_diag_approx_metric_tensor(self, sample_circuit, tol):
        """Test that a metric tensor under the diagonal approximation evaluates
        correctly."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit
        params = np.array(
            [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272],
            requires_grad=True,
        )

        G = qml.metric_tensor(circuit, approx="diag")(params)

        # ============================================
        # Test block-diag metric tensor of first layer is correct.
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

        assert qml.math.allclose(G[:3, :3], G1, atol=tol, rtol=0)

        # =============================================
        # Test block-diag metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G2 = np.zeros([2, 2])

        def layer2_diag(params):
            x, y, z, *_ = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        layer2_diag = qml.QNode(layer2_diag, dev)

        # calculate the diagonal terms
        varK0, varK1 = layer2_diag(params)
        G2[0, 0] = varK0 / 4
        G2[1, 1] = varK1 / 4

        assert qml.math.allclose(G[4:6, 4:6], G2, atol=tol, rtol=0)

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

        def layer3_diag(params):
            x, y, z, *_, f = params
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        layer3_diag = qml.QNode(layer3_diag, dev)
        G3 = layer3_diag(params) / 4
        assert qml.math.allclose(G[3:4, 3:4], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G3, G2)
        assert qml.math.allclose(G, G_expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "interface,array_cls",
        [
            pytest.param("jax", "array", marks=pytest.mark.jax),
            pytest.param("autograd", "array", marks=pytest.mark.autograd),
            pytest.param("tf", "Variable", marks=pytest.mark.tf),
            pytest.param("torch", "Tensor", marks=pytest.mark.torch),
        ],
    )
    def test_argnum_metric_tensor_interfaces(self, tol, interface, array_cls):
        """Test that argnum successfully reduces the number of tapes and gives
        the desired outcome."""
        if interface == "tf":
            interface_name = "tensorflow"
        elif interface == "jax":
            interface_name = "jax.numpy"
        elif interface == "autograd":
            interface_name = "numpy"
        else:
            interface_name = interface

        mod = importlib.import_module(interface_name)
        type_ = type(getattr(mod, array_cls)([])) if interface != "tf" else getattr(mod, "Tensor")

        dev = qml.device("default.qubit", wires=3)

        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[2], wires=1)
            qml.RZ(weights[3], wires=0)

        weights = getattr(mod, array_cls)([0.1, 0.2, 0.3, 0.5])

        with qml.tape.QuantumTape() as tape:
            circuit(weights)

        tapes, proc_fn = qml.metric_tensor(tape)
        res = qml.execute(tapes, dev, None)
        mt = proc_fn(res)

        tapes, proc_fn = qml.metric_tensor(tape, argnum=(0, 1, 3))
        res = qml.execute(tapes, dev, None, interface=interface)
        mt013 = proc_fn(res)
        assert isinstance(mt013, type_)

        assert len(tapes) == 6
        assert mt.shape == mt013.shape
        assert qml.math.allclose(mt[:2, :2], mt013[:2, :2], atol=tol, rtol=0)
        assert qml.math.allclose(mt[3, 3], mt013[3, 3], atol=tol, rtol=0)
        assert qml.math.allclose(0, mt013[2, :], atol=tol, rtol=0)
        assert qml.math.allclose(0, mt013[:, 2], atol=tol, rtol=0)

        tapes, proc_fn = qml.metric_tensor(tape, argnum=(2, 3))
        res = qml.execute(tapes, dev, None, interface=interface)
        mt23 = proc_fn(res)
        assert isinstance(mt23, type_)

        assert len(tapes) == 1
        assert mt.shape == mt23.shape
        assert qml.math.allclose(mt[2:, 2:], mt23[2:, 2:], atol=tol, rtol=0)
        assert qml.math.allclose(0, mt23[:2, :], atol=tol, rtol=0)
        assert qml.math.allclose(0, mt23[:, :2], atol=tol, rtol=0)

        tapes, proc_fn = qml.metric_tensor(tape, argnum=0)
        res = qml.execute(tapes, dev, None, interface=interface)
        mt0 = proc_fn(res)
        assert isinstance(mt0, type_)

        assert len(tapes) == 1
        assert mt.shape == mt0.shape
        assert qml.math.allclose(mt[0, 0], mt0[0, 0], atol=tol, rtol=0)
        assert qml.math.allclose(0, mt0[1:, :], atol=tol, rtol=0)
        assert qml.math.allclose(0, mt0[:, 1:], atol=tol, rtol=0)

    def test_argnum_metric_tensor_errors(self):
        """Test that argnum successfully reduces the number of tapes and gives
        the desired outcome."""

        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[2], wires=1)
            qml.RZ(weights[3], wires=0)

        weights = np.array([0.1, 0.2, 0.3, 0.5], requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            circuit(weights)

        error_msg = (
            "Some parameters specified in argnum are not in the "
            r"trainable parameters \[0, 1, 2, 3\] of the tape "
            "and will be ignored. This may be caused by attempting to "
            "differentiate with respect to parameters that are not marked "
            "as trainable."
        )
        with pytest.raises(ValueError, match=error_msg):
            qml.metric_tensor(tape, argnum=4)

    def test_multi_qubit_gates(self):
        """Test that a tape with Ising gates has the correct metric tensor tapes."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            qml.Hadamard(2)
            qml.IsingXX(0.2, wires=[0, 1])
            qml.IsingXX(-0.6, wires=[1, 2])
            qml.IsingZZ(1.02, wires=[0, 1])
            qml.IsingZZ(-4.2, wires=[1, 2])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.metric_tensor(tape, approx="block-diag")
        assert len(tapes) == 4
        assert [len(tape.operations) for tape in tapes] == [3, 5, 4, 5]
        assert [len(tape.measurements) for tape in tapes] == [1] * 4
        expected_ops = [
            [qml.Hadamard, qml.Hadamard, qml.Hadamard],
            [qml.Hadamard, qml.Hadamard, qml.IsingXX, qml.Hadamard, qml.Hadamard],
            [qml.Hadamard, qml.Hadamard, qml.IsingXX, qml.IsingXX],
            [qml.Hadamard, qml.Hadamard, qml.IsingXX, qml.IsingXX, qml.IsingZZ],
        ]
        assert [[type(op) for op in tape.operations] for tape in tapes] == expected_ops

    @pytest.mark.autograd
    @pytest.mark.filterwarnings("ignore:Attempted to compute the metric tensor")
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    def test_no_trainable_params_qnode_autograd(self, interface):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.metric_tensor(circuit)(weights)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_no_trainable_params_qnode_torch(self, interface):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.metric_tensor(circuit)(weights)

    @pytest.mark.tf
    @pytest.mark.filterwarnings("ignore:Attempted to compute the metric tensor")
    @pytest.mark.parametrize("interface", ["auto"])
    def test_no_trainable_params_qnode_tf(self, interface):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.metric_tensor(circuit)(weights)

    @pytest.mark.jax
    @pytest.mark.filterwarnings("ignore:Attempted to compute the metric tensor")
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_no_trainable_params_qnode_jax(self, interface):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.metric_tensor(circuit)(weights)

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=3)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="tensor of a tape with no trainable parameters"):
            mt_tapes, post_processing = qml.metric_tensor(tape)
        res = post_processing(qml.execute(mt_tapes, dev, None))

        assert mt_tapes == []  # pylint: disable=use-implicit-booleaness-not-comparison
        assert res == ()


fixed_pars = [-0.2, 0.2, 0.5, 0.3, 0.7]


def fubini_ansatz0(params, wires=None):
    qml.RX(params[0], wires=0)
    qml.RY(fixed_pars[0], wires=0)
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[1], wires=0)
    qml.CNOT(wires=[wires[0], wires[1]])


def fubini_ansatz1(params, wires=None):
    qml.RX(fixed_pars[1], wires=0)
    for wire in wires:
        qml.Rot(*params[0][wire], wires=wire)
    qml.CNOT(wires=[0, 1])
    qml.RY(fixed_pars[1], wires=0)
    qml.CNOT(wires=[1, 2])
    for wire in wires:
        qml.Rot(*params[1][wire], wires=wire)
    qml.CNOT(wires=[1, 2])
    qml.RX(fixed_pars[2], wires=1)


def fubini_ansatz2(params, wires=None):
    # pylint: disable=unused-argument
    params0 = params[0]
    params1 = params[1]
    qml.RX(fixed_pars[1], wires=0)
    qml.Rot(*fixed_pars[2:5], wires=1)
    qml.CNOT(wires=[0, 1])
    _ = qml.RY(params0, wires=0) ** 0.4
    qml.RY(params0, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.RX)(params1, wires=0)
    qml.adjoint(qml.RX(params1, wires=1))


def fubini_ansatz3(params, wires=None):
    # pylint: disable=unused-argument
    params0 = params[0]
    params1 = params[1]
    params2 = params[2]
    qml.RX(fixed_pars[1], wires=0)
    qml.RX(fixed_pars[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.adjoint(qml.RX(params0, wires=0))
    qml.RX(params0, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    qml.RY(params1, wires=0)
    qml.RY(params1, wires=1)
    qml.RY(params1, wires=2)
    qml.RZ(params2, wires=0)
    qml.RZ(params2, wires=1)
    qml.RZ(params2, wires=2)


def fubini_ansatz4(params00, params_rest, wires=None):
    # pylint: disable=unused-argument
    params01 = params_rest[0]
    params10 = params_rest[1]
    params11 = params_rest[2]
    qml.RY(fixed_pars[3], wires=0)
    qml.RY(fixed_pars[2], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RY(fixed_pars[4], wires=0)
    qml.RX(params00, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params01, wires=1)
    qml.RZ(params10, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params11, wires=1)


def fubini_ansatz5(params, wires=None):
    fubini_ansatz4(params[0], [params[0], params[1], params[1]], wires=wires)


def fubini_ansatz6(params, wires=None):
    fubini_ansatz4(params[0], [params[0], params[1], -params[1]], wires=wires)


def fubini_ansatz7(x, wires=None):
    qml.RX(fixed_pars[0], wires=wires[0])
    qml.RX(x, wires=0)


def fubini_ansatz8(params, wires=None):
    # pylint: disable=unused-argument
    params0 = params[0]
    params1 = params[1]
    qml.RX(fixed_pars[1], wires=[0])
    qml.RY(fixed_pars[3], wires=[0])
    qml.RZ(fixed_pars[2], wires=[0])
    qml.RX(fixed_pars[2], wires=[1])
    qml.RY(fixed_pars[2], wires=[1])
    qml.RZ(fixed_pars[4], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RX(fixed_pars[0], wires=[0])
    qml.RY(fixed_pars[1], wires=[0])
    qml.RZ(fixed_pars[3], wires=[0])
    qml.RX(fixed_pars[1], wires=[1])
    qml.RY(fixed_pars[2], wires=[1])
    qml.RZ(fixed_pars[0], wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RX(params0, wires=[0])
    qml.RX(params0, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RY(fixed_pars[4], wires=[1])
    qml.RY(params1, wires=[0])
    qml.RY(params1, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RX(fixed_pars[2], wires=[1])


fubini_ansatze = [
    fubini_ansatz0,
    fubini_ansatz1,
    fubini_ansatz2,
    fubini_ansatz3,
    fubini_ansatz4,
    fubini_ansatz5,
    fubini_ansatz6,
    fubini_ansatz7,
    fubini_ansatz8,
]

B = np.array(
    [
        [
            [0.73, 0.49, 0.04],
            [0.29, 0.45, 0.59],
            [0.64, 0.06, 0.26],
        ],
        [
            [0.93, 0.14, 0.46],
            [0.31, 0.83, 0.79],
            [0.25, 0.40, 0.16],
        ],
    ],
    requires_grad=True,
)
fubini_params = [
    (np.array([0.3434, -0.7245345], requires_grad=True),),
    (B,),
    (np.array([-0.1111, -0.2222], requires_grad=True),),
    (np.array([-0.1111, -0.2222, 0.4554], requires_grad=True),),
    (
        np.array(-0.1735, requires_grad=True),
        np.array([-0.1735, -0.2846, -0.2846], requires_grad=True),
    ),
    (np.array([-0.1735, -0.2846], requires_grad=True),),
    (np.array([-0.1735, -0.2846], requires_grad=True),),
    (np.array(-0.1735, requires_grad=True),),
    (np.array([-0.1111, 0.3333], requires_grad=True),),
]


def autodiff_metric_tensor(ansatz, num_wires):
    """Compute the metric tensor by full state vector
    differentiation via autograd."""
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def qnode(*params):
        ansatz(*params, wires=dev.wires)
        return qml.state()

    def mt(*params):
        state = qnode(*params)

        def rqnode(*params):
            return np.real(qnode(*params))

        def iqnode(*params):
            return np.imag(qnode(*params))

        rjac = qml.jacobian(rqnode)(*params)
        ijac = qml.jacobian(iqnode)(*params)

        if isinstance(rjac, tuple):
            out = []
            for rc, ic in zip(rjac, ijac):
                c = rc + 1j * ic
                psidpsi = np.tensordot(np.conj(state), c, axes=([0], [0]))
                out.append(
                    np.real(
                        np.tensordot(np.conj(c), c, axes=([0], [0]))
                        - np.tensordot(np.conj(psidpsi), psidpsi, axes=0)
                    )
                )
            return tuple(out)

        jac = rjac + 1j * ijac
        psidpsi = np.tensordot(np.conj(state), jac, axes=([0], [0]))
        return np.real(
            np.tensordot(np.conj(jac), jac, axes=([0], [0]))
            - np.tensordot(np.conj(psidpsi), psidpsi, axes=0)
        )

    return mt


class TestFullMetricTensor:
    num_wires = 3

    @pytest.mark.external
    def test_catalyst_compatibility(self):
        """Test that the metric tensor can be executed with catalyst."""

        pytest.importorskip("catalyst")
        jax = pytest.importorskip("jax")

        def ansatz(params, wires=None):  # pylint: disable=unused-argument
            qml.RX(params[0], 0)
            qml.RX(params[1], 1)
            qml.CNOT((0, 1))
            qml.RX(params[2], 0)
            qml.RY(params[3], 1)

        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def circuit(params):
            ansatz(params)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        x = jax.numpy.array([1.0, 2.0, 3.0, 4.0])

        qjit_res = qml.qjit(qml.metric_tensor(circuit))(x)

        cost_autograd = autodiff_metric_tensor(ansatz, num_wires=3)(qml.numpy.array(x))

        assert qml.math.allclose(qjit_res, cost_autograd)

    @pytest.mark.autograd
    @pytest.mark.parametrize("ansatz, params", zip(fubini_ansatze, fubini_params))
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    @pytest.mark.parametrize("dev_name", ("default.qubit", "lightning.qubit"))
    def test_correct_output_autograd(self, dev_name, ansatz, params, interface):

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qml.device(dev_name, wires=self.num_wires + 1)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires[:-1])
            return qml.expval(qml.PauliZ(0))

        mt = qml.metric_tensor(circuit, approx=None)(*params)

        if isinstance(mt, tuple):
            assert all(qml.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qml.math.allclose(mt, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("ansatz, params", zip(fubini_ansatze, fubini_params))
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    @pytest.mark.parametrize("dev_name", ("default.qubit", "lightning.qubit"))
    @pytest.mark.parametrize("use_jit", [False, True])
    def test_correct_output_jax(self, dev_name, ansatz, params, interface, use_jit):
        import jax
        from jax import numpy as jnp

        if ansatz == fubini_ansatz2:
            pytest.xfail("Issue involving trainable indices to be resolved.")
        if ansatz == fubini_ansatz3 and dev_name == "lightning.qubit":
            pytest.xfail("Issue invovling trainable_params to be resolved.")

        jax.config.update("jax_enable_x64", True)

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qml.device(dev_name, wires=self.num_wires + 1)

        params = tuple(jnp.array(p) for p in params)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires[:-1])
            return qml.expval(qml.PauliZ(0))

        argnums = range(0, len(params)) if len(params) > 1 else None
        # pylint:disable=unexpected-keyword-arg
        mt_fn = qml.metric_tensor(circuit, argnums=argnums, approx=None)
        if use_jit:
            mt_fn = jax.jit(mt_fn)
        mt = mt_fn(*params)

        if isinstance(mt, tuple):
            assert all(qml.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qml.math.allclose(mt, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("ansatz, params", zip(fubini_ansatze, fubini_params))
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    @pytest.mark.parametrize("dev_name", ("default.qubit", "lightning.qubit"))
    def test_jax_argnum_error(self, dev_name, ansatz, params, interface):
        from jax import numpy as jnp

        dev = qml.device(dev_name, wires=self.num_wires + 1)

        params = tuple(jnp.array(p) for p in params)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires[:-1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            QuantumFunctionError,
            match="argnum does not work with the Jax interface. You should use argnums instead.",
        ):
            qml.metric_tensor(circuit, argnum=range(len(params)), approx=None)(*params)

    @pytest.mark.torch
    @pytest.mark.parametrize("ansatz, params", zip(fubini_ansatze, fubini_params))
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    @pytest.mark.parametrize("dev_name", ("default.qubit", "lightning.qubit"))
    def test_correct_output_torch(self, dev_name, ansatz, params, interface):
        import torch

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qml.device(dev_name, wires=self.num_wires + 1)

        params = tuple(torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in params)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires[:-1])
            return qml.expval(qml.PauliZ(0))

        mt = qml.metric_tensor(circuit, approx=None)(*params)

        if isinstance(mt, tuple):
            assert all(qml.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qml.math.allclose(mt, expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("ansatz, params", zip(fubini_ansatze, fubini_params))
    @pytest.mark.parametrize("interface", ["auto"])
    @pytest.mark.parametrize("dev_name", ("default.qubit", "lightning.qubit"))
    def test_correct_output_tf(self, dev_name, ansatz, params, interface):
        import tensorflow as tf

        expected = autodiff_metric_tensor(ansatz, self.num_wires)(*params)
        dev = qml.device(dev_name, wires=self.num_wires + 1)

        params = tuple(tf.Variable(p, dtype=tf.float64) for p in params)

        @qml.qnode(dev, interface=interface)
        def circuit(*params):
            """Circuit with dummy output to create a QNode."""
            ansatz(*params, dev.wires[:-1])
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape():
            qml.metric_tensor(circuit, approx="block-diag")(*params)
            mt = qml.metric_tensor(circuit, approx=None)(*params)

        if isinstance(mt, tuple):
            assert all(qml.math.allclose(_mt, _exp) for _mt, _exp in zip(mt, expected))
        else:
            assert qml.math.allclose(mt, expected)


def diffability_ansatz_0(weights, wires=None):
    # pylint: disable=unused-argument
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=1)


def expected_diag_jac_0(weights):
    return np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [
                np.cos(weights[0] + weights[1]) * np.sin(weights[0] + weights[1]) / 2,
                np.cos(weights[0] + weights[1]) * np.sin(weights[0] + weights[1]) / 2,
                0,
            ],
        ]
    )


def diffability_ansatz_1(weights, wires=None):
    # pylint: disable=unused-argument
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=1)


def expected_diag_jac_1(weights):
    return np.array(
        [
            [0, 0, 0],
            [-np.sin(2 * weights[0]) / 4, 0, 0],
            [
                np.cos(weights[0]) * np.cos(weights[1]) ** 2 * np.sin(weights[0]) / 2,
                np.cos(weights[0]) ** 2 * np.sin(2 * weights[1]) / 4,
                0,
            ],
        ]
    )


def diffability_ansatz_2(weights, wires=None):
    # pylint: disable=unused-argument
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=1)


def expected_diag_jac_2(weights):
    return np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [
                np.cos(weights[1]) ** 2 * np.sin(2 * weights[0]) / 4,
                np.cos(weights[0]) ** 2 * np.sin(2 * weights[1]) / 4,
                0,
            ],
        ]
    )


weights_diff = np.array([0.432, 0.12, -0.292], requires_grad=True)


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "ansatz, weights, expected_diag_jac",
    [
        (diffability_ansatz_0, (weights_diff,), expected_diag_jac_0),
        (diffability_ansatz_1, (weights_diff,), expected_diag_jac_1),
        (diffability_ansatz_2, (weights_diff,), expected_diag_jac_2),
    ],
)
class TestDifferentiabilityDiag:
    """Test for diagonal metric tensor differentiability"""

    def get_circuit(self, ansatz):
        def circuit(*weights):
            ansatz(*weights)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        return circuit

    dev = qml.device("default.qubit", wires=3)

    @pytest.mark.autograd
    @pytest.mark.filterwarnings("ignore:Attempted to compute the gradient")
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    def test_autograd_diag(self, diff_method, tol, ansatz, weights, expected_diag_jac, interface):
        """Test metric tensor differentiability in the autograd interface"""
        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)
        qnode(*weights)

        def cost_diag(*weights):
            mt = qml.metric_tensor(qnode, approx="block-diag")(*weights)
            if isinstance(mt, tuple):
                diag = qml.math.hstack(
                    [qml.math.diag(_mt) if len(qml.math.shape(_mt)) == 2 else _mt for _mt in mt]
                )

            else:
                diag = qml.math.diag(mt)
            return diag

        jac = qml.jacobian(cost_diag)(*weights)
        if isinstance(jac, tuple):
            assert all(
                qml.math.allclose(j, e, atol=tol, rtol=0)
                for j, e in zip(jac, expected_diag_jac(*weights))
            )
        else:
            assert qml.math.allclose(jac, expected_diag_jac(*weights), atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_jax_diag(self, diff_method, tol, ansatz, weights, expected_diag_jac, interface):
        """Test metric tensor differentiability in the JAX interface"""
        if diff_method == "parameter-shift":
            pytest.skip("Does not support parameter-shift")

        import jax
        from jax import numpy as jnp

        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)

        def cost_diag(*weights):
            return jnp.diag(qml.metric_tensor(qnode, approx="block-diag")(*weights))

        jac = jax.jacobian(cost_diag)(jnp.array(*weights))
        assert qml.math.allclose(jac, expected_diag_jac(*weights), atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf_diag(self, diff_method, tol, ansatz, weights, expected_diag_jac, interface):
        """Test metric tensor differentiability in the TF interface"""
        import tensorflow as tf

        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)

        weights_t = tuple(tf.Variable(w) for w in weights)
        with tf.GradientTape() as tape:
            loss_diag = tf.linalg.diag_part(
                qml.metric_tensor(qnode, approx="block-diag")(*weights_t)
            )
        jac = tape.jacobian(loss_diag, weights_t)
        assert qml.math.allclose(jac, expected_diag_jac(*weights), atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_torch_diag(self, diff_method, tol, ansatz, weights, expected_diag_jac, interface):
        """Test metric tensor differentiability in the torch interface"""
        import torch

        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)

        weights_t = tuple(torch.tensor(w, requires_grad=True) for w in weights)

        def cost_diag(*weights):
            mt = qml.metric_tensor(qnode, approx="block-diag")(*weights)
            if isinstance(mt, tuple):
                diag = qml.math.hstack(
                    [qml.math.diag(_mt) if len(qml.math.shape(_mt)) == 2 else _mt for _mt in mt]
                )

            else:
                diag = qml.math.diag(mt)
            return diag

        jac = torch.autograd.functional.jacobian(cost_diag, weights_t)

        if isinstance(jac, tuple) and len(jac) != 1:
            assert all(
                qml.math.allclose(j.detach().numpy(), e, atol=tol, rtol=0)
                for j, e in zip(jac, expected_diag_jac(*weights))
            )
        else:
            if isinstance(jac, tuple) and len(jac) == 1:
                jac = jac[0]
            assert qml.math.allclose(
                jac.detach().numpy(), expected_diag_jac(*weights), atol=tol, rtol=0
            )


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
@pytest.mark.parametrize(
    "ansatz, weights",
    [
        (diffability_ansatz_0, (weights_diff,)),
        (diffability_ansatz_1, (weights_diff,)),
        (diffability_ansatz_2, (weights_diff,)),
    ],
)
class TestDifferentiability:
    """Test for metric tensor differentiability"""

    def get_circuit(self, ansatz):
        def circuit(*weights):
            ansatz(*weights)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        return circuit

    dev = qml.device("default.qubit", wires=3)

    @pytest.mark.autograd
    @pytest.mark.filterwarnings("ignore:Attempted to compute the gradient")
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    def test_autograd(self, diff_method, tol, ansatz, weights, interface):
        """Test metric tensor differentiability in the autograd interface"""
        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)

        def cost_full(*weights):
            return np.array(qml.metric_tensor(qnode, approx=None)(*weights))

        def _cost_full(*weights):
            return np.array(autodiff_metric_tensor(ansatz, 3)(*weights))

        _c = _cost_full(*weights)
        c = cost_full(*weights)
        assert all(
            qml.math.allclose(_sub_c, sub_c, atol=tol, rtol=0) for _sub_c, sub_c in zip(_c, c)
        )
        for argnum in range(len(weights)):
            expected_full = qml.jacobian(_cost_full, argnum=argnum)(*weights)
            jac = qml.jacobian(cost_full, argnum=argnum)(*weights)
            assert qml.math.allclose(expected_full, jac, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, diff_method, tol, ansatz, weights):
        """Test metric tensor differentiability in the JAX interface"""
        if diff_method == "parameter-shift":
            pytest.skip("Does not support parameter-shift")

        import jax

        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface="jax", diff_method=diff_method)

        def cost_full(*weights):
            return qml.metric_tensor(qnode, approx=None)(*weights)

        weights_jax = tuple(jax.numpy.array(w) for w in weights)
        _cost_full_autograd = autodiff_metric_tensor(ansatz, num_wires=3)
        v1 = _cost_full_autograd(*weights)
        v2 = cost_full(*weights_jax)
        assert qml.math.allclose(v1, v2, atol=tol, rtol=0)
        jac = jax.jacobian(cost_full)(*weights_jax)
        expected_full = qml.jacobian(_cost_full_autograd)(*weights)
        assert qml.math.allclose(expected_full, jac, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["auto"])
    def test_tf(self, diff_method, tol, ansatz, weights, interface):
        """Test metric tensor differentiability in the TF interface"""
        import tensorflow as tf

        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)

        weights_t = tuple(tf.Variable(w) for w in weights)
        with tf.GradientTape() as tape:
            loss_full = qml.metric_tensor(qnode, approx=None)(*weights_t)
        jac = tape.jacobian(loss_full, weights_t)
        _cost_full = autodiff_metric_tensor(ansatz, num_wires=3)
        assert qml.math.allclose(_cost_full(*weights), loss_full, atol=tol, rtol=0)
        expected_full = qml.jacobian(_cost_full)(*weights)
        assert qml.math.allclose(expected_full, jac, atol=tol, rtol=0)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_torch(self, diff_method, tol, ansatz, weights, interface):
        """Test metric tensor differentiability in the torch interface"""
        import torch

        circuit = self.get_circuit(ansatz)
        qnode = qml.QNode(circuit, self.dev, interface=interface, diff_method=diff_method)
        weights_t = tuple(torch.tensor(w, requires_grad=True) for w in weights)
        qnode(*weights_t)
        cost_full = qml.metric_tensor(qnode, approx=None)
        _cost_full = autodiff_metric_tensor(ansatz, num_wires=3)
        jac = torch.autograd.functional.jacobian(cost_full, weights_t)
        expected_full = qml.jacobian(_cost_full)(*weights)
        assert qml.math.allclose(
            _cost_full(*weights), cost_full(*weights_t).detach().numpy(), atol=tol, rtol=0
        )
        if isinstance(jac, tuple) and len(jac) != 1:
            assert all(
                qml.math.allclose(j.detach().numpy(), e, atol=tol, rtol=0)
                for j, e in zip(jac, expected_full)
            )
        else:
            if isinstance(jac, tuple) and len(jac) == 1:
                jac = jac[0]
            assert qml.math.allclose(jac.detach().numpy(), expected_full, atol=tol, rtol=0)


@pytest.mark.parametrize("approx", [True, False, "Invalid", 2])
def test_invalid_value_for_approx(approx):
    """Test exception is raised if ``approx`` is invalid."""
    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(np.array(0.5, requires_grad=True), wires=0)
        qml.expval(qml.PauliX(0))

    tape = qml.tape.QuantumScript.from_queue(q)
    with pytest.raises(ValueError, match="keyword argument approx"):
        qml.metric_tensor(tape, approx=approx)


def test_generator_no_expval(monkeypatch):
    """Test exception is raised if subcircuit contains an
    operation with generator object that is not an observable"""
    with monkeypatch.context() as m:
        m.setattr("pennylane.RX.generator", lambda self: qml.RX(0.1, wires=0))

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(np.array(0.5, requires_grad=True), wires=0)
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.raises(QuantumFunctionError, match="is not hermitian"):
            qml.metric_tensor(tape, approx="block-diag")


def test_error_missing_aux_wire():
    """Tests that a special error is raised if the requested (or default, if not given)
    auxiliary wire for the Hadamard test is missing."""
    dev = qml.device("default.qubit", wires=qml.wires.Wires(["wire1", "wire2"]))

    @qml.qnode(dev)
    def circuit(x, z):
        qml.RX(x, wires="wire1")
        qml.RZ(z, wires="wire2")
        qml.CNOT(wires=["wire1", "wire2"])
        qml.RX(x, wires="wire1")
        qml.RZ(z, wires="wire2")
        return qml.expval(qml.PauliZ("wire2"))

    x = np.array(0.5, requires_grad=True)
    z = np.array(0.1, requires_grad=True)

    with pytest.raises(
        qml.wires.WireError, match="The device has no free wire for the auxiliary wire."
    ):
        qml.metric_tensor(circuit, approx=None)(x, z)


def test_error_not_available_aux_wire():
    """Tests that a special error is raised if aux wires is not available."""

    dev = qml.device("default.qubit", wires=1)
    x = np.array(0.5, requires_grad=True)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(
        qml.wires.WireError, match="The requested auxiliary wire does not exist on the used device."
    ):
        qml.metric_tensor(circuit, aux_wire=404)(x)


@pytest.mark.parametrize("allow_nonunitary", [True, False])
def test_error_generator_not_registered(allow_nonunitary):
    """Tests that an error is raised if an operation doe not have a
    controlled-generator operation registered."""
    dev = qml.device("default.qubit", wires=qml.wires.Wires(["wire1", "wire2", "wire3"]))

    x = np.array(0.5, requires_grad=True)
    z = np.array(0.1, requires_grad=True)

    class RX(qml.RX):
        def generator(self):
            return qml.Hadamard(self.wires)

    @qml.qnode(dev)
    def circuit1(x, z):
        RX(x, wires="wire1")
        qml.RZ(z, wires="wire1")
        return qml.expval(qml.PauliZ("wire2"))

    if allow_nonunitary:
        qml.metric_tensor(circuit1, approx=None, allow_nonunitary=allow_nonunitary)(x, z)
    else:
        with pytest.raises(ValueError, match="Generator for operation"):
            qml.metric_tensor(circuit1, approx=None, allow_nonunitary=allow_nonunitary)(x, z)


def test_no_error_missing_aux_wire_not_used():
    """Tests that a no error is raised if the requested (or default, if not given)
    auxiliary wire for the Hadamard test is missing but it is not used, either
    because ``approx`` is used or because there only is a diagonal contribution."""
    dev = qml.device("default.qubit", wires=qml.wires.Wires(["wire1", "wire2"]))

    @qml.qnode(dev)
    def circuit_single_block(x, z):
        """This circuit has a metric tensor that consists
        of a single block in the block diagonal "approximation"."""
        qml.RX(x, wires="wire1")
        qml.RZ(z, wires="wire2")
        qml.CNOT(wires=["wire1", "wire2"])
        return qml.expval(qml.PauliZ("wire2"))

    @qml.qnode(dev)
    def circuit_multi_block(x, z):
        """This circuit has a metric tensor that consists
        of multiple blocks and thus is approximated when only
        computing the block diagonal."""
        qml.RX(x, wires="wire1")
        qml.RZ(z, wires="wire2")
        qml.CNOT(wires=["wire1", "wire2"])
        qml.RX(x, wires="wire1")
        qml.RZ(z, wires="wire2")
        return qml.expval(qml.PauliZ("wire2"))

    x = np.array(0.5, requires_grad=True)
    z = np.array(0.1, requires_grad=True)

    qml.metric_tensor(circuit_single_block, approx=None)(x, z)
    qml.metric_tensor(circuit_single_block, approx=None, aux_wire="aux_wire")(x, z)
    qml.metric_tensor(circuit_multi_block, approx="block-diag")(x, z)
    qml.metric_tensor(circuit_multi_block, approx="block-diag", aux_wire="aux_wire")(x, z)


def test_raises_circuit_that_uses_missing_wire():
    """Test that an error in the original circuit is reraised properly and not caught. This avoids
    accidentally catching relevant errors, which can lead to a recursion error."""

    dev = qml.device("default.qubit", wires=[0, "b"])

    @qml.qnode(dev)
    def circuit(x):
        """Flawed circuit that uses a wire which is not on the device."""
        qml.RX(x[0], 0)
        qml.CNOT([0, 1])  # wire 1 is not on the device
        qml.RX(x[1], 0)
        return qml.expval(qml.PauliZ(0))

    x = np.array([1.3, 0.2])
    with pytest.raises(qml.wires.WireError, match=r"no free wire"):
        qml.metric_tensor(circuit)(x)


def aux_wire_ansatz_0(x, y):
    qml.RX(x, wires=0)
    qml.RY(y, wires=2)


def aux_wire_ansatz_1(x, y):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)


@pytest.mark.parametrize("aux_wire", [None, "aux", 3])
@pytest.mark.parametrize("ansatz", [aux_wire_ansatz_0, aux_wire_ansatz_1])
def test_get_aux_wire(aux_wire, ansatz):
    """Test ``_get_aux_wire`` without device_wires."""
    x, y = np.array([0.2, 0.1], requires_grad=True)
    with qml.queuing.AnnotatedQueue() as q:
        ansatz(x, y)
    tape = qml.tape.QuantumScript.from_queue(q)
    out = _get_aux_wire(aux_wire, tape, None)

    if aux_wire is not None:
        assert out == aux_wire
    else:
        assert out == (1 if 1 not in tape.wires else 2)


def test_get_aux_wire_with_device_wires():
    """Test ``_get_aux_wire`` with device_wires."""
    x, y = np.array([0.2, 0.1], requires_grad=True)
    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(x, wires=0)
        qml.RX(y, wires="one")

    tape = qml.tape.QuantumScript.from_queue(q)
    device_wires = qml.wires.Wires([0, "aux", "one"])

    assert _get_aux_wire(None, tape, device_wires) == "aux"
    assert _get_aux_wire("aux", tape, device_wires) == "aux"
    _match = "The requested auxiliary wire is already in use by the circuit."
    with pytest.raises(qml.wires.WireError, match=_match):
        _get_aux_wire("one", tape, device_wires)
    with pytest.raises(qml.wires.WireError, match=_match):
        _get_aux_wire(0, tape, device_wires)


def test_get_aux_wire_with_unavailable_aux():
    """Test ``_get_aux_wire`` with device_wires and a requested ``aux_wire`` that is missing."""
    x, y = np.array([0.2, 0.1], requires_grad=True)
    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(x, wires=0)
        qml.RX(y, wires="one")
    tape = qml.tape.QuantumScript.from_queue(q)
    device_wires = qml.wires.Wires([0, "one"])
    with pytest.raises(qml.wires.WireError, match="The requested auxiliary wire does not exist"):
        _get_aux_wire("two", tape, device_wires)


def test_metric_tensor_repeated_parametrized_op():
    """Test that metric tensor works when an operator is repeated."""
    op = qml.RX(0.5, 0)
    tape1 = qml.tape.QuantumScript([op, op], [qml.expval(qml.Z(0))])
    tape2 = qml.tape.QuantumScript([op, qml.RX(0.5, 0)], [qml.expval(qml.Z(0))])

    batch1, _ = qml.metric_tensor(tape1)
    batch2, _ = qml.metric_tensor(tape2)

    for t1, t2 in zip(batch1, batch2, strict=True):
        qml.assert_equal(t1, t2)
