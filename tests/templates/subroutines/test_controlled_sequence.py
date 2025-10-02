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
Unit tests for the ControlledSequence subroutine.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.wires import Wires

# pylint: disable=unidiomatic-typecheck, cell-var-from-loop


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])
    qml.ops.functions.assert_valid(op)


class TestInitialization:

    def test_id(self):
        """Tests that the id attribute can be set."""
        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2], id="a")
        assert op.id == "a"

    def test_overlapping_wires_error(self):
        """Test that an error is raised if the wires of the base
        operator and the control wires overlap"""

        base = qml.PauliX(1)
        control = [0, 1, 2]

        with pytest.raises(
            ValueError, match="The control wires must be different from the base operation wires."
        ):
            _ = qml.ControlledSequence(base, control)

    def test_name(self):
        """Test that the name for the operator is ControlledSequence (must be overwritten for SymbolicOps)"""

        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])

        assert op.name == "ControlledSequence"


class TestProperties:

    def test_hash(self):
        """Test that op.hash uniquely describes a ControlledSequence"""

        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])
        op1 = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])  # identical
        op2 = qml.ControlledSequence(
            qml.RX(0.25, wires=4), control=[0, 1, 2]
        )  # different base wire
        op3 = qml.ControlledSequence(
            qml.RX(0.35, wires=3), control=[0, 1, 2]
        )  # different base param
        op4 = qml.ControlledSequence(
            qml.RY(0.25, wires=3), control=[0, 1, 2]
        )  # different base class
        op5 = qml.ControlledSequence(
            qml.RX(0.25, wires=3), control=[0, 1, 4]
        )  # different control wires

        assert hash(op) == hash(op1)
        for other_op in [op2, op3, op4, op5]:
            assert hash(op) != hash(other_op)

    def test_control(self):
        """Test that the control property returns control wires"""
        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=["a", 1, "blue"])
        assert op.control == Wires(["a", 1, "blue"])

    def test_wires(self):
        """Test that the wires property returns all wires, including both base and control"""
        op = qml.ControlledSequence(qml.CNOT([17, "3"]), control=["b", 2])
        assert op.wires == Wires(["b", 2, 17, "3"])

    def test_has_matrix(self):
        """Test that a ControlledSequence returns False for has_matrix, even if the base returns True"""
        op = qml.ControlledSequence(qml.PauliX(0), control=[1])
        assert op.has_matrix is False
        assert op.base.has_matrix is True


class TestMethods:

    def test_repr(self):
        """Test that the operator repr is as expected"""
        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])
        assert repr(op) == f"ControlledSequence({op.base}, control=[0, 1, 2])"

    def test_map_wires(self):
        """Test mapping wires works as expected, including mapping wires on the base operator"""

        op = qml.ControlledSequence(qml.CNOT([0, 1]), control=[2, 3])
        wire_map = {0: "a", 1: "b", 2: "c", 3: "d"}

        new_op = op.map_wires(wire_map)

        assert type(new_op) == type(op)
        assert type(new_op.base) == type(op.base)
        assert new_op.data == op.data

        assert new_op.wires == Wires(["c", "d", "a", "b"])
        assert new_op.base.wires == Wires(["a", "b"])
        assert new_op.control == Wires(["c", "d"])

    def test_compute_decomposition_lazy(self):
        """Test compute_decomposition with lazy=True"""
        base = qml.RZ(4.3, 1)
        control_wires = [0, 2, 3]

        decomp = qml.ControlledSequence.compute_decomposition(
            base=base, control_wires=control_wires, lazy=True
        )

        assert len(decomp) == len(control_wires)
        for i, op in enumerate(decomp):
            qml.assert_equal(op.base.base, base)
            assert isinstance(op, qml.ops.Pow)
            assert op.z == 2 ** (len(control_wires) - i - 1)

        for op, w in zip(decomp, control_wires):
            assert op.base.control_wires == Wires(w)

    def test_compute_decomposition_not_lazy(self):
        """Test compute_decomposition with lazy=False"""
        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=["a", 1, "blue"])

        decomp = op.compute_decomposition(base=op.base, control_wires=op.control, lazy=False)
        expected_decomp = [
            qml.CRX(0.25 * 4, wires=["a", 3]),
            qml.CRX(0.25 * 2, wires=[1, 3]),
            qml.CRX(0.25 * 1, wires=["blue", 3]),
        ]

        for op1, op2 in zip(decomp, expected_decomp):
            assert op1 == op2

    def test_decomposition(self):
        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=["a", 1, "blue"])

        decomp = op.decomposition()
        expected_decomp = [
            qml.CRX(0.25 * 4, wires=["a", 3]),
            qml.CRX(0.25 * 2, wires=[1, 3]),
            qml.CRX(0.25 * 1, wires=["blue", 3]),
        ]

        for op1, op2 in zip(decomp, expected_decomp):
            assert op1 == op2

    def test_decomposition_new(self):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.ControlledSequence(qml.RX(0.25, wires=3), control=["a", 1, "blue"])
        for rule in qml.list_decomps(qml.ControlledSequence):
            _test_decomposition_rule(op, rule)


class TestIntegration:
    """Tests that the ControlledSequence is executable and differentiable in a QNode context"""

    @staticmethod
    def circuit(x):
        """Test circuit"""
        qml.PauliX(2)
        qml.ControlledSequence(qml.RX(x, wires=3), control=[0, 1, 2])
        return qml.probs(wires=range(4))

    x = np.array(0.25)
    # not calculated analytically, we are only ensuring that the results are consistent accross interfaces
    exp_result = np.array([0, 0, 0.9835, 0.0165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    exp_jac = np.array([0, 0, -0.12342829, 0.12342829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_qnode_numpy(self):
        """Test that the QNode executes with Numpy."""
        dev = qml.device("default.qubit")
        qnode = qml.QNode(self.circuit, dev, interface=None)

        res = qnode(self.x)
        assert res.shape == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_autograd(self, shots, seed):
        """Test that the QNode executes with Autograd."""

        dev = qml.device("default.qubit", wires=4, seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="autograd", diff_method=diff_method), shots=shots
        )
        x = qml.numpy.array(self.x, requires_grad=True)

        res = qnode(x)
        assert qml.math.shape(res) == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

        res = qml.jacobian(qnode)(x)
        assert np.shape(res) == (16,)
        assert np.allclose(res, self.exp_jac, atol=0.005)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_jax(self, shots, use_jit, seed):
        """Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""

        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method), shots=shots
        )
        if use_jit:
            qnode = jax.jit(qnode)

        x = jax.numpy.array(self.x)
        res = qnode(x)
        assert qml.math.shape(res) == (16,)
        assert np.allclose(res, self.exp_result, atol=0.005)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(x)
        assert jac.shape == (16,)
        assert np.allclose(jac, self.exp_jac, atol=0.006)

    @pytest.mark.torch
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_torch(self, shots, seed):
        """Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""

        import torch

        dev = qml.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method), shots=shots
        )

        x = torch.tensor(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == (16,)
        assert qml.math.allclose(res, self.exp_result, atol=0.002)

        jac = torch.autograd.functional.jacobian(qnode, x)
        assert qml.math.shape(jac) == (16,)
        assert qml.math.allclose(jac, self.exp_jac, atol=0.005)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots", [None, 10000])
    @pytest.mark.xfail(reason="tf gradient doesn't seem to be working, returns ()")
    def test_qnode_tf(self, shots, seed):
        """Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""

        import tensorflow as tf

        dev = qml.device("default.qubit", seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method), shots=shots
        )

        x = tf.Variable(self.x)
        with tf.GradientTape() as tape:
            res = qnode(x)

        assert qml.math.shape(res) == (16,)
        assert qml.math.allclose(res, self.exp_result, atol=0.002)

        jac = tape.gradient(res, x)
        assert qml.math.shape(jac) == (16,)

    def test_prod_rx_rx_compiled_circuit(self):
        """Test that a circuit can execute successfully using qml.compile and
        a CompositeOp as a base"""

        @qml.prod
        def U(thetas):
            qml.RX(thetas[0], wires=0)
            qml.RX(thetas[1], wires=1)

        dev = qml.device("default.qubit", wires=4)

        @qml.compile
        @qml.qnode(dev)
        def circuit(thetas):
            qml.ControlledSequence(U(thetas), control=[2, 3])
            return qml.state()

        _ = circuit([1.0, 1.0])

    def test_approx_time_base(self):
        """Test that using ControlledSequence with ApproxTimeEvolution as a base in a
        circuit can be decomposed to execute successfully"""
        H = qml.Hamiltonian([1.0, 2.0], [qml.PauliZ(0), qml.PauliZ(1)])
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.ControlledSequence(qml.ApproxTimeEvolution(H, 1, 1), control=[2, 3])
            return qml.state()

        _ = circuit()

    def test_gradient_with_composite_op_base(self):
        """Test executing and getting the gradient of a circuit with a
        ControlledSequence based on a CompositeOp"""

        @qml.prod
        def U(thetas):
            qml.RX(thetas[0], wires=0)
            qml.RX(thetas[1], wires=1)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(thetas):
            qml.ControlledSequence(U(thetas), control=[2, 3])
            return qml.expval(qml.PauliZ(0))

        thetas = pnp.array([1.0, 1.0], requires_grad=True)
        _ = circuit(thetas)
        _ = qml.grad(circuit)(thetas)


class TestQPEResults:
    """Tests phase estimation using the template"""

    @pytest.mark.parametrize("phase", [7, np.pi, np.pi / 3, 2.3])
    def test_phase_estimated_single_ops(self, phase):
        """Tests that the QPE defined using ControlledSequence works correctly for a single operator"""

        unitary = qml.RX(phase, wires=[0])
        estimation_wires = range(1, 6)
        dev = qml.device("default.qubit", wires=range(6))

        @qml.qnode(dev)
        def circuit():
            for i in estimation_wires:
                qml.Hadamard(wires=i)

            qml.ControlledSequence(unitary, control=estimation_wires)

            qml.adjoint(qml.QFT)(wires=estimation_wires)

            return qml.state()

        qpe = qml.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)

        assert np.allclose(qml.matrix(circuit)(), qml.matrix(qpe))

    @pytest.mark.parametrize("phase", [7, np.pi, np.pi / 3, 2.3])
    def test_phase_estimated_composite_ops(self, phase):
        """Tests that the QPE defined using ControlledSequence works correctly for compound operators"""
        unitary = qml.RX(phase, wires=[0]) @ qml.CNOT(wires=[0, 1])
        estimation_wires = range(2, 6)
        dev = qml.device("default.qubit", wires=range(6))

        @qml.qnode(dev)
        def circuit():
            for i in estimation_wires:
                qml.Hadamard(wires=i)

            qml.ControlledSequence(unitary, control=estimation_wires)

            qml.adjoint(qml.QFT)(wires=estimation_wires)

            return qml.state()

        qpe = qml.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)

        assert np.allclose(qml.matrix(circuit)(), qml.matrix(qpe))
