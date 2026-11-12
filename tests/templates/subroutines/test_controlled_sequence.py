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

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.wires import Wires

# pylint: disable=unidiomatic-typecheck, cell-var-from-loop


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=[0, 1, 2])
    qp.testing.assert_valid(op)


class TestInitialization:

    def test_overlapping_wires_error(self):
        """Test that an error is raised if the wires of the base
        operator and the control wires overlap"""

        base = qp.PauliX(1)
        control = [0, 1, 2]

        with pytest.raises(ValueError, match="base.wires and control_wires must not overlap"):
            _ = qp.ControlledSequence(base, control)

    def test_name(self):
        """Test that the name for the operator is ControlledSequence (must be overwritten for SymbolicOps)"""

        op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=[0, 1, 2])

        assert op.name == "ControlledSequence"


class TestProperties:

    @pytest.mark.parametrize("angle", [0.25, np.array([0.25, 0.5])])
    def test_base_parameter_properties(self, angle):
        """Test that base parameters remain visible to differentiation and broadcasting."""
        base = qp.RX(angle, wires=3)
        op = qp.ControlledSequence(base, control=[0, 1, 2])

        assert op.data == base.data
        assert op.num_params == base.num_params
        assert op.ndim_params == base.ndim_params
        assert op.batch_size == base.batch_size
        assert op.grad_method is None

    def test_hash(self):
        """Test that hash(op) uniquely describes a ControlledSequence"""

        op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=[0, 1, 2])
        op1 = qp.ControlledSequence(qp.RX(0.25, wires=3), control=[0, 1, 2])  # identical
        op2 = qp.ControlledSequence(qp.RX(0.25, wires=4), control=[0, 1, 2])  # different base wire
        op3 = qp.ControlledSequence(qp.RX(0.35, wires=3), control=[0, 1, 2])  # different base param
        op4 = qp.ControlledSequence(qp.RY(0.25, wires=3), control=[0, 1, 2])  # different base class
        op5 = qp.ControlledSequence(
            qp.RX(0.25, wires=3), control=[0, 1, 4]
        )  # different control wires

        assert hash(op) == hash(op1)
        for other_op in [op2, op3, op4, op5]:
            assert hash(op) != hash(other_op)

    def test_control(self):
        """Test that the control property returns control wires"""
        op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=["a", 1, "blue"])
        assert op.control == Wires(["a", 1, "blue"])

    def test_wires(self):
        """Test that the wires property returns all wires, including both base and control"""
        op = qp.ControlledSequence(qp.CNOT([17, "3"]), control=["b", 2])
        assert op.wires == Wires(["b", 2, 17, "3"])

    def test_has_matrix(self):
        """Test that a ControlledSequence returns False for has_matrix, even if the base returns True"""
        op = qp.ControlledSequence(qp.PauliX(0), control=[1])
        assert op.has_matrix is False
        assert op.base.has_matrix is True

    def test_has_sparse_matrix(self):
        """Test that the sequence has no sparse matrix even when its base does."""
        op = qp.ControlledSequence(qp.PauliX(0), control=[1])
        assert op.has_sparse_matrix is False
        assert op.base.has_sparse_matrix is True


class TestMethods:

    def test_repr(self):
        """Test that the operator repr is as expected"""
        op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=[0, 1, 2])
        assert repr(op) == f"ControlledSequence({op.base}, control=[0, 1, 2])"

    def test_map_wires(self):
        """Test mapping wires works as expected, including mapping wires on the base operator"""

        op = qp.ControlledSequence(qp.CNOT([0, 1]), control=[2, 3])
        wire_map = {0: "a", 1: "b", 2: "c", 3: "d"}

        new_op = op.map_wires(wire_map)

        assert type(new_op) == type(op)
        assert type(new_op.base) == type(op.base)
        assert new_op.data == op.data

        assert new_op.wires == Wires(["c", "d", "a", "b"])
        assert new_op.base.wires == Wires(["a", "b"])
        assert new_op.control == Wires(["c", "d"])

        assert op.base.wires == Wires([0, 1])
        assert op.control == Wires([2, 3])

    @pytest.mark.parametrize("control", [[], [0], [0, 1, 2]])
    def test_decomposition(self, control):
        """Test the returned and queued operations from the registered decomposition."""
        op = qp.ControlledSequence(qp.RX(0.25, wires=3), control=control)

        with qp.queuing.AnnotatedQueue() as queue:
            decomp = op.decomposition()
        expected_decomp = [
            qp.ctrl(qp.RX(0.25 * 2 ** (len(control) - 1 - j), wires=3), control=wire)
            for j, wire in enumerate(control)
        ]

        queued_ops = queue.queue
        for operations in (decomp, queued_ops):
            assert len(operations) == len(expected_decomp)
            for actual, expected in zip(operations, expected_decomp):
                qp.assert_equal(actual, expected)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("control", [[], 0, [0, 1], [0, 1, 2]])
    @pytest.mark.parametrize(
        "base",
        [
            qp.RX(0.25, wires=3),
            qp.X(3),
            qp.Hadamard(3),
            qp.CNOT([3, 4]),
            qp.prod(qp.RX(0.25, wires=3), qp.RY(0.5, wires=4)),
        ],
    )
    def test_decomposition_new(self, control, base):
        """Test decomposition resources and matrices for different power simplifications."""
        op = qp.ControlledSequence(base, control=control)
        qp.testing.assert_decomposable(op)

    @pytest.mark.capture
    @pytest.mark.parametrize("control", [[0], [0, 1, 2]])
    def test_power_capture(self, control):
        """Test that capture simplifies static powers without structured loops."""
        import jax

        rule = qp.list_decomps(qp.ControlledSequence)[0]

        def circuit():
            rule(base=qp.RX(0.25, wires=3), control=control)

        plxpr = jax.make_jaxpr(circuit)()
        assert all(eqn.primitive.name != "for_loop" for eqn in plxpr.jaxpr.eqns)
        operations = []

        class CollectOps(qp.capture.PlxprInterpreter):  # pylint: disable=too-few-public-methods
            def interpret_operation(self, op):
                operations.append(op)

        CollectOps().eval(plxpr.jaxpr, plxpr.consts)
        expected = [
            qp.ctrl(qp.RX(0.25 * 2 ** (len(control) - 1 - j), wires=3), control=wire)
            for j, wire in enumerate(control)
        ]
        assert len(operations) == len(expected)
        for actual, expected_op in zip(operations, expected):
            qp.assert_equal(actual, expected_op)


class TestIntegration:
    """Tests that the ControlledSequence is executable and differentiable in a QNode context"""

    @staticmethod
    def circuit(x):
        """Test circuit"""
        qp.PauliX(2)
        qp.ControlledSequence(qp.RX(x, wires=3), control=[0, 1, 2])
        return qp.probs(wires=range(4))

    x = np.array(0.25)
    # not calculated analytically, we are only ensuring that the results are consistent accross interfaces
    exp_result = np.array([0, 0, 0.9835, 0.0165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    exp_jac = np.array([0, 0, -0.12342829, 0.12342829, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_qnode_numpy(self):
        """Test that the QNode executes with Numpy."""
        dev = qp.device("default.qubit")
        qnode = qp.QNode(self.circuit, dev, interface=None)

        res = qnode(self.x)
        assert res.shape == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_autograd(self, shots, seed):
        """Test that the QNode executes with Autograd."""

        dev = qp.device("default.qubit", wires=4, seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qp.set_shots(
            qp.QNode(self.circuit, dev, interface="autograd", diff_method=diff_method), shots=shots
        )
        x = qp.numpy.array(self.x, requires_grad=True)

        res = qnode(x)
        assert qp.math.shape(res) == (16,)
        assert np.allclose(res, self.exp_result, atol=0.002)

        res = qp.jacobian(qnode)(x)
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

        dev = qp.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qp.set_shots(
            qp.QNode(self.circuit, dev, interface="jax", diff_method=diff_method), shots=shots
        )
        if use_jit:
            qnode = jax.jit(qnode)

        x = jax.numpy.array(self.x)
        res = qnode(x)
        assert qp.math.shape(res) == (16,)
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

        dev = qp.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qp.set_shots(
            qp.QNode(self.circuit, dev, interface="torch", diff_method=diff_method), shots=shots
        )

        x = torch.tensor(self.x, requires_grad=True)
        res = qnode(x)
        assert qp.math.shape(res) == (16,)
        assert qp.math.allclose(res, self.exp_result, atol=0.002)

        jac = torch.autograd.functional.jacobian(qnode, x)
        assert qp.math.shape(jac) == (16,)
        assert qp.math.allclose(jac, self.exp_jac, atol=0.005)

    def test_prod_rx_rx_compiled_circuit(self):
        """Test that a circuit can execute successfully using qp.compile and
        a CompositeOp as a base"""

        @qp.prod
        def U(thetas):
            qp.RX(thetas[0], wires=0)
            qp.RX(thetas[1], wires=1)

        dev = qp.device("default.qubit", wires=4)

        @qp.compile
        @qp.qnode(dev)
        def circuit(thetas):
            qp.ControlledSequence(U(thetas), control=[2, 3])
            return qp.state()

        _ = circuit([1.0, 1.0])

    def test_approx_time_base(self):
        """Test that using ControlledSequence with ApproxTimeEvolution as a base in a
        circuit can be decomposed to execute successfully"""
        H = qp.Hamiltonian([1.0, 2.0], [qp.PauliZ(0), qp.PauliZ(1)])
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit():
            qp.ControlledSequence(qp.ApproxTimeEvolution(H, 1, 1), control=[2, 3])
            return qp.state()

        _ = circuit()

    def test_gradient_with_composite_op_base(self):
        """Test executing and getting the gradient of a circuit with a
        ControlledSequence based on a CompositeOp"""

        @qp.prod
        def U(thetas):
            qp.RX(thetas[0], wires=0)
            qp.RX(thetas[1], wires=1)

        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit(thetas):
            qp.ControlledSequence(U(thetas), control=[2, 3])
            return qp.expval(qp.PauliZ(0))

        thetas = pnp.array([1.0, 1.0], requires_grad=True)
        _ = circuit(thetas)
        _ = qp.grad(circuit)(thetas)


class TestQPEResults:
    """Tests phase estimation using the template"""

    @pytest.mark.parametrize("phase", [7, np.pi, np.pi / 3, 2.3])
    def test_phase_estimated_single_ops(self, phase):
        """Tests that the QPE defined using ControlledSequence works correctly for a single operator"""

        unitary = qp.RX(phase, wires=[0])
        estimation_wires = range(1, 6)
        dev = qp.device("default.qubit", wires=range(6))

        @qp.qnode(dev)
        def circuit():
            for i in estimation_wires:
                qp.Hadamard(wires=i)

            qp.ControlledSequence(unitary, control=estimation_wires)

            qp.adjoint(qp.QFT)(wires=estimation_wires)

            return qp.state()

        qpe = qp.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)

        assert np.allclose(qp.matrix(circuit)(), qp.matrix(qpe))

    @pytest.mark.parametrize("phase", [7, np.pi, np.pi / 3, 2.3])
    def test_phase_estimated_composite_ops(self, phase):
        """Tests that the QPE defined using ControlledSequence works correctly for compound operators"""
        unitary = qp.RX(phase, wires=[0]) @ qp.CNOT(wires=[0, 1])
        estimation_wires = range(2, 6)
        dev = qp.device("default.qubit", wires=range(6))

        @qp.qnode(dev)
        def circuit():
            for i in estimation_wires:
                qp.Hadamard(wires=i)

            qp.ControlledSequence(unitary, control=estimation_wires)

            qp.adjoint(qp.QFT)(wires=estimation_wires)

            return qp.state()

        qpe = qp.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)

        assert np.allclose(qp.matrix(circuit)(), qp.matrix(qpe))
