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
"""Unit tests for the qubit parameter-shift QubitParamShiftTape"""
import pytest
from pennylane import numpy as np

import pennylane as qml
from pennylane.interfaces.autograd import AutogradInterface
from pennylane.tape import JacobianTape, ReversibleTape
from pennylane import QNode, qnode
from pennylane.measure import MeasurementProcess


thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)


class TestReversibleTape:
    """Unit tests for the reversible tape"""

    def test_diff_circuit_construction(self, mocker):
        """Test that the diff circuit is correctly constructed"""
        dev = qml.device("default.qubit", wires=2)

        with ReversibleTape() as tape:
            qml.PauliX(wires=0)
            qml.RX(0.542, wires=0)
            qml.RY(0.542, wires=0)
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(dev, "execute")
        tape.jacobian(dev)

        tape0 = spy.call_args_list[0][0][0]
        tape1 = spy.call_args_list[1][0][0]
        tape2 = spy.call_args_list[2][0][0]

        assert tape0 is tape

        assert len(tape1.operations) == 4
        assert len(tape1.measurements) == 1
        assert tape1.operations[0].name == "QubitStateVector"
        assert tape1.operations[1].name == "RY.inv"
        assert tape1.operations[2].name == "PauliX"
        assert tape1.operations[3].name == "RY"

        assert len(tape2.operations) == 2
        assert len(tape1.measurements) == 1
        assert tape1.operations[0].name == "QubitStateVector"
        assert tape2.operations[1].name == "PauliY"

    def test_rot_diff_circuit_construction(self, mocker):
        """Test that the diff circuit is correctly constructed for the Rot gate"""
        dev = qml.device("default.qubit", wires=2)

        with ReversibleTape() as tape:
            qml.PauliX(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(dev, "execute")
        tape.jacobian(dev)

        tape0 = spy.call_args_list[0][0][0]
        tape1 = spy.call_args_list[1][0][0]
        tape2 = spy.call_args_list[2][0][0]
        tape3 = spy.call_args_list[3][0][0]

        assert tape0 is tape

        assert len(tape1.operations) == 6
        assert len(tape1.measurements) == 1
        assert tape1.operations[0].name == "QubitStateVector"
        assert tape1.operations[1].name == "RZ.inv"
        assert tape1.operations[2].name == "RY.inv"
        assert tape1.operations[3].name == "PauliZ"
        assert tape1.operations[4].name == "RY"
        assert tape1.operations[5].name == "RZ"

        assert len(tape2.operations) == 4
        assert len(tape1.measurements) == 1
        assert tape1.operations[0].name == "QubitStateVector"
        assert tape2.operations[1].name == "RZ.inv"
        assert tape2.operations[2].name == "PauliY"
        assert tape2.operations[3].name == "RZ"

        assert len(tape3.operations) == 2
        assert len(tape1.measurements) == 1
        assert tape1.operations[0].name == "QubitStateVector"
        assert tape3.operations[1].name == "PauliZ"

    @pytest.mark.parametrize("op, name", [(qml.CRX, "CRX"), (qml.CRY, "CRY"), (qml.CRZ, "CRZ")])
    def test_controlled_rotation_gates_exception(self, op, name):
        """Tests that an exception is raised when a controlled
        rotation gate is used with the ReversibleTape."""
        # TODO: remove this test when this support is added
        dev = qml.device("default.qubit", wires=2)

        with ReversibleTape() as tape:
            qml.PauliX(wires=0)
            op(0.542, wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="The {} gate is not currently supported".format(name)):
            tape.jacobian(dev)

    def test_var_exception(self):
        """Tests that an exception is raised when variance
        is used with the ReversibleTape."""
        # TODO: remove this test when this support is added
        dev = qml.device("default.qubit", wires=2)

        with ReversibleTape() as tape:
            qml.PauliX(wires=0)
            qml.RX(0.542, wires=0)
            qml.var(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Variance is not supported"):
            tape.jacobian(dev)

    def test_probs_exception(self):
        """Tests that an exception is raised when probability
        is used with the ReversibleTape."""
        # TODO: remove this test when this support is added
        dev = qml.device("default.qubit", wires=2)

        with ReversibleTape() as tape:
            qml.PauliX(wires=0)
            qml.RX(0.542, wires=0)
            qml.probs(wires=[0, 1])

        with pytest.raises(ValueError, match="Probability is not supported"):
            tape.jacobian(dev)

    def test_phaseshift_exception(self):
        """Tests that an exception is raised when a PhaseShift gate
        is used with the ReversibleTape."""
        # TODO: remove this test when this support is added
        dev = qml.device("default.qubit", wires=1)

        with ReversibleTape() as tape:
            qml.PauliX(wires=0)
            qml.PhaseShift(0.542, wires=0)
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="The PhaseShift gate is not currently supported"):
            tape.jacobian(dev)


class TestGradients:
    """Jacobian integration tests for qubit expectations."""

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        dev = qml.device("default.qubit", wires=1)

        with ReversibleTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1}

        autograd_val = tape.jacobian(dev, method="analytic")

        # compare to finite differences
        numeric_val = tape.jacobian(dev, method="numeric")
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol):
        """Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."""
        dev = qml.device("default.qubit", wires=1)
        params = np.array([theta, theta ** 3, np.sqrt(2) * theta])

        with ReversibleTape() as tape:
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        autograd_val = tape.jacobian(dev, method="analytic")

        # compare to finite differences
        numeric_val = tape.jacobian(dev, method="numeric")
        assert np.allclose(autograd_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [1, -2, 1.623, -0.051, 0])  # intergers, floats, zero
    def test_ry_gradient(self, par, mocker, tol):
        """Test that the gradient of the RY gate matches the exact analytic
        formula. Further, make sure the correct gradient methods
        are being called."""

        with ReversibleTape() as tape:
            qml.RY(par, wires=[0])
            qml.expval(qml.PauliX(0))

        tape.trainable_params = {0}

        dev = qml.device("default.qubit", wires=1)

        spy_numeric = mocker.spy(tape, "numeric_pd")
        spy_analytic = mocker.spy(tape, "analytic_pd")

        # gradients
        exact = np.cos(par)
        grad_F = tape.jacobian(dev, method="numeric")

        spy_numeric.assert_called()
        spy_analytic.assert_not_called()

        spy_device = mocker.spy(tape, "execute_device")
        grad_A = tape.jacobian(dev, method="analytic")

        spy_analytic.assert_called()
        spy_device.assert_called_once()  # check that the state was only pre-computed once

        # different methods must agree
        assert np.allclose(grad_F, exact, atol=tol, rtol=0)
        assert np.allclose(grad_A, exact, atol=tol, rtol=0)

    def test_rx_gradient(self, tol):
        """Test that the gradient of the RX gate matches the known formula."""
        dev = qml.device("default.qubit", wires=2)
        a = 0.7418

        with ReversibleTape() as tape:
            qml.RX(a, wires=0)
            qml.expval(qml.PauliZ(0))

        circuit_output = tape.execute(dev)
        expected_output = np.cos(a)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = tape.jacobian(dev, method="analytic")
        expected_jacobian = -np.sin(a)
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_multiple_rx_gradient(self, tol):
        """Tests that the gradient of multiple RX gates in a circuit
        yeilds the correct result."""
        dev = qml.device("default.qubit", wires=3)
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        with ReversibleTape() as tape:
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[2], wires=2)

            for idx in range(3):
                qml.expval(qml.PauliZ(idx))

        circuit_output = tape.execute(dev)
        expected_output = np.cos(params)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = tape.jacobian(dev, method="analytic")
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    qubit_ops = [getattr(qml, name) for name in qml.ops._qubit__ops__]
    analytic_qubit_ops = {cls for cls in qubit_ops if cls.grad_method == "A"}
    analytic_qubit_ops = analytic_qubit_ops - {
        qml.CRX,
        qml.CRY,
        qml.CRZ,
        qml.CRot,
        qml.PhaseShift,
        qml.ControlledPhaseShift,
        qml.CPhase,
        qml.PauliRot,
        qml.MultiRZ,
        qml.U1,
        qml.U2,
        qml.U3,
        qml.IsingXX,
        qml.IsingZZ,
        qml.SingleExcitation,
        qml.SingleExcitationPlus,
        qml.SingleExcitationMinus,
        qml.DoubleExcitation,
        qml.DoubleExcitationPlus,
        qml.DoubleExcitationMinus,
    }

    @pytest.mark.parametrize("obs", [qml.PauliX, qml.PauliY])
    @pytest.mark.parametrize("op", analytic_qubit_ops)
    def test_gradients(self, op, obs, mocker, tol):
        """Tests that the gradients of circuits match between the
        finite difference and analytic methods."""
        args = np.linspace(0.2, 0.5, op.num_params)

        with ReversibleTape() as tape:
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            op(*args, wires=range(op.num_wires))

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            qml.expval(obs(wires=0))
            qml.expval(qml.PauliZ(wires=1))

        dev = qml.device("default.qubit", wires=2)
        res = tape.execute(dev)

        tape._update_gradient_info()
        tape.trainable_params = set(range(1, 1 + op.num_params))

        # check that every parameter is analytic
        for i in range(op.num_params):
            assert tape._par_info[1 + i]["grad_method"][0] == "A"

        grad_F = tape.jacobian(dev, method="numeric")

        spy = mocker.spy(ReversibleTape, "analytic_pd")
        spy_execute = mocker.spy(tape, "execute_device")
        grad_A = tape.jacobian(dev, method="analytic")
        spy.assert_called()

        # check that the execute device method has only been called
        # once, for all parameters.
        spy_execute.assert_called_once()

        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_gradient_gate_with_multiple_parameters(self, tol):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        with ReversibleTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {1, 2, 3}

        dev = qml.device("default.qubit", wires=1)
        grad_A = tape.jacobian(dev, method="analytic")
        grad_F = tape.jacobian(dev, method="numeric")

        # gradient has the correct shape and every element is nonzero
        assert grad_A.shape == (1, 3)
        assert np.count_nonzero(grad_A) == 3
        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)


class TestQNodeIntegration:
    """Test QNode integration with the reversible method"""

    def test_qnode(self, mocker, tol):
        """Test that specifying diff_method allows the reversible
        method to be selected"""
        args = np.array([0.54, 0.1, 0.5], requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.RX(0.543, wires=0)
            qml.CNOT(wires=[0, 1])

            qml.Rot(x, y, z, wires=0)

            qml.Rot(1.3, -2.3, 0.5, wires=[0])
            qml.RZ(-0.5, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        qnode1 = QNode(circuit, dev, diff_method="reversible")
        spy = mocker.spy(ReversibleTape, "analytic_pd")

        grad_fn = qml.grad(qnode1)
        grad_A = grad_fn(*args)

        spy.assert_called()
        assert isinstance(qnode1.qtape, ReversibleTape)

        qnode2 = QNode(circuit, dev, diff_method="finite-diff")
        grad_fn = qml.grad(qnode2)
        grad_F = grad_fn(*args)

        assert not isinstance(qnode2.qtape, ReversibleTape)
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize("reused_p", thetas ** 3 / 19)
    @pytest.mark.parametrize("other_p", thetas ** 2 / 1)
    def test_fanout_multiple_params(self, reused_p, other_p, tol):
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        from gate_data import Rotx as Rx, Roty as Ry, Rotz as Rz

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        dev = qml.device("default.qubit", wires=1)
        extra_param = np.array(0.31, requires_grad=False)

        @qnode(dev)
        def cost(p1, p2):
            qml.RX(extra_param, wires=[0])
            qml.RY(p1, wires=[0])
            qml.RZ(p2, wires=[0])
            qml.RX(p1, wires=[0])
            return qml.expval(qml.PauliZ(0))

        zero_state = np.array([1.0, 0.0])

        # analytic gradient
        grad_fn = qml.grad(cost)
        grad_A = grad_fn(reused_p, other_p)

        # manual gradient
        grad_true0 = (
            expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        grad_true1 = (
            expZ(
                Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
            - expZ(
                Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state
            )
        ) / 2
        expected = grad_true0 + grad_true1  # product rule

        assert np.allclose(grad_A[0], expected, atol=tol, rtol=0)

    def test_gradient_repeated_gate_parameters(self, mocker, tol):
        """Tests that repeated use of a free parameter in a
        multi-parameter gate yield correct gradients."""
        dev = qml.device("default.qubit", wires=1)
        params = np.array([0.8, 1.3], requires_grad=True)

        def circuit(params):
            qml.RX(np.array(np.pi / 4, requires_grad=False), wires=[0])
            qml.Rot(params[1], params[0], 2 * params[0], wires=[0])
            return qml.expval(qml.PauliX(0))

        spy_numeric = mocker.spy(JacobianTape, "numeric_pd")
        spy_analytic = mocker.spy(ReversibleTape, "analytic_pd")

        cost = QNode(circuit, dev, diff_method="finite-diff")
        grad_fn = qml.grad(cost)
        grad_F = grad_fn(params)

        spy_numeric.assert_called()
        spy_analytic.assert_not_called()

        cost = QNode(circuit, dev, diff_method="reversible")
        grad_fn = qml.grad(cost)
        grad_A = grad_fn(params)

        spy_analytic.assert_called()

        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)


class TestHelperFunctions:
    """Tests for additional helper functions."""

    one_qubit_vec1 = np.array([1, 1])
    one_qubit_vec2 = np.array([1, 1j])
    two_qubit_vec = np.array([1, 1, 1, -1])
    single_qubit_obs1 = qml.PauliZ(0)
    single_qubit_obs2 = qml.PauliY(0)
    two_qubit_obs = qml.Hermitian(np.eye(4), wires=[0, 1])

    @pytest.mark.parametrize(
        "wires, vec1, obs, vec2, expected",
        [
            (1, one_qubit_vec1, single_qubit_obs1, one_qubit_vec1, 0),
            (1, one_qubit_vec2, single_qubit_obs1, one_qubit_vec2, 0),
            (1, one_qubit_vec1, single_qubit_obs1, one_qubit_vec2, 1 - 1j),
            (1, one_qubit_vec2, single_qubit_obs1, one_qubit_vec1, 1 + 1j),
            (1, one_qubit_vec1, single_qubit_obs2, one_qubit_vec1, 0),
            (1, one_qubit_vec2, single_qubit_obs2, one_qubit_vec2, 2),
            (1, one_qubit_vec1, single_qubit_obs2, one_qubit_vec2, 1 + 1j),
            (1, one_qubit_vec2, single_qubit_obs2, one_qubit_vec1, 1 - 1j),
            (2, two_qubit_vec, single_qubit_obs1, two_qubit_vec, 0),
            (2, two_qubit_vec, single_qubit_obs2, two_qubit_vec, 0),
            (2, two_qubit_vec, two_qubit_obs, two_qubit_vec, 4),
        ],
    )
    def test_matrix_elem(self, wires, vec1, obs, vec2, expected):
        """Tests for the helper function _matrix_elem"""
        tape = ReversibleTape()
        res = tape._matrix_elem(vec1, obs, vec2, qml.wires.Wires(range(wires)))
        assert res == expected
