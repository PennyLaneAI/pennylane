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
Unit tests for the PennyLane :class:`~.ReversibleQNode` class.
"""
import pytest
import numpy as np

import pennylane as qml
from pennylane._device import Device
from pennylane.operation import CVObservable
from pennylane.qnodes.base import QuantumFunctionError
from pennylane.qnodes.rev import ReversibleQNode


thetas = np.linspace(-2*np.pi, 2*np.pi, 8)


@pytest.fixture(scope="function")
def operable_mock_device_2_wires(monkeypatch):
    """A mock instance of the abstract Device class that can support qfuncs."""

    dev = Device
    with monkeypatch.context() as m:
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, '_capabilities', {"model": "qubit"})
        m.setattr(dev, 'operations', ["BasisState", "RX", "RY", "CNOT", "Rot", "PhaseShift"])
        m.setattr(dev, 'observables', ["PauliX", "PauliY", "PauliZ"])
        m.setattr(dev, 'reset', lambda self: None)
        m.setattr(dev, 'apply', lambda self, x, y, z: None)
        m.setattr(dev, 'expval', lambda self, x, y, z: 1)
        yield Device(wires=2)


class TestBestMethod:
    """Test different flows of _best_method"""

    def test_no_following_observable(self, operable_mock_device_2_wires):
        """Test that the gradient is 0 if no observables succeed"""

        def circuit(x):
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0))

        q = ReversibleQNode(circuit, operable_mock_device_2_wires)
        q._construct([1.0], {})
        assert q.par_to_grad_method == {0: "0"}

    def test_param_unused(self, operable_mock_device_2_wires):
        """Test that the gradient is 0 of an unused parameter"""

        def circuit(x, y):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        q = ReversibleQNode(circuit, operable_mock_device_2_wires)
        q._construct([1.0, 1.0], {})
        assert q.par_to_grad_method == {0: "A", 1: "0"}

    def test_not_differentiable(self, operable_mock_device_2_wires):
        """Test that an operation with grad_method=None is marked as
        non-differentiable"""

        def circuit(x):
            qml.BasisState(x, wires=[1])
            return qml.expval(qml.PauliZ(0))

        q = ReversibleQNode(circuit, operable_mock_device_2_wires)
        q._construct([np.array([1.0])], {})
        assert q.par_to_grad_method == {0: None}


class TestExpectationJacobian:
    """Jacobian integration tests for qubit expectations."""

    @pytest.mark.parametrize("mult", [1, -2, 1.623, -0.051, 0])  # intergers, floats, zero
    def test_parameter_multipliers(self, mult, tol):
        """Test that various types and values of scalar multipliers for differentiable
        qfunc parameters yield the correct gradients."""

        def circuit(x):
            qml.RY(mult * x, wires=[0])
            return qml.expval(qml.PauliX(0))

        dev = qml.device("default.qubit", wires=1)
        q = ReversibleQNode(circuit, dev)

        par = [0.1]

        # gradients
        exact = mult * np.cos(mult * np.array([par]))
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method="A")

        # different methods must agree
        assert grad_F == pytest.approx(exact, abs=tol)
        assert grad_A == pytest.approx(exact, abs=tol)

    @pytest.mark.parametrize("reused_p", thetas ** 3 / 19)
    @pytest.mark.parametrize("other_p", thetas ** 2 / 1)
    def test_fanout_multiple_params(self, reused_p, other_p, tol):
        """Tests that the correct gradient is computed for qnodes which
        use the same parameter in multiple gates."""

        from gate_data import Rotx as Rx, Roty as Ry, Rotz as Rz

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = 0.31
        def circuit(reused_param, other_param):
            qml.RX(extra_param, wires=[0])
            qml.RY(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        f = ReversibleQNode(circuit, dev)
        zero_state = np.array([1., 0.])

        # analytic gradient
        grad_A = f.jacobian([reused_p, other_p])

        # manual gradient
        grad_true0 = (expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state) \
                     -expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state)) / 2
        grad_true1 = (expZ(Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state) \
                     -expZ(Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state)) / 2
        grad_true = grad_true0 + grad_true1 # product rule

        assert grad_A[0, 0] == pytest.approx(grad_true, abs=tol)

    @pytest.mark.parametrize("shape", [(8,), (8, 1), (4, 2), (2, 2, 2), (2, 1, 2, 1, 2)])
    def test_multidim_array_parameter(self, shape, tol):
        """Tests that arguments which are multidimensional arrays are
        properly evaluated and differentiated in ReversibleQNodes."""

        n = np.prod(shape)
        base_array = np.linspace(-1.0, 1.0, n)
        multidim_array = np.reshape(base_array, shape)

        def circuit(w):
            for k in range(n):
                qml.RX(w[np.unravel_index(k, shape)], wires=k)  # base_array[k]
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(n))

        dev = qml.device("default.qubit", wires=n)
        circuit = ReversibleQNode(circuit, dev)

        # circuit evaluations
        circuit_output = circuit(multidim_array)
        expected_output = np.cos(base_array)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([multidim_array])
        expected_jacobian = -np.diag(np.sin(base_array))
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_gradient_gate_with_multiple_parameters(self, tol):
        """Tests that gates with multiple free parameters yield correct gradients."""
        par = [0.5, 0.3, -0.7]

        def qf(x, y, z):
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        q = ReversibleQNode(qf, dev)
        value = q(*par)
        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0: "A", 1: "A", 2: "A"}
        # gradient has the correct shape and every element is nonzero
        assert grad_A.shape == (1, 3)
        assert np.count_nonzero(grad_A) == 3
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)

    def test_gradient_repeated_gate_parameters(self, tol):
        """Tests that repeated use of a free parameter in a
        multi-parameter gate yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(np.pi / 4, wires=[0])
            qml.Rot(y, x, 2 * x, wires=[0])
            return qml.expval(qml.PauliX(0))

        dev = qml.device("default.qubit", wires=1)
        q = ReversibleQNode(qf, dev)
        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)

    def test_gradient_parameters_inside_array(self, tol):
        """Tests that free parameters inside an array passed to
        an Operation yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(x, wires=[0])
            qml.RY(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([y, 1]), 0))

        dev = qml.device("default.qubit", wires=1)
        q = ReversibleQNode(qf, dev)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method="F")

        # par[0] can use the "A" method, par[1] cannot
        assert q.par_to_grad_method == {0: "A", 1: "F"}
        # the different methods agree
        assert grad == pytest.approx(grad_F, abs=tol)

    def test_keywordarg_not_differentiated(self, tol):
        """Tests that qnodes do not differentiate w.r.t. keyword arguments."""
        par = np.array([0.5, 0.54])

        def circuit1(weights, x=0.3):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device("default.qubit", wires=2)
        circuit1 = ReversibleQNode(circuit1, dev)

        def circuit2(weights):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], 0.3, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit2 = ReversibleQNode(circuit2, dev)

        res1 = circuit1.jacobian([par])
        res2 = circuit2.jacobian([par])
        assert res1 == pytest.approx(res2, abs=tol)

    def test_differentiate_all_positional(self, tol):
        """Tests that all positional arguments are differentiated."""

        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(3))

        dev = qml.device("default.qubit", wires=3)
        circuit1 = ReversibleQNode(circuit1, dev)

        vals = np.array([np.pi, np.pi / 2, np.pi / 3])
        circuit_output = circuit1(*vals)
        expected_output = np.cos(vals)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit1.jacobian(vals)
        expected_jacobian = -np.diag(np.sin(vals))
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_first_positional(self, tol):
        """Tests that the first positional arguments are differentiated."""

        def circuit2(a, b):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit2 = ReversibleQNode(circuit2, dev)

        a = 0.7418
        b = -5.0
        circuit_output = circuit2(a, b)
        expected_output = np.cos(a)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit2.jacobian([a, b])
        expected_jacobian = np.array([[-np.sin(a), 0]])
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_second_positional(self, tol):
        """Tests that the second positional arguments are differentiated."""

        def circuit3(a, b):
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit3 = ReversibleQNode(circuit3, dev)

        a = 0.7418
        b = -5.0
        circuit_output = circuit3(a, b)
        expected_output = np.cos(b)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit3.jacobian([a, b])
        expected_jacobian = np.array([[0, -np.sin(b)]])
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_second_third_positional(self, tol):
        """Tests that the second and third positional arguments are differentiated."""

        def circuit4(a, b, c):
            qml.RX(b, wires=0)
            qml.RX(c, wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        dev = qml.device("default.qubit", wires=2)
        circuit4 = ReversibleQNode(circuit4, dev)

        a = 0.7418
        b = -5.0
        c = np.pi / 7
        circuit_output = circuit4(a, b, c)
        expected_output = np.array([np.cos(b), np.cos(c)])
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit4.jacobian([a, b, c])
        expected_jacobian = np.array([[0.0, -np.sin(b), 0.0], [0.0, 0.0, -np.sin(c)]])
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_positional_multidim(self, tol):
        """Tests that all positional arguments are differentiated
        when they are multidimensional."""

        def circuit(a, b):
            qml.RX(a[0], wires=0)
            qml.RX(a[1], wires=1)
            qml.RX(b[2, 1], wires=2)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        dev = qml.device("default.qubit", wires=3)
        circuit = ReversibleQNode(circuit, dev)

        a = np.array([-np.sqrt(2), -0.54])
        b = np.array([np.pi / 7] * 6).reshape([3, 2])
        circuit_output = circuit(a, b)
        expected_output = np.cos(np.array([a[0], a[1], b[-1, 0]]))
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([a, b])
        expected_jacobian = np.array(
            [
                [-np.sin(a[0])] + [0.0] * 7,  # expval 0
                [0.0, -np.sin(a[1])] + [0.0] * 6,  # expval 1
                [0.0] * 2 + [0.0] * 5 + [-np.sin(b[2, 1])],
            ]
        )  # expval 2
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_array_parameters_evaluate(self, tol):
        """Tests that array parameters gives same result as positional arguments."""
        a, b, c = 0.5, 0.54, 0.3
        dev = qml.device("default.qubit", wires=2)

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def circuit1(x, y, z):
            return ansatz(x, y, z)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        def circuit3(array):
            return ansatz(*array)

        circuit1 = ReversibleQNode(circuit1, dev)
        circuit2 = ReversibleQNode(circuit2, dev)
        circuit3 = ReversibleQNode(circuit3, dev)

        positional_res = circuit1(a, b, c)
        positional_grad = circuit1.jacobian([a, b, c])

        array_res = circuit2(a, np.array([b, c]))
        array_grad = circuit2.jacobian([a, np.array([b, c])])

        assert positional_res == pytest.approx(array_res, abs=tol)
        assert positional_grad == pytest.approx(array_grad, abs=tol)

        list_res = circuit2(a, [b, c])
        list_grad = circuit2.jacobian([a, [b, c]])

        assert positional_res == pytest.approx(list_res, abs=tol)
        assert positional_grad == pytest.approx(list_grad, abs=tol)

        array_res = circuit3(np.array([a, b, c]))
        array_grad = circuit3.jacobian([np.array([a, b, c])])

        list_res = circuit3([a, b, c])
        list_grad = circuit3.jacobian([[a, b, c]])

        assert positional_res == pytest.approx(array_res, abs=tol)
        assert positional_grad == pytest.approx(array_grad, abs=tol)

    @pytest.mark.parametrize('theta', thetas)
    @pytest.mark.parametrize('G', [qml.ops.RX, qml.ops.RY, qml.ops.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        circuit = ReversibleQNode(circuit, dev)

        autograd_val = circuit.jacobian([theta])
        manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize('theta', thetas)
    def test_Rot_gradient(self, theta, tol):
        """Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."""

        def circuit(x,y,z):
            qml.Rot(x,y,z, wires=[0])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        circuit = ReversibleQNode(circuit, dev)
        eye = np.eye(3)

        angle_inputs = np.array([theta, theta ** 3, np.sqrt(2) * theta])
        autograd_val = circuit.jacobian(angle_inputs)
        manualgrad_val = np.zeros((1,3))

        for idx in range(3):
            onehot_idx = eye[idx]
            param1 = angle_inputs + np.pi / 2 * onehot_idx
            param2 = angle_inputs - np.pi / 2 * onehot_idx
            manualgrad_val[0, idx] = (circuit(*param1) - circuit(*param2)) / 2

        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_controlled_RX_gradient(self, tol):
        """Test gradient of controlled RX gate"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit = ReversibleQNode(circuit, dev)

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = ReversibleQNode(circuit1, dev)

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_controlled_RY_gradient(self, tol):
        """Test gradient of controlled RY gate"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit = ReversibleQNode(circuit, dev)

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = ReversibleQNode(circuit1, dev)

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_controlled_RZ_gradient(self, tol):
        """Test gradient of controlled RZ gate"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRZ(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit = ReversibleQNode(circuit, dev)

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRZ(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = ReversibleQNode(circuit1, dev)

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)


class TestVarianceJacobian:
    """Variance analytic jacobian integration tests."""

    def test_involutory_variance(self, tol):
        """Tests qubit observable that are involutory"""
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.var(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        circuit = ReversibleQNode(circuit, dev)

        a = 0.54
        var = circuit(a)
        expected = 1 - np.cos(a) ** 2
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = 2 * np.sin(a) * np.cos(a)
        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        def circuit(a):
            qml.RX(a, wires=0)
            return qml.var(qml.Hermitian(A, 0))

        dev = qml.device("default.qubit", wires=1)
        circuit = ReversibleQNode(circuit, dev)

        a = 0.54
        var = circuit(a)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_fanout(self, tol):
        """Tests qubit observable with repeated parameters"""
        def circuit(a):
            qml.RX(a, wires=0)
            qml.RY(a, wires=0)
            return qml.var(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit = ReversibleQNode(circuit, dev)

        a = 0.54
        var = circuit(a)
        expected = 0.5 * np.sin(a) ** 2 * (np.cos(2 * a) + 3)
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = 4 * np.sin(a) * np.cos(a) ** 3
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(c, wires=2)
            return qml.var(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(2))

        circuit = ReversibleQNode(circuit, dev)

        a = 0.54
        b = -0.423
        c = 0.123
        var = circuit(a, b, c)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )
        assert var == pytest.approx(expected, abs=tol)

        # # circuit jacobians
        gradA = circuit.jacobian([a, b, c], method="A")
        gradF = circuit.jacobian([a, b, c], method="F")
        expected = np.array(
            [
                [2 * np.cos(a) * np.sin(a), -np.cos(b) * np.sin(a), 0],
                [
                    0,
                    -np.cos(a) * np.sin(b),
                    0.5 * (2 * np.cos(b) * np.cos(2 * c) * np.sin(b) + np.sin(2 * b)),
                ],
                [0, 0, np.cos(b) ** 2 * np.sin(2 * c)],
            ]
        ).T
        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)
