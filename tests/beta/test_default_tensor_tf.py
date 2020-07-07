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
Unit tests and integration tests for the :mod:`pennylane.plugin.Tensornet.tf` device.
"""
from itertools import product

import numpy as np
import pytest

tensornetwork = pytest.importorskip("tensornetwork", minversion="0.1")
tensorflow = pytest.importorskip("tensorflow", minversion="2.0")

import pennylane as qml
from pennylane.beta.plugins.default_tensor_tf import DefaultTensorTF
from gate_data import I, X, Y, Z, H, CNOT, SWAP, CNOT, Toffoli, CSWAP
from pennylane.qnodes import qnode, QNode
from pennylane.qnodes.decorator import ALLOWED_INTERFACES, ALLOWED_DIFF_METHODS

np.random.seed(42)


#####################################################
# Test matrices
#####################################################

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


#####################################################
# Define standard qubit operations
#####################################################

phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)


single_qubit = [(qml.PauliX, X), (qml.PauliY, Y), (qml.PauliZ, Z), (qml.Hadamard, H)]


single_qubit_param = [
    (qml.PhaseShift, phase_shift),
    (qml.RX, rx),
    (qml.RY, ry),
    (qml.RZ, rz),
]
two_qubit = [(qml.CNOT, CNOT), (qml.SWAP, SWAP)]
two_qubit_param = [(qml.CRZ, crz)]
three_qubit = [(qml.Toffoli, Toffoli), (qml.CSWAP, CSWAP)]


#####################################################
# Fixtures
#####################################################


@pytest.fixture
def init_state(scope="session"):
    """Generates a random initial state"""

    def _init_state(n):
        """random initial state"""
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


#####################################################
# Unit tests
#####################################################


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestApply:
    """Test application of PennyLane operations."""

    def test_basis_state(self, tol, rep):
        """Test basis state initialization"""
        dev = DefaultTensorTF(wires=4, representation=rep)
        state = np.array([0, 0, 1, 0])

        dev.execute([qml.BasisState(state, wires=[0, 1, 2, 3])], [], {})

        res = dev._state().numpy().flatten()
        expected = np.zeros([2 ** 4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_qubit_state_vector(self, init_state, tol, rep):
        """Test qubit state vector application"""
        dev = DefaultTensorTF(wires=1, representation=rep)
        state = init_state(1)

        dev.execute([qml.QubitStateVector(state, wires=[0])], [], {})

        res = dev._state().numpy().flatten()
        expected = state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_invalid_qubit_state_vector(self, rep):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = DefaultTensorTF(wires=2, representation=rep)
        state = np.array([0, 123.432])

        with pytest.raises(
            ValueError, match=r"can apply QubitStateVector only to all of the 2 wires"
        ):
            dev.execute([qml.QubitStateVector(state, wires=[0])], [], {})

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, op, mat, rep, tol):
        """Test non-parametrized single qubit operations"""
        dev = DefaultTensorTF(wires=1, representation=rep)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(wires=0)]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, op, func, theta, rep, tol):
        """Test parametrized single qubit operations"""
        dev = DefaultTensorTF(wires=1, representation=rep)
        state = init_state(1)

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation(self, init_state, rep, tol):
        """Test three axis rotation gate"""
        dev = DefaultTensorTF(wires=1, representation=rep)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.QubitStateVector(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = rot(a, b, c) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, op, mat, rep, tol):
        """Test non-parametrized two qubit operations"""
        dev = DefaultTensorTF(wires=2, representation=rep)
        state = init_state(2)

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [op(wires=[0, 1])]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, mat, rep, tol):
        """Test application of arbitrary qubit unitaries"""
        N = int(np.log2(len(mat)))
        dev = DefaultTensorTF(wires=N, representation=rep)
        state = init_state(N)

        queue = [qml.QubitStateVector(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, op, mat, rep, tol):
        """Test non-parametrized three qubit operations"""

        if rep == "mps":
            pytest.skip("Three-qubit gates not supported for `mps` representation.")

        dev = DefaultTensorTF(wires=3, representation=rep)
        state = init_state(3)

        queue = [qml.QubitStateVector(state, wires=[0, 1, 2])]
        queue += [op(wires=[0, 1, 2])]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, op, func, theta, rep, tol):
        """Test two qubit parametrized operations"""
        dev = DefaultTensorTF(wires=2, representation=rep)
        state = init_state(2)

        queue = [qml.QubitStateVector(state, wires=[0, 1])]
        queue += [op(theta, wires=[0, 1])]
        dev.execute(queue, [], {})

        res = dev._state().numpy().flatten()
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)

# test data; each tuple is of the form (GATE, OBSERVABLE, EXPECTED)
single_wire_expval_test_data = [
    (qml.RX, qml.Identity, lambda t, p: np.array([1, 1])),
    (qml.RX, qml.PauliZ, lambda t, p: np.array([np.cos(t), np.cos(t) * np.cos(p)])),
    (qml.RY, qml.PauliX, lambda t, p: np.array([np.sin(t) * np.sin(p), np.sin(p)])),
    (qml.RX, qml.PauliY, lambda t, p: np.array([0, -np.cos(t) * np.sin(p)])),
    (
        qml.RY,
        qml.Hadamard,
        lambda t, p: np.array(
            [np.sin(t) * np.sin(p) + np.cos(t), np.cos(t) * np.cos(p) + np.sin(p)]
        )
        / np.sqrt(2),
    ),
]


@pytest.mark.parametrize("rep", ("exact", "mps"))
@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation values"""

    @pytest.mark.parametrize("gate,obs,expected", single_wire_expval_test_data)
    def test_single_wire_expectation(self, gate, obs, expected, theta, phi, rep, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        dev = DefaultTensorTF(wires=2, representation=rep)
        queue = [gate(theta, wires=0), gate(phi, wires=1), qml.CNOT(wires=[0, 1])]
        observables = [obs(wires=[i]) for i in range(2)]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Expectation

        res = dev.execute(queue, observables, {})
        assert np.allclose(res, expected(theta, phi), atol=tol, rtol=0)

    def test_hermitian_expectation(self, theta, phi, rep, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        dev = DefaultTensorTF(wires=2, representation=rep)
        queue = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
        observables = [qml.Hermitian(A, wires=[i]) for i in range(2)]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Expectation

        res = dev.execute(queue, observables, {})

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multi_mode_hermitian_expectation(self, theta, phi, rep, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev = DefaultTensorTF(wires=2, representation=rep)
        queue = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
        observables = [qml.Hermitian(A, wires=[0, 1])]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Expectation

        res = dev.execute(queue, observables, {})

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, rep, tol):
        """Tests for variance calculation"""
        dev = DefaultTensorTF(wires=1, representation=rep)
        # test correct variance for <Z> of a rotated state

        queue = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
        observables = [qml.PauliZ(wires=[0])]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Variance

        res = dev.execute(queue, observables, {})
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_hermitian(self, theta, phi, rep, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = DefaultTensorTF(wires=2, representation=rep)

        # test correct variance for <H> of a rotated state
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        queue = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
        observables = [qml.Hermitian(H, wires=[0])]

        for i in range(len(observables)):
            observables[i].return_type = qml.operation.Variance

        res = dev.execute(queue, observables, {})
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)


#####################################################
# Integration tests
#####################################################


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestQNodeIntegration:
    """Integration tests for default.tensor.tf. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_load_tensornet_tf_device(self, rep):
        """Test that the tensor network plugin loads correctly"""
        dev = qml.device("default.tensor.tf", wires=2, representation=rep)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.short_name == "default.tensor.tf"
        assert dev.capabilities()["provides_jacobian"]
        assert dev.capabilities()["passthru_interface"] == "tf"

    @pytest.mark.parametrize("decorator", [qml.qnode, qnode])
    def test_qubit_circuit(self, decorator, rep, tol):
        """Test that the tensor network plugin provides correct
        result for a simple circuit using the old QNode.
        This test is parametrized for both the new and old QNode decorator."""
        p = 0.543

        dev = qml.device("default.tensor.tf", wires=1, representation=rep)

        @decorator(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, rep, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("default.tensor.tf", wires=2, representation=rep)

        state = dev._state()

        expected = np.array([[1, 0], [0, 0]])
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev._state()

        expected = np.array([[1, 0], [1, 0]]) / np.sqrt(2)
        assert np.allclose(state, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestJacobianIntegration:
    """Tests for the Jacobian calculation"""

    def test_jacobian_variable_multiply(self, torch_support, rep, tol):
        """Test that qnode.jacobian applied to the tensornet.tf device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998

        dev = qml.device("default.tensor.tf", wires=1, representation=rep)

        @qnode(dev)
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit([x, y, z])
        expected = np.cos(3 * x) * np.cos(y) * np.cos(z / 2) - np.sin(3 * x) * np.sin(z / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = circuit.jacobian([[x, y, z]])
        expected = np.array(
            [
                -3 * (np.sin(3 * x) * np.cos(y) * np.cos(z / 2) + np.cos(3 * x) * np.sin(z / 2)),
                -np.cos(3 * x) * np.sin(y) * np.cos(z / 2),
                -0.5 * (np.sin(3 * x) * np.cos(z / 2) + np.cos(3 * x) * np.cos(y) * np.sin(z / 2)),
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated(self, torch_support, rep, tol):
        """Test that qnode.jacobian applied to the tensornet.tf device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = np.array([x, y, z])
        dev = qml.device("default.tensor.tf", wires=1, representation=rep)

        @qnode(dev)
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)
        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = circuit.jacobian([p])
        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0,]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "device"])
    def test_jacobian_agrees(self, diff_method, torch_support, rep, tol):
        """Test that qnode.jacobian applied to the tensornet.tf device
        returns the same result as default.qubit."""
        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.tensor.tf", wires=3, representation=rep)
        dev2 = qml.device("default.qubit", wires=3)

        circuit1 = QNode(circuit, dev1, diff_method=diff_method, h=1e-7)
        circuit2 = QNode(circuit, dev2, diff_method="parameter-shift", h=1e-7)

        assert np.allclose(circuit1(p), circuit2(p), atol=tol, rtol=0)
        assert np.allclose(circuit1.jacobian([p]), circuit2.jacobian([p]), atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestInterfaceDeviceIntegration:
    """Integration tests for default.tensor.tf. This test class ensures it integrates
    properly with the PennyLane UI, in particular the classical machine learning
    interfaces, when using the 'device' differentiation method."""

    a = -0.234
    b = 0.654
    p = [a, b]

    # the analytic result of evaluating circuit(a, b)
    expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)

    # the analytic result of evaluating grad(circuit(a, b))
    expected_grad = np.array(
        [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
    )

    @pytest.fixture
    def circuit(self, interface, torch_support, rep):
        """Fixture to create cost function for the test class"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        dev = qml.device("default.tensor.tf", wires=2, representation=rep)

        @qnode(dev, diff_method="device", interface=interface)
        def circuit_fn(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        return circuit_fn

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_autograd_interface(self, circuit, interface, tol):
        """Tests that the gradient of the circuit fixture above is correct
        using the autograd interface"""
        res = circuit(*self.p)
        assert np.allclose(res, self.expected_cost, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit, argnum=[0, 1])
        res = np.asarray(grad_fn(*self.p))
        assert np.allclose(res, self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_torch_interface(self, circuit, interface, tol):
        """Tests that the gradient of the circuit fixture above is correct
        using the Torch interface"""
        import torch
        from torch.autograd import Variable

        params = Variable(torch.tensor(self.p), requires_grad=True)
        res = circuit(*params)
        assert np.allclose(res.detach().numpy(), self.expected_cost, atol=tol, rtol=0)

        res.backward()
        res = params.grad
        assert np.allclose(res.detach().numpy(), self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_tf_interface(self, circuit, interface, tol):
        """Tests that the gradient of the circuit fixture above is correct
        using the TensorFlow interface"""
        import tensorflow as tf

        a = tf.Variable(self.a, dtype=tf.float64)
        b = tf.Variable(self.b, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch([a, b])
            res = circuit(a, b)

        assert np.allclose(res.numpy(), self.expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, [a, b])
        assert np.allclose(res, self.expected_grad, atol=tol, rtol=0)


@pytest.mark.parametrize("rep", ("exact", "mps"))
class TestHybridInterfaceDeviceIntegration:
    """Integration tests for default.tensor.tf. This test class ensures it integrates
    properly with the PennyLane UI, in particular the classical machine learning
    interfaces in the case of hybrid-classical computation, when using the
    device differentiation option."""

    theta = 0.543
    phi = -0.234
    lam = 0.654
    p = [theta, phi, lam]

    # the analytic result of evaluating cost(p)
    expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2

    # the analytic result of evaluating grad(cost(p))
    expected_grad = (
        np.array(
            [
                np.sin(theta) * np.cos(lam) * np.cos(phi),
                np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
            ]
        )
        * 2
        * (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi))
    )

    @pytest.fixture
    def cost(self, diff_method, interface, torch_support, rep):
        """Fixture to create cost function for the test class"""
        dev = qml.device("default.tensor.tf", wires=1, representation=rep)

        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        @qnode(dev, diff_method=diff_method, interface=interface)
        def circuit(x, weights, w=None):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(1j * np.array([1, -1]) / np.sqrt(2), wires=w)
            # the parameterized gate is one that gets decomposed
            # via a template
            qml.U3(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost_fn(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        return cost_fn

    @pytest.mark.parametrize("interface", ["autograd"])
    @pytest.mark.parametrize("diff_method", ["device"])
    def test_autograd_interface(self, cost, interface, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the autograd interface"""
        res = cost(self.p)
        assert np.allclose(res, self.expected_cost, atol=tol, rtol=0)

        grad_fn = qml.grad(cost, argnum=[0])
        res = np.asarray(grad_fn(self.p))
        assert np.allclose(res, self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["torch"])
    @pytest.mark.parametrize("diff_method", ["device"])
    def test_torch_interface(self, cost, interface, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the Torch interface"""
        import torch
        from torch.autograd import Variable

        params = Variable(torch.tensor(self.p), requires_grad=True)
        res = cost(params)
        assert np.allclose(res.detach().numpy(), self.expected_cost, atol=tol, rtol=0)

        res.backward()
        res = params.grad
        assert np.allclose(res.detach().numpy(), self.expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["tf"])
    @pytest.mark.parametrize("diff_method", ["backprop", "device"])
    def test_tf_interface(self, cost, interface, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct using the
        TensorFlow interface with the allowed differentiation methods"""
        import tensorflow as tf

        params = tf.Variable(self.p, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(params)
            res = cost(params)

        assert np.allclose(res.numpy(), self.expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, params)
        assert np.allclose(res.numpy(), self.expected_grad, atol=tol, rtol=0)

    def test_error_backprop_diff_torch(self, torch_support, tol, rep):
        """Tests that an error is raised if for the backprop differentiation
        method when using the Torch interface"""
        if not torch_support:
            pytest.skip("Skipped, no torch support")

        import torch
        from torch.autograd import Variable

        interface = "torch"
        diff_method = "backprop"

        params = Variable(torch.tensor(self.p), requires_grad=True)

        def cost_raising_error(params):
            # Cost within the test case such that the error can be caught
            dev = qml.device("default.tensor.tf", wires=1)

            if interface == "torch" and not torch_support:
                pytest.skip("Skipped, no torch support")

            @qnode(dev, diff_method=diff_method, interface=interface)
            def circuit(x, w=None):
                qml.RZ(x, wires=w)
                return qml.expval(qml.PauliX(w))

            return circuit(params[0], w=0)

        with pytest.raises(ValueError, match="Device default.tensor.tf only supports diff_method='backprop' when using the tf interface"):
            res = cost_raising_error(params)

    def test_error_backprop_diff_autograd(self, tol, rep):
        """Tests that an error is raised if for the backprop differentiation
        method when using the autograd interface"""
        interface = "autograd"
        diff_method = "backprop"

        params = self.p

        def cost_raising_error(params):
            # Cost within the test case such that the error can be caught
            dev = qml.device("default.tensor.tf", wires=1)

            @qnode(dev, diff_method=diff_method, interface=interface)
            def circuit(x, w=None):
                qml.RZ(x, wires=w)
                return qml.expval(qml.PauliX(w))

            return circuit(params[0], w=0)

        with pytest.raises(ValueError, match="Device default.tensor.tf only supports diff_method='backprop' when using the tf interface"):
            res = cost_raising_error(params)


class TestWiresIntegration:
    """Test that the device integrates with PennyLane's wire management."""

    def make_circuit_expval(self, wires, representation):
        """Factory for a qnode returning expvals using arbitrary wire labels."""
        dev = qml.device("default.tensor.tf", wires=wires, representation=representation)
        n_wires = len(wires)

        @qml.qnode(dev, interface='tf')
        def circuit():
            # use modulo to make circuit independent of number of wires
            qml.RX(0.5, wires=wires[0 % n_wires])
            qml.Hadamard(wires=wires[3 % n_wires])
            qml.RY(2., wires=wires[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires[0 % n_wires], wires[1 % n_wires]])
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires]

        return circuit

    @pytest.mark.parametrize("representation", ["exact", "mps"])
    @pytest.mark.parametrize("wires1, wires2", [(['a', 'c', 'd'], [2, 3, 0]),
                                                ([-1, -2, -3], ['q1', 'ancilla', 2]),
                                                (['a', 'c'], [3, 0]),
                                                ([-1, -2], ['ancilla', 2]),
                                                (['a'], ['nothing']),
                                                ])
    def test_wires_expval(self, wires1, wires2, representation, tol):
        """Test that the expectation of a circuit is independent from the wire labels used."""

        circuit1 = self.make_circuit_expval(wires1, representation)
        circuit2 = self.make_circuit_expval(wires2, representation)

        assert np.allclose(circuit1(), circuit2(), tol)
