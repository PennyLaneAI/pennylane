# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit and integration tests for the adjoint_jacobian function for DefaultQubit"""
import pytest
import pennylane as qml
from pennylane.devices.qubit import adjoint_jacobian, adjoint_jvp, adjoint_vjp
from pennylane.tape import QuantumScript
import pennylane.numpy as np


class TestAdjointJacobian:
    """Tests for adjoint_jacobian"""

    def test_custom_wire_labels(self, tol):
        """Test that adjoint_jacbonian works as expected when custom wire labels are used."""
        qs = QuantumScript(
            [qml.RX(0.123, wires="a"), qml.RY(0.456, wires="b")], [qml.expval(qml.PauliX("a"))]
        )
        qs.trainable_params = {0, 1}

        calculated_val = adjoint_jacobian(qs)

        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        numeric_val = fn(results)
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        prep_op = qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
        qs = QuantumScript(
            ops=[G(theta, wires=[0])], measurements=[qml.expval(qml.PauliZ(0))], prep=[prep_op]
        )

        qs.trainable_params = {1}

        calculated_val = adjoint_jacobian(qs)
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        numeric_val = fn(results)
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)
        assert isinstance(calculated_val, np.ndarray)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])
        prep_op = qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)

        qs = QuantumScript(
            ops=[qml.Rot(*params, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            prep=[prep_op],
        )

        qs.trainable_params = {1, 2, 3}
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs = qs_valid[0]

        qs.trainable_params = {1, 2, 3}

        calculated_val = adjoint_jacobian(qs)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        numeric_val = fn(results)
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)
        assert isinstance(calculated_val, tuple)
        assert all(isinstance(val, np.ndarray) for val in calculated_val)

    @pytest.mark.autograd
    @pytest.mark.parametrize("obs", [qml.PauliY])
    @pytest.mark.parametrize(
        "op", [qml.RX(0.4, wires=0), qml.CRZ(1.0, wires=[0, 1]), qml.Rot(0.2, -0.1, 0.2, wires=0)]
    )
    def test_gradients(self, op, obs, tol):
        """Tests that the gradients of circuits match between the finite difference and device
        methods."""

        ops = [
            qml.Hadamard(wires=0),
            qml.RX(0.543, wires=0),
            qml.CNOT(wires=[0, 1]),
            op,
            qml.Rot(1.3, -2.3, 0.5, wires=[0]),
            qml.RZ(-0.5, wires=0),
            qml.adjoint(qml.RY(0.5, wires=1)),
            qml.CNOT(wires=[0, 1]),
        ]
        measurements = [qml.expval(obs(wires=0)), qml.expval(qml.PauliZ(wires=1))]

        qs = QuantumScript(ops, measurements)
        qs.trainable_params = set(range(1, 1 + op.num_params))

        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        qs_valid.trainable_params = set(range(1, 1 + op.num_params))

        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        numeric_val = fn(results)
        grad_D = adjoint_jacobian(qs_valid)

        grad_F = np.squeeze(numeric_val)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)
        assert isinstance(grad_D, tuple)

    def test_multiple_rx_gradient(self, tol):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        qs = QuantumScript(
            [qml.RX(params[0], wires=0), qml.RX(params[1], wires=1), qml.RX(params[2], wires=2)],
            [qml.expval(qml.PauliZ(idx)) for idx in range(3)],
        )
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        # circuit jacobians
        jacobian = adjoint_jacobian(qs_valid)
        expected_jacobian = -np.diag(np.sin(params))
        assert np.allclose(jacobian, expected_jacobian, atol=tol, rtol=0)
        assert isinstance(jacobian, tuple)
        assert all(isinstance(j, tuple) for j in jacobian)

    def test_custom_op_gradient(self, tol):
        """Tests that the gradient of a custom operation that only provides a
        matrix and a generator works."""

        class MyOp(qml.operation.Operation):
            """Custom operation that only defines a generator and a matrix representation."""

            num_wires = 1

            def generator(self):
                """Generator of MyOp, just a multiple of Pauli X."""
                return qml.PauliX(self.wires) * 1.2

            @staticmethod
            def compute_matrix(angle):  # pylint: disable=arguments-differ
                """Matrix representation of MyOp, just the same as a reparametrized RX."""
                return qml.RX.compute_matrix(-2.4 * angle)

        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        qs = QuantumScript(
            [MyOp(p, w) for p, w in zip(params, [0, 1, 2])],
            [qml.expval(qml.PauliZ(idx)) for idx in range(3)],
        )
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        # circuit jacobians
        jacobian = adjoint_jacobian(qs_valid)
        expected_jacobian = 2.4 * np.diag(np.sin(-2.4 * params))
        assert np.allclose(jacobian, expected_jacobian, atol=tol, rtol=0)
        assert isinstance(jacobian, tuple)
        assert all(isinstance(j, tuple) for j in jacobian)

    @pytest.mark.autograd
    def test_gradient_gate_with_multiple_parameters(self, tol):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        qs = QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
        )

        qs.trainable_params = {1, 2, 3}
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        qs_valid.trainable_params = {1, 2, 3}

        grad_D = adjoint_jacobian(qs_valid)
        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        grad_F = fn(results)
        grad_F = np.squeeze(grad_F)

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "prep_op", [qml.BasisState([1], wires=0), qml.StatePrep([0, 1], wires=0)]
    )
    def test_state_prep(self, prep_op, tol):
        """Tests provides correct answer when provided state preparation operation."""
        x, y, z = [0.5, 0.3, -0.7]

        qs = QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
            [prep_op],
        )

        qs.trainable_params = {2, 3, 4}
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        qs_valid.trainable_params = {2, 3, 4}

        grad_D = adjoint_jacobian(qs_valid)
        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        grad_F = fn(results)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    def test_gradient_of_tape_with_hermitian(self, tol):
        """Test that computing the gradient of a tape that obtains the
        expectation value of a Hermitian operator works correctly."""
        a, b, c = [0.5, 0.3, -0.7]

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        qs = QuantumScript(
            [
                qml.RX(a, wires=0),
                qml.RX(b, wires=1),
                qml.RX(c, wires=2),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            [qml.expval(qml.Hermitian(mx, wires=[0, 2]))],
        )

        qs.trainable_params = {0, 1, 2}
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        qs_valid.trainable_params = {0, 1, 2}

        res = adjoint_jacobian(qs_valid)

        expected = [
            np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(b) * np.sin(a) * np.sin(c),
            np.cos(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_gradient_of_tape_with_tensor(self, tol):
        """Test that computing the gradient of a tape that obtains the
        expectation value of a Tensor operator works correctly."""
        a, b, c = [0.5, 0.3, -0.7]

        qs = QuantumScript(
            [
                qml.RX(a, wires=0),
                qml.RX(b, wires=1),
                qml.RX(c, wires=2),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            [qml.expval(qml.PauliX(0) @ qml.PauliY(2))],
        )

        qs.trainable_params = {0, 1, 2}
        qs_valid, _ = qml.devices.preprocess.decompose(
            qs, qml.devices.default_qubit.adjoint_stopping_condition
        )
        qs_valid = qs_valid[0]

        res = adjoint_jacobian(qs_valid)

        expected = [
            np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(b) * np.sin(a) * np.sin(c),
            np.cos(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestAdjointJVP:
    """Test for adjoint_jvp"""

    @pytest.mark.parametrize("tangents", [(0,), (1.232,)])
    def test_single_param_single_obs(self, tangents, tol):
        """Test JVP is correct for a single parameter and observable"""
        x = np.array(0.654)
        qs = QuantumScript([qml.RY(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs.trainable_params = {0}

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, np.ndarray)

        expected = -tangents[0] * np.sin(x)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("tangents", [(0,), (1.232,)])
    def test_single_param_multi_obs(self, tangents, tol):
        """Test JVP is correct for a single parameter and multiple observables"""
        x = np.array(0.654)
        qs = QuantumScript([qml.RY(x, 0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))])
        qs.trainable_params = {0}

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2
        assert all(isinstance(r, np.ndarray) for r in actual)

        expected = tangents[0] * np.array([-np.sin(x), np.cos(x)])
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("tangents", [(0, 0), (0, 0.653), (1.232, 2.963)])
    def test_multi_param_single_obs(self, tangents, tol):
        """Test JVP is correct for multiple parameters and a single observable"""
        x = np.array(0.654)
        y = np.array(1.221)

        qs = QuantumScript([qml.RY(x, 0), qml.RZ(y, 0)], [qml.expval(qml.PauliY(0))])
        qs.trainable_params = {0, 1}

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, np.ndarray)

        expected = np.dot(
            np.array([np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)]), np.array(tangents)
        )
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("tangents", [(0, 0), (0, 0.653), (1.232, 2.963)])
    def test_multi_param_multi_obs(self, tangents, tol):
        """Test JVP is correct for multiple parameters and observables"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))]
        qs = QuantumScript([qml.RY(x, 0), qml.RZ(y, 0)], obs)
        qs.trainable_params = {0, 1}

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 3
        assert all(isinstance(r, np.ndarray) for r in actual)

        jac = np.array(
            [
                [-np.sin(x), 0],
                [np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y)],
                [np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)],
            ]
        )
        expected = jac @ np.array(tangents)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("tangents", [(0, 0), (0, 0.653), (1.232, 2.963)])
    @pytest.mark.parametrize("wires", [[1, 0], ["a", "b"]])
    def test_custom_wire_labels(self, tangents, wires, tol):
        """Test JVP is correct for custom wire labels"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [
            qml.expval(qml.PauliZ(wires[0])),
            qml.expval(qml.PauliY(wires[1])),
            qml.expval(qml.PauliX(wires[0])),
        ]
        qs = QuantumScript([qml.RY(x, wires[0]), qml.RX(y, wires[1])], obs)
        qs.trainable_params = {0, 1}
        assert qs.wires.tolist() == wires

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 3
        assert all(isinstance(r, np.ndarray) for r in actual)

        jac = np.array([[-np.sin(x), 0], [0, -np.cos(y)], [np.cos(x), 0]])
        expected = jac @ np.array(tangents)
        assert np.allclose(actual, expected, atol=tol)


class TestAdjointVJP:
    """Test for adjoint_vjp"""

    @pytest.mark.parametrize("cotangents", [(0,), (1.232,)])
    def test_single_param_single_obs(self, cotangents, tol):
        """Test VJP is correct for a single parameter and observable"""
        x = np.array(0.654)
        qs = QuantumScript([qml.RY(x, 0)], [qml.expval(qml.PauliZ(0))])
        qs.trainable_params = {0}

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, np.ndarray)

        expected = -cotangents[0] * np.sin(x)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("cotangents", [(0, 0), (0, 0.653), (1.232, 2.963)])
    def test_single_param_multi_obs(self, cotangents, tol):
        """Test VJP is correct for a single parameter and multiple observables"""
        x = np.array(0.654)
        qs = QuantumScript([qml.RY(x, 0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))])
        qs.trainable_params = {0}

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, np.ndarray)

        expected = np.dot(np.array([-np.sin(x), np.cos(x)]), np.array(cotangents))
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("cotangents", [(0,), (1.232,)])
    def test_multi_param_single_obs(self, cotangents, tol):
        """Test VJP is correct for multiple parameters and a single observable"""
        x = np.array(0.654)
        y = np.array(1.221)

        qs = QuantumScript([qml.RY(x, 0), qml.RZ(y, 0)], [qml.expval(qml.PauliY(0))])
        qs.trainable_params = {0, 1}

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2
        assert all(isinstance(r, np.ndarray) for r in actual)

        expected = cotangents[0] * np.array([np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)])
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents", [(0, 0, 0), (0, 0.653, 0), (1.236, 0, 0.573), (1.232, 2.963, 1.942)]
    )
    def test_multi_param_multi_obs(self, cotangents, tol):
        """Test VJP is correct for multiple parameters and observables"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))]
        qs = QuantumScript([qml.RY(x, 0), qml.RZ(y, 0)], obs)
        qs.trainable_params = {0, 1}

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2
        assert all(isinstance(r, np.ndarray) for r in actual)

        jac = np.array(
            [
                [-np.sin(x), 0],
                [np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y)],
                [np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)],
            ]
        )
        expected = np.array(cotangents) @ jac
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents", [(0, 0, 0), (0, 0.653, 0), (1.236, 0, 0.573), (1.232, 2.963, 1.942)]
    )
    @pytest.mark.parametrize("wires", [[1, 0], ["a", "b"]])
    def test_custom_wire_labels(self, cotangents, wires, tol):
        """Test VJP is correct for custom wire labels"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [
            qml.expval(qml.PauliZ(wires[0])),
            qml.expval(qml.PauliY(wires[1])),
            qml.expval(qml.PauliX(wires[0])),
        ]
        qs = QuantumScript([qml.RY(x, wires[0]), qml.RX(y, wires[1])], obs)
        qs.trainable_params = {0, 1}
        assert qs.wires.tolist() == wires

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2
        assert all(isinstance(r, np.ndarray) for r in actual)

        jac = np.array([[-np.sin(x), 0], [0, -np.cos(y)], [np.cos(x), 0]])
        expected = np.array(cotangents) @ jac
        assert np.allclose(actual, expected, atol=tol)
