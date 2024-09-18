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
import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit import adjoint_jacobian, adjoint_jvp, adjoint_vjp
from pennylane.tape import QuantumScript


def adjoint_ops(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    return op.num_params == 0 or op.num_params == 1 and op.has_generator


class TestAdjointJacobian:
    """Tests for adjoint_jacobian"""

    def test_custom_wire_labels(self, tol):
        """Test that adjoint_jacbonian works as expected when custom wire labels are used."""
        qs = QuantumScript(
            [qml.RX(0.123, wires="a"), qml.RY(0.456, wires="b")],
            [qml.expval(qml.PauliX("a"))],
            trainable_params=[0, 1],
        )

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

        prep_op = qml.StatePrep(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
        qs = QuantumScript(
            ops=[prep_op, G(theta, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            trainable_params=[1],
        )

        calculated_val = adjoint_jacobian(qs)
        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        numeric_val = fn(results)
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol):
        """Tests that the device gradient of an arbitrary Euler-angle-parametrized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])
        prep_op = qml.StatePrep(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)

        qs = QuantumScript(
            ops=[prep_op, qml.Rot(*params, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            trainable_params=[1, 2, 3],
        )

        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
        qs = qs_valid[0]

        qs.trainable_params = {1, 2, 3}

        calculated_val = adjoint_jacobian(qs)

        # compare to finite differences
        tapes, fn = qml.gradients.finite_diff(qs)
        results = tuple(qml.devices.qubit.simulate(t) for t in tapes)
        numeric_val = fn(results)
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)
        assert isinstance(calculated_val, tuple)

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

        qs = QuantumScript(ops, measurements, trainable_params=list(range(1, 1 + op.num_params)))

        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
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
        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
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
        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
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
            trainable_params=[1, 2, 3],
        )

        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
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
            [prep_op, qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
            trainable_params=[2, 3, 4],
        )

        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
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
            trainable_params=[0, 1, 2],
        )

        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
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
            trainable_params=[0, 1, 2],
        )

        qs_valid, _ = qml.devices.preprocess.decompose(qs, adjoint_ops)
        qs_valid = qs_valid[0]

        res = adjoint_jacobian(qs_valid)

        expected = [
            np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(b) * np.sin(a) * np.sin(c),
            np.cos(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_with_nontrainable_parametrized(self):
        """Test that a parametrized `QubitUnitary` is accounted for correctly
        when it is not trainable."""

        par = np.array(0.6)

        ops = [
            qml.RY(par, wires=0),
            qml.QubitUnitary(np.eye(2), wires=0),
        ]
        qs = QuantumScript(ops, [qml.expval(qml.PauliZ(0))], trainable_params=[0])

        grad_adjoint = adjoint_jacobian(qs)
        expected = [-np.sin(par)]
        assert np.allclose(grad_adjoint, expected)


class TestAdjointJacobianState:
    """Tests for differentiating a state vector."""

    def test_simple_state_derivative(self):
        """Test state differentiation for a single parameter."""
        x = 1.2
        tape = qml.tape.QuantumScript([qml.RX(x, wires=0)], [qml.state()])
        jac = adjoint_jacobian(tape)
        expected = [-0.5 * np.sin(x / 2), -0.5j * np.cos(x / 2)]
        assert qml.math.allclose(jac, expected)

        dy = np.array([0.5, 2.0], dtype=np.complex128)
        vjp = adjoint_vjp(tape, dy)
        expected_vjp = dy[0] * expected[0] + dy[1] * expected[1]
        assert qml.math.allclose(expected_vjp, vjp)

    def test_two_wires_two_parameters(self):
        """Test a more complicated circuit with two parameters and two wires."""

        x = 0.5
        y = 0.6
        tape = qml.tape.QuantumScript([qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))], [qml.state()])
        x_jac, y_jac = adjoint_jacobian(tape)

        c_x, s_x = np.cos(x / 2), np.sin(x / 2)
        c_y, s_y = np.cos(y / 2), np.sin(y / 2)
        x_jac_expected = np.array(
            [-0.5 * c_y * s_x, -0.5 * s_y * s_x, -0.5j * s_y * c_x, -0.5j * c_x * c_y]
        )
        assert qml.math.allclose(x_jac, x_jac_expected)

        y_jac_expected = np.array(
            [-0.5 * c_x * s_y, 0.5 * c_x * c_y, -0.5j * s_x * c_y, 0.5j * s_x * s_y]
        )
        assert qml.math.allclose(y_jac, y_jac_expected)

        dy = np.array([0.5, 1.0, 2.0, 2.5], dtype=np.complex128)
        x_vjp, y_vjp = adjoint_vjp(tape, dy)
        x_vjp_expected = np.dot(x_jac_expected, dy)
        assert qml.math.allclose(x_vjp, x_vjp_expected)
        y_vjp_expected = np.dot(y_jac_expected, dy)
        assert qml.math.allclose(y_vjp, y_vjp_expected)


class TestAdjointJVP:
    """Test for adjoint_jvp"""

    @pytest.mark.parametrize("tangents", [(0,), (1.232,)])
    def test_single_param_single_obs(self, tangents, tol):
        """Test JVP is correct for a single parameter and observable"""
        x = np.array(0.654)
        qs = QuantumScript([qml.RY(x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0])

        actual = adjoint_jvp(qs, tangents)

        expected = -tangents[0] * np.sin(x)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("tangents", [(0,), (1.232,)])
    def test_single_param_multi_obs(self, tangents, tol):
        """Test JVP is correct for a single parameter and multiple observables"""
        x = np.array(0.654)
        qs = QuantumScript(
            [qml.RY(x, 0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))],
            trainable_params=[0],
        )

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        expected = tangents[0] * np.array([-np.sin(x), np.cos(x)])
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("tangents", [(0, 0), (0, 0.653), (1.232, 2.963)])
    def test_multi_param_single_obs(self, tangents, tol):
        """Test JVP is correct for multiple parameters and a single observable"""
        x = np.array(0.654)
        y = np.array(1.221)

        qs = QuantumScript(
            [qml.RY(x, 0), qml.RZ(y, 0)], [qml.expval(qml.PauliY(0))], trainable_params=[0, 1]
        )

        actual = adjoint_jvp(qs, tangents)

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
        qs = QuantumScript([qml.RY(x, 0), qml.RZ(y, 0)], obs, trainable_params=[0, 1])

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 3

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
        qs = QuantumScript([qml.RY(x, wires[0]), qml.RX(y, wires[1])], obs, trainable_params=[0, 1])
        assert qs.wires.tolist() == wires

        actual = adjoint_jvp(qs, tangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 3

        jac = np.array([[-np.sin(x), 0], [0, -np.cos(y)], [np.cos(x), 0]])
        expected = jac @ np.array(tangents)
        assert np.allclose(actual, expected, atol=tol)

    def test_with_nontrainable_parametrized(self):
        """Test that a parametrized `QubitUnitary` is accounted for correctly
        when it is not trainable."""

        par = np.array(0.6)
        tangents = (0.45,)

        ops = [
            qml.RY(par, wires=0),
            qml.QubitUnitary(np.eye(2), wires=0),
        ]
        qs = QuantumScript(ops, [qml.expval(qml.PauliZ(0))], trainable_params=[0])

        jvp_adjoint = adjoint_jvp(qs, tangents)
        expected = [-np.sin(par) * tangents[0]]
        assert np.allclose(jvp_adjoint, expected)


class TestAdjointVJP:
    """Test for adjoint_vjp"""

    @pytest.mark.parametrize("cotangents", [0, 1.232, 5.2])
    def test_single_param_single_obs(self, cotangents, tol):
        """Test VJP is correct for a single parameter and observable"""
        x = np.array(0.654)
        qs = QuantumScript([qml.RY(x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0])

        actual = adjoint_vjp(qs, cotangents)

        expected = -cotangents * np.sin(x)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("cotangents", [(0, 0), (0, 0.653), (1.232, 2.963)])
    def test_single_param_multi_obs(self, cotangents, tol):
        """Test VJP is correct for a single parameter and multiple observables"""
        x = np.array(0.654)
        qs = QuantumScript(
            [qml.RY(x, 0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))],
            trainable_params=[0],
        )

        actual = adjoint_vjp(qs, cotangents)

        expected = np.dot(np.array([-np.sin(x), np.cos(x)]), np.array(cotangents))
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize("cotangents", [0, 1.232])
    def test_multi_param_single_obs(self, cotangents, tol):
        """Test VJP is correct for multiple parameters and a single observable"""
        x = np.array(0.654)
        y = np.array(1.221)

        qs = QuantumScript(
            [qml.RY(x, 0), qml.RZ(y, 0)], [qml.expval(qml.PauliY(0))], trainable_params=[0, 1]
        )

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        expected = cotangents * np.array([np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)])
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents", [(0, 0, 0), (0, 0.653, 0), (1.236, 0, 0.573), (1.232, 2.963, 1.942)]
    )
    def test_multi_param_multi_obs(self, cotangents, tol):
        """Test VJP is correct for multiple parameters and observables"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))]
        qs = QuantumScript([qml.RY(x, 0), qml.RZ(y, 0)], obs, trainable_params=[0, 1])

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

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
        qs = QuantumScript([qml.RY(x, wires[0]), qml.RX(y, wires[1])], obs, trainable_params=[0, 1])
        assert qs.wires.tolist() == wires

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        jac = np.array([[-np.sin(x), 0], [0, -np.cos(y)], [np.cos(x), 0]])
        expected = np.array(cotangents) @ jac
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        ((0, 1.23), (1.232, -2.098, 0.323, 1.112), (5.212, -0.354, -2.575), (0.0, 0.0, 0.0)),
    )
    def test_single_param_single_obs_batched(self, cotangents, tol):
        """Test that batched cotangents with adjoint VJP give correct results when
        the tape has a single trainable parameter and a single observable"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [qml.expval(qml.PauliZ(wires=[0]))]
        qs = QuantumScript([qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0])

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 1

        jac = np.array([[-np.sin(x)]])
        expected = jac.T @ np.expand_dims(np.array(cotangents), 0)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.653, 0, 0]), np.array([0, 0.573, 0]), np.array([0, 0, 1.232])),
            (np.array([0.653, -1.456]), np.array([0.498, 0.573]), np.array([0, 1.232])),
            (
                np.array([0.653, 0, 0, -1.234]),
                np.array([-0.323, 0.573, -1.449, -0.573]),
                np.array([0, 1, 1.232, 1.232]),
            ),
            (
                np.array([0, 0, 0]),
                np.array([0, 0, 0]),
                np.array([0, 0, 0]),
            ),
        ],
    )
    def test_single_param_multi_obs_batched(self, cotangents, tol):
        """Test that batched cotangents with adjoint VJP give correct results when
        the tape has a single trainable parameter and multiple observables"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [
            qml.expval(qml.PauliZ(wires=[0])),
            qml.expval(qml.PauliY(wires=[1])),
            qml.expval(qml.PauliX(wires=[0])),
        ]
        qs = QuantumScript([qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0])

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 1

        jac = np.array([[-np.sin(x)], [0], [np.cos(x)]])
        expected = jac.T @ np.array(cotangents)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        ((0.0, 1.23), (1.232, -2.098, 0.323, 1.112), (5.212, -0.354, -2.575), (0.0, 0.0, 0.0)),
    )
    def test_multi_param_single_obs_batched(self, cotangents, tol):
        """Test that batched cotangents with adjoint VJP give correct results when
        the tape has multiple trainable parameters and a single observable"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [qml.expval(qml.PauliZ(wires=[0]))]
        qs = QuantumScript(
            [qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0, 1]
        )

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        jac = np.array([[-np.sin(x), 0]])
        expected = jac.T @ np.expand_dims(np.array(cotangents), 0)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.653, 0, 0]), np.array([0, 0.573, 0]), np.array([0, 0, 1.232])),
            (np.array([0.653, -1.456]), np.array([0.498, 0.573]), np.array([0, 1.232])),
            (
                np.array([0.653, 0, 0, -1.234]),
                np.array([-0.323, 0.573, -1.449, -0.573]),
                np.array([0, 1, 1.232, 1.232]),
            ),
            (
                np.array([0.0, 0, 0]),
                np.array([0.0, 0, 0]),
                np.array([0.0, 0, 0]),
            ),
        ],
    )
    def test_multi_param_multi_obs_batched(self, cotangents, tol):
        """Test that batched cotangents with adjoint VJP give correct results when
        the tape has multiple trainable parameters and multiple observables"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [
            qml.expval(qml.PauliZ(wires=[0])),
            qml.expval(qml.PauliY(wires=[1])),
            qml.expval(qml.PauliX(wires=[0])),
        ]
        qs = QuantumScript(
            [qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0, 1]
        )

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        jac = np.array([[-np.sin(x), 0], [0, -np.cos(y)], [np.cos(x), 0]])
        expected = jac.T @ np.array(cotangents)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        [
            (np.array([0.498, 0.573]), np.array([0.653, -1.456]), 0.0),
            (np.array([0.498, 0.573, -1.456]), 0.0, 0.0),
        ],
    )
    def test_inhomogenous_cotangents(self, cotangents, tol):
        """Test that inhomogenous cotangents give the correct VJP"""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [
            qml.expval(qml.PauliZ(wires=[0])),
            qml.expval(qml.PauliY(wires=[1])),
            qml.expval(qml.PauliX(wires=[0])),
        ]
        qs = QuantumScript(
            [qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0, 1]
        )

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        new_cotangents = []
        inner_shape = len(cotangents[0])
        for c in cotangents:
            if isinstance(c, float):
                new_cotangents.append(np.zeros(inner_shape))
            else:
                new_cotangents.append(c)

        jac = np.array([[-np.sin(x), 0], [0, -np.cos(y)], [np.cos(x), 0]])
        expected = jac.T @ np.array(new_cotangents)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        [
            np.array(
                [
                    [0.123 + 0j, 0.456, 0.789, -0.123],
                    [-0.456, -0.789, 1.234, 5.678],
                    [-0.345, -4.345, -2.589, 3.456],
                ],
                dtype=np.complex128,
            ),
            np.array(
                [[0.0 + 0j, 0.123, 0.765, 4.123], [-7.698, -3.465, -1.289, 4.697]],
                dtype=np.complex128,
            ),
        ],
    )
    def test_single_param_state_batched(self, cotangents, tol):
        """Test that computing the VJP with batched cotangents for state measurements
        gives the correct results for a single trainable parameter."""
        x = np.array(0.654)
        y = np.array(1.221)

        obs = [qml.state()]
        qs = QuantumScript([qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0])

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 1

        jac = np.array(
            [
                [-0.5 * np.sin(x / 2) * np.cos(y / 2)],
                [0.5j * np.sin(x / 2) * np.sin(y / 2)],
                [0.5 * np.cos(x / 2) * np.cos(y / 2)],
                [-0.5j * np.cos(x / 2) * np.sin(y / 2)],
            ]
        )
        expected = jac.T @ cotangents.T
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.parametrize(
        "cotangents",
        [
            np.array(
                [
                    [0.123, 0.456, 0.789, -0.123],
                    [-0.456, -0.789, 1.234, 5.678],
                    [-0.345, -4.345, -2.589, 3.456],
                ],
                dtype=np.complex128,
            ),
            np.array(
                [[0.0, 0.123, 0.765, 4.123], [-7.698, -3.465, -1.289, 4.697]], dtype=np.complex128
            ),
        ],
    )
    def test_multi_param_state_batched(self, cotangents, tol):
        """Test that computing the VJP with batched cotangents for state measurements
        gives the correct results for multiple trainable parameters."""
        x = np.array(0.654 + 0j)
        y = np.array(1.221 + 0j)

        obs = [qml.state()]
        qs = QuantumScript(
            [qml.RY(x, wires=[0]), qml.RX(y, wires=[1])], obs, trainable_params=[0, 1]
        )

        actual = adjoint_vjp(qs, cotangents)
        assert isinstance(actual, tuple)
        assert len(actual) == 2

        jac = np.array(
            [
                [-0.5 * np.sin(x / 2) * np.cos(y / 2), -0.5 * np.cos(x / 2) * np.sin(y / 2)],
                [0.5j * np.sin(x / 2) * np.sin(y / 2), -0.5j * np.cos(x / 2) * np.cos(y / 2)],
                [0.5 * np.cos(x / 2) * np.cos(y / 2), -0.5 * np.sin(x / 2) * np.sin(y / 2)],
                [-0.5j * np.cos(x / 2) * np.sin(y / 2), -0.5j * np.sin(x / 2) * np.cos(y / 2)],
            ]
        )
        expected = jac.T @ cotangents.T
        assert np.allclose(actual, expected, atol=tol)

    def test_with_nontrainable_parametrized(self):
        """Test that a parametrized `QubitUnitary` is accounted for correctly
        when it is not trainable."""

        par = np.array(0.6)
        cotangents = (0.45,)

        ops = [
            qml.RY(par, wires=0),
            qml.QubitUnitary(np.eye(2), wires=0),
        ]
        qs = QuantumScript(ops, [qml.expval(qml.PauliZ(0))], trainable_params=[0])

        vjp_adjoint = adjoint_vjp(qs, cotangents)
        expected = [-np.sin(par) * cotangents[0]]
        assert np.allclose(vjp_adjoint, expected)

    def test_hermitian_expval(self):
        """Test adjoint_vjp works with a hermitian expectation value."""

        x = 1.2
        H = qml.Hermitian(np.array([[1, 0], [0, -1]]), wires=0)
        cotangent = (0.5,)

        qs = QuantumScript([qml.RX(x, wires=0)], [qml.expval(H)], trainable_params=[0])

        [vjp_adjoint] = adjoint_vjp(qs, cotangent)
        assert qml.math.allclose(vjp_adjoint, -0.5 * np.sin(x))
