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
"""Unit and integration tests for the adjoint_jacobian function for DefaultQubit2"""
import pytest
import pennylane as qml
from pennylane.devices.qubit import adjoint_jacobian
from pennylane.tape import QuantumScript
import pennylane.numpy as np


class TestAdjointJacobian:
    """Unit tests for adjoint_jacobian"""

    @pytest.fixture
    def dev(self):
        """Fixture that creates a two-qubit default qubit device for comparisions."""
        return qml.device("default.qubit", wires=2)

    def test_not_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        measurements = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(3)), qml.sample()]
        qs = QuantumScript(ops=[], measurements=measurements)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement",
        ):
            adjoint_jacobian(qs)

    def test_unsupported_op(self):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        qs = QuantumScript([qml.U2(0.1, 0.2, wires=[0])], [qml.expval(qml.PauliZ(2))])

        with pytest.raises(
            qml.QuantumFunctionError, match="The U2 operation is not supported using"
        ):
            adjoint_jacobian(qs)

    @pytest.mark.parametrize(
        "obs",
        [
            qml.Hamiltonian([2, 0.5], [qml.PauliZ(0), qml.PauliY(1)]),
        ],
    )
    def test_unsupported_obs(self, obs):
        """Test that the correct error is raised if a Hamiltonian or Sum measurement is differentiated"""
        qs = QuantumScript([qml.RX(0.5, wires=1)], [qml.expval(obs)])

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support observable",
        ):
            adjoint_jacobian(qs)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_pauli_rotation_gradient(self, G, theta, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""

        prep_op = qml.QubitStateVector(
            np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = QuantumScript(
            ops=[G(theta, wires=[0])], measurements=[qml.expval(qml.PauliZ(0))], prep=[prep_op]
        )

        qs.trainable_params = {1}

        calculated_val = adjoint_jacobian(qs)

        # compare to finite differences
        dev_single_qubit = qml.device("default.qubit", wires=1)
        tapes, fn = qml.gradients.finite_diff(qs)
        numeric_val = fn(qml.execute(tapes, dev_single_qubit, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)
        assert isinstance(calculated_val, np.ndarray)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_Rot_gradient(self, theta, tol):
        """Tests that the device gradient of an arbitrary Euler-angle-parameterized gate is
        correct."""
        params = np.array([theta, theta**3, np.sqrt(2) * theta])
        prep_op = qml.QubitStateVector(
            np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )

        qs = QuantumScript(
            ops=[qml.Rot(*params, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            prep=[prep_op],
        )

        qs.trainable_params = {1, 2, 3}

        calculated_val = adjoint_jacobian(qs)

        # compare to finite differences
        dev_single_qubit = qml.device("default.qubit", wires=1)
        tapes, fn = qml.gradients.finite_diff(qs)
        numeric_val = fn(qml.execute(tapes, dev_single_qubit, None))
        assert np.allclose(calculated_val, numeric_val, atol=tol, rtol=0)
        assert isinstance(calculated_val, tuple)
        assert all(isinstance(val, np.ndarray) for val in calculated_val)

    @pytest.mark.autograd
    @pytest.mark.parametrize("obs", [qml.PauliY])
    @pytest.mark.parametrize(
        "op", [qml.RX(0.4, wires=0), qml.CRZ(1.0, wires=[0, 1]), qml.Rot(0.2, -0.1, 0.2, wires=0)]
    )
    def test_gradients(self, op, obs, tol, dev):
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

        tapes, fn = qml.gradients.finite_diff(qs)
        grad_F = fn(qml.execute(tapes, dev, None))
        grad_D = adjoint_jacobian(qs)

        grad_F = np.squeeze(grad_F)

        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)
        assert isinstance(grad_D, tuple)

    def test_multiple_rx_gradient(self, tol):
        """Tests that the gradient of multiple RX gates in a circuit yields the correct result."""
        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        qs = QuantumScript(
            [qml.RX(params[0], wires=0), qml.RX(params[1], wires=1), qml.RX(params[2], wires=2)],
            [qml.expval(qml.PauliZ(idx)) for idx in range(3)],
        )

        # circuit jacobians
        jacobian = adjoint_jacobian(qs)
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
            def compute_matrix(angle):
                """Matrix representation of MyOp, just the same as a reparametrized RX."""
                return qml.RX.compute_matrix(-2.4 * angle)

        params = np.array([np.pi, np.pi / 2, np.pi / 3])

        qs = QuantumScript(
            [MyOp(p, w) for p, w in zip(params, [0, 1, 2])],
            [qml.expval(qml.PauliZ(idx)) for idx in range(3)],
        )

        # circuit jacobians
        jacobian = adjoint_jacobian(qs)
        expected_jacobian = 2.4 * np.diag(np.sin(-2.4 * params))
        assert np.allclose(jacobian, expected_jacobian, atol=tol, rtol=0)
        assert isinstance(jacobian, tuple)
        assert all(isinstance(j, tuple) for j in jacobian)

    @pytest.mark.autograd
    def test_gradient_gate_with_multiple_parameters(self, tol, dev):
        """Tests that gates with multiple free parameters yield correct gradients."""
        x, y, z = [0.5, 0.3, -0.7]

        qs = QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
        )

        qs.trainable_params = {1, 2, 3}

        grad_D = adjoint_jacobian(qs)
        tapes, fn = qml.gradients.finite_diff(qs)
        grad_F = fn(qml.execute(tapes, dev, None))
        grad_F = np.squeeze(grad_F)

        # gradient has the correct shape and every element is nonzero
        assert len(grad_D) == 3
        assert np.count_nonzero(grad_D) == 3
        # the different methods agree
        assert np.allclose(grad_D, grad_F, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "prep_op", [qml.BasisState([1], wires=0), qml.QubitStateVector([0, 1], wires=0)]
    )
    def test_state_prep(self, prep_op, tol, dev):
        """Tests provides correct answer when provided state preparation operation."""
        x, y, z = [0.5, 0.3, -0.7]

        qs = QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.Rot(x, y, z, wires=[0]), qml.RY(-0.2, wires=[0])],
            [qml.expval(qml.PauliZ(0))],
            [prep_op],
        )

        qs.trainable_params = {1, 2, 3}

        grad_D = adjoint_jacobian(qs)
        qml.enable_return()
        grad_F = dev.adjoint_jacobian(qs)
        qml.disable_return()

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
        res = adjoint_jacobian(qs)

        expected = [
            np.cos(a) * np.sin(b) * np.sin(c),
            np.cos(b) * np.sin(a) * np.sin(c),
            np.cos(c) * np.sin(b) * np.sin(a),
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_trainable_hermitian_warns(self):
        """Test attempting to compute the gradient of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable."""

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        qs = QuantumScript([], [qml.expval(qml.Hermitian(mx, wires=[0, 2]))])

        qs.trainable_params = {0}
        with pytest.warns(
            UserWarning, match="Differentiating with respect to the input parameters of Hermitian"
        ):
            _ = adjoint_jacobian(qs)
