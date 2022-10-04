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
Unit tests for the qubit matrix-based operations.
"""
import numpy as np
import pytest
from gate_data import H, I, S, T, X, Z
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.operation import DecompositionUndefinedError
from pennylane.wires import Wires


class TestQubitUnitary:
    """Tests for the QubitUnitary class."""

    def test_qubit_unitary_noninteger_pow(self):
        """Test QubitUnitary raised to a non-integer power raises an error."""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )

        op = qml.QubitUnitary(U, wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.123)

    def test_qubit_unitary_noninteger_pow_broadcasted(self):
        """Test broadcasted QubitUnitary raised to a non-integer power raises an error."""
        U = np.array(
            [
                [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]],
                [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]],
            ]
        )

        op = qml.QubitUnitary(U, wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.123)

    @pytest.mark.parametrize("n", (1, 3, -1, -3))
    def test_qubit_unitary_pow(self, n):
        """Test qubit unitary raised to an integer power."""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )

        op = qml.QubitUnitary(U, wires="a")
        new_ops = op.pow(n)

        assert len(new_ops) == 1
        assert new_ops[0].wires == op.wires

        mat_to_pow = qml.math.linalg.matrix_power(qml.matrix(op), n)
        new_mat = qml.matrix(new_ops[0])

        assert qml.math.allclose(mat_to_pow, new_mat)

    @pytest.mark.parametrize("n", (1, 3, -1, -3))
    def test_qubit_unitary_pow_broadcasted(self, n):
        """Test broadcasted qubit unitary raised to an integer power."""
        U = np.array(
            [
                [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]],
                [[0.4125124 + 0.0j, 0.0 - 0.91095199j], [0.0 - 0.91095199j, 0.4125124 + 0.0j]],
            ]
        )

        op = qml.QubitUnitary(U, wires="a")
        new_ops = op.pow(n)

        assert len(new_ops) == 1
        assert new_ops[0].wires == op.wires

        mat_to_pow = qml.math.linalg.matrix_power(qml.matrix(op), n)
        new_mat = qml.matrix(new_ops[0])

        assert qml.math.allclose(mat_to_pow, new_mat)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_autograd(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with autograd."""

        out = qml.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_torch(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with torch."""
        import torch

        U = torch.tensor(U)
        out = qml.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, torch.Tensor)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U.detach().clone()
        U3[0, 0] += 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_tf(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with tensorflow."""
        import tensorflow as tf

        U = tf.Variable(U)
        out = qml.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, tf.Variable)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = tf.Variable(U + 0.5)
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_jax(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with jax."""
        from jax import numpy as jnp

        U = jnp.array(U)
        out = qml.QubitUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, jnp.ndarray)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U + 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QubitUnitary(U3, wires=range(num_wires), unitary_check=True).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QubitUnitary(U, wires=range(num_wires + 1)).matrix()

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "U,num_wires", [(H, 1), (np.kron(H, H), 2), (np.tensordot([1j, -1, 1], H, axes=0), 1)]
    )
    def test_qubit_unitary_jax_jit(self, U, num_wires):
        """Tests that QubitUnitary works with jitting."""
        import jax
        from jax import numpy as jnp

        U = jnp.array(U)
        f = lambda m: qml.QubitUnitary(m, wires=range(num_wires)).matrix()
        out = jax.jit(f)(U)
        assert qml.math.allclose(out, qml.QubitUnitary(U, wires=range(num_wires)).matrix())

    @pytest.mark.parametrize(
        "U,expected_gate,expected_params",
        [
            (I, qml.RZ, [0.0]),
            (Z, qml.RZ, [np.pi]),
            (S, qml.RZ, [np.pi / 2]),
            (T, qml.RZ, [np.pi / 4]),
            (qml.matrix(qml.RZ(0.3, wires=0)), qml.RZ, [0.3]),
            (qml.matrix(qml.RZ(-0.5, wires=0)), qml.RZ, [-0.5]),
            (
                np.array(
                    [
                        [0, -9.831019270939975e-01 + 0.1830590094588862j],
                        [9.831019270939975e-01 + 0.1830590094588862j, 0],
                    ]
                ),
                qml.Rot,
                [-0.18409714468526372, np.pi, 0.18409714468526372],
            ),
            (H, qml.Rot, [np.pi, np.pi / 2, 0.0]),
            (X, qml.Rot, [np.pi / 2, np.pi, -np.pi / 2]),
            (qml.matrix(qml.Rot(0.2, 0.5, -0.3, wires=0)), qml.Rot, [0.2, 0.5, -0.3]),
            (
                np.exp(1j * 0.02) * qml.matrix(qml.Rot(-1.0, 2.0, -3.0, wires=0)),
                qml.Rot,
                [-1.0, 2.0, -3.0],
            ),
        ],
    )
    def test_qubit_unitary_decomposition(self, U, expected_gate, expected_params):
        """Tests that single-qubit QubitUnitary decompositions are performed."""
        decomp = qml.QubitUnitary.compute_decomposition(U, wires=0)
        decomp2 = qml.QubitUnitary(U, wires=0).decomposition()

        assert len(decomp) == 1 == len(decomp2)
        assert isinstance(decomp[0], expected_gate)
        assert np.allclose(decomp[0].parameters, expected_params, atol=1e-7)
        assert isinstance(decomp2[0], expected_gate)
        assert np.allclose(decomp2[0].parameters, expected_params, atol=1e-7)

    def test_error_qubit_unitary_decomposition_broadcasted(self):
        """Tests that broadcasted QubitUnitary decompositions are not supported."""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )
        U = np.tensordot([1j, -1.0, (1 + 1j) / np.sqrt(2)], U, axes=0)
        with pytest.raises(DecompositionUndefinedError, match="QubitUnitary does not support"):
            qml.QubitUnitary.compute_decomposition(U, wires=0)
        with pytest.raises(DecompositionUndefinedError, match="QubitUnitary does not support"):
            qml.QubitUnitary(U, wires=0).decomposition()

    def test_qubit_unitary_decomposition_multiqubit_invalid(self):
        """Test that QubitUnitary is not decomposed for more than two qubits."""
        U = qml.Toffoli(wires=[0, 1, 2]).matrix()

        with pytest.raises(qml.operation.DecompositionUndefinedError):
            qml.QubitUnitary.compute_decomposition(U, wires=[0, 1, 2])

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )
        res_static = qml.QubitUnitary.compute_matrix(U)
        res_dynamic = qml.QubitUnitary(U, wires=0).matrix()
        expected = U
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(
            [[0.98877108 + 0.0j, 0.0 - 0.14943813j], [0.0 - 0.14943813j, 0.98877108 + 0.0j]]
        )
        U = np.tensordot([1j, -1.0, (1 + 1j) / np.sqrt(2)], U, axes=0)
        res_static = qml.QubitUnitary.compute_matrix(U)
        res_dynamic = qml.QubitUnitary(U, wires=0).matrix()
        expected = U
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    @pytest.mark.parametrize("inverse", (True, False))
    def test_controlled(self, inverse):
        """Test QubitUnitary's controlled method."""
        U = qml.PauliX.compute_matrix()
        base = qml.QubitUnitary(U, wires=0)
        base.inverse = inverse

        expected = qml.ControlledQubitUnitary(U, control_wires="a", wires=0)
        expected.inverse = inverse

        out = base._controlled("a")
        assert qml.equal(out, expected)


class TestDiagonalQubitUnitary:
    """Test the DiagonalQubitUnitary operation."""

    def test_decomposition(self):
        """Test that DiagonalQubitUnitary falls back to QubitUnitary."""
        D = np.array([1j, 1, 1, -1, -1j, 1j, 1, -1])

        decomp = qml.DiagonalQubitUnitary.compute_decomposition(D, [0, 1, 2])
        decomp2 = qml.DiagonalQubitUnitary(D, wires=[0, 1, 2]).decomposition()

        assert len(decomp) == 1 == len(decomp2)
        assert decomp[0].name == "QubitUnitary" == decomp2[0].name
        assert decomp[0].wires == Wires([0, 1, 2]) == decomp2[0].wires
        assert np.allclose(decomp[0].data[0], np.diag(D))
        assert np.allclose(decomp2[0].data[0], np.diag(D))

    def test_decomposition_broadcasted(self):
        """Test that the broadcasted DiagonalQubitUnitary falls back to QubitUnitary."""
        D = np.outer([1.0, -1.0], [1.0, -1.0, 1j, 1.0])

        decomp = qml.DiagonalQubitUnitary.compute_decomposition(D, [0, 1])
        decomp2 = qml.DiagonalQubitUnitary(D, wires=[0, 1]).decomposition()

        assert len(decomp) == 1 == len(decomp2)
        assert decomp[0].name == "QubitUnitary" == decomp2[0].name
        assert decomp[0].wires == Wires([0, 1]) == decomp2[0].wires

        expected = np.array([np.diag([1.0, -1.0, 1j, 1.0]), np.diag([-1.0, 1.0, -1j, -1.0])])
        assert np.allclose(decomp[0].data[0], expected)
        assert np.allclose(decomp2[0].data[0], expected)

    def test_controlled(self):
        """Test that the correct controlled operation is created when controlling a qml.DiagonalQubitUnitary."""
        D = np.array([1j, 1, 1, -1, -1j, 1j, 1, -1])
        op = qml.DiagonalQubitUnitary(D, wires=[1, 2, 3])
        with qml.tape.QuantumTape() as tape:
            op._controlled(control=0)
        mat = qml.matrix(tape)
        assert qml.math.allclose(
            mat, qml.math.diag(qml.math.append(qml.math.ones(8, dtype=complex), D))
        )

    def test_controlled_broadcasted(self):
        """Test that the correct controlled operation is created when
        controlling a qml.DiagonalQubitUnitary with a broadcasted diagonal."""
        D = np.array([[1j, 1, -1j, 1], [1, -1, 1j, -1]])
        op = qml.DiagonalQubitUnitary(D, wires=[1, 2])
        with qml.tape.QuantumTape() as tape:
            op._controlled(control=0)
        mat = qml.matrix(tape)
        expected = np.array(
            [np.diag([1, 1, 1, 1, 1j, 1, -1j, 1]), np.diag([1, 1, 1, 1, 1, -1, 1j, -1])]
        )
        assert qml.math.allclose(mat, expected)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        diag = np.array([1, -1])
        res_static = qml.DiagonalQubitUnitary.compute_matrix(diag)
        res_dynamic = qml.DiagonalQubitUnitary(diag, wires=0).matrix()
        expected = np.array([[1, 0], [0, -1]])
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly for a broadcasted diagonal."""
        diag = np.array([[1, -1], [1j, -1], [-1j, -1]])
        res_static = qml.DiagonalQubitUnitary.compute_matrix(diag)
        res_dynamic = qml.DiagonalQubitUnitary(diag, wires=0).matrix()
        expected = np.array([[[1, 0], [0, -1]], [[1j, 0], [0, -1]], [[-1j, 0], [0, -1]]])
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    @pytest.mark.parametrize("n", (2, -1, 0.12345))
    @pytest.mark.parametrize("diag", ([1.0, -1.0], np.array([1.0, -1.0])))
    def test_pow(self, n, diag):
        """Test pow method returns expected results."""
        op = qml.DiagonalQubitUnitary(diag, wires="b")
        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        for x_op, x_pow in zip(op.data[0], pow_ops[0].data[0]):
            assert (x_op + 0.0j) ** n == x_pow

    @pytest.mark.parametrize("n", (2, -1, 0.12345))
    @pytest.mark.parametrize(
        "diag", ([[1.0, -1.0]] * 5, np.array([[1.0, -1j], [1j, 1j], [-1j, 1]]))
    )
    def test_pow_broadcasted(self, n, diag):
        """Test pow method returns expected results for broadcasted diagonals."""
        op = qml.DiagonalQubitUnitary(diag, wires="b")
        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        qml.math.allclose(np.array(op.data[0], dtype=complex) ** n, pow_ops[0].data[0])

    @pytest.mark.parametrize("D", [[1, 2], [[0.2, 1.0, -1.0], [1.0, -1j, 1j]]])
    def test_error_matrix_not_unitary(self, D):
        """Tests that error is raised if diagonal by `compute_matrix` does not lead to a unitary"""
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qml.DiagonalQubitUnitary.compute_matrix(np.array(D))
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qml.DiagonalQubitUnitary(np.array(D), wires=1).matrix()

    @pytest.mark.parametrize("D", [[1, 2], [[0.2, 1.0, -1.0], [1.0, -1j, 1j]]])
    def test_error_eigvals_not_unitary(self, D):
        """Tests that error is raised if diagonal by `compute_matrix` does not lead to a unitary"""
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qml.DiagonalQubitUnitary.compute_eigvals(np.array(D))
        with pytest.raises(ValueError, match="Operator must be unitary"):
            qml.DiagonalQubitUnitary(np.array(D), wires=0).eigvals()

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test that the diagonal matrix unitary operation works
        within a QNode that uses the JAX JIT"""
        import jax

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=1, shots=None)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            diag = jnp.exp(1j * x * jnp.array([1, -1]) / 2)
            qml.Hadamard(wires=0)
            qml.DiagonalQubitUnitary(diag, wires=0)
            return qml.expval(qml.PauliX(0))

        x = 0.654
        grad = jax.grad(circuit)(x)
        expected = -jnp.sin(x)
        assert np.allclose(grad, expected)

    @pytest.mark.jax
    def test_jax_jit_broadcasted(self):
        """Test that the diagonal matrix unitary operation works
        within a QNode that uses the JAX JIT"""
        import jax

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=1, shots=None)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            diag = jnp.exp(1j * jnp.outer(x, jnp.array([1, -1])) / 2)
            qml.Hadamard(wires=0)
            qml.DiagonalQubitUnitary(diag, wires=0)
            return qml.expval(qml.PauliX(0))

        x = jnp.array([0.654, 0.321])
        jac = jax.jacobian(circuit)(x)
        expected = jnp.diag(-jnp.sin(x))
        assert np.allclose(jac, expected)

    @pytest.mark.tf
    @pytest.mark.slow  # test takes 12 seconds due to tf.function
    def test_tf_function(self):
        """Test that the diagonal matrix unitary operation works
        within a QNode that uses TensorFlow autograph"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1, shots=None)

        @tf.function
        @qml.qnode(dev, interface="tf")
        def circuit(x):
            x = tf.cast(x, tf.complex128)
            diag = tf.math.exp(1j * x * tf.constant([1.0 + 0j, -1.0 + 0j]) / 2)
            qml.Hadamard(wires=0)
            qml.DiagonalQubitUnitary(diag, wires=0)
            return qml.expval(qml.PauliX(0))

        x = tf.Variable(0.452)

        with tf.GradientTape() as tape:
            loss = circuit(x)

        grad = tape.gradient(loss, x)
        expected = -tf.math.sin(x)
        assert np.allclose(grad, expected)


X = np.array([[0, 1], [1, 0]])
X_broadcasted = np.array([X] * 3)


class TestControlledQubitUnitary:
    """Tests for the ControlledQubitUnitary operation"""

    def test_no_control(self):
        """Test if ControlledQubitUnitary raises an error if control wires are not specified"""
        with pytest.raises(ValueError, match="Must specify control wires"):
            qml.ControlledQubitUnitary(X, wires=2)

    def test_shared_control(self):
        """Test if ControlledQubitUnitary raises an error if control wires are shared with wires"""
        with pytest.raises(ValueError, match="The control wires must be different from the wires"):
            qml.ControlledQubitUnitary(X, control_wires=[0, 2], wires=2)

    def test_wrong_shape(self):
        """Test if ControlledQubitUnitary raises a ValueError if a unitary of shape inconsistent
        with wires is provided"""
        with pytest.raises(ValueError, match=r"Input unitary must be of shape \(2, 2\)"):
            qml.ControlledQubitUnitary(np.eye(4), control_wires=[0, 1], wires=2).matrix()

    @pytest.mark.parametrize("target_wire", range(3))
    def test_toffoli(self, target_wire):
        """Test if ControlledQubitUnitary acts like a Toffoli gate when the input unitary is a
        single-qubit X. This test allows the target wire to be any of the three wires."""
        control_wires = list(range(3))
        del control_wires[target_wire]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(8, random_state=1)
        U2 = unitary_group.rvs(8, random_state=2)

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(3))
            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wire)
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(3))
            qml.Toffoli(wires=control_wires + [target_wire])
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    @pytest.mark.parametrize("target_wire", range(3))
    def test_toffoli_broadcasted(self, target_wire):
        """Test if ControlledQubitUnitary acts like a Toffoli gate when the input unitary is a
        broadcasted single-qubit X. Allows the target wire to be any of the three wires."""
        control_wires = list(range(3))
        del control_wires[target_wire]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(8, random_state=1)
        U2 = unitary_group.rvs(8, random_state=2)

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(3))
            qml.ControlledQubitUnitary(
                X_broadcasted, control_wires=control_wires, wires=target_wire
            )
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(3))
            qml.Toffoli(wires=control_wires + [target_wire])
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.shape(state_1) == (3, 8)
        assert np.allclose(state_1, state_1[0])  # Check that all broadcasted results are equal
        assert np.allclose(state_1, state_2)

    def test_arbitrary_multiqubit(self):
        """Test if ControlledQubitUnitary applies correctly for a 2-qubit unitary with 2-qubit
        control, where the control and target wires are not ordered."""
        control_wires = [1, 3]
        target_wires = [2, 0]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(16, random_state=1)
        U2 = unitary_group.rvs(16, random_state=2)

        # the two-qubit unitary
        U = unitary_group.rvs(4, random_state=3)

        # the 4-qubit representation of the unitary if the control wires were [0, 1] and the target
        # wires were [2, 3]
        U_matrix = np.eye(16, dtype=np.complex128)
        U_matrix[12:16, 12:16] = U

        # We now need to swap wires so that the control wires are [1, 3] and the target wires are
        # [2, 0]
        swap = qml.SWAP.compute_matrix()

        # initial wire permutation: 0123
        # target wire permutation: 1302
        swap1 = np.kron(swap, np.eye(4))  # -> 1023
        swap2 = np.kron(np.eye(4), swap)  # -> 1032
        swap3 = np.kron(np.kron(np.eye(2), swap), np.eye(2))  # -> 1302
        swap4 = np.kron(np.eye(4), swap)  # -> 1320

        all_swap = swap4 @ swap3 @ swap2 @ swap1
        U_matrix = all_swap.T @ U_matrix @ all_swap

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(4))
            qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=target_wires)
            qml.QubitUnitary(U2, wires=range(4))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(4))
            qml.QubitUnitary(U_matrix, wires=range(4))
            qml.QubitUnitary(U2, wires=range(4))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_error_message",
        [
            ([0, 1], 2, "ab", "String of control values can contain only '0' or '1'."),
            ([0, 1], 2, "011", "Length of control bit string must equal number of control wires."),
            ([0, 1], 2, [0, 1], "Alternative control values must be passed as a binary string."),
        ],
    )
    def test_invalid_mixed_polarity_controls(
        self, control_wires, wires, control_values, expected_error_message
    ):
        """Test if ControlledQubitUnitary properly handles invalid mixed-polarity
        control values."""
        target_wires = Wires(wires)

        with pytest.raises(ValueError, match=expected_error_message):
            qml.ControlledQubitUnitary(
                X, control_wires=control_wires, wires=target_wires, control_values=control_values
            ).matrix()

    @pytest.mark.parametrize(
        "control_wires,wires,control_values",
        [
            ([0], 1, "0"),
            ([0, 1], 2, "00"),
            ([0, 1], 2, "10"),
            ([0, 1], 2, "11"),
            ([1, 0], 2, "01"),
            ([0, 1], [2, 3], "11"),
            ([0, 2], [3, 1], "10"),
            ([1, 2, 0], [3, 4], "100"),
            ([1, 0, 2], [4, 3], "110"),
        ],
    )
    def test_mixed_polarity_controls(self, control_wires, wires, control_values):
        """Test if ControlledQubitUnitary properly applies mixed-polarity
        control values."""
        target_wires = Wires(wires)

        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick a random unitary
        U = unitary_group.rvs(2 ** len(target_wires), random_state=1967)

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=(2 ** (len(control_wires) + 1) - 2))
        target_state_weights = np.random.normal(size=(2 ** (len(target_wires) + 1) - 2))

        @qml.qnode(dev)
        def circuit_mixed_polarity():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.ControlledQubitUnitary(
                U, control_wires=control_wires, wires=target_wires, control_values=control_values
            )
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mixed_polarity_state = circuit_mixed_polarity()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mixed_polarity_state, pauli_x_state)

    def test_same_as_Toffoli(self):
        """Test if ControlledQubitUnitary returns the correct matrix for a control-control-X
        (Toffoli) gate"""
        mat = qml.ControlledQubitUnitary(X, control_wires=[0, 1], wires=2).matrix()
        mat2 = qml.Toffoli(wires=[0, 1, 2]).matrix()
        assert np.allclose(mat, mat2)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array([[0.94877869, 0.31594146], [-0.31594146, 0.94877869]])
        res_static = qml.ControlledQubitUnitary.compute_matrix(U, control_wires=[1], u_wires=[0])
        res_dynamic = qml.ControlledQubitUnitary(U, control_wires=[1], wires=0).matrix()
        expected = np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.94877869 + 0.0j, 0.31594146 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, -0.31594146 + 0.0j, 0.94877869 + 0.0j],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(
            [
                [[0.94877869, 0.31594146], [-0.31594146, 0.94877869]],
                [[0.4125124, -0.91095199], [0.91095199, 0.4125124]],
                [[0.31594146, 0.94877869j], [0.94877869j, 0.31594146]],
            ]
        )

        res_static = qml.ControlledQubitUnitary.compute_matrix(U, control_wires=[1], u_wires=[0])
        res_dynamic = qml.ControlledQubitUnitary(U, control_wires=[1], wires=0).matrix()
        expected = np.array(
            [
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.94877869 + 0.0j, 0.31594146 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, -0.31594146 + 0.0j, 0.94877869 + 0.0j],
                ],
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.4125124 + 0.0j, -0.91095199 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.91095199 + 0.0j, 0.4125124 + 0.0j],
                ],
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.31594146 + 0.0j, 0.0 + 0.94877869j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.94877869j, 0.31594146 + 0.0j],
                ],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_no_decomp(self):
        """Test that ControlledQubitUnitary raises a decomposition undefined
        error."""
        mat = qml.PauliX(0).matrix()
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            qml.ControlledQubitUnitary(mat, wires=0, control_wires=1).decomposition()
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            qml.ControlledQubitUnitary(X_broadcasted, wires=0, control_wires=1).decomposition()

    @pytest.mark.parametrize("n", (2, -1, -2))
    def test_pow(self, n):
        """Tests the metadata and unitary for a ControlledQubitUnitary raised to a power."""
        U1 = np.array(
            [
                [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
            ]
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        assert pow_ops[0].hyperparameters["u_wires"] == op.hyperparameters["u_wires"]
        assert pow_ops[0].control_wires == op.control_wires

        op_mat_to_pow = qml.math.linalg.matrix_power(op.data[0], n)
        assert qml.math.allclose(pow_ops[0].data[0], op_mat_to_pow)

    @pytest.mark.parametrize("n", (2, -1, -2))
    def test_pow_broadcasted(self, n):
        """Tests the metadata and unitary for a broadcasted
        ControlledQubitUnitary raised to a power."""
        U1 = np.tensordot(
            np.array([1j, -1.0, 1j]),
            np.array(
                [
                    [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                    [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
                ]
            ),
            axes=0,
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        assert pow_ops[0].hyperparameters["u_wires"] == op.hyperparameters["u_wires"]
        assert pow_ops[0].control_wires == op.control_wires

        op_mat_to_pow = qml.math.linalg.matrix_power(op.data[0], n)
        assert qml.math.allclose(pow_ops[0].data[0], op_mat_to_pow)

    def test_noninteger_pow(self):
        """Test that a ControlledQubitUnitary raised to a non-integer power raises an error."""
        U1 = np.array(
            [
                [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
            ]
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    def test_noninteger_pow_broadcasted(self):
        """Test that a ControlledQubitUnitary raised to a non-integer power raises an error."""
        U1 = np.array(
            [
                [0.73708696 + 0.61324932j, 0.27034258 + 0.08685028j],
                [-0.24979544 - 0.1350197j, 0.95278437 + 0.1075819j],
            ]
            * 3
        )

        op = qml.ControlledQubitUnitary(U1, control_wires=("b", "c"), wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    @pytest.mark.parametrize("inverse", (True, False))
    def test_controlled(self, inverse):
        """Test the _controlled method for ControlledQubitUnitary."""

        U = qml.PauliX(0).compute_matrix()

        original = qml.ControlledQubitUnitary(U, control_wires=(0, 1), wires=4, control_values="01")
        original.inverse = inverse
        expected = qml.ControlledQubitUnitary(
            U, control_wires=(0, 1, "a"), wires=4, control_values="011"
        )
        expected.inverse = inverse

        out = original._controlled("a")
        assert qml.equal(out, expected)


label_data = [
    (X, qml.QubitUnitary(X, wires=0)),
    (X, qml.ControlledQubitUnitary(X, control_wires=0, wires=1)),
    ([1, 1], qml.DiagonalQubitUnitary([1, 1], wires=0)),
]


@pytest.mark.parametrize("mat, op", label_data)
class TestUnitaryLabels:
    def test_no_cache(self, mat, op):
        """Test labels work without a provided cache."""
        assert op.label() == "U"

    def test_matrices_not_in_cache(self, mat, op):
        """Test provided cache doesn't have a 'matrices' keyword."""
        assert op.label(cache={}) == "U"

    def test_cache_matrices_not_list(self, mat, op):
        """Test 'matrices' key pair is not a list."""
        assert op.label(cache={"matrices": 0}) == "U"

    def test_empty_cache_list(self, mat, op):
        """Test matrices list is provided, but empty. Operation should have `0` label and matrix
        should be added to cache."""
        cache = {"matrices": []}
        assert op.label(cache=cache) == "U(M0)"
        assert qml.math.allclose(cache["matrices"][0], mat)

    def test_something_in_cache_list(self, mat, op):
        """If something exists in the matrix list, but parameter is not in the list, then parameter
        added to list and label given number of its position."""
        cache = {"matrices": [Z]}
        assert op.label(cache=cache) == "U(M1)"

        assert len(cache["matrices"]) == 2
        assert qml.math.allclose(cache["matrices"][1], mat)

    def test_matrix_already_in_cache_list(self, mat, op):
        """If the parameter already exists in the matrix cache, then the label uses that index and the
        matrix cache is unchanged."""
        cache = {"matrices": [Z, mat, S]}
        assert op.label(cache=cache) == "U(M1)"

        assert len(cache["matrices"]) == 3


class TestInterfaceMatricesLabel:
    """Test different interface matrices with qubit."""

    def check_interface(self, mat):
        """Interface independent helper method."""

        op = qml.QubitUnitary(mat, wires=0)

        cache = {"matrices": []}
        assert op.label(cache=cache) == "U(M0)"
        assert qml.math.allclose(cache["matrices"][0], mat)

        cache = {"matrices": [0, mat, 0]}
        assert op.label(cache=cache) == "U(M1)"
        assert len(cache["matrices"]) == 3

    @pytest.mark.torch
    def test_labelling_torch_tensor(self):
        """Test matrix cache labelling with torch interface."""

        import torch

        mat = torch.tensor([[1, 0], [0, -1]])
        self.check_interface(mat)

    @pytest.mark.tf
    def test_labelling_tf_variable(self):
        """Test matrix cache labelling with tf interface."""

        import tensorflow as tf

        mat = tf.Variable([[1, 0], [0, -1]])

        self.check_interface(mat)

    @pytest.mark.jax
    def test_labelling_jax_variable(self):
        """Test matrix cache labelling with jax interface."""

        import jax.numpy as jnp

        mat = jnp.array([[1, 0], [0, -1]])

        self.check_interface(mat)


control_data = [
    (qml.QubitUnitary(X, wires=0), Wires([])),
    (qml.DiagonalQubitUnitary([1, 1], wires=1), Wires([])),
    (qml.ControlledQubitUnitary(X, control_wires=0, wires=1), Wires([0])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for matrix operations."""
    assert op.control_wires == control_wires
