# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the qutrit matrix-based operations.
"""
import pytest
import numpy as np
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import DecompositionUndefinedError

from gate_data import TSWAP

U_thadamard_01 = np.multiply(
    1 / np.sqrt(2),
    np.array(
        [[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]],
    ),
)

# TODO: Add tests for adding controls to `QutritUnitary` once `ControlledQutritUnitary` is implemented


class TestQutritUnitary:
    """Tests for the QutritUnitary class."""

    def test_qutrit_unitary_noninteger_pow(self):
        """Test QutritUnitary raised to a non-integer power raises an error."""
        op = qml.QutritUnitary(U_thadamard_01, wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.123)

    def test_qutrit_unitary_noninteger_pow_broadcasted(self):
        """Test broadcasted QutritUnitary raised to a non-integer power raises an error."""
        U = np.array(
            [
                U_thadamard_01,
                U_thadamard_01,
            ]
        )

        op = qml.QutritUnitary(U, wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.123)

    @pytest.mark.parametrize("n", (1, 3, -1, -3))
    def test_qutrit_unitary_pow(self, n):
        """Test qutrit unitary raised to an integer power."""
        op = qml.QutritUnitary(U_thadamard_01, wires="a")
        new_ops = op.pow(n)

        assert len(new_ops) == 1
        assert new_ops[0].wires == op.wires

        mat_to_pow = qml.math.linalg.matrix_power(qml.matrix(op), n)
        new_mat = qml.matrix(new_ops[0])

        assert qml.math.allclose(mat_to_pow, new_mat)

    @pytest.mark.parametrize("n", (1, 3, -1, -3))
    def test_qutrit_unitary_pow_broadcasted(self, n):
        """Test broadcasted qutrit unitary raised to an integer power."""
        U = np.array(
            [
                U_thadamard_01,
                U_thadamard_01,
            ]
        )

        op = qml.QutritUnitary(U, wires="a")
        new_ops = op.pow(n)

        assert len(new_ops) == 1
        assert new_ops[0].wires == op.wires

        mat_to_pow = qml.math.linalg.matrix_power(qml.matrix(op), n)
        new_mat = qml.matrix(new_ops[0])

        assert qml.math.allclose(mat_to_pow, new_mat)

    interface_and_decomp_data = [
        (U_thadamard_01, 1),
        (TSWAP, 2),
        (unitary_group.rvs(3, random_state=10), 1),
        (unitary_group.rvs(9, random_state=10), 2),
        (np.eye(3), 1),
        (np.eye(9), 2),
        (np.tensordot([1j, -1, 1], U_thadamard_01, axes=0), 1),
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("U,num_wires", interface_and_decomp_data)
    def test_qutrit_unitary_autograd(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with autograd."""

        out = qml.QutritUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QutritUnitary(U3, wires=range(num_wires)).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U, wires=range(num_wires + 1)).matrix()

        # verify adjoint behaves correctly
        op = qml.QutritUnitary(U, wires=range(num_wires)).adjoint()
        mat = op.matrix()
        expected = (
            np.conj(np.transpose(U)) if len(np.shape(U)) == 2 else np.conj(np.swapaxes(U, -2, -1))
        )
        assert qml.math.allclose(mat, expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("U,num_wires", interface_and_decomp_data)
    def test_qutrit_unitary_torch(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with torch."""
        import torch

        U_adjoint = (
            np.conj(np.transpose(U)) if len(np.shape(U)) == 2 else np.conj(np.swapaxes(U, -2, -1))
        )
        U = torch.tensor(U)
        out = qml.QutritUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, torch.Tensor)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U.detach().clone()
        U3[0, 0] += 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QutritUnitary(U3, wires=range(num_wires)).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U, wires=range(num_wires + 1)).matrix()

        # verify adjoint behaves correctly
        op = qml.QutritUnitary(U, wires=range(num_wires)).adjoint()
        mat = op.matrix()
        expected = torch.tensor(U_adjoint)
        assert qml.math.allclose(mat, expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("U,num_wires", interface_and_decomp_data)
    def test_qutrit_unitary_tf(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with tensorflow."""
        import tensorflow as tf

        U_adjoint = (
            np.conj(np.transpose(U)) if len(np.shape(U)) == 2 else np.conj(np.swapaxes(U, -2, -1))
        )
        U = tf.Variable(U)
        out = qml.QutritUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, tf.Variable)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = tf.Variable(U + 0.5)
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QutritUnitary(U3, wires=range(num_wires)).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U, wires=range(num_wires + 1)).matrix()

        # verify adjoint behaves correctly
        op = qml.QutritUnitary(U, wires=range(num_wires)).adjoint()
        mat = op.matrix()
        expected = tf.Variable(U_adjoint)
        assert qml.math.allclose(mat, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("U,num_wires", interface_and_decomp_data)
    def test_qutrit_unitary_jax(self, U, num_wires):
        """Test that the unitary operator produces the correct output and
        catches incorrect input with jax."""
        from jax import numpy as jnp

        U = jnp.array(U)
        out = qml.QutritUnitary(U, wires=range(num_wires)).matrix()

        # verify output type
        assert isinstance(out, jnp.ndarray)

        # verify equivalent to input state
        assert qml.math.allclose(out, U)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U[:, 1:], wires=range(num_wires)).matrix()

        # test non-unitary matrix
        U3 = U + 0.5
        with pytest.warns(UserWarning, match="may not be unitary"):
            qml.QutritUnitary(U3, wires=range(num_wires)).matrix()

        # test an error is thrown when constructed with incorrect number of wires
        with pytest.raises(ValueError, match="must be of shape"):
            qml.QutritUnitary(U, wires=range(num_wires + 1)).matrix()

        # verify adjoint behaves correctly
        op = qml.QutritUnitary(U, wires=range(num_wires)).adjoint()
        mat = op.matrix()
        expected = (
            jnp.conj(jnp.transpose(U))
            if len(jnp.shape(U)) == 2
            else jnp.conj(jnp.swapaxes(U, -2, -1))
        )
        assert qml.math.allclose(mat, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("U,num_wires", interface_and_decomp_data)
    def test_qutrit_unitary_jax_jit(self, U, num_wires):
        """Tests that QutritUnitary works with jitting."""
        import jax
        from jax import numpy as jnp

        U = jnp.array(U)
        f = lambda m: qml.QutritUnitary(m, wires=range(num_wires)).matrix()
        out = jax.jit(f)(U)
        assert qml.math.allclose(out, qml.QutritUnitary(U, wires=range(num_wires)).matrix())

    @pytest.mark.parametrize("U, num_wires", interface_and_decomp_data)
    def test_qutrit_unitary_decomposition_error(self, U, num_wires):
        """Tests that QutritUnitary is not decomposed and throws error"""
        with pytest.raises(DecompositionUndefinedError):
            qml.QutritUnitary.compute_decomposition(U, wires=range(num_wires))

    def test_matrix_representation(self):
        """Test that the matrix representation is defined correctly"""
        U = np.array([[1, -1j, -1 + 1j], [1j, 1, 1 + 1j], [1 + 1j, -1 + 1j, 0]]) * 0.5

        res_static = qml.QutritUnitary.compute_matrix(U)
        res_dynamic = qml.QutritUnitary(U, wires=0).matrix()
        expected = U
        assert np.allclose(res_static, expected)
        assert np.allclose(res_dynamic, expected)

    def test_matrix_representation_broadcasted(self):
        """Test that the matrix representation is defined correctly for
        broadcasted matrices"""
        U = np.array([[1, -1j, -1 + 1j], [1j, 1, 1 + 1j], [1 + 1j, -1 + 1j, 0]]) * 0.5
        U = np.tensordot([1j, -1.0, (1 + 1j) / np.sqrt(2)], U, axes=0)

        res_static = qml.QutritUnitary.compute_matrix(U)
        res_dynamic = qml.QutritUnitary(U, wires=0).matrix()
        expected = U
        assert np.allclose(res_static, expected)
        assert np.allclose(res_dynamic, expected)

    def test_controlled(self):
        dev = qml.device("default.qutrit", wires=2)

        op = qml.QutritUnitary(U_thadamard_01, wires=1)

        # Controlling 1 qutrit operation with 1 control wire at value "2"
        with qml.tape.QuantumTape() as tape:
            qml.TShift(wires=0)
            qml.TShift(wires=0)
            op._controlled(wire=0)
            qml.state()

        res = dev.execute(tape)

        expected = np.array([0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        assert np.allclose(res, expected)

        # Controlling 1 qutrit operation with 1 control wire at value "2" when
        # control wire is not in "2" state
        dev.reset()
        with qml.tape.QuantumTape() as tape:
            op._controlled(wire=0)
            qml.state()

        res = dev.execute(tape)

        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.allclose(res, expected)

        # Controlling 1 qutrit operation with 1 control wire at value "1"
        dev.reset()
        with qml.tape.QuantumTape() as tape:
            qml.TShift(wires=0)
            op._controlled(wire=0)
            qml.state()

        res = dev.execute(tape)

        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert np.allclose(res, expected)

        # # Controlling 1 qutrit operation with 1 control wire at value "1"
        # # using `qml.ctrl`
        # dev.reset()
        # with qml.tape.QuantumTape() as tape:
        #     qml.TShift(wires=0)
        #     qml.ctrl(op, 0, control_values="1")
        #     qml.state()

        res = dev.execute(tape)

        expected = np.array([0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        assert np.allclose(res, expected)

        # # Controlling when the control wire is in a superposition
        # dev.reset()
        # with qml.tape.QuantumTape() as tape:
        #     qml.QutritUnitary(U_thadamard_01, wires=0)
        #     qml.ctrl(op, 0, control_values="1")
        #     qml.state()

        res = dev.execute(tape)

        expected = np.array([1 / np.sqrt(2), 0, 0, 0.5, 0.5, 0, 0, 0, 0])
        assert np.allclose(res, expected)


class TestControlledQutritUnitary:
    """Tests for the ControlledQutritUnitary operation"""

    def test_no_control(self):
        """Test if ControlledQutritUnitary raises an error if control wires are not specified"""
        with pytest.raises(ValueError, match="Must specify control wires"):
            qml.ControlledQutritUnitary(U_thadamard_01, wires=2)

    def test_shared_control(self):
        """Test if ControlledQutritUnitary raises an error if control wires are shared with wires"""
        with pytest.raises(ValueError, match="The control wires must be different from the wires"):
            qml.ControlledQutritUnitary(U_thadamard_01, control_wires=[0, 2], wires=2)

    def test_wrong_shape(self):
        """Test if ControlledQutritUnitary raises a ValueError if a unitary of shape inconsistent
        with wires is provided"""
        with pytest.raises(ValueError, match=r"Input unitary must be of shape \(3, 3\)"):
            qml.ControlledQutritUnitary(np.eye(9), control_wires=[0, 1], wires=2).matrix()

    def test_arbitrary_multiqutrit(self):
        """Test if ControlledQutritUnitary applies correctly for a 2-qutrit unitary with 2-qutrit
        control, where the control and target wires are not ordered."""
        control_wires = [1, 3]
        target_wires = [2, 0]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(81, random_state=1)
        U2 = unitary_group.rvs(81, random_state=2)

        # the two-qutrit unitary
        U = unitary_group.rvs(9, random_state=3)

        # the 4-qutrit representation of the unitary if the control wires were [0, 1] and the target
        # wires were [2, 3]
        U_matrix = np.eye(81, dtype=np.complex128)
        U_matrix[72:81, 72:81] = U

        # We now need to swap wires so that the control wires are [1, 3] and the target wires are
        # [2, 0]
        swap = TSWAP

        # initial wire permutation: 0123
        # target wire permutation: 1320
        swap1 = np.kron(swap, np.eye(9))  # -> 1023
        swap2 = np.kron(np.eye(9), swap)  # -> 1032
        swap3 = np.kron(np.kron(np.eye(3), swap), np.eye(3))  # -> 1302
        swap4 = np.kron(np.eye(9), swap)  # -> 1320

        all_swap = swap4 @ swap3 @ swap2 @ swap1
        U_matrix = all_swap.T @ U_matrix @ all_swap

        dev = qml.device("default.qutrit", wires=4)

        @qml.qnode(dev)
        def f1():
            qml.QutritUnitary(U1, wires=range(4))
            qml.ControlledQutritUnitary(U, control_wires=control_wires, wires=target_wires)
            qml.QutritUnitary(U2, wires=range(4))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QutritUnitary(U1, wires=range(4))
            qml.QutritUnitary(U_matrix, wires=range(4))
            qml.QutritUnitary(U2, wires=range(4))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_error_message",
        [
            ([0, 1], 2, "ab", "String of control values can contain only '0' or '1' or '2'."),
            ([0, 1], 2, "012", "Length of control trit string must equal number of control wires."),
            ([0, 1], 2, [0, 1], "Alternative control values must be passed as a ternary string."),
        ],
    )
    def test_invalid_mixed_polarity_controls(
        self, control_wires, wires, control_values, expected_error_message
    ):
        """Test if ControlledQutritUnitary properly handles invalid mixed-polarity
        control values."""
        target_wires = Wires(wires)

        with pytest.raises(ValueError, match=expected_error_message):
            qml.ControlledQutritUnitary(
                U_thadamard_01,
                control_wires=control_wires,
                wires=target_wires,
                control_values=control_values,
            ).matrix()

    mixed_polarity_data = [
        ([0], 1, "0"),
        ([0], 1, "1"),
        ([0], 1, "2"),
        ([0, 1], 2, "00"),
        ([0, 1], 2, "01"),
        ([0, 1], 2, "02"),
        ([1, 0], 2, "10"),
        ([0, 1], 2, "11"),
        ([0, 1], 2, "12"),
        ([0, 1], 2, "20"),
        ([1, 0], 2, "21"),
        ([1, 0], 2, "22"),
        ([0, 1], [2, 3], "01"),
        ([0, 2], [3, 1], "12"),
        ([1, 2, 0], [3, 4], "012"),
        ([1, 0, 2], [4, 3], "210"),
    ]

    @pytest.mark.parametrize("control_wires,wires,control_values", mixed_polarity_data)
    def test_mixed_polarity_controls(self, control_wires, wires, control_values, tol):
        """Test if ControlledQutritUnitary properly applies mixed-polarity
        control values."""
        target_wires = Wires(wires)

        dev = qml.device("default.qutrit", wires=len(control_wires + target_wires))

        # Pick a random unitary
        U = unitary_group.rvs(3 ** len(target_wires), random_state=1967)

        # Pick random starting state for the control and target qutrits
        starting_state = np.random.normal(size=3 ** (len(control_wires + target_wires)))
        starting_state = starting_state / np.linalg.norm(starting_state)
        dev._state = starting_state.reshape([3] * len(control_wires + target_wires))
        dev.apply(
            [
                qml.ControlledQutritUnitary(
                    U,
                    control_wires=control_wires,
                    wires=target_wires,
                    control_values=control_values,
                ),
            ]
        )
        final_state = dev._state.flatten()

        dev.reset()
        dev._state = starting_state.reshape([3] * len(control_wires + target_wires))

        # Find which wires are controlled on "1" and "0", and shift the state of the device
        # accordingly to apply the controls, and then undo them after the controlled operation
        one_locations = [x for x in range(len(control_values)) if control_values[x] == "1"]
        zero_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        ops = []
        for wire in one_locations:
            ops.append(qml.TShift(wires=control_wires[wire]))

        for wire in zero_locations:
            ops.append(qml.TShift(wires=control_wires[wire]))
            ops.append(qml.TShift(wires=control_wires[wire]))

        ops.append(
            qml.ControlledQutritUnitary(
                U,
                control_wires=control_wires,
                wires=target_wires,
            )
        )

        for wire in zero_locations:
            ops.append(qml.TShift(wires=control_wires[wire]))

        for wire in one_locations:
            ops.append(qml.TShift(wires=control_wires[wire]))
            ops.append(qml.TShift(wires=control_wires[wire]))

        dev.apply(ops)

        assert np.allclose(dev._state.flatten(), final_state, atol=tol)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        U = np.array(unitary_group.rvs(3, random_state=10))
        res_static = qml.ControlledQutritUnitary.compute_matrix(U, control_wires=[1], u_wires=[0])
        res_dynamic = qml.ControlledQutritUnitary(U, control_wires=[1], wires=0).matrix()
        expected = np.eye(9, dtype=np.complex128)
        expected[6:9, 6:9] = U

        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_matrix_representation_broadcasted(self, tol):
        """Test that the matrix representation is defined correctly when broadcasting"""
        U1 = unitary_group.rvs(3, random_state=10)
        U2 = unitary_group.rvs(3, random_state=10)
        U3 = unitary_group.rvs(3, random_state=10)
        U = np.array([U1, U2, U3])

        res_static = qml.ControlledQutritUnitary.compute_matrix(U, control_wires=[1], u_wires=[0])
        res_dynamic = qml.ControlledQutritUnitary(U, control_wires=[1], wires=0).matrix()
        expected = np.array([np.eye(9), np.eye(9), np.eye(9)], dtype=np.complex128)
        expected[0, 6:, 6:] = U1
        expected[1, 6:, 6:] = U2
        expected[2, 6:, 6:] = U3
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_no_decomp(self):
        """Test that ControlledQutritUnitary raises a DecompositionUndefinedError."""
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            qml.ControlledQutritUnitary(U_thadamard_01, wires=0, control_wires=1).decomposition()

    @pytest.mark.parametrize("n", (2, -1, -2))
    def test_pow(self, n):
        """Tests the metadata and unitary for a ControlledQutritUnitary raised to a power."""
        U1 = unitary_group.rvs(3, random_state=10)

        op = qml.ControlledQutritUnitary(U1, control_wires=("b", "c"), wires="a")

        pow_ops = op.pow(n)
        assert len(pow_ops) == 1

        assert pow_ops[0].hyperparameters["u_wires"] == op.hyperparameters["u_wires"]
        assert pow_ops[0].control_wires == op.control_wires

        op_mat_to_pow = qml.math.linalg.matrix_power(op.data[0], n)
        assert qml.math.allclose(pow_ops[0].data[0], op_mat_to_pow)

    @pytest.mark.parametrize("n", (2, -1, -2))
    def test_pow_broadcasted(self, n):
        """Tests the metadata and unitary for a broadcasted
        ControlledQutritUnitary raised to a power."""
        U1 = np.tensordot(
            np.array([1j, -1.0, 1j]),
            U_thadamard_01,
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
        """Test that a ControlledQutritUnitary raised to a non-integer power raises an error."""
        U1 = unitary_group.rvs(3, random_state=10)

        op = qml.ControlledQutritUnitary(U1, control_wires=("b", "c"), wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    def test_noninteger_pow_broadcasted(self):
        """Test that a ControlledQutritUnitary raised to a non-integer power raises an error when broadcasting."""
        U1 = unitary_group.rvs(3, random_state=10)
        U = np.array([U1, U1, U1])

        op = qml.ControlledQutritUnitary(U, control_wires=("b", "c"), wires="a")

        with pytest.raises(qml.operation.PowUndefinedError):
            op.pow(0.12)

    def test_controlled(self):
        """Test that adding a control to a `ControlledQutritUnitary` operation using `_controlled` behaves
        as expected"""
        dev = qml.device("default.qutrit", wires=3)

        op = qml.ControlledQutritUnitary(U_thadamard_01, control_wires=1, wires=2)
        with qml.tape.QuantumTape() as tape:
            qml.TShift(wires=1)
            qml.TShift(wires=1)
            qml.TShift(wires=0)
            qml.TShift(wires=0)
            op._controlled(wire=0)
            qml.state()

        res = dev.execute(tape)

        expected = np.zeros(27)
        expected[24] = 1 / np.sqrt(2)
        expected[25] = 1 / np.sqrt(2)
        assert np.allclose(res, expected)

        dev.reset()
        with qml.tape.QuantumTape() as tape:
            op._controlled(wire=0)
            qml.state()

        res = dev.execute(tape)

        expected = np.zeros(27)
        expected[0] = 1
        assert np.allclose(res, expected)

        dev.reset()
        with qml.tape.QuantumTape() as tape:
            qml.TShift(wires=0)
            op._controlled(wire=0)
            qml.state()

        res = dev.execute(tape)

        expected = np.zeros(27)
        expected[9] = 1
        assert np.allclose(res, expected)


label_data = [
    (U_thadamard_01, qml.QutritUnitary(U_thadamard_01, wires=0)),
    (U_thadamard_01, qml.ControlledQutritUnitary(U_thadamard_01, control_wires=0, wires=1)),
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
        cache = {"matrices": [TSWAP]}
        assert op.label(cache=cache) == "U(M1)"

        assert len(cache["matrices"]) == 2
        assert qml.math.allclose(cache["matrices"][1], mat)

    def test_matrix_already_in_cache_list(self, mat, op):
        """If the parameter already exists in the matrix cache, then the label uses that index and the
        matrix cache is unchanged."""
        cache = {"matrices": [TSWAP, mat]}
        assert op.label(cache=cache) == "U(M1)"

        assert len(cache["matrices"]) == 2


class TestInterfaceMatricesLabel:
    """Test different interface matrices with qutrit."""

    def check_interface(self, mat):
        """Interface independent helper method."""

        op = qml.QutritUnitary(mat, wires=0)

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

        mat = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.check_interface(mat)

    @pytest.mark.tf
    def test_labelling_tf_variable(self):
        """Test matrix cache labelling with tf interface."""

        import tensorflow as tf

        mat = tf.Variable([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        self.check_interface(mat)

    @pytest.mark.jax
    def test_labelling_jax_variable(self):
        """Test matrix cache labelling with jax interface."""

        import jax.numpy as jnp

        mat = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        self.check_interface(mat)


control_data = [
    (qml.QutritUnitary(U_thadamard_01, wires=0), Wires([])),
    (qml.ControlledQutritUnitary(U_thadamard_01, control_wires=0, wires=1), Wires([0])),
]


@pytest.mark.parametrize("op, control_wires", control_data)
def test_control_wires(op, control_wires):
    """Test ``control_wires`` attribute for matrix operations."""
    assert op.control_wires == control_wires
