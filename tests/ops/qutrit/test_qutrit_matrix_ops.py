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


label_data = [
    (U_thadamard_01, qml.QutritUnitary(U_thadamard_01, wires=0)),
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
