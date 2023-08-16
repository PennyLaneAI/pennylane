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
"""
Unit tests for the dot function
"""
import pytest

import pennylane as qml
from pennylane.ops import Hamiltonian, Prod, SProd, Sum
from pennylane.pauli.pauli_arithmetic import PauliSentence


class TestDotSum:
    """Unittests for the dot function when ``pauli=False``."""

    def test_dot_returns_sum(self):
        """Test that the dot function returns a Sum operator when ``pauli=False``."""
        c = [1.0, 2.0, 3.0]
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        S = qml.dot(coeffs=c, ops=o)
        assert isinstance(S, Sum)
        for summand, coeff in zip(S.operands, c):
            if coeff == 1:
                assert isinstance(summand, qml.PauliX)
            else:
                assert isinstance(summand, SProd)
                assert summand.scalar == coeff

    def test_dot_returns_sprod(self):
        """Test that the dot function returns a SProd operator when only one operator is input."""
        O = qml.dot(coeffs=[2.0], ops=[qml.PauliX(0)])
        assert isinstance(O, SProd)
        assert O.scalar == 2

    def test_cast_tensor_to_prod(self):
        """Test that `dot` casts all `Tensor` objects to `Prod`."""
        result = qml.dot(
            coeffs=[1, 1, 1],
            ops=[
                qml.PauliX(0) @ qml.PauliY(0),
                qml.PauliX(0) @ qml.PauliY(0),
                qml.PauliX(0) @ qml.PauliY(0),
            ],
        )
        assert isinstance(result, Sum)
        for op in result:
            assert isinstance(op, Prod)

    def test_dot_groups_coeffs(self):
        """Test that the `dot` function groups the coefficients."""
        result = qml.dot(coeffs=[4, 4, 4], ops=[qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)])
        assert isinstance(result, SProd)
        assert result.scalar == 4
        assert isinstance(result.base, Sum)
        assert len(result.base) == 3
        for op in result.base:
            assert isinstance(op, qml.PauliX)

    def test_dot_groups_coeffs_with_different_sign(self):
        """test that the `dot` function groups coefficients with different signs."""
        cs = [4, -4, 4]
        result = qml.dot(coeffs=cs, ops=[qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)])
        assert isinstance(result, SProd)
        assert result.scalar == 4
        for op, c in zip(result.base, cs):
            if c == -4:
                assert isinstance(op, SProd)
                assert op.scalar == -1
            else:
                assert isinstance(op, qml.PauliX)

    def test_dot_different_number_of_coeffs_and_ops(self):
        """Test that a ValueError is raised when the number of coefficients and operators does
        not match."""
        with pytest.raises(
            ValueError,
            match="Number of coefficients and operators does not match",
        ):
            qml.dot([1.0], [qml.PauliX(0), qml.PauliY(1)])

    def test_dot_empty_coeffs_or_ops(self):
        """Test that a ValueError is raised when the number of coefficients and operators does
        not match."""
        with pytest.raises(
            ValueError,
            match="Cannot compute the dot product of an empty sequence",
        ):
            qml.dot([], [])

    @pytest.mark.autograd
    @pytest.mark.parametrize("dtype", (float, complex))
    def test_dot_autograd(self, dtype):
        """Test the dot function with the autograd interface."""
        c = qml.numpy.array([1.0, 2.0, 3.0], dtype=dtype)
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        op_sum = qml.dot(c, o)
        op_sum_2 = Sum(
            qml.PauliX(0),
            SProd(qml.numpy.array(2.0, dtype=dtype), qml.PauliY(1)),
            SProd(qml.numpy.array(3.0, dtype=dtype), qml.PauliZ(2)),
        )
        assert qml.equal(op_sum, op_sum_2)

    @pytest.mark.tf
    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    def test_dot_tf(self, dtype):
        """Test the dot function with the tensorflow interface."""
        import tensorflow as tf

        c = tf.constant([1.0, 2.0, 3.0], dtype=getattr(tf, dtype))
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        op_sum = qml.dot(c, o)
        op_sum_2 = Sum(
            qml.PauliX(0),
            SProd(tf.constant(2.0, dtype=getattr(tf, dtype)), qml.PauliY(1)),
            SProd(tf.constant(3.0, dtype=getattr(tf, dtype)), qml.PauliZ(2)),
        )
        assert qml.equal(op_sum, op_sum_2)

    @pytest.mark.torch
    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    def test_dot_torch(self, dtype):
        """Test the dot function with the torch interface."""
        import torch

        c = torch.tensor([1.0, 2.0, 3.0], dtype=getattr(torch, dtype))
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        op_sum = qml.dot(c, o)
        op_sum_2 = Sum(
            qml.PauliX(0),
            SProd(torch.tensor(2.0, dtype=getattr(torch, dtype)), qml.PauliY(1)),
            SProd(torch.tensor(3.0, dtype=getattr(torch, dtype)), qml.PauliZ(2)),
        )
        assert qml.equal(op_sum, op_sum_2)

    @pytest.mark.jax
    @pytest.mark.parametrize("dtype", (float, complex))
    def test_dot_jax(self, dtype):
        """Test the dot function with the torch interface."""
        import jax

        c = jax.numpy.array([1.0, 2.0, 3.0], dtype=dtype)
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        op_sum = qml.dot(c, o)
        op_sum_2 = Sum(
            qml.PauliX(0),
            SProd(jax.numpy.array(2.0, dtype=dtype), qml.PauliY(1)),
            SProd(jax.numpy.array(3.0, dtype=dtype), qml.PauliZ(2)),
        )
        assert qml.equal(op_sum, op_sum_2)


coeffs = [0.12345, 1.2345, 12.345, 123.45, 1234.5, 12345]
ops = [
    qml.PauliX(0),
    qml.PauliY(1),
    qml.PauliZ(2),
    qml.PauliX(3),
    qml.PauliY(4),
    qml.PauliZ(5),
]


class TestDotPauliSentence:
    """Unittest for the dot function when ``pauli=True``"""

    def test_dot_returns_pauli_sentence(self):
        """Test that the dot function returns a PauliSentence class."""
        ps = qml.dot(coeffs, ops, pauli=True)
        assert isinstance(ps, PauliSentence)

    def test_coeffs_and_ops(self):
        """Test that the coefficients and operators of the returned PauliSentence are correct."""
        ps = qml.dot(coeffs, ops, pauli=True)
        h = ps.hamiltonian()
        assert qml.math.allequal(h.coeffs, coeffs)
        assert all(qml.equal(op1, op2) for op1, op2 in zip(h.ops, ops))

    def test_dot_simplifies_linear_combination(self):
        """Test that the dot function groups equal pauli words."""
        ps = qml.dot(
            coeffs=[0.12, 1.2, 12], ops=[qml.PauliX(0), qml.PauliX(0), qml.PauliX(0)], pauli=True
        )
        assert len(ps) == 1
        h = ps.hamiltonian()
        assert len(h.ops) == 1
        assert qml.equal(h.ops[0], qml.PauliX(0))

    def test_dot_returns_hamiltonian_simplified(self):
        """Test that hamiltonian computed from the PauliSentence created by the dot function is equal
        to the simplified hamiltonian."""
        ps = qml.dot(coeffs, ops, pauli=True)
        h_ps = ps.hamiltonian()
        h = Hamiltonian(coeffs, ops)
        h.simplify()
        assert qml.equal(h_ps, h)

    @pytest.mark.autograd
    def test_dot_autograd(self):
        """Test the dot function with the autograd interface."""
        c = qml.numpy.array([1.0, 2.0, 3.0])
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ps = qml.dot(c, o, pauli=True)

        ps_2 = qml.pauli.PauliSentence(
            {
                qml.pauli.PauliWord({0: "X"}): 1.0,
                qml.pauli.PauliWord({1: "Y"}): 2.0,
                qml.pauli.PauliWord({2: "Z"}): 3.0,
            }
        )
        assert ps == ps_2

    @pytest.mark.tf
    def test_dot_tf(self):
        """Test the dot function with the tensorflow interface."""
        import tensorflow as tf

        c = tf.constant([1.0, 2.0, 3.0])
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ps = qml.dot(c, o, pauli=True)

        ps_2 = qml.pauli.PauliSentence(
            {
                qml.pauli.PauliWord({0: "X"}): tf.constant(1.0),
                qml.pauli.PauliWord({1: "Y"}): tf.constant(2.0),
                qml.pauli.PauliWord({2: "Z"}): tf.constant(3.0),
            }
        )
        assert ps == ps_2

    @pytest.mark.torch
    def test_dot_torch(self):
        """Test the dot function with the torch interface."""
        import torch

        c = torch.tensor([1.0, 2.0, 3.0])
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ps = qml.dot(c, o, pauli=True)

        ps_2 = qml.pauli.PauliSentence(
            {
                qml.pauli.PauliWord({0: "X"}): torch.tensor(1.0),
                qml.pauli.PauliWord({1: "Y"}): torch.tensor(2.0),
                qml.pauli.PauliWord({2: "Z"}): torch.tensor(3.0),
            }
        )
        assert ps == ps_2

    @pytest.mark.jax
    def test_dot_jax(self):
        """Test the dot function with the torch interface."""
        import jax

        c = jax.numpy.array([1.0, 2.0, 3.0])
        o = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        ps = qml.dot(c, o, pauli=True)

        ps_2 = qml.pauli.PauliSentence(
            {
                qml.pauli.PauliWord({0: "X"}): jax.numpy.array(1.0),
                qml.pauli.PauliWord({1: "Y"}): jax.numpy.array(2.0),
                qml.pauli.PauliWord({2: "Z"}): jax.numpy.array(3.0),
            }
        )
        assert ps == ps_2
