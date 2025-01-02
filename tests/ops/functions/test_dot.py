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
from pennylane.ops import Prod, SProd, Sum
from pennylane.pauli.pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z

pw1 = PauliWord({0: I, 1: X, 2: Y})
pw2 = PauliWord({0: Z, 2: Z, 4: Z})
pw3 = PauliWord({"a": X, "b": X, "c": Z})
pw4 = PauliWord({"a": Y, "b": Z, "c": X})
pw5 = PauliWord({4: X, 5: X})

ps1 = PauliSentence({pw1: 1.0, pw2: 2.0})
ps2 = PauliSentence({pw3: 1.0, pw4: 2.0})

op1 = qml.prod(qml.PauliX(1), qml.PauliY(2))
op2 = qml.prod(qml.PauliZ(0), qml.PauliZ(2), qml.PauliZ(4))
op3 = qml.prod(qml.PauliX("a"), qml.PauliX("b"), qml.PauliZ("c"))
op4 = qml.prod(qml.PauliY("a"), qml.PauliZ("b"), qml.PauliX("c"))
op5 = qml.prod(qml.PauliX(4), qml.PauliX(5))

pw_id = PauliWord({})
ps_id = PauliSentence({pw_id: 1.0})
op_id = qml.Identity(0)

X0, Y0, Z0 = PauliWord({0: "X"}), PauliWord({0: "Y"}), PauliWord({0: "Z"})
XX, YY, ZZ = PauliWord({0: "X", 1: "X"}), PauliWord({0: "Y", 1: "Y"}), PauliWord({0: "Z", 1: "Z"})

H1 = X0 + Y0 + Z0
H2 = XX + YY + ZZ
H3 = 1.0 * X0 + 2.0 * Y0 + 3.0 * Z0


class TestDotSum:
    """Unittests for the dot function when ``pauli=False``."""

    def test_error_if_ops_operator(self):
        """Test that dot raises an error if ops is an operator itself."""
        with pytest.raises(ValueError, match=r"ops must be an Iterable of Operator's"):
            qml.dot([1, 1], qml.X(0) @ qml.Y(1))

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

    def test_coeffs_all_ones(self):
        """Test when the coeffs are all ones that we get a sum of the individual products."""
        result = qml.dot([1, 1, 1], [qml.X(0), qml.Y(1), qml.Z(2)])
        assert isinstance(result, Sum)
        assert result[0] == qml.X(0)
        assert result[1] == qml.Y(1)
        assert result[2] == qml.Z(2)

    @pytest.mark.parametrize("coeffs", ([4, 4, 4], [4, -4, 4]))
    def test_dot_does_not_groups_coeffs(self, coeffs):
        """Test that the `dot` function does not groups the coefficients."""
        result = qml.dot(coeffs=coeffs, ops=[qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)])
        assert isinstance(result, Sum)
        assert result[0] == qml.s_prod(coeffs[0], qml.PauliX(0))
        assert result[1] == qml.s_prod(coeffs[1], qml.PauliX(1))
        assert result[2] == qml.s_prod(coeffs[2], qml.PauliX(2))

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
        qml.assert_equal(op_sum, op_sum_2)

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
        qml.assert_equal(op_sum, op_sum_2)

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
        qml.assert_equal(op_sum, op_sum_2)

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
        qml.assert_equal(op_sum, op_sum_2)

    data_just_words_pauli_false = (
        ([1.0, 2.0, 3.0], [pw1, pw2, pw_id], [op1, op2, op_id]),
        ([1.0, 2.0, 3.0], [pw1, pw2, pw_id], [op1, op2, op_id]),
        ([1.0, 2.0, 3.0, 4.0], [pw1, pw2, pw3, pw_id], [op1, op2, op3, op_id]),
        ([1.0, 2.0, 3j, 4j], [pw1, pw2, pw3, pw_id], [op1, op2, op3, op_id]),
        ([1, 1, 1], [pw1, pw1, pw1], [op1, op1, op1]),
        ([1, 1, 1], [pw_id, pw_id, pw_id], [op_id, op_id, op_id]),
    )

    @pytest.mark.parametrize("coeff, words, ops", data_just_words_pauli_false)
    def test_dot_with_just_words_pauli_false(self, coeff, words, ops):
        """Test operators that are just pauli words"""
        dot_res = qml.dot(coeff, words, pauli=False)
        true_res = qml.dot(coeff, ops)
        assert dot_res == true_res

    data_words_and_sentences_pauli_false = (
        (
            [1.0, 2.0, 3.0],
            [pw1, pw2, ps1],
        ),
        (
            [1.0, 2.0, 3.0, 4.0],
            [X0, Y0, XX, YY],
        ),
        (
            [1.0, 2.0, 3.0],
            [H1, H2, H3],
        ),
        (
            [1.0, 2.0, 3.0],
            [pw3, pw4, ps2],
        ),  # comparisons for Sum objects with string valued wires
    )

    @pytest.mark.parametrize("coeff, ops", data_words_and_sentences_pauli_false)
    def test_dot_with_words_and_sentences_pauli_false(self, coeff, ops):
        """Test operators that are a mix of pauli words and pauli sentences"""
        dot_res = qml.dot(coeff, ops, pauli=False)
        true_res = qml.dot(coeff, [op.operation() for op in ops], pauli=False)
        assert dot_res == true_res

    data_op_words_and_sentences_pauli_false = (
        (
            [1.0, 2.0, 3.0, 1j],
            [pw1, pw2, ps1, op1],
            qml.sum(qml.s_prod(3.0 * 1 + 1 + 1j, op1), qml.s_prod(3 * 2.0 + 2, op2)),
        ),
        (
            [2.0, 3.0, 1j],
            [pw2, ps1, op1],
            qml.sum(qml.s_prod(3.0 * 1 + 1j, op1), qml.s_prod(3 * 2.0 + 2, op2)),
        ),
        (
            [2.0, 3.0, 1j],
            [pw2, ps1, op5],
            qml.sum(qml.s_prod(3.0 * 1, op1), qml.s_prod(3 * 2.0 + 2, op2), qml.s_prod(1j, op5)),
        ),
        (
            [1.0, 2.0, 3.0, 1j],
            [pw3, pw4, ps2, op3],
            qml.sum(qml.s_prod(3.0 * 1 + 1 + 1j, op3), qml.s_prod(3 * 2.0 + 2, op4)),
        ),  # string valued wires
    )

    @pytest.mark.parametrize("coeff, ops, res", data_op_words_and_sentences_pauli_false)
    def test_dot_with_ops_words_and_sentences(self, coeff, ops, res):
        """Test operators that are a mix of PL operators, pauli words and pauli sentences with pauli=False (i.e. returning operators)"""
        dot_res = qml.dot(coeff, ops, pauli=False).simplify()
        assert dot_res == res

    def test_identities_with_pauli_words_pauli_false(self):
        """Test that identities in form of empty PauliWords are treated correctly"""
        _pw1 = PauliWord({0: I, 1: X, 2: Y})
        res = qml.dot([2.0, 2.0], [pw_id, _pw1], pauli=False)
        true_res = qml.sum(qml.s_prod(2.0, qml.I()), 2.0 * qml.prod(qml.X(1), qml.Y(2)))
        assert res == true_res

    def test_identities_with_pauli_sentences_pauli_false(self):
        """Test that identities in form of PauliSentences with empty PauliWords are treated correctly"""
        _pw1 = PauliWord({0: I, 1: X, 2: Y})
        res = qml.dot([2.0, 2.0], [ps_id, _pw1], pauli=False)
        true_res = qml.s_prod(2, qml.I()) + qml.s_prod(2.0, qml.prod(qml.X(1), qml.Y(2)))
        assert res == true_res


coeffs0 = [0.12345, 1.2345, 12.345, 123.45, 1234.5, 12345]
ops0 = [
    qml.PauliX(0),
    qml.PauliY(1),
    qml.PauliZ(2),
    qml.PauliX(3),
    qml.PauliY(4),
    qml.PauliZ(5),
]


class TestDotPauliSentence:
    """Unittest for the dot function when ``pauli=True``"""

    def test_error_if_ops_PauliWord(self):
        """Test that dot raises an error if ops is a PauliWord itself."""
        _pw = qml.pauli.PauliWord({0: "X", 1: "Y"})
        with pytest.raises(ValueError, match=r"ops must be an Iterable of PauliWord's"):
            qml.dot([1, 2], _pw)

    def test_error_if_ops_PauliSentence(self):
        """Test that dot raises an error if ops is a PauliSentence itself."""
        _pw1 = qml.pauli.PauliWord({0: "X", 1: "Y"})
        _pw2 = qml.pauli.PauliWord({2: "Z"})
        ps = 2 * _pw1 + 3 * _pw2
        with pytest.raises(ValueError, match=r"ops must be an Iterable of PauliSentence's"):
            qml.dot([1, 2], ps)

    def test_dot_returns_pauli_sentence(self):
        """Test that the dot function returns a PauliSentence class."""
        ps = qml.dot(coeffs0, ops0, pauli=True)
        assert isinstance(ps, PauliSentence)

    def test_coeffs_and_ops(self):
        """Test that the coefficients and operators of the returned PauliSentence are correct."""
        ps = qml.dot(coeffs0, ops0, pauli=True)
        h = ps.operation()
        hcoeffs, hops = h.terms()
        assert qml.math.allequal(hcoeffs, coeffs0)
        for _op1, _op2 in zip(hops, ops0):
            qml.assert_equal(_op1, _op2)

    def test_dot_simplifies_linear_combination(self):
        """Test that the dot function groups equal pauli words."""
        ps = qml.dot(
            coeffs=[0.12, 1.2, 12], ops=[qml.PauliX(0), qml.PauliX(0), qml.PauliX(0)], pauli=True
        )
        assert len(ps) == 1
        h = ps.operation()
        assert isinstance(h, qml.ops.SProd)
        qml.assert_equal(h.base, qml.PauliX(0))

    def test_dot_returns_hamiltonian_simplified(self):
        """Test that hamiltonian computed from the PauliSentence created by the dot function is equal
        to the simplified hamiltonian."""
        ps = qml.dot(coeffs0, ops0, pauli=True)
        h_ps = ps.operation()
        h = qml.Hamiltonian(coeffs0, ops0)
        h.simplify()
        qml.assert_equal(h_ps, h)

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

    data_just_words = (
        (
            [1.0, 2.0, 3.0, 4.0],
            [pw1, pw2, pw3, pw_id],
            PauliSentence({pw1: 1.0, pw2: 2.0, pw3: 3.0, pw_id: 4.0}),
        ),
        (
            [1.5, 2.5, 3.5, 4.5j],
            [pw1, pw2, pw3, pw_id],
            PauliSentence({pw1: 1.5, pw2: 2.5, pw3: 3.5, pw_id: 4.5j}),
        ),
        ([1.5, 2.5, 3.5], [pw3, pw2, pw1], PauliSentence({pw3: 1.5, pw2: 2.5, pw1: 3.5})),
        ([1, 1, 1], [PauliWord({0: "X"})] * 3, PauliSentence({PauliWord({0: "X"}): 3.0})),
        ([1, 1, 1], [PauliWord({})] * 3, PauliSentence({PauliWord({}): 3.0})),
    )

    @pytest.mark.parametrize("coeff, ops, res", data_just_words)
    def test_dot_with_just_words(self, coeff, ops, res):
        """Test operators that are just pauli words"""
        dot_res = qml.dot(coeff, ops, pauli=True)
        assert dot_res == res

    data_words_and_sentences = (
        ([1.0, 2.0, 3.0], [pw1, pw2, ps1], PauliSentence({pw1: 3.0 * 1 + 1, pw2: 3 * 2.0 + 2})),
        ([1.0, 2.0, 3.0], [pw3, pw4, ps2], PauliSentence({pw3: 3.0 * 1 + 1, pw4: 3 * 2.0 + 2})),
    )

    @pytest.mark.parametrize("coeff, ops, res", data_words_and_sentences)
    def test_dot_with_words_and_sentences(self, coeff, ops, res):
        """Test operators that are a mix of pauli words and pauli sentences"""
        dot_res = qml.dot(coeff, ops, pauli=True)
        assert dot_res == res

    data_op_words_and_sentences = (
        (
            [1.0, 2.0, 3.0, 1j],
            [pw1, pw2, ps1, op1],
            PauliSentence({pw1: 3.0 * 1 + 1 + 1j, pw2: 3 * 2.0 + 2}),
        ),
        (
            [1.0, 2.0, 3.0, 1j],
            [pw3, pw4, ps2, op3],
            PauliSentence({pw3: 3.0 * 1 + 1 + 1j, pw4: 3 * 2.0 + 2}),
        ),
        ([2.0, 3.0, 1j], [pw2, ps1, op1], PauliSentence({pw1: 3.0 * 1 + 1j, pw2: 3 * 2.0 + 2})),
        ([2.0, 3.0, 1j], [pw2, ps1, op5], PauliSentence({pw1: 3.0 * 1, pw2: 3 * 2.0 + 2, pw5: 1j})),
    )

    @pytest.mark.parametrize("coeff, ops, res", data_op_words_and_sentences)
    def test_dot_with_ops_words_and_sentences(self, coeff, ops, res):
        """Test operators that are a mix of PL operators, pauli words and pauli sentences"""
        dot_res = qml.dot(coeff, ops, pauli=True)
        assert dot_res == res

    def test_identities_with_pauli_words_pauli_true(self):
        """Test that identities in form of empty PauliWords are treated correctly"""
        res = qml.dot([2.0, 2.0], [pw_id, pw1], pauli=True)
        true_res = PauliSentence({pw_id: 2, pw1: 2})
        assert res == true_res

    def test_identities_with_pauli_sentences_pauli_true(self):
        """Test that identities in form of PauliSentences with empty PauliWords are treated correctly"""
        res = qml.dot([2.0, 2.0], [ps_id, pw1], pauli=True)
        true_res = PauliSentence({pw_id: 2.0, pw1: 2.0})
        assert res == true_res
