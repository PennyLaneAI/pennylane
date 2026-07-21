# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Pow2 class."""

import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

import pennylane as qp
from pennylane.core.operator import Operator2
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.ops import ISWAP, Identity, PhaseShift, S, T, Z
from pennylane.ops.op_math.controlled2 import ControlledOp2
from pennylane.ops.op_math.pow import pow
from pennylane.ops.op_math.pow2 import Pow2
from tests.core.operator.operator2_utils import DynOp
from tests.ops.op_math.test_adjoint2 import RX2, SX2

# pylint: disable=unused-argument,arguments-differ,useless-parent-delegation,too-few-public-methods


class NoPowOp(Operator2):
    """A base operator whose ``pow`` always raises ``PowUndefinedError``."""

    wire_sizes = (1,)

    def __init__(self, wires):
        super().__init__(wires)

    def pow(self, z):
        raise PowUndefinedError


class BadPowOp(Operator2):
    """A base operator whose ``pow`` raises a generic (non-``Pow``) error."""

    wire_sizes = (1,)

    def __init__(self, wires):
        super().__init__(wires)

    def pow(self, z):
        raise ValueError("bad pow")


class RealPowOp(Operator2):
    """A base operator with a well-defined ``pow`` for any exponent."""

    dynamic_argnames = ("phi",)

    wire_sizes = (1,)

    def __init__(self, phi, wires):
        super().__init__(phi, wires)

    def pow(self, z):
        return [RealPowOp(self.phi * z, self.wires)]


def test_initialization():
    """Tests initializing a Pow2 operator."""

    base = DynOp(0.5, wires=0)

    # lazy
    op = pow(base, z=0.5, lazy=True)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 0.5
    assert op.base == base

    # eager
    op = pow(base, z=0.5, lazy=False)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 0.5
    assert op.base == base

    # has a custom pow()
    op = pow(T(0), z=3, lazy=False)
    assert isinstance(op, PhaseShift)

    # produces no ops
    op = pow(Identity(0), z=2, lazy=False)
    assert isinstance(op, Identity)

    # produces multiple ops
    ops = pow(ISWAP((0, 1)), z=6, lazy=False)
    assert ops == Z(0) @ Z(1)

    # we call Pow2 directly
    op = Pow2(base, z=1.5)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 1.5
    assert op.base == base


@pytest.mark.parametrize(
    "op", [pow(S(0), z=3), pow(ControlledOp2(S(0) @ S(1), control_wires=2), z=3)]
)
def test_repr(op):
    """Tests __repr__ method."""
    if op.base.arithmetic_depth > 0:
        assert repr(op) == f"({op.base})**{op.static_args["z"]}"
    else:
        assert repr(op) == f"{op.base}**{op.static_args["z"]}"


def test_has_sparse_matrix():
    """Tests the has_sparse_matrix property."""

    # base has a sparse matrix and z is an integer
    assert pow(SX2(0), z=2).has_sparse_matrix

    # base has a sparse matrix but z is not an integer
    assert not pow(SX2(0), z=0.5).has_sparse_matrix

    # base does not have a sparse matrix
    assert not pow(RX2(0.5, wires=0), z=2).has_sparse_matrix


def test_compute_matrix():
    """Tests the compute_matrix method for integer and fractional powers."""

    base_matrix = SX2(0).matrix()

    # integer power uses matrix_power
    op_int = pow(SX2(0), z=2)
    assert qp.math.allclose(op_int.matrix(), np.linalg.matrix_power(base_matrix, 2))

    # fractional power uses scipy's fractional_matrix_power
    op_frac = pow(SX2(0), z=0.5)
    assert qp.math.allclose(op_frac.matrix(), fractional_matrix_power(base_matrix, 0.5))


def test_compute_sparse_matrix():
    """Tests the compute_sparse_matrix method."""

    # integer power returns the base sparse matrix raised to the power
    op = pow(SX2(0), z=2)
    expected = (SX2(0).sparse_matrix() ** 2).todense()
    assert qp.math.allclose(op.sparse_matrix().todense(), expected)

    # a non-integer power raises SparseMatrixUndefinedError
    with pytest.raises(SparseMatrixUndefinedError):
        pow(SX2(0), z=0.5).sparse_matrix()


def test_has_decomposition():
    """Tests the has_decomposition property across all branches."""

    # positive integer power always has a decomposition
    assert pow(NoPowOp(0), z=2).has_decomposition

    # base.pow succeeds for a non-integer power
    assert pow(RealPowOp(0.5, wires=0), z=0.5).has_decomposition

    # base.pow raises PowUndefinedError
    assert not pow(NoPowOp(0), z=0.5).has_decomposition

    # base.pow raises a generic error with a batched (non-scalar) z
    assert not pow(BadPowOp(0), z=[0.5, 0.5]).has_decomposition

    # base.pow raises a generic error with a scalar z, which is re-raised
    with pytest.raises(ValueError, match="bad pow"):
        _ = pow(BadPowOp(0), z=0.5).has_decomposition


def test_compute_decomposition():
    """Tests the decomposition method across all branches."""

    # base.pow defines the decomposition directly
    decomp = Pow2.compute_decomposition(RealPowOp(2.0, wires=0), z=3)
    qp.assert_equal(decomp[0], RealPowOp(6.0, wires=0))

    # PowUndefinedError with a positive integer power repeats the base
    with qp.queuing.AnnotatedQueue():
        repeated = Pow2.compute_decomposition(NoPowOp(0), z=3)
    assert len(repeated) == 3
    assert all(isinstance(op, NoPowOp) for op in repeated)

    # PowUndefinedError with a non positive-integer power is undefined
    with pytest.raises(DecompositionUndefinedError):
        Pow2.compute_decomposition(NoPowOp(0), z=0.5)

    # a generic (non-Pow) error during base.pow is wrapped in DecompositionUndefinedError
    with pytest.raises(DecompositionUndefinedError):
        Pow2.compute_decomposition(BadPowOp(0), z=0.5)


def test_diagonalizing_gates():
    """Tests the compute_diagonalizing_gates method and has_diagonalizing_gates property."""

    op = pow(SX2(0), z=2)
    assert op.has_diagonalizing_gates
    assert op.compute_diagonalizing_gates(SX2(0), z=2) == [qp.H(0)]

    # base without diagonalizing gates
    assert not pow(RX2(0.5, wires=0), z=2).has_diagonalizing_gates


def test_eigvals():
    """Tests the compute_eigvals method."""

    # SX2 has eigenvalues [1, 1j], which squared are [1, -1]
    op = pow(SX2(0), z=2)
    assert qp.math.allclose(op.compute_eigvals(SX2(0), z=2), [1, -1])


def test_generator():
    """Tests the generator method and has_generator property."""

    op = pow(RX2(0.5, wires=0), z=3)
    assert op.has_generator
    assert op.generator().simplify() == qp.Hamiltonian([-1.5], [qp.X(0)])

    # base without a generator
    assert not pow(SX2(0), z=2).has_generator


def test_has_adjoint_and_adjoint():
    """Tests the adjoint method and has_adjoint property."""

    op = pow(RX2(0.5, wires=0), z=2)
    assert op.has_adjoint
    adj = op.adjoint()
    assert isinstance(adj, Pow2)
    qp.assert_equal(adj, Pow2(qp.adjoint(RX2(0.5, wires=0)), z=2))

    # fractional powers do not have a well-defined adjoint
    frac = pow(RX2(0.5, wires=0), z=0.5)
    assert not frac.has_adjoint
    with pytest.raises(AdjointUndefinedError, match="only is well-defined for integer powers"):
        frac.adjoint()


def test_pow():
    """Tests that the pow method combines exponents by multiplication."""

    op = pow(DynOp(0.5, wires=0), z=2)
    combined = op.pow(3)
    assert len(combined) == 1
    assert isinstance(combined[0], Pow2)
    qp.assert_equal(combined[0], Pow2(DynOp(0.5, wires=0), z=6))

    # a fractional exponent is also combined by multiplication
    frac = pow(DynOp(0.5, wires=0), z=2)
    qp.assert_equal(frac.pow(0.5)[0], Pow2(DynOp(0.5, wires=0), z=1.0))


def test_simplify_pauli_rep():
    """Tests that simplify uses the pauli representation when available."""

    # SX2 squared simplifies to X via its pauli representation
    op = pow(SX2(0), z=2)
    qp.assert_equal(op.simplify(), qp.X(0))


def test_simplify_to_identity():
    """Tests that simplify returns the identity when base.pow returns no ops."""

    op = pow(DynOp(0.5, wires=0), z=0)
    simplified = op.simplify()
    assert isinstance(simplified, Identity)
    assert simplified.wires == qp.wires.Wires([0])


def test_simplify_single_op():
    """Tests that simplify returns a single op when base.pow returns one op."""

    op = pow(RealPowOp(2.0, wires=0), z=0.5)
    qp.assert_equal(op.simplify(), RealPowOp(1.0, wires=0))


def test_simplify_multiple_ops():
    """Tests that simplify returns a product when base.pow returns several ops."""

    op = pow(DynOp(0.5, wires=0), z=2)
    qp.assert_equal(op.simplify(), qp.prod(DynOp(0.5, wires=0), DynOp(0.5, wires=0)))


def test_simplify_pow_undefined():
    """Tests that simplify falls back to a Pow2 when base.pow is undefined."""

    op = pow(NoPowOp(0), z=0.5)
    simplified = op.simplify()
    assert isinstance(simplified, Pow2)
    qp.assert_equal(simplified.base, NoPowOp(0))
    assert simplified.static_args["z"] == 0.5


def test_label():
    """Test that the label draws the exponent as superscript."""
    base = DynOp(1.2, wires=0)
    op = Pow2(base, -1.23456789)

    assert op.label() == "DynOp⁻¹⋅²³⁴⁵⁶⁷⁸⁹"
    assert op.label(decimals=2) == "DynOp\n(1.20)⁻¹⋅²³⁴⁵⁶⁷⁸⁹"
