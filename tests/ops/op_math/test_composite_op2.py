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
Unit tests for the composite operator class of qubit operations
"""

import inspect
from typing import Sequence

import numpy as np
import pytest

import pennylane as qp
from pennylane import math
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.ops.op_math import CompositeOp2
from pennylane.pauli.pauli_arithmetic import PauliWord
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires, WiresLike

# pylint:disable=protected-access, use-implicit-booleaness-not-comparison


ops = (
    (qp.PauliX(wires=0), qp.PauliZ(wires=0), qp.Hadamard(wires=0)),
    (qp.CNOT(wires=[0, 1]), qp.RX(1.23, wires=1), qp.Identity(wires=0)),
    (
        qp.IsingXX(4.56, wires=[2, 3]),
        qp.Toffoli(wires=[1, 2, 3]),
        qp.Rot(0.34, 1.0, 0, wires=0),
    ),
)

ops_rep = (
    "X(0) # Z(0) # H(0)",
    "(CNOT(wires=[0, 1])) # RX(1.23, wires=[1]) # I(0)",
    "IsingXX(4.56, wires=[2, 3]) # (Toffoli(wires=[1, 2, 3])) # Rot(0.34, 1.0, 0, wires=[0])",
)


class ValidOp(CompositeOp2):
    # pylint:disable=unused-argument
    _op_symbol = "#"
    _math_op = math.prod

    hybrid_argnames = ("operands", "_init_pauli_rep")
    wire_argnames = ()

    def __init__(self, operands: Sequence[Operator], _init_pauli_rep=None):
        super().__init__(operands, _init_pauli_rep=_init_pauli_rep)

    def _build_pauli_rep(self):
        return qp.pauli.PauliSentence({})

    @property
    def is_verified_hermitian(self):
        return False

    def matrix(self, wire_order=None):
        if wire_order is None:
            wire_order = self.wires
        mat = np.eye(2 ** len(wire_order))
        for op in self:
            mat = mat @ math.expand_matrix(op.matrix(), op.wires, wire_order=wire_order)
        return mat

    @classmethod
    # pylint: disable-next=unused-argument
    def _sort(cls, op_list, wire_map: dict = None):
        return op_list


# pylint: disable-next=too-few-public-methods
class NoMatrixOp(Operator2):

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires)


class NonOverlappingOp(ValidOp):

    def __init__(self, operands: Sequence[Operator], _init_pauli_rep=None):
        super().__init__(operands, _init_pauli_rep=_init_pauli_rep)
        self._overlapping_ops = []


class TestConstruction:
    """Test the construction of composite ops."""

    simple_operands = (qp.S(0), qp.T(1))

    def test_direct_initialization_fails(self):
        """Test directly initializing a CompositeOp2 fails"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class CompositeOp2"):
            _ = CompositeOp2(self.simple_operands)  # pylint:disable=abstract-class-instantiated

    @pytest.mark.xfail
    def test_raise_error_fewer_than_2_operands(self):
        """Test that initializing a composite operator with less than 2 operands raises a ValueError."""
        with pytest.raises(ValueError, match="Require at least two operators to combine;"):
            _ = ValidOp(qp.PauliX(0))

    def test_raise_error_with_mcm_input(self):
        """Test that composite ops of mid-circuit measurements are not supported."""
        mcm_0 = qp.ops.MidMeasure(0)
        mcm_1 = qp.ops.MidMeasure(1)
        ppm = qp.ops.PauliMeasure("XY", wires=[0, 1])
        op = qp.RX(0.5, 2)
        with pytest.raises(ValueError, match="Composite operators of mid-circuit"):
            _ = ValidOp((mcm_0, mcm_1))
        with pytest.raises(ValueError, match="Composite operators of mid-circuit"):
            _ = ValidOp((op, mcm_1))
        with pytest.raises(ValueError, match="Composite operators of mid-circuit"):
            _ = ValidOp((mcm_0, op))
        with pytest.raises(ValueError, match="Composite operators of mid-circuit"):
            _ = ValidOp((ppm, op))

    def test_initialization(self):
        """Test that valid child classes can be initialized without error"""
        op = ValidOp(self.simple_operands)
        assert op._name == "ValidOp"
        assert op._op_symbol == "#"

    def test_abstract_init(self):
        """Test that building a composite op from abstract operands routes through
        ``__abstract_init__`` rather than the concrete ``__init__``."""
        operands = (abstractify(qp.X(0)), abstractify(qp.PauliZ(1)))
        op = ValidOp(operands)

        assert op.operands[0] == abstractify(qp.X(0))
        assert op.operands[1] == abstractify(qp.PauliZ(1))
        assert op._pauli_rep is None

        assert op._hash is None
        assert op._has_overlapping_wires is None
        assert op._overlapping_ops is None

    def test_map_wires(self):
        """Test the map_wires method."""
        pr = PauliWord({0: "Y"}) + PauliWord({1: "Y"})
        op = ValidOp([qp.X(0), qp.Z(1)], _init_pauli_rep=pr)
        op_mapped = op.map_wires({0: 40, 1: 41})
        assert op_mapped.wires == (40, 41)
        assert op_mapped.operands == [qp.X(40), qp.Z(41)]
        assert op_mapped.pauli_rep.wires == (40, 41)

    def test_data(self):
        """Test that the data property flattens the data of all operands in order."""
        op = ValidOp((qp.RX(9.87, wires=0), qp.Rot(1.23, 4.0, 5.67, wires=1), qp.PauliX(0)))
        assert op.data == (9.87, 1.23, 4.0, 5.67)

    def test_data_is_read_only(self):
        """Test that composite operator data is read-only."""
        op = ValidOp((qp.RX(9.87, wires=0), qp.Rot(1.23, 4.0, 5.67, wires=1), qp.PauliX(0)))
        assert op.data == (9.87, 1.23, 4.0, 5.67)

        with pytest.raises(
            AttributeError, match="property 'data' of 'ValidOp' object has no setter"
        ):
            op.data = (1.23, 0.0, -1.0, -2.0)  # pylint:disable=attribute-defined-outside-init

    def test_initialization_in_queuing_context(self):
        """Test that valid child classes can be initialized in a queuing context"""
        with AnnotatedQueue() as q:
            op = ValidOp(self.simple_operands)
            assert op._name == "ValidOp"
            assert op._op_symbol == "#"
        assert op in q.queue
        assert len(q.queue) == 1

    def test_decomposition_raises_error(self):
        """Test that calling decomposition() raises a ValueError."""
        op = ValidOp(self.simple_operands)

        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_diagonalizing_gates_non_overlapping(self):
        """Test that the diagonalizing gates are correct when wires do not overlap."""
        diag_op = ValidOp((qp.PauliZ(wires=0), qp.Identity(wires=1)))
        assert diag_op.diagonalizing_gates() == []

    def test_diagonalizing_gates_overlapping(self):
        """Test that the diagonalizing gates are correct when wires overlap."""
        diag_op = ValidOp((qp.S(0), qp.PauliX(0)))
        diagonalizing_gates = diag_op.diagonalizing_gates()

        assert len(diagonalizing_gates) == 1
        u = diagonalizing_gates[0].matrix()

        # The diagonalizing gate rotates the (overlapping) operator into its eigenbasis, so the
        # rotated matrix must be diagonal.
        rotated = u @ diag_op.matrix() @ np.conj(u.T)
        off_diagonal = rotated - np.diag(np.diagonal(rotated))
        assert np.allclose(off_diagonal, 0)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_op = ValidOp(self.simple_operands)
        eig_decomp = diag_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = diag_op._eigs[diag_op]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)

    def test_build_pauli_rep(self):
        """Test the build_pauli_rep"""
        op = ValidOp(self.simple_operands)
        assert op._build_pauli_rep() == qp.pauli.PauliSentence({})


@pytest.mark.parametrize("math_op", [qp.prod, qp.sum])
def test_no_recursion_error_raised(math_op):
    """Tests that no RecursionError is raised from any property of method of a nested op."""

    op = qp.RX(np.random.uniform(0, 2 * np.pi), wires=1)
    for _ in range(2000):
        op = math_op(op, qp.RY(np.random.uniform(0, 2 * np.pi), wires=1))
    _assert_method_and_property_no_recursion_error(op)


def test_no_recursion_error_raised_sprod():
    """Tests that no RecursionError is raised from any property of method of a nested SProd."""

    op = qp.RX(np.random.uniform(0, 2 * np.pi), wires=1)
    for _ in range(5000):
        op = qp.s_prod(1, op)
    _assert_method_and_property_no_recursion_error(op)


def test_no_recursion_error_raised_prod():
    """Tests that no RecursionError is raised from any property of method of a nested Prod."""

    op = qp.RX(np.random.uniform(0, 2 * np.pi), wires=1)
    for _ in range(5000):
        op = qp.prod(qp.I(0), op)
    _assert_method_and_property_no_recursion_error(op)


def _assert_method_and_property_no_recursion_error(instance):
    """Checks that all methods and properties do not raise a RecursionError when accessed."""

    for name, attr in inspect.getmembers(instance.__class__):
        if inspect.isfunction(attr) and _is_method_with_no_argument(attr):
            _assert_method_no_recursion_error(instance, name)

        if isinstance(attr, property):
            _assert_property_no_recursion_error(instance, name)


def _assert_method_no_recursion_error(instance, method_name):
    """Checks that the method does not raise a RecursionError when called."""
    try:
        getattr(instance, method_name)()
    except Exception as e:  # pylint: disable=broad-except
        assert not isinstance(e, RecursionError)
        if isinstance(e, RuntimeError) and not isinstance(e, NotImplementedError):
            assert "This is likely due to nesting too many levels" in str(e)


def _assert_property_no_recursion_error(instance, property_name):
    """Checks that the property does not raise a RecursionError when accessed."""
    try:
        getattr(instance, property_name)
    except Exception as e:  # pylint: disable=broad-except
        assert not isinstance(e, RecursionError)
        if isinstance(e, RuntimeError) and not isinstance(e, NotImplementedError):
            assert "This is likely due to nesting too many levels" in str(e)


def _is_method_with_no_argument(method):
    """Checks if a method has no argument other than self."""
    parameters = list(inspect.signature(method).parameters.values())
    if not (parameters and parameters[0].name == "self"):
        return False
    for param in parameters[1:]:
        if param.kind is not param.POSITIONAL_OR_KEYWORD or param.default == param.empty:
            return False
    return True


class TestMscMethods:
    """Test dunder and other miscellaneous methods."""

    def test_has_diagonalizing_gates(self):
        """Test that the has_diagonalizing_gates property is correct."""
        # has overlapping wires and no diag gates case
        op = ValidOp((NoMatrixOp(0), qp.PauliX(0)))
        assert op.has_diagonalizing_gates is False
        # has overlapping wires and diag gates case
        op = ValidOp((qp.PauliZ(0), qp.PauliX(0)))
        assert op.has_diagonalizing_gates is True
        # no overlapping wires and no diag gates case
        op = ValidOp((NoMatrixOp(0), NoMatrixOp(1)))
        assert op.has_diagonalizing_gates is False
        # no overlapping wires and diag gates case
        op = ValidOp((qp.PauliZ(0), qp.PauliX(1)))
        assert op.has_diagonalizing_gates is True
        # One sub op with no diagonalizing gates
        op = ValidOp((NoMatrixOp(0), qp.PauliZ(1), qp.PauliX(1)))
        assert op.has_diagonalizing_gates is False

    @pytest.mark.parametrize(
        "operators",
        [
            (qp.S(0),),
            (qp.S(0), qp.T(1)),
            (qp.T(0), qp.S(1)),
            (qp.Identity(0), qp.Hadamard(1)),
            (qp.T(0), qp.Identity(1)),
            (qp.T(0), qp.Identity(1), qp.S(1)),
        ],
    )
    def test_eigvals(self, operators):
        """Test that the eigvals method is correct."""
        op = ValidOp(operators)
        vals = op.eigvals()

        def _expand_two(sub_op):
            return (
                np.kron(sub_op.matrix(), np.eye(2))
                if sub_op.wires == (0,)
                else np.kron(np.eye(2), sub_op.matrix())
            )

        if len(operators) > 1:
            sub_mat = _expand_two(operators[0])
            for sub in operators[1:]:
                sub_mat = sub_mat @ _expand_two(sub)
            assert np.allclose(vals, math.linalg.eig(sub_mat)[0])
        else:
            assert np.allclose(vals, math.linalg.eig(operators[0].matrix())[0])

    def test_has_matrix(self):
        """Test that the has_matrix property is correct."""
        op = ValidOp((NoMatrixOp(0), NoMatrixOp(1)))
        assert op.has_matrix is False
        op = ValidOp((qp.PauliX(0), qp.PauliX(0)))
        assert op.has_matrix is True
        op = ValidOp((qp.PauliX(0), qp.PauliY(1)))
        assert op.has_matrix is True
        op = ValidOp((qp.PauliX(0), NoMatrixOp(1)))
        assert op.has_matrix is False

    def test_has_overlapping_wires(self):
        """Test that the has_overlapping_wires property is correct."""
        op = ValidOp((qp.PauliX(0), qp.PauliX(0)))
        assert op.has_overlapping_wires is True

        op = ValidOp((qp.PauliX(0), qp.PauliY(1)))
        assert op.has_overlapping_wires is False

        op = ValidOp((qp.PauliX(0), qp.PauliY(0)))
        assert op.has_overlapping_wires is True

    @pytest.mark.parametrize("ops_lst, op_rep", tuple((i, j) for i, j in zip(ops, ops_rep)))
    def test_repr(self, ops_lst, op_rep):
        """Test __repr__ method."""
        op = ValidOp(ops_lst)
        assert op_rep == repr(op)

    def test_nested_repr(self):
        """Test nested repr values while other nested features such as equality are not ready"""
        op = ValidOp((qp.PauliX(0), ValidOp((qp.RY(1, wires=1), qp.PauliX(0)))))
        assert repr(op) == "X(0) # (RY(1, wires=[1]) # X(0))"

    def test_label(self):
        """Test label method."""
        op = ValidOp((qp.RY(1, wires=1), qp.PauliX(1)))
        assert op.label() == "RY#X"
        with pytest.raises(ValueError):
            op.label(base_label=["only_first"])

        nested_op = ValidOp((qp.PauliX(0), op))
        assert nested_op.label() == "X#(RY#X)"
        assert nested_op.label(decimals=2) == "X#(RY\n(1.00)#X)"
        assert nested_op.label(base_label=["x0", ["ry", "x1"]]) == "x0#(ry#x1)"

        U = np.array([[1, 0], [0, -1]])
        cache = {"matrices": []}
        op = ValidOp((qp.PauliX(0), ValidOp((qp.PauliY(1), qp.QubitUnitary(U, wires=0)))))
        assert op.label(cache=cache) == "X#(Y#U\n(M0))"
        assert cache["matrices"] == [U]

    @pytest.mark.parametrize("ops_lst", ops)
    def test_len(self, ops_lst):
        """Test __len__ method."""
        op = ValidOp(ops_lst)
        assert len(op) == len(ops_lst)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_iter(self, ops_lst):
        """Test __iter__ method."""
        op = ValidOp(ops_lst)
        for i, j in zip(op, ops_lst):
            assert i == j

    @pytest.mark.parametrize("ops_lst", ops)
    def test_getitem(self, ops_lst):
        """Test __getitem__ method."""
        op = ValidOp(ops_lst)
        for i, operand in enumerate(ops_lst):
            assert op[i] == operand


class TestProperties:
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        """Test num_params property updates correctly."""
        op = ValidOp(ops_lst)
        true_num_params = sum(op.num_params for op in ops_lst)

        assert op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        """Test num_wires property updates correctly."""
        valid_op = ValidOp(ops_lst)
        true_wires = set()

        for op in ops_lst:
            true_wires = true_wires.union(op.wires.toset())

        assert valid_op.num_wires == len(true_wires)

    def test_depth_property(self):
        """Test depth property."""
        op = ValidOp((qp.RZ(1.32, wires=0), qp.Identity(wires=0), qp.RX(1.9, wires=1)))
        assert op.arithmetic_depth == 1

        op = ValidOp((qp.PauliX(0), ValidOp((qp.Identity(wires=0), qp.RX(1.9, wires=1)))))
        assert op.arithmetic_depth == 2

    def test_overlapping_ops_property(self):
        """Test the overlapping_ops property."""
        valid_op = ValidOp(
            (
                qp.sum(qp.PauliX(0), qp.PauliY(5), qp.PauliZ(10)),
                qp.sum(qp.PauliX(1), qp.PauliY(4), qp.PauliZ(6)),
                qp.prod(qp.PauliX(10), qp.PauliY(2)),
                qp.PauliY(7),
                qp.Hamiltonian([1, 1], [qp.PauliX(2), qp.PauliZ(7)]),
                qp.prod(qp.PauliX(4), qp.PauliY(3), qp.PauliZ(8)),
            )
        )
        overlapping_ops = [
            [
                qp.sum(qp.PauliX(0), qp.PauliY(5), qp.PauliZ(10)),
                qp.prod(qp.PauliX(10), qp.PauliY(2)),
                qp.PauliY(7),
                qp.Hamiltonian([1, 1], [qp.PauliX(2), qp.PauliZ(7)]),
            ],
            [
                qp.sum(qp.PauliX(1), qp.PauliY(4), qp.PauliZ(6)),
                qp.prod(qp.PauliX(4), qp.PauliY(3), qp.PauliZ(8)),
            ],
        ]
        for list_op1, list_op2 in zip(overlapping_ops, valid_op.overlapping_ops):
            for op1, op2 in zip(list_op1, list_op2):
                qp.assert_equal(op1, op2)

    def test_overlapping_ops_private_attribute(self):
        """Test that the private `_overlapping_ops` attribute gets updated after a call to
        the `overlapping_ops` property."""
        op = ValidOp((qp.RZ(1.32, wires=0), qp.Identity(wires=0), qp.RX(1.9, wires=1)))
        overlapping_ops = op.overlapping_ops
        assert op._overlapping_ops == overlapping_ops

        op = NonOverlappingOp((qp.RZ(1.32, wires=0),))
        assert op.overlapping_ops == []


@pytest.mark.capture
# pylint: disable-next=too-few-public-methods
class TestCapture:
    """Test that a CompositeOp2 subclass integrates with program capture."""

    @pytest.mark.jax
    def test_capture_valid_op(self):
        """Test that a ValidOp can be captured into and reconstructed from jaxpr."""
        import jax

        from tests.capture.capture_utils import assert_eqn_matches_op

        def qfunc():
            return ValidOp((qp.RX(1.2, wires=0), qp.PauliZ(0)))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]
        assert_eqn_matches_op(eqn, ValidOp)

        with AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q.queue) == 1
        qp.assert_equal(q.queue[0], ValidOp((qp.RX(1.2, wires=0), qp.PauliZ(0))))
