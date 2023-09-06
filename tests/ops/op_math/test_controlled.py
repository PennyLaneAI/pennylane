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
"""Unit tests for Controlled"""

from copy import copy
from functools import partial

import numpy as onp
import pytest
from gate_data import CNOT, CSWAP, CZ, CRot3, CRotx, CRoty, CRotz, Toffoli
from scipy import sparse

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import DecompositionUndefinedError, Operation, Operator
from pennylane.ops.op_math.controlled import (
    Controlled,
    ControlledOp,
    _decompose_no_control_values,
    ctrl,
)
from pennylane.tape import QuantumScript
from pennylane.tape.tape import expand_tape
from pennylane.wires import Wires

# pylint: disable=too-few-public-methods
# pylint: disable=protected-access
# pylint: disable=pointless-statement
# pylint: disable=expression-not-assigned


def equal_list(lhs, rhs):
    if not isinstance(lhs, list):
        lhs = [lhs]
    if not isinstance(rhs, list):
        rhs = [rhs]
    return len(lhs) == len(rhs) and all(qml.equal(l, r) for l, r in zip(lhs, rhs))


base_num_control_mats = [
    (qml.PauliX("a"), 1, CNOT),
    (qml.PauliZ("a"), 1, CZ),
    (qml.SWAP(("a", "b")), 1, CSWAP),
    (qml.PauliX("a"), 2, Toffoli),
    (qml.RX(1.234, "b"), 1, CRotx(1.234)),
    (qml.RY(-0.432, "a"), 1, CRoty(-0.432)),
    (qml.RZ(6.78, "a"), 1, CRotz(6.78)),
    (qml.Rot(1.234, -0.432, 9.0, "a"), 1, CRot3(1.234, -0.432, 9.0)),
]


class TempOperator(Operator):
    num_wires = 1


class TempOperation(Operation):
    num_wires = 1


class TestControlledInheritance:
    """Test the inheritance structure modified through dynamic __new__ method."""

    def test_plain_operator(self):
        """Test when base directly inherits from Operator only inherits from Operator."""

        base = TempOperator(1.234, wires=0)
        op = Controlled(base, 1.2)

        assert isinstance(op, Controlled)
        assert isinstance(op, Operator)
        assert not isinstance(op, Operation)
        assert not isinstance(op, ControlledOp)

    def test_operation(self):
        """When the operation inherits from `Operation`, then a `ControlledOp` should
        be created instead and the Controlled should now have Operation functionality."""

        class CustomOp(Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op = Controlled(base, 6.5)

        assert type(op) is ControlledOp  # pylint: disable=unidiomatic-typecheck

        assert isinstance(op, Controlled)
        assert isinstance(op, Operator)
        assert isinstance(op, Operation)
        assert isinstance(op, ControlledOp)

    def test_controlledop_new(self):
        """Test that if a `ControlledOp` is directly requested, it is created
        even if the base isn't an operation."""

        base = TempOperator(1.234, wires="a")
        op = ControlledOp(base, "b")

        assert type(op) is ControlledOp  # pylint: disable=unidiomatic-typecheck


class TestInitialization:
    """Test the initialization process and standard properties."""

    temp_op = TempOperator("a")

    def test_nonparametric_ops(self):
        """Test pow initialization for a non parameteric operation."""

        op = Controlled(
            self.temp_op, (0, 1), control_values=[True, False], work_wires="aux", id="something"
        )

        assert op.base is self.temp_op
        assert op.hyperparameters["base"] is self.temp_op

        assert op.wires == Wires((0, 1, "a", "aux"))

        assert op.control_wires == Wires((0, 1))
        assert op.hyperparameters["control_wires"] == Wires((0, 1))

        assert op.target_wires == Wires("a")

        assert op.control_values == [True, False]
        assert op.hyperparameters["control_values"] == [True, False]

        assert op.work_wires == Wires(("aux"))

        assert op.name == "C(TempOperator)"
        assert op.id == "something"

        assert op.num_params == 0
        assert op.parameters == []  # pylint: disable=use-implicit-booleaness-not-comparison
        assert op.data == ()

        assert op.num_wires == 4

    def test_default_control_values(self):
        """Test assignment of default control_values."""
        op = Controlled(self.temp_op, (0, 1))
        assert op.control_values == [True, True]

    def test_zero_one_control_values(self):
        """Test assignment of provided control_values."""
        op = Controlled(self.temp_op, (0, 1), control_values=[0, 1])
        assert op.control_values == [False, True]

    def test_string_control_values(self):
        """Test warning and conversion of string control_values."""

        with pytest.warns(UserWarning, match="Specifying control values as a string"):
            op = Controlled(self.temp_op, (0, 1), "01")

        assert op.control_values == [False, True]

    def test_non_boolean_control_values(self):
        """Test control values are converted to booleans."""
        op = Controlled(self.temp_op, (0, 1, 2), control_values=["", None, 5])
        assert op.control_values == [False, False, True]

    def test_control_values_wrong_length(self):
        """Test checking control_values length error."""
        with pytest.raises(ValueError, match="control_values should be the same length"):
            Controlled(self.temp_op, (0, 1), [True])

    def test_target_control_wires_overlap(self):
        """Test checking overlap of target wires and control_wires"""
        with pytest.raises(ValueError, match="The control wires must be different"):
            Controlled(self.temp_op, "a")

    def test_work_wires_overlap_target(self):
        """Test checking work wires are not in target wires."""
        with pytest.raises(ValueError, match="Work wires must be different"):
            Controlled(self.temp_op, "b", work_wires="a")

    def test_work_wires_overlap_control(self):
        """Test checking work wires are not in contorl wires."""
        with pytest.raises(ValueError, match="Work wires must be different."):
            Controlled(self.temp_op, control_wires="b", work_wires="b")


class TestProperties:
    """Test the properties of the ``Controlled`` symbolic operator."""

    def test_data(self):
        """Test that the base data can be get and set through Controlled class."""

        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        op = Controlled(base, (0, 1))

        assert op.data == (x,)

        x_new = (np.array(2.3454),)
        op.data = x_new
        assert op.data == (x_new,)
        assert base.data == (x_new,)

        x_new2 = (np.array(3.456),)
        base.data = x_new2
        assert op.data == (x_new2,)
        assert op.parameters == [x_new2]

    @pytest.mark.parametrize(
        "val, arr", ((4, [1, 0, 0]), (6, [1, 1, 0]), (1, [0, 0, 1]), (5, [1, 0, 1]))
    )
    def test_control_int(self, val, arr):
        """Test private `_control_int` property converts control_values to integer representation."""
        op = Controlled(TempOperator(5), (0, 1, 2), control_values=arr)
        assert op._control_int == val

    @pytest.mark.parametrize("value", (True, False))
    def test_has_matrix(self, value):
        """Test that Controlled defers has_matrix to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            has_matrix = value

        op = Controlled(DummyOp(1), 0)
        assert op.has_matrix is value

    @pytest.mark.parametrize(
        "base", (qml.RX(1.23, 0), qml.Rot(1.2, 2.3, 3.4, 0), qml.QubitUnitary([[0, 1], [1, 0]], 0))
    )
    def test_ndim_params(self, base):
        """Test that Controlled defers to base ndim_params"""

        op = Controlled(base, 1)
        assert op.ndim_params == base.ndim_params

    @pytest.mark.parametrize("cwires, cvalues", [(0, [0]), ([3, 0, 2], [1, 1, 0])])
    def test_has_decomposition_true_via_control_values(self, cwires, cvalues):
        """Test that Controlled claims `has_decomposition` to be true if there are
        any negated control values."""

        op = Controlled(TempOperation(0.2, wires=1), cwires, cvalues)
        assert op.has_decomposition is True

    def test_has_decomposition_true_via_base_has_ctrl_single_cwire(self):
        """Test that Controlled claims `has_decomposition` to be true if
        only one control wire is used and the base has a `_controlled` method."""

        op = Controlled(qml.RX(0.2, wires=1), 4)
        assert op.has_decomposition is True

    def test_has_decomposition_true_via_pauli_x(self):
        """Test that Controlled claims `has_decomposition` to be true if
        the base is a `PauliX` operator"""

        op = Controlled(qml.PauliX(3), [0, 4])
        assert op.has_decomposition is True

    def test_has_decomposition_true_via_base_has_decomp(self):
        """Test that Controlled claims `has_decomposition` to be true if
        the base has a decomposition and indicates this via `has_decomposition`."""

        op = Controlled(qml.IsingXX(0.6, [1, 3]), [0, 4])
        assert op.has_decomposition is True

    def test_has_decomposition_false_single_cwire(self):
        """Test that Controlled claims `has_decomposition` to be false if
        no path of decomposition would work, here we use a single control wire."""

        # all control values are 1, there is only one control wire but TempOperator does
        # not have `_controlled`, is not `PauliX`, doesn't have a ZYZ decomposition,
        # and reports `has_decomposition=False`
        op = Controlled(TempOperator(0.5, 1), 0)
        assert op.has_decomposition is False

    def test_has_decomposition_false_multi_cwire(self):
        """Test that Controlled claims `has_decomposition` to be false if
        no path of decomposition would work, here we use multiple control wires."""

        # all control values are 1, there are multiple control wires,
        # `TempOperator` is not `PauliX`, and reports `has_decomposition=False`
        op = Controlled(TempOperator(0.5, 1), [0, 5])
        assert op.has_decomposition is False

    @pytest.mark.parametrize("value", (True, False))
    def test_has_adjoint(self, value):
        """Test that Controlled defers has_adjoint to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            has_adjoint = value

        op = Controlled(DummyOp(1), 0)
        assert op.has_adjoint is value

    @pytest.mark.parametrize("value", (True, False))
    def test_has_diagonalizing_gates(self, value):
        """Test that Controlled defers has_diagonalizing_gates to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            has_diagonalizing_gates = value

        op = Controlled(DummyOp(1), 0)
        assert op.has_diagonalizing_gates is value

    @pytest.mark.parametrize("value", ("_ops", None))
    def test_queue_cateogry(self, value):
        """Test that Controlled defers `_queue_category` to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            _queue_category = value

        op = Controlled(DummyOp(1), 0)
        assert op._queue_category == value

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value):
        """Test that Controlled defers `is_hermitian` to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            is_hermitian = value

        op = Controlled(DummyOp(1), 0)
        assert op.is_hermitian is value

    def test_map_wires(self):
        """Test that we can get and set private wires."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        op = Controlled(base, (3, 4), work_wires="aux")

        assert op.wires == Wires((3, 4, 0, 1, "aux"))

        op = op.map_wires(wire_map={3: "a", 4: "b", 0: "c", 1: "d", "aux": "extra"})

        assert op.base.wires == Wires(("c", "d"))
        assert op.control_wires == Wires(("a", "b"))
        assert op.work_wires == Wires(("extra"))


class TestMiscMethods:
    """Test miscellaneous minor Controlled methods."""

    def test_repr(self):
        """Test __repr__ method."""
        assert repr(Controlled(qml.S(0), [1])) == "Controlled(S(wires=[0]), control_wires=[1])"

        base = qml.S(0) + qml.T(1)
        op = Controlled(base, [2])
        assert repr(op) == "Controlled(S(wires=[0]) + T(wires=[1]), control_wires=[2])"

        op = Controlled(base, [2, 3], control_values=[True, False], work_wires=[4])
        assert (
            repr(op)
            == "Controlled(S(wires=[0]) + T(wires=[1]), control_wires=[2, 3], work_wires=[4], control_values=[True, False])"
        )

    def test_flatten_unflatten(self):
        """Tests the _flatten and _unflatten methods."""
        target = qml.S(0)
        control_wires = qml.wires.Wires((1, 2))
        control_values = (0, 0)
        work_wires = qml.wires.Wires(3)

        op = Controlled(target, control_wires, control_values=control_values, work_wires=work_wires)

        data, metadata = op._flatten()
        assert data[0] is target
        assert len(data) == 1

        assert metadata == (control_wires, control_values, work_wires)

        # make sure metadata is hashable
        assert hash(metadata)

        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)
        assert new_op._name == "C(S)"  # make sure initialization was called

    def test_copy(self):
        """Test that a copy of a controlled oeprator can have its parameters updated
        independently of the original operator."""

        param1 = 1.234
        base_wire = "a"
        control_wires = [0, 1]
        base = qml.RX(param1, base_wire)
        op = Controlled(base, control_wires, control_values=[0, 1])

        copied_op = copy(op)

        assert copied_op.__class__ is op.__class__
        assert copied_op.control_wires == op.control_wires
        assert copied_op.control_values == op.control_values
        assert copied_op.data == (param1,)

        copied_op.data = (6.54,)
        assert op.data == (param1,)

    def test_label(self):
        """Test that the label method defers to the label of the base."""
        base = qml.U1(1.23, wires=0)
        op = Controlled(base, "a")

        assert op.label() == base.label()
        assert op.label(decimals=2) == base.label(decimals=2)
        assert op.label(base_label="hi") == base.label(base_label="hi")

    def test_label_matrix_param(self):
        """Test that the label method simply returns the label of the base and updates the cache."""
        U = np.eye(2)
        base = qml.QubitUnitary(U, wires=0)
        op = Controlled(base, ["a", "b"])

        cache = {"matrices": []}
        assert op.label(cache=cache) == base.label(cache=cache)
        assert cache["matrices"] == [U]

    def test_eigvals(self):
        """Test the eigenvalues against the matrix eigenvalues."""
        base = qml.IsingXX(1.234, wires=(0, 1))
        op = Controlled(base, (2, 3))

        mat = op.matrix()
        mat_eigvals = np.sort(qml.math.linalg.eigvals(mat))

        eigs = op.eigvals()
        sort_eigs = np.sort(eigs)

        assert qml.math.allclose(mat_eigvals, sort_eigs)

    def test_has_generator_true(self):
        """Test `has_generator` property carries over when base op defines generator."""
        base = qml.RX(0.5, 0)
        op = Controlled(base, ("b", "c"))

        assert op.has_generator is True

    def test_has_generator_false(self):
        """Test `has_generator` property carries over when base op does not define a generator."""
        base = qml.PauliX(0)
        op = Controlled(base, ("b", "c"))

        assert op.has_generator is False

    def test_generator(self):
        """Test that the generator is a tensor product of projectors and the base's generator."""

        base = qml.RZ(-0.123, wires="a")
        op = Controlled(base, ("b", "c"))

        base_gen, base_gen_coeff = qml.generator(base, format="prefactor")
        gen_tensor, gen_coeff = qml.generator(op, format="prefactor")

        assert base_gen_coeff == gen_coeff

        for wire, ob in zip(op.control_wires, gen_tensor.operands):
            assert isinstance(ob, qml.Projector)
            assert ob.data == ([1],)
            assert ob.wires == qml.wires.Wires(wire)

        assert gen_tensor.operands[-1].__class__ is base_gen.__class__
        assert gen_tensor.operands[-1].wires == base_gen.wires

    def test_diagonalizing_gates(self):
        """Test that the Controlled diagonalizing gates is the same as the base diagonalizing gates."""
        base = qml.PauliX(0)
        op = Controlled(base, (1, 2))

        op_gates = op.diagonalizing_gates()
        base_gates = base.diagonalizing_gates()

        assert len(op_gates) == len(base_gates)

        for op1, op2 in zip(op_gates, base_gates):
            assert op1.__class__ is op2.__class__
            assert op1.wires == op2.wires

    def test_hash(self):
        """Test that op.hash uniquely describes an op up to work wires."""

        base = qml.RY(1.2, wires=0)
        # different control wires
        op1 = Controlled(base, (1, 2), [0, 1])
        op2 = Controlled(base, (2, 1), [0, 1])
        assert op1.hash != op2.hash

        # different control values
        op3 = Controlled(base, (1, 2), [1, 0])
        assert op1.hash != op3.hash
        assert op2.hash != op3.hash

        # all variations on default control_values
        op4 = Controlled(base, (1, 2))
        op5 = Controlled(base, (1, 2), [True, True])
        op6 = Controlled(base, (1, 2), [1, 1])
        assert op4.hash == op5.hash
        assert op4.hash == op6.hash

        # work wires
        op7 = Controlled(base, (1, 2), [0, 1], work_wires="aux")
        assert op7.hash != op1.hash


class TestOperationProperties:
    """Test ControlledOp specific properties."""

    # pylint:disable=no-member

    @pytest.mark.parametrize("gm", (None, "A", "F"))
    def test_grad_method(self, gm):
        """Check grad_method defers to that of the base operation."""

        class DummyOp(Operation):
            num_wires = 1
            grad_method = gm

        base = DummyOp(1)
        op = Controlled(base, 2)
        assert op.grad_method == gm

    def test_basis(self):
        """Test that controlled mimics the basis attribute of the base op."""

        class DummyOp(Operation):
            num_wires = 1
            basis = "Z"

        base = DummyOp(1)
        op = Controlled(base, 2)
        assert op.basis == "Z"

    @pytest.mark.parametrize(
        "base, expected",
        [
            (qml.RX(1.23, wires=0), [(0.5, 1.0)]),
            (qml.PhaseShift(-2.4, wires=0), [(1,)]),
            (qml.IsingZZ(-9.87, (0, 1)), [(0.5, 1.0)]),
        ],
    )
    def test_parameter_frequencies(self, base, expected):
        """Test parameter-frequencies against expected values."""

        op = Controlled(base, (3, 4))
        assert op.parameter_frequencies == expected

    def test_parameter_frequencies_no_generator_error(self):
        """An error should be raised if the base doesn't have a generator."""
        base = TempOperation(1.234, 1)
        op = Controlled(base, 2)

        with pytest.raises(
            qml.operation.ParameterFrequenciesUndefinedError,
            match=r"does not have parameter frequencies",
        ):
            op.parameter_frequencies

    def test_parameter_frequencies_multiple_params_error(self):
        """An error should be raised if the base has more than one parameter."""
        base = TempOperation(1.23, 2.234, 1)
        op = Controlled(base, (2, 3))

        with pytest.raises(
            qml.operation.ParameterFrequenciesUndefinedError,
            match=r"does not have parameter frequencies",
        ):
            op.parameter_frequencies


class TestSimplify:
    """Test qml.sum simplify method and depth property."""

    def test_depth_property(self):
        """Test depth property."""
        controlled_op = Controlled(qml.RZ(1.32, wires=0) + qml.Identity(wires=0), control_wires=1)
        assert controlled_op.arithmetic_depth == 2

    def test_simplify_method(self):
        """Test that the simplify method reduces complexity to the minimum."""
        controlled_op = Controlled(
            qml.RZ(1.32, wires=0) + qml.Identity(wires=0) + qml.RX(1.9, wires=1), control_wires=2
        )
        final_op = Controlled(
            qml.sum(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1)),
            control_wires=2,
        )
        simplified_op = controlled_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Controlled)
        for s1, s2 in zip(final_op.base.operands, simplified_op.base.operands):
            assert s1.name == s2.name
            assert s1.wires == s2.wires
            assert s1.data == s2.data
            assert s1.arithmetic_depth == s2.arithmetic_depth

    def test_simplify_nested_controlled_ops(self):
        """Test the simplify method with nested control operations on different wires."""
        controlled_op = Controlled(Controlled(qml.Hadamard(0), 1), 2)
        final_op = Controlled(qml.Hadamard(0), [2, 1])
        simplified_op = controlled_op.simplify()

        # TODO: Use qml.equal when supported for nested operators

        assert isinstance(simplified_op, Controlled)
        assert isinstance(simplified_op.base, qml.Hadamard)
        assert simplified_op.name == final_op.name
        assert simplified_op.wires == final_op.wires
        assert simplified_op.data == final_op.data
        assert simplified_op.arithmetic_depth == final_op.arithmetic_depth


class TestQueuing:
    """Test that Controlled operators queue and update base metadata."""

    def test_queuing(self):
        """Test that `Controlled` is queued upon initialization and updates base metadata."""
        with qml.queuing.AnnotatedQueue() as q:
            base = qml.Rot(1.234, 2.345, 3.456, wires=2)
            op = Controlled(base, (0, 1))

        assert base not in q
        assert qml.equal(q.queue[0], op)

    def test_queuing_base_defined_outside(self):
        """Test that base isn't added to queue if its defined outside the recording context."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        with qml.queuing.AnnotatedQueue() as q:
            op = Controlled(base, ("a", "b"))

        assert len(q) == 1
        assert q.queue[0] is op


base_num_control_mats = [
    (qml.PauliX("a"), 1, CNOT),
    (qml.PauliZ("a"), 1, CZ),
    (qml.SWAP(("a", "b")), 1, CSWAP),
    (qml.PauliX("a"), 2, Toffoli),
    (qml.RX(1.234, "b"), 1, CRotx(1.234)),
    (qml.RY(-0.432, "a"), 1, CRoty(-0.432)),
    (qml.RZ(6.78, "a"), 1, CRotz(6.78)),
    (qml.Rot(1.234, -0.432, 9.0, "a"), 1, CRot3(1.234, -0.432, 9.0)),
]


class TestMatrix:
    """Tests of Controlled.matrix and Controlled.sparse_matrix"""

    def test_correct_matrix_dimenions_with_batching(self):
        """Test batching returns a matrix of the correct dimensions"""
        x = np.array([1.0, 2.0, 3.0])
        base = qml.RX(x, 0)
        op = Controlled(base, 1)

        matrix = op.matrix()

        assert matrix.shape == (3, 4, 4)

    @pytest.mark.parametrize("base, num_control, mat", base_num_control_mats)
    def test_matrix_compare_with_gate_data(self, base, num_control, mat):
        """Test the matrix against matrices provided by `gate_data` file."""
        op = Controlled(base, list(range(num_control)))
        assert qml.math.allclose(op.matrix(), mat)

    def test_aux_wires_included(self):
        """Test that matrix expands to have identity on work wires."""

        base = qml.PauliX(1)
        op = Controlled(
            base,
            0,
            work_wires="aux",
        )
        mat = op.matrix()
        assert mat.shape == (8, 8)

    def test_wire_order(self):
        """Test that the ``wire_order`` keyword argument alters the matrix as expected."""
        base = qml.RX(-4.432, wires=1)
        op = Controlled(base, 0)

        method_order = op.matrix(wire_order=(1, 0))
        function_order = qml.math.expand_matrix(op.matrix(), op.wires, (1, 0))

        assert qml.math.allclose(method_order, function_order)

    @pytest.mark.parametrize("control_values", ([0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]))
    def test_control_values(self, control_values):
        """Test that the matrix with specified control_values is the same as using PauliX flips
        to reverse the control values."""
        control_wires = (0, 1, 2)

        base = qml.RX(3.456, wires=3)
        op = Controlled(base, control_wires, control_values=control_values)

        mat = op.matrix()
        with qml.queuing.AnnotatedQueue() as q:
            [qml.PauliX(w) for w, val in zip(control_wires, control_values) if not val]
            Controlled(base, control_wires, control_values=[1, 1, 1])
            [qml.PauliX(w) for w, val in zip(control_wires, control_values) if not val]
        tape = qml.tape.QuantumScript.from_queue(q)
        decomp_mat = qml.matrix(tape, wire_order=op.wires)

        assert qml.math.allclose(mat, decomp_mat)

    def test_sparse_matrix_base_defines(self):
        """Check that an op that defines a sparse matrix has it used in the controlled
        sparse matrix."""

        Hmat = (1.0 * qml.PauliX(0)).sparse_matrix()
        H_sparse = qml.SparseHamiltonian(Hmat, wires="0")
        op = Controlled(H_sparse, "a")

        sparse_mat = op.sparse_matrix()
        assert isinstance(sparse_mat, sparse.csr_matrix)
        assert qml.math.allclose(sparse_mat.toarray(), op.matrix())

    @pytest.mark.parametrize("control_values", ([0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]))
    def test_sparse_matrix_only_matrix_defined(self, control_values):
        """Check that an base doesn't define a sparse matrix but defines a dense matrix
        still provides a controlled sparse matrix."""
        control_wires = (0, 1, 2)
        base = qml.U2(1.234, -3.2, wires=3)
        op = Controlled(base, control_wires, control_values=control_values)

        sparse_mat = op.sparse_matrix()
        assert isinstance(sparse_mat, sparse.csr_matrix)
        assert qml.math.allclose(op.sparse_matrix().toarray(), op.matrix())

    def test_sparse_matrix_wire_order_error(self):
        """Check a NonImplementedError is raised if the user requests specific wire order."""
        control_wires = (0, 1, 2)
        base = qml.U2(1.234, -3.2, wires=3)
        op = Controlled(base, control_wires)

        with pytest.raises(NotImplementedError):
            op.sparse_matrix(wire_order=[3, 2, 1, 0])

    def test_no_matrix_defined_sparse_matrix_error(self):
        """Check that if the base gate defines neither a sparse matrix nor a dense matrix, a
        SparseMatrixUndefined error is raised."""

        base = TempOperator(1)
        op = Controlled(base, 2)

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_sparse_matrix_format(self):
        """Test format keyword determines output type of sparse matrix."""
        base = qml.PauliX(0)
        op = Controlled(base, 1)

        lil_mat = op.sparse_matrix(format="lil")
        assert isinstance(lil_mat, sparse.lil_matrix)


class TestHelperMethod:
    """Unittests for the _decompose_no_control_values helper function."""

    def test_crx(self):
        """Test case with single control wire and defined _controlled"""
        base = qml.RX(1.0, wires=0)
        op = Controlled(base, 1)
        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 1
        assert qml.equal(decomp[0], qml.CRX(1.0, wires=(1, 0)))

    def test_toffoli(self):
        """Test case when PauliX with two controls."""
        op = Controlled(qml.PauliX("c"), ("a", 2))
        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 1
        assert equal_list(decomp, qml.MultiControlledX(wires=("a", 2, "c")))

    def test_multicontrolledx(self):
        """Test case when PauliX has many controls."""
        op = Controlled(qml.PauliX(4), (0, 1, 2, 3))
        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 1
        assert qml.equal(decomp[0], qml.MultiControlledX(wires=(0, 1, 2, 3, 4)))

    def test_decomposes_target(self):
        """Test that we decompose the target if we don't have a special case."""
        target = qml.IsingXX(1.0, wires=(0, 1))
        op = Controlled(target, (3, 4))

        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 3

        target_decomp = target.expand().circuit
        for op1, target in zip(decomp, target_decomp):
            assert isinstance(op1, Controlled)
            assert op1.control_wires == (3, 4)

            assert qml.equal(op1.base, target)

    def test_None_default(self):
        """Test that helper returns None if no special decomposition."""
        op = Controlled(TempOperator(0), (1, 2))
        assert _decompose_no_control_values(op) is None


@pytest.mark.parametrize("test_expand", (False, True))
class TestDecomposition:
    """Test controlled's decomposition method."""

    def test_control_values_no_special_decomp(self, test_expand):
        """Test decomposition applies PauliX gates to flip any control-on-zero wires."""

        control_wires = (0, 1, 2)
        control_values = [True, False, False]

        base = TempOperator("a")
        op = Controlled(base, control_wires, control_values)

        decomp = op.expand().circuit if test_expand else op.decomposition()

        assert qml.equal(decomp[0], qml.PauliX(1))
        assert qml.equal(decomp[1], qml.PauliX(2))

        assert isinstance(decomp[2], Controlled)
        assert decomp[2].control_values == [True, True, True]

        assert qml.equal(decomp[3], qml.PauliX(1))
        assert qml.equal(decomp[4], qml.PauliX(2))

    def test_control_values_special_decomp(self, test_expand):
        """Test decomposition when needs control_values flips and special decomp exists."""

        base = qml.PauliX(2)
        op = Controlled(base, (0, 1), (True, False))

        decomp = op.expand().circuit if test_expand else op.decomposition()
        expected = [qml.PauliX(1), qml.MultiControlledX(wires=(0, 1, 2)), qml.PauliX(1)]
        assert equal_list(decomp, expected)

    def test_no_control_values_special_decomp(self, test_expand):
        """Test a case with no control values but a special decomposition."""
        base = qml.RX(1.0, 2)
        op = Controlled(base, 1)
        decomp = op.expand().circuit if test_expand else op.decomposition()
        assert len(decomp) == 1
        assert qml.equal(decomp[0], qml.CRX(1.0, (1, 2)))

    def test_no_control_values_target_decomposition(self, test_expand):
        """Tests a case with no control values and no special decomposition but
        the ability to decompose the target."""
        base = qml.IsingXX(1.23, wires=(0, 1))
        op = Controlled(base, "a")

        decomp = op.expand().circuit if test_expand else op.decomposition()
        base_decomp = base.decomposition()
        for cop, base_op in zip(decomp, base_decomp):
            assert isinstance(cop, Controlled)
            assert qml.equal(cop.base, base_op)

    def test_no_control_values_no_special_decomp(self, test_expand):
        """Test if all control_values are true and no special decomposition exists,
        the method raises a DecompositionUndefinedError."""

        base = TempOperator("a")
        op = Controlled(base, (0, 1, 2))

        with pytest.raises(DecompositionUndefinedError):
            # pylint: disable=unused-variable
            decomp = op.expand().circuit if test_expand else op.decomposition()


class TestArithmetic:
    """Test arithmetic decomposition methods."""

    control_wires = qml.wires.Wires((3, 4))
    work_wires = qml.wires.Wires("aux")
    control_values = [True, False]

    def test_adjoint(self):
        """Test the adjoint method for Controlled Operators."""

        class DummyOp(Operator):
            num_wires = 1

            def adjoint(self):
                return DummyOp("adjointed", self.wires)

        base = DummyOp("basic", 2)
        op = Controlled(base, self.control_wires, self.control_values, self.work_wires)

        adj_op = op.adjoint()
        assert isinstance(adj_op, Controlled)
        assert adj_op.base.parameters == ["adjointed"]

        assert adj_op.control_wires == self.control_wires
        assert adj_op.control_values == self.control_values
        assert adj_op.work_wires == self.work_wires

    @pytest.mark.parametrize("z", (2, -1, 0.5))
    def test_pow(self, z):
        """Test the pow method for Controlled Operators."""

        class DummyOp(Operator):
            num_wires = 1

            def pow(self, z):
                return [DummyOp(z, self.wires)]

        base = DummyOp(wires=0)
        op = Controlled(base, self.control_wires, self.control_values, self.work_wires)

        pow_op = op.pow(z)[0]
        assert isinstance(pow_op, Controlled)
        assert pow_op.base.parameters == [z]

        assert pow_op.control_wires == self.control_wires
        assert pow_op.control_values == self.control_values
        assert pow_op.work_wires == self.work_wires


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
class TestDifferentiation:
    """Tests for differentiation"""

    @pytest.mark.autograd
    def test_autograd(self, diff_method):
        """Test differentiation using autograd"""

        dev = qml.device("default.qubit", wires=2)
        init_state = np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = np.array(0.123, requires_grad=True)
        res = qml.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.torch
    def test_torch(self, diff_method):
        """Test differentiation using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)
        init_state = torch.tensor(
            [1.0, -1.0], requires_grad=False, dtype=torch.complex128
        ) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = torch.tensor(0.123, requires_grad=True, dtype=torch.float64)
        loss = circuit(b)
        loss.backward()  # pylint:disable=no-member

        res = b.grad.detach()
        expected = np.sin(b.detach() / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["auto", "jax", "jax-python"])
    def test_jax(self, diff_method, jax_interface):
        """Test differentiation using JAX"""

        import jax

        jax.config.update("jax_enable_x64", True)

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method, interface=jax_interface)
        def circuit(b):
            init_state = onp.array([1.0, -1.0]) / np.sqrt(2)
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = jnp.array(0.123)
        res = jax.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.tf
    def test_tf(self, diff_method):
        """Test differentiation using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        init_state = tf.constant([1.0, -1.0], dtype=tf.complex128) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = tf.Variable(0.123, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = circuit(b)

        res = tape.gradient(loss, b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)


class TestControlledSupportsBroadcasting:
    """Test that the Controlled version of qubit operations with the ``supports_broadcasting`` attribute
    actually support broadcasting."""

    single_scalar_single_wire_ops = [
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "U1",
    ]

    single_scalar_multi_wire_ops = [
        "ControlledPhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "OrbitalRotation",
        "FermionicSWAP",
    ]

    two_scalar_single_wire_ops = [
        "U2",
    ]

    three_scalar_single_wire_ops = [
        "Rot",
        "U3",
    ]

    three_scalar_multi_wire_ops = [
        "CRot",
    ]

    # When adding an operation to the following list, you
    # actually need to write a new test!
    separately_tested_ops = [
        "QubitUnitary",
        "ControlledQubitUnitary",
        "DiagonalQubitUnitary",
        "PauliRot",
        "MultiRZ",
        "StatePrep",
        "AmplitudeEmbedding",
        "AngleEmbedding",
        "IQPEmbedding",
        "QAOAEmbedding",
    ]

    @pytest.mark.parametrize("name", single_scalar_single_wire_ops)
    def test_controlled_of_single_scalar_single_wire_ops(self, name):
        """Test that a Controlled operation whose base is a single-scalar-parameter operations
        on a single wire marked as supporting parameter broadcasting actually do support broadcasting.
        """
        par = np.array([0.25, 2.1, -0.42])
        wires = ["wire0"]

        cls = getattr(qml, name)
        base = cls(par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_mats = [Controlled(cls(p, wires=wires), "wire1").matrix() for p in par]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize("name", single_scalar_multi_wire_ops)
    def test_controlled_single_scalar_multi_wire_ops(self, name):
        """Test that a Controlled operation whose base is a single-scalar-parameter operations
        on multiple wires marked as supporting parameter broadcasting actually do support broadcasting.
        """
        par = np.array([0.25, 2.1, -0.42])
        cls = getattr(qml, name)

        # Provide up to 6 wires and take as many as the class requires
        # This assumes that the class does *not* have `num_wires=qml.operation.AnyWires`
        wires = ["wire0", 5, 41, "aux_wire", -1, 9][: cls.num_wires]
        base = cls(par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_mats = [Controlled(cls(p, wires=wires), "wire1").matrix() for p in par]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize("name", two_scalar_single_wire_ops)
    def test_controlled_two_scalar_single_wire_ops(self, name):
        """Test that a Controlled operation whose base is a two-scalar-parameter operations
        on a single wire marked as supporting parameter broadcasting actually do support broadcasting.
        """
        par = (np.array([0.25, 2.1, -0.42]), np.array([-6.2, 0.12, 0.421]))
        wires = ["wire0"]

        cls = getattr(qml, name)
        base = cls(*par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [Controlled(cls(*p, wires=wires), "wire1").matrix() for p in single_pars]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize("name", three_scalar_single_wire_ops)
    def test_controlled_three_scalar_single_wire_ops(self, name):
        """Test that a Controlled operation whose base is a three-scalar-parameter operations
        on a single wire marked as supporting parameter broadcasting actually do support broadcasting.
        """
        par = (
            np.array([0.25, 2.1, -0.42]),
            np.array([-6.2, 0.12, 0.421]),
            np.array([0.2, 1.1, -5.2]),
        )
        wires = ["wire0"]

        cls = getattr(qml, name)
        base = cls(*par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [Controlled(cls(*p, wires=wires), "wire1").matrix() for p in single_pars]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize("name", three_scalar_multi_wire_ops)
    def test_controlled_three_scalar_multi_wire_ops(self, name):
        """Test that a Controlled operation whose base is a three-scalar-parameter operations
        on multiple wires marked as supporting parameter broadcasting actually do support broadcasting.
        """
        par = (
            np.array([0.25, 2.1, -0.42]),
            np.array([-6.2, 0.12, 0.421]),
            np.array([0.2, 1.1, -5.2]),
        )
        wires = ["wire0", 214]

        cls = getattr(qml, name)
        base = cls(*par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [Controlled(cls(*p, wires=wires), "wire1").matrix() for p in single_pars]

        assert qml.math.allclose(mat, single_mats)

    def test_controlled_diagonal_qubit_unitary(self):
        """Test that a Controlled operation whose base is a DiagonalQubitUnitary, which is marked
        as supporting parameter broadcasting, actually does support broadcasting."""
        diag = np.array([[1j, 1, 1, -1j], [-1j, 1j, 1, -1], [1j, -1j, 1.0, -1]])
        wires = ["a", 5]

        base = qml.DiagonalQubitUnitary(diag, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_mats = [
            Controlled(qml.DiagonalQubitUnitary(d, wires=wires), "wire1").matrix() for d in diag
        ]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize(
        "pauli_word, wires", [("XYZ", [0, "4", 1]), ("II", [1, 5]), ("X", [7])]
    )
    def test_controlled_pauli_rot(self, pauli_word, wires):
        """Test that a Controlled operation whose base is PauliRot, which is marked as supporting
        parameter broadcasting, actually does support broadcasting."""
        par = np.array([0.25, 2.1, -0.42])

        base = qml.PauliRot(par, pauli_word, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_mats = [
            Controlled(qml.PauliRot(p, pauli_word, wires=wires), "wire1").matrix() for p in par
        ]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize("wires", [[0, "4", 1], [1, 5], [7]])
    def test_controlled_multi_rz(self, wires):
        """Test that a Controlled operation whose base is MultiRZ, which is marked as supporting
        parameter broadcasting, actually does support broadcasting."""
        par = np.array([0.25, 2.1, -0.42])

        base = qml.MultiRZ(par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_mats = [Controlled(qml.MultiRZ(p, wires=wires), "wire1").matrix() for p in par]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize(
        "state_, num_wires",
        [([1.0, 0.0], 1), ([0.5, -0.5j, 0.5, -0.5], 2), (np.ones(8) / np.sqrt(8), 3)],
    )
    def test_controlled_qubit_state_vector(self, state_, num_wires):
        """Test that StatePrep, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        state = np.array([state_])
        base = qml.StatePrep(state, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")

        assert op.batch_size == 1
        qml.StatePrep.compute_decomposition(state, list(range(num_wires)))
        op.decomposition()

        state = np.array([state_] * 3)
        base = qml.StatePrep(state, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 3
        qml.StatePrep.compute_decomposition(state, list(range(num_wires)))
        op.decomposition()

    @pytest.mark.parametrize(
        "state, num_wires",
        [([1.0, 0.0], 1), ([0.5, -0.5j, 0.5, -0.5], 2), (np.ones(8) / np.sqrt(8), 3)],
    )
    def test_controlled_amplitude_embedding(self, state, num_wires):
        """Test that AmplitudeEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        features = np.array([state])
        base = qml.AmplitudeEmbedding(features, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 1
        qml.AmplitudeEmbedding.compute_decomposition(features, list(range(num_wires)))
        op.decomposition()

        features = np.array([state] * 3)
        base = qml.AmplitudeEmbedding(features, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 3
        qml.AmplitudeEmbedding.compute_decomposition(features, list(range(num_wires)))
        op.decomposition()

    @pytest.mark.parametrize(
        "angles, num_wires",
        [
            (np.array([[0.5], [2.1]]), 1),
            (np.array([[0.5, -0.5], [0.2, 1.5]]), 2),
            (np.ones((2, 5)), 5),
        ],
    )
    def test_controlled_angle_embedding(self, angles, num_wires):
        """Test that AngleEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        base = qml.AngleEmbedding(angles, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 2
        qml.AngleEmbedding.compute_decomposition(angles, list(range(num_wires)), rotation=qml.RX)
        op.decomposition()

    @pytest.mark.parametrize(
        "features, num_wires",
        [
            (np.array([[0.5], [2.1]]), 1),
            (np.array([[0.5, -0.5], [0.2, 1.5]]), 2),
            (np.ones((2, 5)), 5),
        ],
    )
    def test_controlled_iqp_embedding(self, features, num_wires):
        """Test that IQPEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        base = qml.IQPEmbedding(features, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 2
        qml.IQPEmbedding.compute_decomposition(
            features,
            list(range(num_wires)),
            n_repeats=2,
            pattern=op.base.hyperparameters["pattern"],
        )
        op.decomposition()

    @pytest.mark.parametrize(
        "features, weights, num_wires, batch_size",
        [
            (np.array([[0.5], [2.1]]), np.array([[0.61], [0.3]]), 1, 2),
            (np.array([[0.5, -0.5], [0.2, 1.5]]), np.ones((2, 4, 3)), 2, 2),
            (np.array([0.5, -0.5, 0.2]), np.ones((3, 2, 6)), 3, 3),
        ],
    )
    def test_controlled_qaoa_embedding(self, features, weights, num_wires, batch_size):
        """Test that QAOAEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        base = qml.QAOAEmbedding(features, weights, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == batch_size
        qml.QAOAEmbedding.compute_decomposition(
            features, weights, wires=list(range(num_wires)), local_field=qml.RY
        )
        op.decomposition()


##### TESTS FOR THE ctrl TRANSFORM #####


def test_invalid_input_error():
    """Test that a ValueError is raised upon invalid inputs."""
    err = r"The object 1 of type <class 'int'> is not an Operator or callable."
    with pytest.raises(ValueError, match=err):
        qml.ctrl(1, control=2)


def test_ctrl_sanity_check():
    """Test that control works on a very standard usecase."""

    def make_ops():
        qml.RX(0.123, wires=0)
        qml.RY(0.456, wires=2)
        qml.RX(0.789, wires=0)
        qml.Rot(0.111, 0.222, 0.333, wires=2)
        qml.PauliX(wires=2)
        qml.PauliY(wires=4)
        qml.PauliZ(wires=0)

    with qml.queuing.AnnotatedQueue() as q_tape:
        cmake_ops = ctrl(make_ops, control=1)
        # Execute controlled version.
        cmake_ops()

    tape = QuantumScript.from_queue(q_tape)
    expanded_tape = tape.expand()

    expected = [
        qml.CRX(0.123, wires=[1, 0]),
        qml.CRY(0.456, wires=[1, 2]),
        qml.CRX(0.789, wires=[1, 0]),
        qml.CRot(0.111, 0.222, 0.333, wires=[1, 2]),
        qml.CNOT(wires=[1, 2]),
        *qml.CY(wires=[1, 4]).decomposition(),
        *qml.CZ(wires=[1, 0]).decomposition(),
    ]
    assert len(tape.operations) == 7
    for op1, op2 in zip(expanded_tape, expected):
        assert qml.equal(op1, op2)


def test_adjoint_of_ctrl():
    """Test adjoint(ctrl(fn)) and ctrl(adjoint(fn))"""

    def my_op(a, b, c):
        qml.RX(a, wires=2)
        qml.RY(b, wires=3)
        qml.RZ(c, wires=0)

    with qml.queuing.AnnotatedQueue() as q1:
        cmy_op_dagger = qml.simplify(qml.adjoint(ctrl(my_op, 5)))
        # Execute controlled and adjointed version of my_op.
        cmy_op_dagger(0.789, 0.123, c=0.456)

    tape1 = QuantumScript.from_queue(q1)
    with qml.queuing.AnnotatedQueue() as q2:
        cmy_op_dagger = qml.simplify(ctrl(qml.adjoint(my_op), 5))
        # Execute adjointed and controlled version of my_op.
        cmy_op_dagger(0.789, 0.123, c=0.456)

    tape2 = QuantumScript.from_queue(q2)
    expected = [
        qml.CRZ(4 * onp.pi - 0.456, wires=[5, 0]),
        qml.CRY(4 * onp.pi - 0.123, wires=[5, 3]),
        qml.CRX(4 * onp.pi - 0.789, wires=[5, 2]),
    ]
    for tape in [tape1.expand(depth=1), tape2.expand(depth=1)]:
        for op1, op2 in zip(tape, expected):
            assert qml.equal(op1, op2)


def test_nested_ctrl():
    """Test nested use of control"""
    with qml.queuing.AnnotatedQueue() as q_tape:
        CCS = ctrl(ctrl(qml.S, 7), 3)
        CCS(wires=0)
    tape = QuantumScript.from_queue(q_tape)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, Controlled)
    new_tape = tape.expand(depth=2)
    assert qml.equal(new_tape[0], Controlled(qml.ControlledPhaseShift(np.pi / 2, [7, 0]), [3]))


def test_multi_ctrl():
    """Test control with a list of wires."""
    with qml.queuing.AnnotatedQueue() as q_tape:
        CCS = ctrl(qml.S, control=[3, 7])
        CCS(wires=0)
    tape = QuantumScript.from_queue(q_tape)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, Controlled)
    new_tape = tape.expand(depth=1)
    assert qml.equal(new_tape[0], Controlled(qml.PhaseShift(np.pi / 2, 0), [3, 7]))


def test_ctrl_with_qnode():
    """Test ctrl works when in a qnode cotext."""
    dev = qml.device("default.qubit", wires=3)

    def my_ansatz(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(params[2], wires=1)
        qml.RX(params[3], wires=0)
        qml.CNOT(wires=[1, 0])

    def controlled_ansatz(params):
        qml.CRY(params[0], wires=[2, 0])
        qml.CRY(params[1], wires=[2, 1])
        qml.Toffoli(wires=[2, 0, 1])
        qml.CRX(params[2], wires=[2, 1])
        qml.CRX(params[3], wires=[2, 0])
        qml.Toffoli(wires=[2, 1, 0])

    def circuit(ansatz, params):
        qml.RX(np.pi / 4.0, wires=2)
        ansatz(params)
        return qml.state()

    params = [0.123, 0.456, 0.789, 1.345]
    circuit1 = qml.qnode(dev)(partial(circuit, ansatz=ctrl(my_ansatz, 2)))
    circuit2 = qml.qnode(dev)(partial(circuit, ansatz=controlled_ansatz))
    res1 = circuit1(params=params)
    res2 = circuit2(params=params)
    assert qml.math.allclose(res1, res2)


def test_ctrl_within_ctrl():
    """Test using ctrl on a method that uses ctrl."""

    def ansatz(params):
        qml.RX(params[0], wires=0)
        ctrl(qml.PauliX, control=0)(wires=1)
        qml.RX(params[1], wires=0)

    controlled_ansatz = ctrl(ansatz, 2)

    with qml.queuing.AnnotatedQueue() as q_tape:
        controlled_ansatz([0.123, 0.456])

    tape = QuantumScript.from_queue(q_tape)
    tape = tape.expand(2, stop_at=lambda op: not isinstance(op, Controlled))

    expected = [
        qml.CRX(0.123, wires=[2, 0]),
        qml.Toffoli(wires=[2, 0, 1]),
        qml.CRX(0.456, wires=[2, 0]),
    ]
    for op1, op2 in zip(tape, expected):
        assert qml.equal(op1, op2)


def test_diagonal_ctrl():
    """Test ctrl on diagonal gates."""
    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(qml.DiagonalQubitUnitary, 1)(onp.array([-1.0, 1.0j]), wires=0)
    tape = QuantumScript.from_queue(q_tape)
    tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))
    assert qml.equal(
        tape[0], qml.DiagonalQubitUnitary(onp.array([1.0, 1.0, -1.0, 1.0j]), wires=[1, 0])
    )


@pytest.mark.parametrize(
    "M",
    [
        qml.PauliX.compute_matrix(),
        qml.PauliY.compute_matrix(),
        qml.PauliZ.compute_matrix(),
        qml.Hadamard.compute_matrix(),
        np.array(
            [
                [1 + 2j, -3 + 4j],
                [3 + 4j, 1 - 2j],
            ]
        )
        * 30 ** -0.5,
    ],
)
def test_qubit_unitary(M):
    """Test ctrl on QubitUnitary and ControlledQubitUnitary"""
    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(qml.QubitUnitary, 1)(M, wires=0)

    tape = QuantumScript.from_queue(q_tape)
    expected = qml.ControlledQubitUnitary(M, control_wires=1, wires=0)
    assert equal_list(list(tape), expected)

    # causes decomposition into more basic operators
    tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))
    assert not equal_list(list(tape), expected)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "M",
    [
        qml.PauliX.compute_matrix(),
        qml.PauliY.compute_matrix(),
        qml.PauliZ.compute_matrix(),
        qml.Hadamard.compute_matrix(),
        np.array(
            [
                [1 + 2j, -3 + 4j],
                [3 + 4j, 1 - 2j],
            ]
        )
        * 30 ** -0.5,
    ],
)
def test_controlledqubitunitary(M):
    """Test ctrl on ControlledQubitUnitary."""
    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(qml.ControlledQubitUnitary, 1)(M, control_wires=2, wires=0)

    tape = QuantumScript.from_queue(q_tape)
    # will immediately decompose according to selected decomposition algorithm
    tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))

    expected = qml.ControlledQubitUnitary(M, control_wires=[2, 1], wires=0).decomposition()
    assert equal_list(list(tape), expected)


def test_no_control_defined():
    """Test a custom operation with no control transform defined."""
    # QFT has no control rule defined.
    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(qml.templates.QFT, 2)(wires=[0, 1])
    tape = QuantumScript.from_queue(q_tape)
    tape = tape.expand(depth=3, stop_at=lambda op: not isinstance(op, Controlled))
    assert len(tape.operations) == 8
    # Check that all operations are updated to their controlled version.
    for op in tape.operations:
        assert type(op) in {qml.ControlledPhaseShift, qml.Toffoli, qml.CRX, qml.CSWAP, qml.CH}


def test_decomposition_defined():
    """Test that a controlled gate that has no control transform defined,
    and a decomposition transformed defined, still works correctly"""

    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(qml.CY, 0)(wires=[1, 2])

    tape = QuantumScript.from_queue(q_tape)
    tape = tape.expand()

    assert len(tape.operations) == 2

    assert tape.operations[0].name == "C(CRY)"
    assert tape.operations[1].name == "C(S)"


def test_ctrl_template():
    """Test that a controlled template correctly expands
    on a device that doesn't support it"""

    weights = np.ones([3, 2])

    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(qml.templates.BasicEntanglerLayers, 0)(weights, wires=[1, 2])

    tape = QuantumScript.from_queue(q_tape)
    tape = expand_tape(tape, depth=2)
    assert len(tape) == 9
    assert all(o.name in {"CRX", "Toffoli"} for o in tape.operations)


def test_ctrl_template_and_operations():
    """Test that a combination of controlled templates and operations correctly expands
    on a device that doesn't support it"""

    weights = np.ones([3, 2])

    def ansatz(weights, wires):
        qml.PauliX(wires=wires[0])
        qml.templates.BasicEntanglerLayers(weights, wires=wires)

    with qml.queuing.AnnotatedQueue() as q_tape:
        ctrl(ansatz, 0)(weights, wires=[1, 2])

    tape = QuantumScript.from_queue(q_tape)
    tape = tape.expand(depth=2, stop_at=lambda obj: not isinstance(obj, Controlled))
    assert len(tape.operations) == 10
    assert all(o.name in {"CNOT", "CRX", "Toffoli"} for o in tape.operations)


custom_controlled_ops = [  # operators with their own controlled class
    (qml.PauliX, 1, qml.CNOT),
    (qml.PauliY, 1, qml.CY),
    (qml.PauliZ, 1, qml.CZ),
    (qml.PauliX, 2, qml.Toffoli),
]


class TestCtrlCustomOperator:
    @pytest.mark.parametrize("op_cls, num_ctrl_wires, custom_op_cls", custom_controlled_ops)
    def test_ctrl_custom_operators(self, op_cls, num_ctrl_wires, custom_op_cls):
        """Test that ctrl returns operators with their own controlled class."""
        ctrl_wires = list(range(1, num_ctrl_wires + 1))
        op = op_cls(wires=0)
        ctrl_op = qml.ctrl(op, control=ctrl_wires)
        custom_op = custom_op_cls(wires=ctrl_wires + [0])
        assert qml.equal(ctrl_op, custom_op)
        assert ctrl_op.name == custom_op.name

    @pytest.mark.parametrize("op_cls, _, custom_op_cls", custom_controlled_ops)
    def test_no_ctrl_custom_operators_excess_wires(self, op_cls, _, custom_op_cls):
        """Test that ctrl returns a `Controlled` class when there is an excess of control wires."""
        if op_cls is qml.PauliX:
            pytest.skip("ctrl(PauliX) becomes MultiControlledX, not Controlled")

        control_wires = list(range(1, 6))
        op = op_cls(wires=0)
        ctrl_op = qml.ctrl(op, control=control_wires)
        expected = Controlled(op, control_wires=control_wires)
        assert not isinstance(ctrl_op, custom_op_cls)
        assert qml.equal(ctrl_op, expected)

    @pytest.mark.parametrize("op_cls, num_ctrl_wires, custom_op_cls", custom_controlled_ops)
    def test_no_ctrl_custom_operators_control_values(self, op_cls, num_ctrl_wires, custom_op_cls):
        """Test that ctrl returns a `Controlled` class when the control value is not `True`."""
        if op_cls is qml.PauliX:
            pytest.skip("ctrl(PauliX) becomes MultiControlledX, not Controlled")

        ctrl_wires = list(range(1, num_ctrl_wires + 1))
        op = op_cls(wires=0)
        ctrl_op = qml.ctrl(op, ctrl_wires, control_values=[False] * num_ctrl_wires)
        expected = Controlled(op, ctrl_wires, control_values=[False] * num_ctrl_wires)
        assert not isinstance(ctrl_op, custom_op_cls)
        assert qml.equal(ctrl_op, expected)

    @pytest.mark.parametrize(
        "control_wires,control_values,expected_values",
        [
            ([1], (False), "0"),
            ([1, 2], (0, 1), "01"),
            ([1, 2, 3], (True, True, True), "111"),
            ([1, 2, 3], (True, True, False), "110"),
            ([1, 2, 3], None, None),
        ],
    )
    def test_ctrl_PauliX_MultiControlledX(self, control_wires, control_values, expected_values):
        """Tests that ctrl(PauliX) with 3+ control wires or Falsy control values make a MCX"""
        with qml.queuing.AnnotatedQueue() as q:
            op = qml.ctrl(qml.PauliX(0), control_wires, control_values=control_values)

        expected = qml.MultiControlledX(wires=control_wires + [0], control_values=expected_values)
        assert len(q) == 1
        assert qml.equal(op, expected)
        assert qml.equal(q.queue[0], expected)


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
class TestCtrlTransformDifferentiation:
    """Tests for differentiation"""

    @pytest.mark.autograd
    def test_autograd(self, diff_method):
        """Test differentiation using autograd"""

        dev = qml.device("default.qubit", wires=2)
        init_state = np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = np.array(0.123, requires_grad=True)
        res = qml.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.torch
    def test_torch(self, diff_method):
        """Test differentiation using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)
        init_state = torch.tensor(
            [1.0, -1.0], requires_grad=False, dtype=torch.complex128
        ) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = torch.tensor(0.123, requires_grad=True, dtype=torch.float64)
        loss = circuit(b)
        loss.backward()  # pylint:disable=no-member

        res = b.grad.detach()
        expected = np.sin(b.detach() / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["auto", "jax", "jax-python"])
    def test_jax(self, diff_method, jax_interface):
        """Test differentiation using JAX"""

        import jax

        jax.config.update("jax_enable_x64", True)

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method, interface=jax_interface)
        def circuit(b):
            init_state = onp.array([1.0, -1.0]) / onp.sqrt(2)
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = jnp.array(0.123)
        res = jax.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.tf
    def test_tf(self, diff_method):
        """Test differentiation using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        init_state = tf.constant([1.0, -1.0], dtype=tf.complex128) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = tf.Variable(0.123, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = circuit(b)

        res = tape.gradient(loss, b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)


def test_ctrl_values_sanity_check():
    """Test that control works with control values on a very standard usecase."""

    def make_ops():
        qml.RX(0.123, wires=0)
        qml.RY(0.456, wires=2)
        qml.RX(0.789, wires=0)
        qml.Rot(0.111, 0.222, 0.333, wires=2)
        qml.PauliX(wires=2)
        qml.PauliY(wires=4)
        qml.PauliZ(wires=0)

    with qml.queuing.AnnotatedQueue() as q_tape:
        cmake_ops = ctrl(make_ops, control=1, control_values=0)
        # Execute controlled version.
        cmake_ops()

    tape = QuantumScript.from_queue(q_tape)
    expected = [
        qml.PauliX(wires=1),
        qml.CRX(0.123, wires=[1, 0]),
        qml.CRY(0.456, wires=[1, 2]),
        qml.CRX(0.789, wires=[1, 0]),
        qml.CRot(0.111, 0.222, 0.333, wires=[1, 2]),
        qml.CNOT(wires=[1, 2]),
        *qml.CY(wires=[1, 4]).decomposition(),
        *qml.CZ(wires=[1, 0]).decomposition(),
        qml.PauliX(wires=1),
    ]
    assert len(tape) == 9
    expanded = tape.expand(stop_at=lambda obj: not isinstance(obj, Controlled))
    for op1, op2 in zip(expanded, expected):
        assert qml.equal(op1, op2)


@pytest.mark.parametrize("ctrl_values", [[0, 0], [0, 1], [1, 0], [1, 1]])
def test_multi_ctrl_values(ctrl_values):
    """Test control with a list of wires and control values."""

    def expected_ops(ctrl_val):
        exp_op = []
        ctrl_wires = [3, 7]
        for i, j in enumerate(ctrl_val):
            if not bool(j):
                exp_op.append(qml.PauliX(ctrl_wires[i]))
        exp_op.append(Controlled(qml.PhaseShift(np.pi / 2, 0), [3, 7]))
        for i, j in enumerate(ctrl_val):
            if not bool(j):
                exp_op.append(qml.PauliX(ctrl_wires[i]))

        return exp_op

    with qml.queuing.AnnotatedQueue() as q_tape:
        CCS = ctrl(qml.S, control=[3, 7], control_values=ctrl_values)
        CCS(wires=0)
    tape = QuantumScript.from_queue(q_tape)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, Controlled)
    new_tape = expand_tape(tape, 1)
    assert equal_list(list(new_tape), expected_ops(ctrl_values))
