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

import numpy as np
import pytest
import scipy as sp
from gate_data import (
    CCZ,
    CH,
    CNOT,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    CRot3,
    CRotx,
    CRoty,
    CRotz,
    Toffoli,
)
from scipy import sparse

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.exceptions import DecompositionUndefinedError, PennyLaneDeprecationWarning
from pennylane.operation import Operation, Operator
from pennylane.ops.op_math.controlled import Controlled, ControlledOp, ctrl
from pennylane.tape import QuantumScript, expand_tape
from pennylane.wires import Wires

# pylint: disable=too-few-public-methods
# pylint: disable=protected-access
# pylint: disable=pointless-statement
# pylint: disable=expression-not-assigned
# pylint: disable=too-many-arguments


def equal_list(lhs, rhs):
    if not isinstance(lhs, list):
        lhs = [lhs]
    if not isinstance(rhs, list):
        rhs = [rhs]
    return len(lhs) == len(rhs) and all(qml.equal(l, r) for l, r in zip(lhs, rhs))


class TempOperator(Operator):
    num_wires = 1


class TempOperation(Operation):
    num_wires = 1


class OpWithDecomposition(Operation):
    @staticmethod
    def compute_decomposition(*params, wires=None, **_):
        return [
            qml.Hadamard(wires=wires[0]),
            qml.S(wires=wires[1]),
            qml.RX(params[0], wires=wires[0]),
        ]


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


class TestControlledInit:
    """Test the initialization process and standard properties."""

    temp_op = TempOperator("a")

    def test_nonparametric_ops(self):
        """Test pow initialization for a non parameteric operation."""

        op = Controlled(
            self.temp_op, (0, 1), control_values=[True, False], work_wires="aux", id="something"
        )

        assert op.base is self.temp_op
        assert op.hyperparameters["base"] is self.temp_op

        assert op.wires == Wires((0, 1, "a"))

        assert op.control_wires == Wires((0, 1))
        assert op.hyperparameters["control_wires"] == Wires((0, 1))

        assert op.target_wires == Wires("a")

        assert op.control_values == [True, False]
        assert op.hyperparameters["control_values"] == [True, False]

        assert op.work_wires == Wires("aux")

        assert op.name == "C(TempOperator)"
        assert op.id == "something"

        assert op.num_params == 0
        assert op.parameters == []  # pylint: disable=use-implicit-booleaness-not-comparison
        assert op.data == ()

        assert op.num_wires == 3

    def test_default_control_values(self):
        """Test assignment of default control_values."""
        op = Controlled(self.temp_op, (0, 1))
        assert op.control_values == [True, True]

    def test_zero_one_control_values(self):
        """Test assignment of provided control_values."""
        op = Controlled(self.temp_op, (0, 1), control_values=[0, 1])
        assert op.control_values == [False, True]

    @pytest.mark.parametrize("control_values", [True, False, 0, 1])
    def test_scalar_control_values(self, control_values):
        """Test assignment of provided control_values."""
        op = Controlled(self.temp_op, 0, control_values=control_values)
        assert op.control_values == [control_values]

    def test_tuple_control_values(self):
        """Test assignment of provided control_values."""
        op = Controlled(self.temp_op, (0, 1), control_values=(0, 1))
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

    @pytest.mark.parametrize("old_name, new_name", [("clean", "zeroed"), ("dirty", "borrowed")])
    def test_old_work_wire_type_deprecated(self, old_name, new_name):
        """Tests that specifying work_wire_type as 'clean' or 'dirty' is deprecated"""
        with pytest.warns(PennyLaneDeprecationWarning, match="work_wire_type"):
            op = Controlled(self.temp_op, "b", work_wires="c", work_wire_type=old_name)
        assert op.work_wire_type == new_name


class TestControlledProperties:
    """Test the properties of the ``Controlled`` symbolic operator."""

    def test_resource_params(self):
        """Tests that a controlled op has the correct resource params."""

        op = Controlled(
            qml.MultiRZ(0.5, wires=[0, 1, 2]),
            control_wires=[3, 4],
            control_values=[True, False],
            work_wires=[5],
        )
        assert op.resource_params == {
            "base_class": qml.MultiRZ,
            "base_params": {"num_wires": 3},
            "num_control_wires": 2,
            "num_zero_control_values": 1,
            "num_work_wires": 1,
            "work_wire_type": "borrowed",
        }

    def test_data(self):
        """Test that the base data can be get and set through Controlled class."""

        x = pnp.array(1.234)

        base = qml.RX(x, wires="a")
        op = Controlled(base, (0, 1))

        assert op.data == (x,)

        x_new = (pnp.array(2.3454),)
        op.data = x_new
        assert op.data == (x_new,)
        assert base.data == (x_new,)

        x_new2 = (pnp.array(3.456),)
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

    def test_has_decomposition_multicontrolled_special_unitary(self):
        """Test that a one qubit special unitary with any number of control
        wires has a decomposition."""
        op = Controlled(qml.RX(1.234, wires=0), (1, 2, 3, 4, 5))
        assert op.has_decomposition

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

        assert op.wires == Wires((3, 4, 0, 1))

        op = op.map_wires(wire_map={3: "a", 4: "b", 0: "c", 1: "d", "aux": "extra"})

        assert op.base.wires == Wires(("c", "d"))
        assert op.control_wires == Wires(("a", "b"))
        assert op.work_wires == Wires("extra")


class TestControlledMiscMethods:
    """Test miscellaneous minor Controlled methods."""

    def test_repr(self):
        """Test __repr__ method."""
        assert repr(Controlled(qml.S(0), [1])) == "Controlled(S(0), control_wires=[1])"

        base = qml.S(0) + qml.T(1)
        op = Controlled(base, [2])
        assert repr(op) == "Controlled(S(0) + T(1), control_wires=[2])"

        op = Controlled(base, [2, 3], control_values=[True, False], work_wires=[4])
        assert (
            repr(op)
            == "Controlled(S(0) + T(1), control_wires=[2, 3], work_wires=[4], control_values=[True, False])"
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

        assert metadata == (control_wires, control_values, work_wires, "borrowed")

        # make sure metadata is hashable
        assert hash(metadata)

        new_op = type(op)._unflatten(*op._flatten())
        qml.assert_equal(op, new_op)
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
        U = pnp.eye(2)
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
        mat_eigvals = pnp.sort(qml.math.linalg.eigvals(mat))

        eigs = op.eigvals()
        sort_eigs = pnp.sort(eigs)

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
        control_values = [0, 1]
        op = Controlled(base, ("b", "c"), control_values=control_values)

        base_gen, base_gen_coeff = qml.generator(base, format="prefactor")
        gen_tensor, gen_coeff = qml.generator(op, format="prefactor")

        assert base_gen_coeff == gen_coeff

        for wire, val in zip(op.control_wires, control_values):
            ob = list(op for op in gen_tensor.operands if op.wires == qml.wires.Wires(wire))
            assert len(ob) == 1
            assert ob[0].data == ([val],)

        ob = list(op for op in gen_tensor.operands if op.wires == base.wires)
        assert len(ob) == 1
        assert ob[0].__class__ is base_gen.__class__

        expected = qml.exp(op.generator(), 1j * op.data[0])
        assert qml.math.allclose(
            expected.matrix(wire_order=["a", "b", "c"]), op.matrix(wire_order=["a", "b", "c"])
        )

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


class TestControlledOperationProperties:
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
            (qml.DoubleExcitationMinus(0.7, [0, 1, 2, 3]), [(0.5, 1.0)]),
        ],
    )
    def test_parameter_frequencies(self, base, expected):
        """Test parameter-frequencies against expected values."""

        op = Controlled(base, (4, 5))
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


class TestControlledSimplify:
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

        assert isinstance(simplified_op, Controlled)
        for s1, s2 in zip(final_op.base.operands, simplified_op.base.operands):
            qml.assert_equal(s1, s2)

    def test_simplify_nested_controlled_ops(self):
        """Test the simplify method with nested control operations on different wires."""
        controlled_op = Controlled(Controlled(qml.Hadamard(0), 1), 2)
        final_op = Controlled(qml.Hadamard(0), [2, 1])
        simplified_op = controlled_op.simplify()
        qml.assert_equal(simplified_op, final_op)


class TestControlledQueuing:
    """Test that Controlled operators queue and update base metadata."""

    def test_queuing(self):
        """Test that `Controlled` is queued upon initialization and updates base metadata."""
        with qml.queuing.AnnotatedQueue() as q:
            base = qml.Rot(1.234, 2.345, 3.456, wires=2)
            op = Controlled(base, (0, 1))

        assert base not in q
        qml.assert_equal(q.queue[0], op)

    def test_queuing_base_defined_outside(self):
        """Test that base isn't added to queue if its defined outside the recording context."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        with qml.queuing.AnnotatedQueue() as q:
            op = Controlled(base, ("a", "b"))

        assert len(q) == 1
        assert q.queue[0] is op


base_num_control_mats = [
    (qml.PauliX("a"), 1, CNOT),
    (qml.PauliX("a"), 2, Toffoli),
    (qml.CNOT(["a", "b"]), 1, Toffoli),
    (qml.PauliY("a"), 1, CY),
    (qml.PauliZ("a"), 1, CZ),
    (qml.PauliZ("a"), 2, CCZ),
    (qml.SWAP(("a", "b")), 1, CSWAP),
    (qml.Hadamard("a"), 1, CH),
    (qml.RX(1.234, "b"), 1, CRotx(1.234)),
    (qml.RY(-0.432, "a"), 1, CRoty(-0.432)),
    (qml.RZ(6.78, "a"), 1, CRotz(6.78)),
    (qml.Rot(1.234, -0.432, 9.0, "a"), 1, CRot3(1.234, -0.432, 9.0)),
    (qml.PhaseShift(1.234, wires="a"), 1, ControlledPhaseShift(1.234)),
]


class TestMatrix:
    """Tests of Controlled.matrix and Controlled.sparse_matrix"""

    def test_correct_matrix_dimensions_with_batching(self):
        """Test batching returns a matrix of the correct dimensions"""

        x = pnp.array([1.0, 2.0, 3.0])
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
        assert mat.shape == (4, 4)

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
        assert op.has_sparse_matrix

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
        assert op.has_sparse_matrix

    def test_sparse_matrix_wire_order(self):
        """Check if the user requests specific wire order, sparse_matrix() returns the same as matrix()."""
        control_wires = (0, 1, 2)
        base = qml.U2(1.234, -3.2, wires=3)
        op = Controlled(base, control_wires)

        op_sparse = op.sparse_matrix(wire_order=[3, 2, 1, 0])
        op_dense = op.matrix(wire_order=[3, 2, 1, 0])

        assert qml.math.allclose(op_sparse.toarray(), op_dense)

    def test_no_matrix_defined_sparse_matrix_error(self):
        """Check that if the base gate defines neither a sparse matrix nor a dense matrix, a
        SparseMatrixUndefined error is raised."""

        base = TempOperator(1)
        op = Controlled(base, 2)
        assert not op.has_sparse_matrix

        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_sparse_matrix_format(self):
        """Test format keyword determines output type of sparse matrix."""
        base = qml.PauliX(0)
        op = Controlled(base, 1)

        lil_mat = op.sparse_matrix(format="lil")
        assert isinstance(lil_mat, sparse.lil_matrix)


special_non_par_op_decomps = [
    (
        qml.Identity,
        [],
        [3],
        [0, 1, 2],
        (lambda wires: qml.ctrl(qml.Identity(wires[-1]), control=wires[:-1])),
        [qml.Identity([0, 1, 2, 3])],
    ),
    (qml.PauliY, [], [0], [1], qml.CY, [qml.CRY(np.pi, wires=[1, 0]), qml.S(1)]),
    (qml.PauliZ, [], [1], [0], qml.CZ, [qml.ControlledPhaseShift(np.pi, wires=[0, 1])]),
    (
        qml.Hadamard,
        [],
        [1],
        [0],
        qml.CH,
        [qml.RY(-np.pi / 4, wires=1), qml.CZ(wires=[0, 1]), qml.RY(np.pi / 4, wires=1)],
    ),
    (
        qml.PauliZ,
        [],
        [0],
        [2, 1],
        qml.CCZ,
        [
            qml.CNOT(wires=[1, 0]),
            qml.adjoint(qml.T(wires=0)),
            qml.CNOT(wires=[2, 0]),
            qml.T(wires=0),
            qml.CNOT(wires=[1, 0]),
            qml.adjoint(qml.T(wires=0)),
            qml.CNOT(wires=[2, 0]),
            qml.T(wires=0),
            qml.T(wires=1),
            qml.CNOT(wires=[2, 1]),
            qml.Hadamard(wires=0),
            qml.T(wires=2),
            qml.adjoint(qml.T(wires=1)),
            qml.CNOT(wires=[2, 1]),
            qml.Hadamard(wires=0),
        ],
    ),
    (
        qml.CZ,
        [],
        [1, 2],
        [0],
        qml.CCZ,
        [
            qml.CNOT(wires=[1, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[0, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[1, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[0, 2]),
            qml.T(wires=2),
            qml.T(wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=2),
            qml.T(wires=0),
            qml.adjoint(qml.T(wires=1)),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[2]),
        ],
    ),
    (
        qml.SWAP,
        [],
        [1, 2],
        [0],
        qml.CSWAP,
        [qml.Toffoli(wires=[0, 2, 1]), qml.Toffoli(wires=[0, 1, 2]), qml.Toffoli(wires=[0, 2, 1])],
    ),
]

special_par_op_decomps = [
    (
        qml.RX,
        [0.123],
        [1],
        [0],
        qml.CRX,
        [
            qml.RZ(np.pi / 2, wires=1),
            qml.RY(0.123 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RY(-0.123 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(-np.pi / 2, wires=1),
        ],
    ),
    (
        qml.RY,
        [0.123],
        [1],
        [0],
        qml.CRY,
        [
            qml.RY(0.123 / 2, 1),
            qml.CNOT(wires=(0, 1)),
            qml.RY(-0.123 / 2, 1),
            qml.CNOT(wires=(0, 1)),
        ],
    ),
    (
        qml.RZ,
        [0.123],
        [0],
        [1],
        qml.CRZ,
        [
            qml.PhaseShift(0.123 / 2, wires=0),
            qml.CNOT(wires=[1, 0]),
            qml.PhaseShift(-0.123 / 2, wires=0),
            qml.CNOT(wires=[1, 0]),
        ],
    ),
    (
        qml.Rot,
        [0.1, 0.2, 0.3],
        [1],
        [0],
        qml.CRot,
        [
            qml.RZ((0.1 - 0.3) / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(-(0.1 + 0.3) / 2, wires=1),
            qml.RY(-0.2 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.RY(0.2 / 2, wires=1),
            qml.RZ(0.3, wires=1),
        ],
    ),
    (
        qml.PhaseShift,
        [0.123],
        [1],
        [0],
        qml.ControlledPhaseShift,
        [
            qml.PhaseShift(0.123 / 2, wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(-0.123 / 2, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(0.123 / 2, wires=1),
        ],
    ),
    (
        qml.GlobalPhase,
        [0.123],
        [1],
        [0],
        (lambda x, wires: qml.ctrl(qml.GlobalPhase(x, wires[-1]), control=wires[:-1])),
        [qml.PhaseShift(-0.123, wires=0)],
    ),
    (
        qml.GlobalPhase,
        [0.123],
        [3],
        [0, 1, 2],
        (lambda x, wires: qml.ctrl(qml.GlobalPhase(x, wires[-1]), control=wires[:-1])),
        [qml.ctrl(qml.PhaseShift(-0.123, wires=2), control=[0, 1])],
    ),
]

custom_ctrl_op_decomps = special_non_par_op_decomps + special_par_op_decomps

pauli_x_based_op_decomps = [
    (qml.PauliX, [0], [1], [qml.CNOT([1, 0])]),
    (
        qml.PauliX,
        [2],
        [0, 1],
        qml.Toffoli.compute_decomposition(wires=[0, 1, 2]),
    ),
    (
        qml.CNOT,
        [1, 2],
        [0],
        qml.Toffoli.compute_decomposition(wires=[0, 1, 2]),
    ),
    (
        qml.PauliX,
        [3],
        [0, 1, 2],
        qml.MultiControlledX.compute_decomposition(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
    ),
    (
        qml.CNOT,
        [2, 3],
        [0, 1],
        qml.MultiControlledX.compute_decomposition(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
    ),
    (
        qml.Toffoli,
        [1, 2, 3],
        [0],
        qml.MultiControlledX.compute_decomposition(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
    ),
]


class TestDecomposition:
    """Test decomposition of Controlled."""

    @pytest.mark.parametrize(
        "target, decomp",
        [
            (
                OpWithDecomposition(0.123, wires=[0, 1]),
                [
                    qml.CH(wires=[2, 0]),
                    Controlled(qml.S(wires=1), control_wires=2),
                    qml.CRX(0.123, wires=[2, 0]),
                ],
            ),
            (
                qml.IsingXX(0.123, wires=[0, 1]),
                [
                    qml.Toffoli(wires=[2, 0, 1]),
                    qml.CRX(0.123, wires=[2, 0]),
                    qml.Toffoli(wires=[2, 0, 1]),
                ],
            ),
        ],
    )
    def test_decomposition(self, target, decomp):
        """Test that we decompose a normal controlled operation"""
        op = Controlled(target, 2)
        assert op.decomposition() == decomp

    def test_non_differentiable_one_qubit_special_unitary(self):
        """Assert that a non-differentiable on qubit special unitary uses the bisect decomposition."""

        op = qml.ctrl(qml.RZ(1.2, wires=0), (1, 2, 3, 4))
        decomp = op.decomposition()

        qml.assert_equal(decomp[0], qml.Toffoli(wires=(1, 2, 0)))
        assert isinstance(decomp[1], qml.QubitUnitary)
        qml.assert_equal(decomp[2], qml.Toffoli(wires=(3, 4, 0)))
        assert isinstance(decomp[3].base, qml.QubitUnitary)
        qml.assert_equal(decomp[4], qml.Toffoli(wires=(1, 2, 0)))
        assert isinstance(decomp[5], qml.QubitUnitary)
        qml.assert_equal(decomp[6], qml.Toffoli(wires=(3, 4, 0)))
        assert isinstance(decomp[7].base, qml.QubitUnitary)

        decomp_mat = qml.matrix(op.decomposition, wire_order=op.wires)()
        assert qml.math.allclose(op.matrix(), decomp_mat)

    def test_differentiable_one_qubit_special_unitary_single_ctrl(self):
        """
        Assert that a differentiable qubit special unitary uses the zyz decomposition with a single controlled wire.
        """

        theta = 1.2
        op = qml.ctrl(qml.RZ(qml.numpy.array(theta), 0), (1))
        decomp = op.decomposition()

        qml.assert_equal(decomp[0], qml.PhaseShift(qml.numpy.array(theta / 2), 0))
        qml.assert_equal(decomp[1], qml.CNOT(wires=(1, 0)))
        qml.assert_equal(decomp[2], qml.PhaseShift(qml.numpy.array(-theta / 2), 0))
        qml.assert_equal(decomp[3], qml.CNOT(wires=(1, 0)))

        decomp_mat = qml.matrix(op.decomposition, wire_order=op.wires)()
        assert qml.math.allclose(op.matrix(), decomp_mat)

    def test_differentiable_one_qubit_special_unitary_multiple_ctrl(self):
        """Assert that a differentiable qubit special unitary uses the zyz decomposition with multiple controlled wires."""

        theta = 1.2
        op = qml.ctrl(qml.RZ(qml.numpy.array(theta), 0), (1, 2, 3, 4))
        decomp = op.decomposition()

        qml.assert_equal(decomp[0], qml.CRZ(qml.numpy.array(theta), [4, 0]))
        qml.assert_equal(decomp[1], qml.MultiControlledX(wires=[1, 2, 3, 0]))
        qml.assert_equal(decomp[2], qml.CRZ(qml.numpy.array(-theta / 2), wires=[4, 0]))
        qml.assert_equal(decomp[3], qml.MultiControlledX(wires=[1, 2, 3, 0]))
        qml.assert_equal(decomp[4], qml.CRZ(qml.numpy.array(-theta / 2), wires=[4, 0]))

        decomp_mat = qml.matrix(op.decomposition, wire_order=op.wires)()
        assert qml.math.allclose(op.matrix(), decomp_mat)

    # pylint: disable=too-many-positional-arguments
    @pytest.mark.parametrize(
        "base_cls, params, base_wires, ctrl_wires, custom_ctrl_cls, expected",
        custom_ctrl_op_decomps,
    )
    def test_decomposition_custom_ops(
        self,
        base_cls,
        params,
        base_wires,
        ctrl_wires,
        custom_ctrl_cls,
        expected,
        tol,
    ):
        """Tests decompositions of custom operations"""

        active_wires = ctrl_wires + base_wires
        base_op = base_cls(*params, wires=base_wires)
        ctrl_op = Controlled(base_op, control_wires=ctrl_wires)
        custom_ctrl_op = custom_ctrl_cls(*params, active_wires)

        assert ctrl_op.decomposition() == expected
        assert qml.tape.QuantumScript(ctrl_op.decomposition()).circuit == expected
        assert custom_ctrl_op.decomposition() == expected
        # There is not custom ctrl class for GlobalPhase (yet), so no `compute_decomposition`
        # to test, just the controlled decompositions logic.
        if base_cls not in (qml.GlobalPhase, qml.Identity):
            assert custom_ctrl_cls.compute_decomposition(*params, active_wires) == expected

        mat = qml.matrix(ctrl_op.decomposition, wire_order=active_wires)()
        assert np.allclose(mat, custom_ctrl_op.matrix(), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "base_cls, params, base_wires, ctrl_wires, custom_ctrl_cls, expected",
        special_par_op_decomps,
    )
    def test_decomposition_custom_par_ops_broadcasted(
        self,
        base_cls,
        params,
        base_wires,
        ctrl_wires,
        custom_ctrl_cls,
        expected,
        tol,
    ):
        """Tests broadcasted decompositions of custom controlled ops"""
        broad_casted_params = [np.array([p, p]) for p in params]
        self.test_decomposition_custom_ops(
            base_cls,
            broad_casted_params,
            base_wires,
            ctrl_wires,
            custom_ctrl_cls,
            expected,
            tol,
        )

    @pytest.mark.parametrize(
        "base_cls, base_wires, ctrl_wires, expected",
        pauli_x_based_op_decomps,
    )
    def test_decomposition_pauli_x(self, base_cls, base_wires, ctrl_wires, expected):
        """Tests decompositions where the base is PauliX"""

        base_op = base_cls(wires=base_wires)
        ctrl_op = Controlled(base_op, control_wires=ctrl_wires, work_wires=Wires("aux"))

        assert ctrl_op.decomposition() == expected
        assert qml.tape.QuantumScript(ctrl_op.decomposition()).circuit == expected

    def test_decomposition_nested(self):
        """Tests decompositions of nested controlled operations"""

        ctrl_op = Controlled(Controlled(qml.RZ(0.123, wires=0), control_wires=1), control_wires=2)
        expected = [
            qml.ControlledPhaseShift(0.123 / 2, wires=[2, 0]),
            qml.Toffoli(wires=[2, 1, 0]),
            qml.ControlledPhaseShift(-0.123 / 2, wires=[2, 0]),
            qml.Toffoli(wires=[2, 1, 0]),
        ]
        assert ctrl_op.decomposition() == expected
        assert qml.tape.QuantumScript(ctrl_op.decomposition()).circuit == expected

    def test_decomposition_undefined(self):
        """Tests error raised when decomposition is undefined"""
        op = Controlled(TempOperator(0), (1, 2))
        with pytest.raises(DecompositionUndefinedError):
            op.decomposition()

    def test_control_on_zero(self):
        """Test decomposition applies PauliX gates to flip any control-on-zero wires."""

        control_wires = (0, 1, 2)
        control_values = [True, False, False]

        base = TempOperator("a")
        op = Controlled(base, control_wires, control_values)

        decomp1 = op.decomposition()
        decomp2 = qml.tape.QuantumScript(op.decomposition()).circuit

        for decomp in [decomp1, decomp2]:
            qml.assert_equal(decomp[0], qml.PauliX(1))
            qml.assert_equal(decomp[1], qml.PauliX(2))

            assert isinstance(decomp[2], Controlled)
            assert decomp[2].control_values == [True, True, True]

            qml.assert_equal(decomp[3], qml.PauliX(1))
            qml.assert_equal(decomp[4], qml.PauliX(2))

    @pytest.mark.parametrize(
        "base_cls, params, base_wires, ctrl_wires, _, expected",
        custom_ctrl_op_decomps,
    )
    def test_control_on_zero_custom_ops(
        self, base_cls, params, base_wires, ctrl_wires, _, expected
    ):
        """Tests that custom ops are not converted when wires are control-on-zero."""

        base_op = base_cls(*params, wires=base_wires)
        op = Controlled(base_op, control_wires=ctrl_wires, control_values=[False] * len(ctrl_wires))

        decomp = op.decomposition()

        i = 0
        for ctrl_wire in ctrl_wires:
            assert decomp[i] == qml.PauliX(wires=ctrl_wire)
            i += 1

        for exp in expected:
            assert decomp[i] == exp
            i += 1

        for ctrl_wire in ctrl_wires:
            assert decomp[i] == qml.PauliX(wires=ctrl_wire)
            i += 1


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
        init_state = pnp.array([1.0, -1.0], requires_grad=False) / pnp.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = pnp.array(0.123, requires_grad=True)
        res = qml.grad(circuit)(b)
        expected = pnp.sin(b / 2) / 2

        assert pnp.allclose(res, expected)

    @pytest.mark.torch
    def test_torch(self, diff_method):
        """Test differentiation using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)
        init_state = torch.tensor(
            [1.0, -1.0], requires_grad=False, dtype=torch.complex128
        ) / pnp.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = torch.tensor(0.123, requires_grad=True, dtype=torch.float64)
        loss = circuit(b)
        loss.backward()  # pylint:disable=no-member

        res = b.grad.detach()
        expected = pnp.sin(b.detach() / 2) / 2

        assert pnp.allclose(res, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["auto", "jax"])
    def test_jax(self, diff_method, jax_interface):
        """Test differentiation using JAX"""

        import jax

        jax.config.update("jax_enable_x64", True)

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method, interface=jax_interface)
        def circuit(b):
            init_state = np.array([1.0, -1.0]) / pnp.sqrt(2)
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = jnp.array(0.123)
        res = jax.grad(circuit)(b)
        expected = pnp.sin(b / 2) / 2

        assert pnp.allclose(res, expected)

    @pytest.mark.tf
    def test_tf(self, diff_method):
        """Test differentiation using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        init_state = tf.constant([1.0, -1.0], dtype=tf.complex128) / pnp.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = tf.Variable(0.123, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = circuit(b)

        res = tape.gradient(loss, b)
        expected = pnp.sin(b / 2) / 2

        assert pnp.allclose(res, expected)


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
        par = pnp.array([0.25, 2.1, -0.42])
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
        par = pnp.array([0.25, 2.1, -0.42])
        cls = getattr(qml, name)

        # Provide up to 6 wires and take as many as the class requires
        # This assumes that the class does *not* have `num_wires=None`
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
        par = (pnp.array([0.25, 2.1, -0.42]), pnp.array([-6.2, 0.12, 0.421]))
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
            pnp.array([0.25, 2.1, -0.42]),
            pnp.array([-6.2, 0.12, 0.421]),
            pnp.array([0.2, 1.1, -5.2]),
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
            pnp.array([0.25, 2.1, -0.42]),
            pnp.array([-6.2, 0.12, 0.421]),
            pnp.array([0.2, 1.1, -5.2]),
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
        diag = pnp.array([[1j, 1, 1, -1j], [-1j, 1j, 1, -1], [1j, -1j, 1.0, -1]])
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
        par = pnp.array([0.25, 2.1, -0.42])

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
        par = pnp.array([0.25, 2.1, -0.42])

        base = qml.MultiRZ(par, wires=wires)
        op = Controlled(base, "wire1")

        mat = op.matrix()
        single_mats = [Controlled(qml.MultiRZ(p, wires=wires), "wire1").matrix() for p in par]

        assert qml.math.allclose(mat, single_mats)

    @pytest.mark.parametrize(
        "state_, num_wires",
        [([1.0, 0.0], 1), ([0.5, -0.5j, 0.5, -0.5], 2), (pnp.ones(8) / pnp.sqrt(8), 3)],
    )
    def test_controlled_qubit_state_vector(self, state_, num_wires):
        """Test that StatePrep, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        state = pnp.array([state_])
        base = qml.StatePrep(state, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")

        assert op.batch_size == 1
        qml.StatePrep.compute_decomposition(state, list(range(num_wires)))
        op.decomposition()

        state = pnp.array([state_] * 3)
        base = qml.StatePrep(state, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 3
        qml.StatePrep.compute_decomposition(state, list(range(num_wires)))
        op.decomposition()

    @pytest.mark.parametrize(
        "state, num_wires",
        [([1.0, 0.0], 1), ([0.5, -0.5j, 0.5, -0.5], 2), (pnp.ones(8) / pnp.sqrt(8), 3)],
    )
    def test_controlled_amplitude_embedding(self, state, num_wires):
        """Test that AmplitudeEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        features = pnp.array([state])
        base = qml.AmplitudeEmbedding(features, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 1
        qml.AmplitudeEmbedding.compute_decomposition(features, list(range(num_wires)))
        op.decomposition()

        features = pnp.array([state] * 3)
        base = qml.AmplitudeEmbedding(features, wires=list(range(num_wires)))
        op = Controlled(base, "wire1")
        assert op.batch_size == 3
        qml.AmplitudeEmbedding.compute_decomposition(features, list(range(num_wires)))
        op.decomposition()

    @pytest.mark.parametrize(
        "angles, num_wires",
        [
            (pnp.array([[0.5], [2.1]]), 1),
            (pnp.array([[0.5, -0.5], [0.2, 1.5]]), 2),
            (pnp.ones((2, 5)), 5),
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
            (pnp.array([[0.5], [2.1]]), 1),
            (pnp.array([[0.5, -0.5], [0.2, 1.5]]), 2),
            (pnp.ones((2, 5)), 5),
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
            (pnp.array([[0.5], [2.1]]), pnp.array([[0.61], [0.3]]), 1, 2),
            (pnp.array([[0.5, -0.5], [0.2, 1.5]]), pnp.ones((2, 4, 3)), 2, 2),
            (pnp.array([0.5, -0.5, 0.2]), pnp.ones((3, 2, 6)), 3, 3),
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


custom_ctrl_ops = [
    (qml.PauliY(wires=0), [1], qml.CY(wires=[1, 0])),
    (qml.PauliZ(wires=0), [1], qml.CZ(wires=[1, 0])),
    (qml.RX(0.123, wires=0), [1], qml.CRX(0.123, wires=[1, 0])),
    (qml.RY(0.123, wires=0), [1], qml.CRY(0.123, wires=[1, 0])),
    (qml.RZ(0.123, wires=0), [1], qml.CRZ(0.123, wires=[1, 0])),
    (
        qml.Rot(0.123, 0.234, 0.456, wires=0),
        [1],
        qml.CRot(0.123, 0.234, 0.456, wires=[1, 0]),
    ),
    (qml.PhaseShift(0.123, wires=0), [1], qml.ControlledPhaseShift(0.123, wires=[1, 0])),
    (
        qml.QubitUnitary(np.array([[0, 1], [1, 0]]), wires=0),
        [1, 2],
        qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), wires=[1, 2, 0]),
    ),
    (qml.Barrier(), [1], qml.Barrier()),
    (qml.Barrier(wires=(0, 1)), [2], qml.Barrier(wires=(0, 1))),
    (qml.Barrier(wires=(0, 1), only_visual=True), [2], qml.Barrier(wires=(0, 1), only_visual=True)),
]


class TestCtrl:
    """Tests for the ctrl transform."""

    def test_sparse_qubit_unitary(self):
        """Test that the controlled sparse QubitUnitary works correctly"""
        data = sp.sparse.eye(2)
        op = qml.QubitUnitary(data, wires=2)
        c_op = qml.ctrl(op, 3)

        data_dense = data.toarray()
        op_dense = qml.QubitUnitary(data_dense, wires=2)
        c_op_dense = qml.ctrl(op_dense, 3)

        assert qml.math.allclose(c_op.sparse_matrix(), c_op_dense.matrix())

    def test_no_redundant_queue(self):
        """Test that the ctrl transform does not add redundant operations to the queue. https://github.com/PennyLaneAI/pennylane/pull/6926"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.ctrl(qml.QubitUnitary(np.eye(2), 0), 1)

        assert len(q.queue) == 1

    def test_invalid_input_error(self):
        """Test that a ValueError is raised upon invalid inputs."""
        with pytest.raises(ValueError, match=r"<class 'int'> is not an Operator or callable."):
            qml.ctrl(1, control=2)

    def test_ctrl_barrier_queueing(self):
        """Test that a ctrl Barrier is queued where the ctrl happens."""

        with qml.queuing.AnnotatedQueue() as q:
            op = qml.Barrier()
            qml.X(0)
            qml.ctrl(op, [1])

        assert len(q.queue) == 2
        assert q.queue[0] == qml.X(0)
        assert q.queue[1] == qml.Barrier()

    @pytest.mark.parametrize("op, ctrl_wires, expected_op", custom_ctrl_ops)
    def test_custom_controlled_ops(self, op, ctrl_wires, expected_op):
        """Tests custom controlled operations are handled correctly."""
        assert qml.ctrl(op, control=ctrl_wires) == expected_op

    @pytest.mark.parametrize("op, ctrl_wires, _", custom_ctrl_ops)
    def test_custom_controlled_ops_ctrl_on_zero(self, op, ctrl_wires, _):
        """Tests custom controlled ops with control on zero are handled correctly."""

        if isinstance(op, (qml.QubitUnitary, qml.Barrier)):
            pytest.skip("ControlledQubitUnitary and Barrier can accept any control values.")

        ctrl_values = [False] * len(ctrl_wires)

        if isinstance(op, Controlled):
            expected = Controlled(
                op.base,
                control_wires=ctrl_wires + op.control_wires,
                control_values=ctrl_values + op.control_values,
            )
        else:
            expected = Controlled(op, control_wires=ctrl_wires, control_values=ctrl_values)

        assert qml.ctrl(op, control=ctrl_wires, control_values=ctrl_values) == expected

    @pytest.mark.parametrize("op, ctrl_wires, _", custom_ctrl_ops)
    def test_custom_controlled_ops_wrong_wires(self, op, ctrl_wires, _):
        """Tests custom controlled ops with wrong number of wires are handled correctly."""
        # pylint: disable=possibly-used-before-assignment

        ctrl_wires = ctrl_wires + ["a", "b", "c"]

        if isinstance(op, (qml.QubitUnitary, qml.Barrier)):
            pytest.skip(
                "ControlledQubitUnitary and Barrier can accept any number of control wires."
            )
        elif isinstance(op, Controlled):
            expected = Controlled(
                op.base,
                control_wires=ctrl_wires + op.control_wires,
            )
        else:
            expected = Controlled(op, control_wires=ctrl_wires)

        assert qml.ctrl(op, control=ctrl_wires) == expected

    def test_nested_controls(self):
        """Tests that nested controls are flattened correctly."""

        with qml.queuing.AnnotatedQueue() as q:
            op = qml.ctrl(
                Controlled(
                    Controlled(qml.S(wires=[0]), control_wires=[1]),
                    control_wires=[2],
                    control_values=[0],
                ),
                control=[3],
            )

        assert len(q) == 1
        assert q.queue[0] is op
        expected = Controlled(
            qml.S(wires=[0]),
            control_wires=[3, 2, 1],
            control_values=[1, 0, 1],
        )
        assert op == expected

    @pytest.mark.parametrize("op, ctrl_wires, ctrl_op", custom_ctrl_ops)
    def test_nested_custom_controls(self, op, ctrl_wires, ctrl_op):
        """Tests that nested controls of custom controlled ops are flattened correctly."""

        if isinstance(ctrl_op, (qml.ControlledQubitUnitary, qml.Barrier)):
            pytest.skip("ControlledQubitUnitary and Barrier have their own logic")

        expected_base = op.base if isinstance(op, Controlled) else op
        base_ctrl_wires = (
            ctrl_wires + op.control_wires if isinstance(op, Controlled) else ctrl_wires
        )
        ctrl_values = [1] * len(ctrl_wires)
        base_ctrl_values = (
            ctrl_values + op.control_values if isinstance(op, Controlled) else ctrl_values
        )

        op = qml.ctrl(
            Controlled(
                ctrl_op,
                control_wires=["b"],
                control_values=[0],
            ),
            control=["a"],
        )
        expected = Controlled(
            expected_base,
            control_wires=["a", "b"] + base_ctrl_wires,
            control_values=[1, 0] + base_ctrl_values,
        )
        assert op == expected

    def test_nested_ctrl_qubit_unitaries(self):
        """Tests that nested controlled qubit unitaries are flattened correctly."""

        op = qml.ctrl(
            Controlled(
                qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), wires=[1, 0]),
                control_wires=[2],
                control_values=[0],
            ),
            control=[3],
        )
        expected = qml.ControlledQubitUnitary(
            np.array([[0, 1], [1, 0]]), wires=[3, 2, 1, 0], control_values=[1, 0, 1]
        )
        assert op == expected

    @pytest.mark.parametrize(
        "op, ctrl_wires, ctrl_values, expected_op",
        [
            (qml.PauliX(wires=[0]), [1], [1], qml.CNOT([1, 0])),
            (
                qml.PauliX(wires=[2]),
                [0, 1],
                [1, 1],
                qml.Toffoli(wires=[0, 1, 2]),
            ),
            (
                qml.CNOT(wires=[1, 2]),
                [0],
                [1],
                qml.Toffoli(wires=[0, 1, 2]),
            ),
            (
                qml.PauliX(wires=[0]),
                [1],
                [0],
                qml.MultiControlledX(wires=[1, 0], control_values=[0], work_wires=["aux"]),
            ),
            (
                qml.PauliX(wires=[2]),
                [0, 1],
                [1, 0],
                qml.MultiControlledX(wires=[0, 1, 2], control_values=[1, 0], work_wires=["aux"]),
            ),
            (
                qml.CNOT(wires=[1, 2]),
                [0],
                [0],
                qml.MultiControlledX(wires=[0, 1, 2], control_values=[0, 1], work_wires=["aux"]),
            ),
            (
                qml.PauliX(wires=[3]),
                [0, 1, 2],
                [1, 1, 1],
                qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
            ),
            (
                qml.CNOT(wires=[2, 3]),
                [0, 1],
                [1, 1],
                qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
            ),
            (
                qml.Toffoli(wires=[1, 2, 3]),
                [0],
                [1],
                qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=Wires("aux")),
            ),
        ],
    )
    def test_pauli_x_based_ctrl_ops(self, op, ctrl_wires, ctrl_values, expected_op):
        """Tests that PauliX-based ops are handled correctly."""
        op = qml.ctrl(op, control=ctrl_wires, control_values=ctrl_values, work_wires=["aux"])
        assert op == expected_op

    def test_nested_pauli_x_based_ctrl_ops(self):
        """Tests that nested PauliX-based ops are handled correctly."""

        op = qml.ctrl(
            Controlled(
                qml.CNOT(wires=[1, 0]),
                control_wires=[2],
                control_values=[0],
            ),
            control=[3],
        )
        expected = qml.MultiControlledX(wires=[3, 2, 1, 0], control_values=[1, 0, 1])
        assert op == expected

    def test_correct_queued_operators(self):
        """Test that args and kwargs do not add operators to the queue."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.ctrl(qml.QSVT, control=0)(qml.X(1), [qml.Z(1)])
            qml.ctrl(qml.QSVT(qml.X(1), [qml.Z(1)]), control=0)
        for op in q.queue:
            assert op.name == "C(QSVT)"

        assert len(q.queue) == 2


class _Rot(Operation):
    """A rotation operation that is not an instance of Rot

    Used to test the default behaviour of expanding tapes without custom handling
    of custom controlled operators (bypass automatic simplification of controlled
    Rot to CRot gates in decompositions).

    """

    @staticmethod
    def compute_decomposition(*params, wires=None):
        return qml.Rot.compute_decomposition(*params, wires=wires)

    def decomposition(self):
        return self.compute_decomposition(*self.parameters, wires=self.wires)


unitaries = (
    [
        qml.PauliX.compute_matrix(),
        qml.PauliY.compute_matrix(),
        qml.PauliZ.compute_matrix(),
        qml.Hadamard.compute_matrix(),
        pnp.array(
            [
                [1 + 2j, -3 + 4j],
                [3 + 4j, 1 - 2j],
            ]
        )
        * 30**-0.5,
    ],
)


class TestTapeExpansionWithControlled:
    """Tests expansion of tapes containing Controlled operations"""

    def test_ctrl_values_sanity_check(self):
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
            ctrl(make_ops, control=1, control_values=0)()

        tape = QuantumScript.from_queue(q_tape)
        expected = [
            qml.PauliX(wires=1),
            *qml.CRX(0.123, wires=[1, 0]).decomposition(),
            *qml.CRY(0.456, wires=[1, 2]).decomposition(),
            *qml.CRX(0.789, wires=[1, 0]).decomposition(),
            *qml.CRot(0.111, 0.222, 0.333, wires=[1, 2]).decomposition(),
            qml.CNOT(wires=[1, 2]),
            *qml.CY(wires=[1, 4]).decomposition(),
            *qml.CZ(wires=[1, 0]).decomposition(),
            qml.PauliX(wires=1),
        ]
        assert len(tape) == 9
        expanded = tape.expand(stop_at=lambda obj: not isinstance(obj, Controlled))
        assert expanded.circuit == expected

    @pytest.mark.parametrize(
        "op",
        [
            qml.ctrl(qml.ctrl(_Rot, 7), 3),  # nested control
            qml.ctrl(_Rot, [3, 7]),  # multi-wire control
        ],
    )
    def test_nested_ctrl(self, op, tol):
        """Tests that nested controlled ops are expanded correctly"""

        with qml.queuing.AnnotatedQueue() as q_tape:
            op(0.1, 0.2, 0.3, wires=0)

        tape = QuantumScript.from_queue(q_tape)
        assert tape.expand(depth=1).circuit == [
            Controlled(qml.RZ(0.1, 0), control_wires=[3, 7]),
            Controlled(qml.RY(0.2, 0), control_wires=[3, 7]),
            Controlled(qml.RZ(0.3, 0), control_wires=[3, 7]),
        ]

        # Tests that the decomposition of the nested controlled _Rot gate is ultimately
        # equivalent to the decomposition of the controlled CRot
        with qml.queuing.AnnotatedQueue() as q_tape:
            for op_ in qml.CRot.compute_decomposition(0.1, 0.2, 0.3, wires=[7, 0]):
                qml.ctrl(op_, control=3)
        tape_expected = QuantumScript.from_queue(q_tape)

        def stopping_condition(o):
            return not isinstance(o, Controlled) or not o.has_decomposition

        actual = tape.expand(depth=10, stop_at=stopping_condition)
        expected = tape_expected.expand(depth=10, stop_at=stopping_condition)
        actual_mat = qml.matrix(actual, wire_order=[3, 7, 0])
        expected_mat = qml.matrix(expected, wire_order=[3, 7, 0])
        assert qml.math.allclose(actual_mat, expected_mat, atol=tol, rtol=0)

    def test_adjoint_of_ctrl(self):
        """Tests that adjoint(ctrl(fn)) and ctrl(adjoint(fn)) are equivalent"""

        def my_op(a, b, c):
            qml.RX(a, wires=2)
            qml.RY(b, wires=3)
            qml.RZ(c, wires=0)

        with qml.queuing.AnnotatedQueue() as q1:
            # Execute controlled and adjoint version of my_op.
            cmy_op_dagger = qml.simplify(qml.adjoint(ctrl(my_op, 5)))
            cmy_op_dagger(0.789, 0.123, c=0.456)
        tape1 = QuantumScript.from_queue(q1)

        with qml.queuing.AnnotatedQueue() as q2:
            # Execute adjoint and controlled version of my_op.
            cmy_op_dagger = qml.simplify(ctrl(qml.adjoint(my_op), 5))
            cmy_op_dagger(0.789, 0.123, c=0.456)
        tape2 = QuantumScript.from_queue(q2)

        expected = [
            *qml.CRZ(4 * np.pi - 0.456, wires=[5, 0]).decomposition(),
            *qml.CRY(4 * np.pi - 0.123, wires=[5, 3]).decomposition(),
            *qml.CRX(4 * np.pi - 0.789, wires=[5, 2]).decomposition(),
        ]
        assert tape1.expand(depth=1).circuit == expected
        assert tape2.expand(depth=1).circuit == expected

    def test_ctrl_with_qnode(self):
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
            qml.RX(pnp.pi / 4.0, wires=2)
            ansatz(params)
            return qml.state()

        params = [0.123, 0.456, 0.789, 1.345]
        circuit1 = qml.qnode(dev)(partial(circuit, ansatz=ctrl(my_ansatz, 2)))
        circuit2 = qml.qnode(dev)(partial(circuit, ansatz=controlled_ansatz))
        res1 = circuit1(params=params)
        res2 = circuit2(params=params)
        assert qml.math.allclose(res1, res2)

    def test_ctrl_within_ctrl(self):
        """Test using ctrl on a method that uses ctrl."""

        def ansatz(params):
            qml.RX(params[0], wires=0)
            ctrl(qml.PauliX, control=0)(wires=1)
            qml.RX(params[1], wires=0)

        controlled_ansatz = ctrl(ansatz, 2)

        with qml.queuing.AnnotatedQueue() as q_tape:
            controlled_ansatz([0.123, 0.456])

        tape = QuantumScript.from_queue(q_tape)
        assert tape.expand(1).circuit == [
            *qml.CRX(0.123, wires=[2, 0]).decomposition(),
            *qml.Toffoli(wires=[2, 0, 1]).decomposition(),
            *qml.CRX(0.456, wires=[2, 0]).decomposition(),
        ]

    @pytest.mark.parametrize("ctrl_values", [[0, 0], [0, 1], [1, 0], [1, 1]])
    def test_multi_ctrl_values(self, ctrl_values):
        """Test control with a list of wires and control values."""

        def expected_ops(ctrl_val):
            exp_op = []
            ctrl_wires = [3, 7]
            for i, j in enumerate(ctrl_val):
                if not bool(j):
                    exp_op.append(qml.PauliX(ctrl_wires[i]))
            exp_op.append(Controlled(qml.PhaseShift(pnp.pi / 2, [0]), [3, 7]))
            for i, j in enumerate(ctrl_val):
                if not bool(j):
                    exp_op.append(qml.PauliX(ctrl_wires[i]))

            return exp_op

        with qml.queuing.AnnotatedQueue() as q_tape:
            ctrl(qml.S, control=[3, 7], control_values=ctrl_values)(wires=0)
        tape = QuantumScript.from_queue(q_tape)
        assert len(tape.operations) == 1
        op = tape.operations[0]
        assert isinstance(op, Controlled)
        new_tape = expand_tape(tape, 1)
        assert equal_list(list(new_tape), expected_ops(ctrl_values))

    def test_diagonal_ctrl(self):
        """Test ctrl on diagonal gates."""
        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.ctrl(qml.DiagonalQubitUnitary, 1)(np.array([-1.0, 1.0j]), wires=0)
        tape = QuantumScript.from_queue(q_tape)
        tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))
        assert tape[0] == qml.DiagonalQubitUnitary(np.array([1.0, 1.0, -1.0, 1.0j]), wires=[1, 0])

    @pytest.mark.parametrize("M", unitaries)
    def test_qubit_unitary(self, M):
        """Test ctrl on QubitUnitary"""
        with qml.queuing.AnnotatedQueue() as q_tape:
            ctrl(qml.QubitUnitary, 1)(M, wires=0)

        tape = QuantumScript.from_queue(q_tape)
        expected = qml.ControlledQubitUnitary(M, wires=[1, 0])
        assert equal_list(list(tape), expected)

        # causes decomposition into more basic operators
        tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))
        assert not equal_list(list(tape), expected)

    @pytest.mark.parametrize("M", unitaries)
    def test_controlled_qubit_unitary(self, M):
        """Test ctrl on ControlledQubitUnitary."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            ctrl(qml.ControlledQubitUnitary, 1)(M, wires=[2, 0])

        tape = QuantumScript.from_queue(q_tape)
        # will immediately decompose according to selected decomposition algorithm
        tape = tape.expand(1, stop_at=lambda op: not isinstance(op, Controlled))

        expected = qml.ControlledQubitUnitary(M, wires=[1, 2, 0]).decomposition()
        assert tape.circuit == expected

    @pytest.mark.parametrize(
        "op, params, depth, expected",
        [
            (qml.templates.QFT, [], 2, 11),
            (qml.templates.BasicEntanglerLayers, [pnp.ones([3, 2])], 1, 9),
        ],
    )
    def test_ctrl_templates(self, op, params, depth, expected):
        """Test ctrl on two different templates."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            ctrl(op, 2)(*params, wires=[0, 1])
        tape = QuantumScript.from_queue(q_tape)
        expanded_tape = tape.expand(depth=depth)
        assert len(expanded_tape.operations) == expected

    def test_ctrl_template_and_operations(self):
        """Test that a combination of controlled templates and operations correctly expands
        on a device that doesn't support it"""

        weights = pnp.ones([3, 2])

        def ansatz(weights, wires):
            qml.PauliX(wires=wires[0])
            qml.templates.BasicEntanglerLayers(weights, wires=wires)

        with qml.queuing.AnnotatedQueue() as q_tape:
            ctrl(ansatz, 0)(weights, wires=[1, 2])

        tape = QuantumScript.from_queue(q_tape)
        tape = tape.expand(depth=1, stop_at=lambda obj: not isinstance(obj, Controlled))
        assert len(tape.operations) == 10
        assert all(o.name in {"CNOT", "CRX", "Toffoli"} for o in tape.operations)


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
class TestCtrlTransformDifferentiation:
    """Tests for differentiation"""

    @pytest.mark.autograd
    def test_autograd(self, diff_method):
        """Test differentiation using autograd"""

        dev = qml.device("default.qubit", wires=2)
        init_state = pnp.array([1.0, -1.0], requires_grad=False) / pnp.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = pnp.array(0.123, requires_grad=True)
        res = qml.grad(circuit)(b)
        expected = pnp.sin(b / 2) / 2

        assert pnp.allclose(res, expected)

    @pytest.mark.torch
    def test_torch(self, diff_method):
        """Test differentiation using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)
        init_state = torch.tensor(
            [1.0, -1.0], requires_grad=False, dtype=torch.complex128
        ) / pnp.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = torch.tensor(0.123, requires_grad=True, dtype=torch.float64)
        loss = circuit(b)
        loss.backward()  # pylint:disable=no-member

        res = b.grad.detach()
        expected = pnp.sin(b.detach() / 2) / 2

        assert pnp.allclose(res, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["auto", "jax"])
    def test_jax(self, diff_method, jax_interface):
        """Test differentiation using JAX"""

        import jax

        jax.config.update("jax_enable_x64", True)

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method, interface=jax_interface)
        def circuit(b):
            init_state = np.array([1.0, -1.0]) / np.sqrt(2)
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = jnp.array(0.123)
        res = jax.grad(circuit)(b)
        expected = pnp.sin(b / 2) / 2

        assert pnp.allclose(res, expected)

    @pytest.mark.tf
    def test_tf(self, diff_method):
        """Test differentiation using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        init_state = tf.constant([1.0, -1.0], dtype=tf.complex128) / pnp.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.StatePrep(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = tf.Variable(0.123, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = circuit(b)

        res = tape.gradient(loss, b)
        expected = pnp.sin(b / 2) / 2

        assert pnp.allclose(res, expected)
