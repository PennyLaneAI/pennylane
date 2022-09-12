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

from copy import copy

import numpy as onp
import pytest
from gate_data import CNOT, CSWAP, CZ, CRot3, CRotx, CRoty, CRotz, Toffoli
from scipy import sparse

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import (
    DecompositionUndefinedError,
    GeneratorUndefinedError,
    Operation,
    Operator,
)
from pennylane.ops.op_math.controlled_class import (
    Controlled,
    ControlledOp,
    _decompose_no_control_values,
)
from pennylane.wires import Wires

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

        assert type(op) is ControlledOp

        assert isinstance(op, Controlled)
        assert isinstance(op, Operator)
        assert isinstance(op, Operation)
        assert isinstance(op, ControlledOp)

    def test_controlledop_new(self):
        """Test that if a `ControlledOp` is directly requested, it is created
        even if the base isn't an operation."""

        base = TempOperator(1.234, wires="a")
        op = ControlledOp(base, "b")

        assert type(op) is ControlledOp


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
        assert op.parameters == []
        assert op.data == []

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

        assert op.data == [x]

        x_new = np.array(2.3454)
        op.data = x_new
        assert op.data == [x_new]
        assert base.data == [x_new]

        x_new2 = np.array(3.456)
        base.data = x_new2
        assert op.data == [x_new2]
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
        """Test that controlled defers has_matrix to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            has_matrix = value

        op = Controlled(DummyOp(1), 0)
        assert op.has_matrix is value

    def test_no_matrix_if_batching(self):
        """Test that has_matrix is false if the base operator has batching."""

        x = np.array([1.2, 3.4, 5.6])
        base = qml.RX(x, wires=0)
        op = Controlled(base, 1)
        assert op.has_matrix is False

    @pytest.mark.parametrize("value", ("_ops", "_prep", None))
    def test_queue_cateogry(self, value):
        """Test that Controlled defers `_queue_category` to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            _queue_category = value

        op = Controlled(DummyOp(1), 0)
        assert op._queue_category == value

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value):
        """Test that controlled defers `is_hermitian` to base operator."""

        class DummyOp(Operator):
            num_wires = 1
            is_hermitian = value

        op = Controlled(DummyOp(1), 0)
        assert op.is_hermitian is value

    def test_private_wires_getter_setter(self):
        """Test that we can get and set private wires."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        op = Controlled(base, (3, 4), work_wires="aux")

        assert op._wires == Wires((3, 4, 0, 1, "aux"))

        op._wires = ("a", "b", "c", "d", "extra")

        assert base.wires == Wires(("c", "d"))
        assert op.control_wires == Wires(("a", "b"))
        assert op.work_wires == Wires(("extra"))

    def test_private_wires_setter_too_few_wires(self):
        """Test that an assertionerror is raised if wires are set with fewer active wires
        than the operation originally had."""
        base = qml.IsingXX(1.234, wires=(0, 1))
        op = Controlled(base, (3, 4), work_wires="aux")

        with pytest.raises(AssertionError, match=r"C\(IsingXX\) needs at least 4 wires."):
            op._wires = ("a", "b")

    def test_private_wires_setter_no_work_wires(self):
        """Test work wires made empty if no left over wires provided to private setter."""
        base = TempOperator(1)
        op = Controlled(base, 2, work_wires="aux")

        op._wires = [3, 4]
        assert len(op.work_wires) == 0
        assert isinstance(op.work_wires, qml.wires.Wires)


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

    def test_copy(self):
        """Test that a copy of a controlled oeprator can have its parameters updated
        independently of the original operator."""

        param1 = 1.234
        base_wire = "a"
        control_wires = [0, 1]
        base = qml.RX(param1, base_wire)
        op = Controlled(base, control_wires, control_values=[0, 1])

        copied_op = copy(op)

        assert qml.equal(copied_op, op)

        copied_op.data = [6.54]
        assert op.data == [param1]

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

    def test_generator(self):
        """Test that the generator is a tensor product of projectors and the base's generator."""

        base = qml.RZ(-0.123, wires="a")
        op = Controlled(base, ("b", "c"))

        base_gen, base_gen_coeff = qml.generator(base, format="prefactor")
        gen_tensor, gen_coeff = qml.generator(op, format="prefactor")

        assert base_gen_coeff == gen_coeff

        for wire, ob in zip(op.control_wires, gen_tensor.obs):
            assert isinstance(ob, qml.Projector)
            assert ob.data == [[1]]
            assert ob.wires == qml.wires.Wires(wire)

        assert gen_tensor.obs[-1].__class__ is base_gen.__class__
        assert gen_tensor.obs[-1].wires == base_gen.wires

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

    def test_invert_controlled_op(self):
        """Test that in-place inversion of a power operator inverts the base operator."""

        base = qml.S(0)
        op = Controlled(base, 2)

        assert op.inverse == base.inverse == False
        assert op.name == "C(S)"

        op.inv()

        assert op.inverse == False
        assert base.inverse == True
        assert op.name == "C(S.inv)"
        assert op.base_name == "C(S)"

    def test_inverse_setter(self):
        """Teest that the inverse property can be set."""
        base = qml.T(0)
        op = Controlled(base, 1)

        assert op.inverse == base.inverse == False
        assert op.name == "C(T)"

        op.inverse = True

        assert op.inverse == False
        assert base.inverse == True
        assert op.name == "C(T.inv)"
        assert op.base_name == "C(T)"

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
    """Test qml.op_sum simplify method and depth property."""

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
            qml.op_sum(qml.RZ(1.32, wires=0), qml.Identity(wires=0), qml.RX(1.9, wires=1)),
            control_wires=2,
        )
        simplified_op = controlled_op.simplify()

        assert qml.equal(final_op, simplified_op)

    def test_simplify_nested_controlled_ops(self):
        """Test the simplify method with nested control operations on different wires."""
        controlled_op = Controlled(Controlled(qml.PauliX(0), 1), 2)
        final_op = Controlled(qml.PauliX(0), [2, 1])
        simplified_op = controlled_op.simplify()

        assert qml.equal(final_op, simplified_op)


class TestQueuing:
    """Test that Controlled operators queue and update base metadata."""

    def test_queuing(self):
        """Test that `Controlled` is queued upon initialization and updates base metadata."""
        with qml.tape.QuantumTape() as tape:
            base = qml.Rot(1.234, 2.345, 3.456, wires=2)
            op = Controlled(base, (0, 1))

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_queuing_base_defined_outside(self):
        """Test that base isn't added to queue if its defined outside the recording context."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        with qml.tape.QuantumTape() as tape:
            op = Controlled(base, ("a", "b"))

        assert len(tape._queue) == 1
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_do_queue_false(self):
        """Test that when `do_queue=False` is specified, the controlled op is not queued."""

        base = qml.PauliX(0)
        with qml.tape.QuantumTape() as tape:
            op = Controlled(base, 1, do_queue=False)

        assert len(tape._queue) == 0


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

    def test_raises_error_if_batching(self):
        """Test a MatrixUndefinedError is raised if the base is batched."""
        x = np.array([1.0, 2.0, 3.0])
        base = qml.RX(x, 0)
        op = Controlled(base, 1)

        with pytest.raises(qml.operation.MatrixUndefinedError):
            op.matrix()

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
        with qml.tape.QuantumTape() as tape:
            [qml.PauliX(w) for w, val in zip(control_wires, control_values) if not val]
            Controlled(base, control_wires, control_values=[1, 1, 1])
            [qml.PauliX(w) for w, val in zip(control_wires, control_values) if not val]
        decomp_mat = qml.matrix(tape, wire_order=op.wires)

        assert qml.math.allclose(mat, decomp_mat)

    def test_sparse_matrix_base_defines(self):
        """Check that an op that defines a sparse matrix has it used in the controlled
        sparse matrix."""

        Hmat = qml.utils.sparse_hamiltonian(1.0 * qml.PauliX(0))
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

    def test_inverts_decomp_if_target_inverted(self):
        """Tests that the method preserves the inverse parameter."""
        base = qml.RX(1.0, wires=0).inv()
        op = Controlled(base, 1)
        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 1
        assert decomp[0].inverse

    def test_toffoli(self):
        """Test case when PauliX with two controls."""
        op = Controlled(qml.PauliX("c"), ("a", 2))
        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 1
        assert qml.equal(decomp[0], qml.Toffoli(("a", 2, "c")))

    def test_multicontrolledx(self):
        """Test case when PauliX has many controls."""
        op = Controlled(qml.PauliX(4), (0, 1, 2, 3))
        decomp = _decompose_no_control_values(op)
        assert len(decomp) == 1
        assert qml.equal(decomp[0], qml.MultiControlledX(wires=(0, 1, 2, 3, 4)))

    @pytest.mark.parametrize("inverse", (True, False))
    def test_decomposes_target(self, inverse):
        """Test that we decompose the target if we don't have a special case."""
        target = qml.IsingXX(1.0, wires=(0, 1))
        target.inverse = inverse
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
        op = Controlled(base, (0, 1, 2), [True, False, False])

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
        expected = [qml.PauliX(1), qml.Toffoli((0, 1, 2)), qml.PauliX(1)]
        for op1, op2 in zip(decomp, expected):
            assert qml.equal(op1, op2)

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
        from pennylane import numpy as pnp

        dev = qml.device("default.qubit", wires=2)
        init_state = pnp.array([1.0, -1.0], requires_grad=False) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.QubitStateVector(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = pnp.array(0.123, requires_grad=True)
        res = qml.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.torch
    def test_torch(self, diff_method):
        """Test differentiation using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)
        init_state = torch.tensor([1.0, -1.0], requires_grad=False) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(b):
            qml.QubitStateVector(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = torch.tensor(0.123, requires_grad=True)
        loss = circuit(b)
        loss.backward()

        res = b.grad.detach()
        expected = np.sin(b.detach() / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["jax", "jax-python", "jax-jit"])
    def test_jax(self, diff_method, jax_interface):
        """Test differentiation using JAX"""

        import jax

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method, interface=jax_interface)
        def circuit(b):
            init_state = onp.array([1.0, -1.0]) / np.sqrt(2)
            qml.QubitStateVector(init_state, wires=0)
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

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(b):
            qml.QubitStateVector(init_state, wires=0)
            Controlled(qml.RY(b, wires=1), control_wires=0)
            return qml.expval(qml.PauliX(0))

        b = tf.Variable(0.123, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = circuit(b)

        res = tape.gradient(loss, b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)
