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

import pytest
from copy import copy
from scipy import sparse

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import DecompositionUndefinedError
from pennylane.ops.op_math.controlled_class import Controlled, ControlledOperation
from pennylane.wires import Wires


from gate_data import CNOT, CZ, CSWAP, Toffoli, CRotx, CRoty, CRotz, CRot3

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


class TempOperator(qml.operation.Operator):
    num_wires = 1


class TestInheritanceMixins:
    """Test the inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self):
        """Test when base directly inherits from Operator only inherits from Operator."""

        base = TempOperator(1.234, wires=0)
        op = Controlled(base, 1.2)

        assert isinstance(op, Controlled)
        assert isinstance(op, qml.operation.Operator)
        assert not isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
        assert not isinstance(op, ControlledOperation)

        # checking we can call `dir` without problems
        assert "num_params" in dir(op)

    def test_operation(self):
        """When the operation inherits from `Operation`, the `ControlledOperation` mixin should
        be added and the Controlled should now have Operation functionality."""

        class CustomOp(qml.operation.Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op = Controlled(base, 6.5)

        assert isinstance(op, Controlled)
        assert isinstance(op, qml.operation.Operator)
        assert isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
        assert isinstance(op, ControlledOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)

    def test_observable(self):
        """Test that when the base is an Observable, Adjoint will also inherit from Observable."""

        class CustomObs(qml.operation.Observable):
            num_wires = 1
            num_params = 0

        base = CustomObs(wires=0)
        ob = Controlled(base, -1.2)

        assert isinstance(ob, Controlled)
        assert isinstance(ob, qml.operation.Operator)
        assert not isinstance(ob, qml.operation.Operation)
        assert isinstance(ob, qml.operation.Observable)
        assert not isinstance(ob, ControlledOperation)

        # Check some basic observable functionality
        assert ob.compare(ob)
        assert isinstance(1.0 * ob @ ob, qml.Hamiltonian)

        # check the dir
        assert "return_type" in dir(ob)
        assert "grad_recipe" not in dir(ob)


class TestInitialization:
    """Test the initialization process and standard properties."""

    paulix_op = qml.PauliX("a")

    def test_nonparametric_ops(self):
        """Test pow initialization for a non parameteric operation."""

        op = Controlled(
            self.paulix_op, (0, 1), control_values=[True, False], work_wires="aux", id="something"
        )

        assert op.base is self.paulix_op
        assert op.hyperparameters["base"] is self.paulix_op

        assert op.wires == Wires((0, 1, "a", "aux"))

        assert op.control_wires == Wires((0, 1))
        assert op.hyperparameters["control_wires"] == Wires((0, 1))

        assert op.target_wires == Wires("a")

        assert op.control_values == [True, False]
        assert op.hyperparameters["control_values"] == [True, False]

        assert op.work_wires == Wires(("aux"))

        assert op.name == "CPauliX"
        assert op.id == "something"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == []

        assert op.num_wires == 4

    def test_default_control_values(self):

        op = Controlled(self.paulix_op, (0, 1))
        assert op.control_values == [True, True]

    def test_zero_one_control_values(self):

        op = Controlled(self.paulix_op, (0, 1), control_values=[0, 1])
        assert op.control_values == [False, True]

    def test_string_control_values(self):

        with pytest.warns(UserWarning, match="Specifying control values as a string"):
            op = Controlled(self.paulix_op, (0, 1), "01")

        assert op.control_values == [False, True]

    def test_non_boolean_control_values(self):

        with pytest.raises(AssertionError, match="control_values can only take on"):
            Controlled(self.paulix_op, (0, 1), ["b", 2])

    def test_control_values_wrong_length(self):

        with pytest.raises(AssertionError, match="control_values should be the same length"):
            Controlled(self.paulix_op, (0, 1), [True])

    def test_target_control_wires_overlap(self):

        with pytest.raises(AssertionError, match="The control wires must be different"):
            Controlled(self.paulix_op, "a")


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

        class DummyOp(qml.operation.Operator):
            num_wires = 1
            has_matrix = value

        op = Controlled(DummyOp(1), 0)
        assert op.has_matrix is value

    @pytest.mark.parametrize("value", ("_ops", "_prep", None))
    def test_queue_cateogry(self, value):
        """Test that Controlled defers `_queue_category` to base operator."""

        class DummyOp(qml.operation.Operator):
            num_wires = 1
            _queue_category = value

        op = Controlled(DummyOp(1), 0)
        assert op._queue_category == value

    @pytest.mark.parametrize("value", (True, False))
    def test_is_hermitian(self, value):
        """Test that controlled defers `is_hermitian` to base operator."""

        class DummyOp(qml.operation.Operator):
            num_wires = 1
            is_hermitian = value

        op = Controlled(DummyOp(1), 0)
        assert op.is_hermitian is value

    def test_batching_properties(self):
        """Test that Adjoint batching behavior mirrors that of the base."""

        class DummyOp(qml.operation.Operator):
            ndim_params = (0, 2)
            num_wires = 1

        param1 = [0.3] * 3
        param2 = [[[0.3, 1.2]]] * 3

        base = DummyOp(param1, param2, wires=0)
        op = Controlled(base, 1)

        assert op.ndim_params == (0, 2)
        assert op.batch_size == 3

    def test_private_wires_getter_setter(self):
        """Test that we can get and set private wires."""

        base = qml.IsingXX(1.234, wires=(0, 1))
        op = Controlled(base, (3, 4), work_wires="aux")

        assert op._wires == Wires((3, 4, 0, 1, "aux"))

        op._wires = ("a", "b", "c", "d", "extra")

        assert base.wires == Wires(("c", "d"))
        assert op.control_wires == Wires(("a", "b"))
        assert op.work_wires == Wires(("extra"))

    def test_wires_setter_too_few_wires(self):
        """Test that an assertionerror is raised if wires are set with fewer active wires
        than the operation originally had."""
        base = qml.IsingXX(1.234, wires=(0, 1))
        op = Controlled(base, (3, 4), work_wires="aux")

        with pytest.raises(AssertionError, match="CIsingXX needs at least 4 wires."):
            op._wires = ("a", "b")


class TestMiscMethods:
    """Test miscellaneous minor Controlled methods."""

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
        assert copied_op.data == [param1]

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


class TestOperationProperties:
    """Test ControlledOperation specific properties."""

    def test_invert_controlled_op(self):
        """Test that in-place inversion of a power operator inverts the base operator."""

        base = qml.S(0)
        op = Controlled(base, 2)

        assert op.inverse == base.inverse == False
        assert op.name == "CS"

        op.inv()

        assert op.inverse == False
        assert base.inverse == True
        assert op.name == "CS.inv"
        assert op.base_name == "CS"

    def test_inverse_setter(self):

        base = qml.T(0)
        op = Controlled(base, 1)

        assert op.inverse == base.inverse == False
        assert op.name == "CT"

        op.inverse = True

        assert op.inverse == False
        assert base.inverse == True
        assert op.name == "CT.inv"
        assert op.base_name == "CT"


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

    @pytest.mark.parametrize("base, num_control, mat", base_num_control_mats)
    def test_matrix_compare_with_gate_data(self, base, num_control, mat):
        """Test the matrix against matrices provided by `gate_data` file."""
        op = Controlled(base, list(range(num_control)))
        assert qml.math.allclose(op.matrix(), mat)

    def test_wire_order(self):
        """Test that the ``wire_order`` keyword argument alters the matrix as expected."""
        base = qml.RX(-4.432, wires=1)
        op = Controlled(base, 0)

        method_order = op.matrix(wire_order=(1, 0))
        function_order = qml.operation.expand_matrix(op.matrix(), op.wires, (1, 0))

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

    @pytest.mark.parametrize("control_values", ([0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]))
    def test_sparse_matrix(self, control_values):
        """Check that the dense form"""
        control_wires = (0, 1, 2)
        base = qml.U2(1.234, -3.2, wires=3)
        op = Controlled(base, control_wires, control_values=control_values)

        sparse_mat = op.sparse_matrix()
        assert isinstance(sparse_mat, sparse.csr_matrix)
        assert qml.math.allclose(op.sparse_matrix().toarray(), op.matrix())
