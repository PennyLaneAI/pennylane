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
"""Tests for the @op_transform framework"""
# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.op_transforms import OperationTransformError


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*op_transform.*:pennylane.PennyLaneDeprecationWarning"
)


class TestValidation:
    """Test for validation and exceptions"""

    def test_sphinx_build(self, monkeypatch):
        """Test that op transforms are not created during Sphinx builds"""

        @qml.op_transform
        def my_transform0(op):
            return op.name

        assert isinstance(my_transform0, qml.op_transform)

        monkeypatch.setenv("SPHINX_BUILD", "1")

        with pytest.warns(UserWarning, match="Operator transformations have been disabled"):

            @qml.op_transform
            def my_transform1(op):
                return op.name

        assert not isinstance(my_transform1, qml.batch_transform)

    def test_error_invalid_callable(self):
        """Test that an error is raised if the transform
        is applied to an invalid function"""

        with pytest.raises(
            OperationTransformError, match="does not appear to be a valid Python function"
        ):
            qml.op_transform(5)

    def test_unknown_object(self):
        """Test that an error is raised if the transform
        is applied to an unknown object"""

        @qml.op_transform
        def my_transform(op):
            return op.name

        with pytest.raises(
            OperationTransformError,
            match="Input is not an Operator, tape, QNode, or quantum function",
        ):
            my_transform(5)(5)

    def test_empty_qfunc(self):
        """Test that an error is raised if the qfunc has no quantum operations
        (e.g., it is not a qfunc)"""

        @qml.op_transform
        def my_transform(op):
            return op.name

        def qfunc(x):
            return x**2

        with pytest.raises(
            OperationTransformError,
            match="Quantum function contains no quantum operations",
        ):
            my_transform(qfunc)(0.5)


class TestUI:
    """Test the user interface of the op_transform, and ensure it applies
    and works well for all combinations of inputs and styles"""

    def test_instantiated_operator(self):
        """Test that a transform can be applied to an instantiated operator"""

        @qml.op_transform
        def my_transform(op):
            return op.name

        op = qml.CRX(0.5, wires=[0, 2])
        res = my_transform(op)
        assert res == "CRX"

    def test_single_operator_qfunc(self, mocker):
        """Test that a transform can be applied to a quantum function
        that contains a single operation"""
        spy = mocker.spy(qml.op_transform, "_make_tape")

        @qml.op_transform
        def my_transform(op):
            return op.name

        # pylint:disable=assignment-from-no-return
        res = my_transform(qml.CRX)(0.5, wires=[0, "a"])
        assert res == "CRX"

        # check default wire order
        assert spy.spy_return[1].tolist() == [0, "a"]

    def test_multiple_operator_error(self):
        """Test that an exception is raised if the transform
        is applied to a multi-op quantum function, without
        the corresponding behaviour being registered"""

        @qml.op_transform
        def my_transform(op):
            return op.name

        @my_transform
        def multi_op_qfunc(x):
            qml.RX(x, wires=0)
            qml.RY(0.65, wires=1)

        with pytest.raises(
            OperationTransformError, match="transform does not support .+ multiple operations"
        ):
            multi_op_qfunc(1.5)

    def test_multiple_operator_tape(self, mocker):
        """Test that a transform can be applied to a quantum function
        with multiple operations as long as it is registered _how_
        the transform applies to multiple operations."""
        spy = mocker.spy(qml.op_transform, "_make_tape")

        @qml.op_transform
        def my_transform(op):
            return op.name

        @my_transform.tape_transform
        def my_transform(tape):
            return [op.name for op in tape.operations]

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.6, wires=0)
            qml.RY(0.65, wires="a")

        tape = qml.tape.QuantumScript.from_queue(q)
        res = my_transform(tape)
        assert res == ["RX", "RY"]

        # check default wire order
        assert spy.spy_return[1].tolist() == [0, "a"]

        qs = qml.tape.QuantumScript(tape.operations)
        res_qs = my_transform(qs)
        assert res_qs == ["RX", "RY"]

    def test_multiple_operator_qfunc(self, mocker):
        """Test that a transform can be applied to a quantum function
        with multiple operations as long as it is registered _how_
        the transform applies to multiple operations."""
        spy = mocker.spy(qml.op_transform, "_make_tape")

        @qml.op_transform
        def my_transform(op):
            return op.name

        @my_transform.tape_transform
        def my_transform(tape):
            return [op.name for op in tape.operations]

        @my_transform
        def multi_op_qfunc(x):
            if x > 1:
                qml.RX(x, wires=0)
                qml.RY(0.65, wires="a")
            else:
                qml.RZ(x, wires="b")

        res = multi_op_qfunc(1.5)
        assert res == ["RX", "RY"]
        # check default wire order
        assert spy.spy_return[1].tolist() == [0, "a"]

        res = multi_op_qfunc(0.5)
        assert res == ["RZ"]
        # check default wire order
        assert spy.spy_return[1].tolist() == ["b"]

    def test_qnode(self, mocker):
        """Test that a transform can be applied to a QNode
        with multiple operations as long as it is registered _how_
        the transform applies to multiple operations."""
        dev = qml.device("default.qubit", wires=["a", 0, 3])
        spy = mocker.spy(qml.op_transform, "_make_tape")

        @qml.op_transform
        def my_transform(op):
            return op.name

        @my_transform.tape_transform
        def my_transform(tape):
            return [op.name for op in tape.operations]

        @my_transform
        @qml.qnode(dev)
        def multi_op_qfunc(x):
            if x > 1:
                qml.RX(x, wires=0)
                qml.RY(0.65, wires="a")
            else:
                qml.RZ(x, wires=0)

            return qml.probs(wires="a")

        res = multi_op_qfunc(1.5)
        assert res == ["RX", "RY"]
        # check default wire order
        assert spy.spy_return[1] == dev.wires

        res = multi_op_qfunc(0.5)
        assert res == ["RZ"]
        # check default wire order
        assert spy.spy_return[1] == dev.wires


class TestTransformParameters:
    def test_instantiated_operator(self):
        """Test that a transform can be applied to an instantiated operator"""

        @qml.op_transform
        def my_transform(op, lower=False):
            return op.name.lower() if lower else op.name

        op = qml.RX(0.5, wires=0)
        res = my_transform(op, lower=True)
        assert res == "rx"

    def test_single_operator_qfunc(self):
        """Test that a transform can be applied to a quantum function"""

        @qml.op_transform
        def my_transform(op, lower=False):
            return op.name.lower() if lower else op.name

        res = my_transform(qml.RX, lower=True)(0.5, wires=0)
        assert res == "rx"

    def test_transform_parameters_qfunc_decorator(self):
        """Test that transform parameters correctly work
        when used as a decorator"""

        @qml.op_transform
        def my_transform(op, lower=False):
            return op.name.lower() if lower else op.name

        @my_transform.tape_transform
        def my_transform(tape, lower=False):
            if lower:
                return [op.name.lower() for op in tape.operations]
            return [op.name for op in tape.operations]

        @my_transform(True)
        def multi_op_qfunc(x):
            if x > 1:
                qml.RX(x, wires=0)
                qml.RY(0.65, wires=1)
            else:
                qml.RZ(x, wires=0)

        res = multi_op_qfunc(1.5)
        assert res == ["rx", "ry"]

        res = multi_op_qfunc(0.5)
        assert res == ["rz"]


def simplify_rotation(op):
    """Simplify Rot(x, 0, 0) to RZ(x) or Rot(0,0,0) to Identity"""
    if op.name == "Rot":
        params = op.parameters
        wires = op.wires

        if qml.math.allclose(params, 0):
            return None

        if qml.math.allclose(params[1:2], 0):
            return qml.RZ(params[0], wires)

    return op


with pytest.warns(qml.PennyLaneDeprecationWarning):
    simplify_rotation = qml.op_transform(simplify_rotation)

    @simplify_rotation.tape_transform
    @qml.qfunc_transform
    def simplify_rotation(tape):
        """Define how simplify rotation works on a tape"""
        for op in tape.operations:
            if op.name == "Rot":
                simplify_rotation(op)
            else:
                qml.apply(op)
        for mp in tape.measurements:
            qml.apply(mp)


class TestQFuncTransformIntegration:
    """Test that @qfunc_transform seamlessly integrates
    with an operator transform."""

    def test_instantiated_operator(self):
        """Test a qfunc and operator transform applied to
        an op"""
        dev = qml.device("default.qubit", wires=2)
        assert simplify_rotation.is_qfunc_transform

        weights = np.array([0.5, 0, 0])
        op = qml.Rot(*weights, wires=0)
        res = simplify_rotation(op)
        assert res.name == "RZ"

        @qml.qnode(dev)
        def circuit():
            simplify_rotation(op)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(1))

        circuit()
        ops = circuit.tape.operations
        assert len(ops) == 2
        assert ops[0].name == "RZ"
        assert ops[1].name == "CNOT"

    def test_qfunc_inside(self):
        """Test a qfunc and operator transform
        applied to a qfunc inside a qfunc"""
        # pylint: disable=not-callable
        dev = qml.device("default.qubit", wires=2)

        def ansatz(weights):
            qml.Rot(*weights, wires=0)
            qml.CRX(0.5, wires=[0, 1])

        @qml.qnode(dev)
        def circuit(weights):
            simplify_rotation(ansatz)(weights)  # <--- qfunc is applied within circuit (inside)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(1))

        weights = np.array([0.1, 0.0, 0.0])
        circuit(weights)

        ops = circuit.tape.operations
        assert len(ops) == 3
        assert ops[0].name == "RZ"
        assert ops[1].name == "CRX"
        assert ops[2].name == "CNOT"

        weights = np.array([0.0, 0.0, 0.0])
        circuit(weights)

        ops = circuit.tape.operations
        assert len(ops) == 2
        assert ops[0].name == "CRX"
        assert ops[1].name == "CNOT"

    def test_qfunc_outside(self):
        """Test a qfunc and operator transform
        applied to qfunc"""
        dev = qml.device("default.qubit", wires=2)

        def ansatz(weights):
            qml.Rot(*weights, wires=0)
            qml.CRX(0.5, wires=[0, 1])

        @qml.qnode(dev)
        @simplify_rotation  # <--- qfunc is applied to circuit (outside)
        def circuit(weights):
            ansatz(weights)
            qml.CNOT(wires=[0, 1])
            qml.Rot(0.0, 0.0, 0.0, wires=0)
            return qml.expval(qml.PauliX(1))

        weights = np.array([0.1, 0.0, 0.0])
        circuit(weights)

        ops = circuit.tape.operations
        assert len(ops) == 3
        assert ops[0].name == "RZ"
        assert ops[1].name == "CRX"
        assert ops[2].name == "CNOT"

    @pytest.mark.xfail(reason="op transform not done yet")
    def test_compilation_pipeline(self):
        """Test a qfunc and operator transform
        applied to qfunc"""
        dev = qml.device("default.qubit", wires=2)

        def ansatz(weights):
            qml.Rot(*weights, wires=0)
            qml.CRX(0.5, wires=[0, 1])

        @qml.qnode(dev)
        @qml.compile(pipeline=[simplify_rotation])
        def circuit(weights):
            ansatz(weights)
            qml.CNOT(wires=[0, 1])
            qml.Rot(0.0, 0.0, 0.0, wires=0)
            return qml.expval(qml.PauliX(1))

        weights = np.array([0.1, 0.0, 0.0])
        circuit(weights)

        ops = circuit.tape.operations
        assert len(ops) == 3
        assert ops[0].name == "RZ"
        assert ops[1].name == "CRX"
        assert ops[2].name == "CNOT"

    def test_qnode_error(self):
        """Since a qfunc transform always returns a qfunc,
        it cannot be applied to a QNode."""
        dev = qml.device("default.qubit", wires=2)

        def ansatz(weights):
            qml.Rot(*weights, wires=0)
            qml.CRX(0.5, wires=[0, 1])

        @simplify_rotation
        @qml.qnode(dev)
        def circuit(weights):
            ansatz(weights)
            qml.CNOT(wires=[0, 1])
            qml.Rot(0.0, 0.0, 0.0, wires=0)
            return qml.expval(qml.PauliX(1))

        weights = np.array([0.1, 0.0, 0.0])

        with pytest.raises(ValueError, match="QNodes cannot be declared as qfunc transforms"):
            circuit(weights)


class TestExpansion:
    """Test for operator and tape expansion"""

    def test_auto_expansion(self):
        """Test that an operator is automatically expanded as needed"""

        @qml.op_transform
        def get_matrix(op):
            return op.matrix()

        weights = np.ones([2, 3, 3])
        op = qml.StronglyEntanglingLayers(weights, wires=[0, 2, "a"])

        # strongly entangling layers does not define a matrix representation

        with pytest.raises(qml.operation.MatrixUndefinedError):
            op.matrix()

        # attempting to call our operator transform will fail

        with pytest.raises(qml.operation.MatrixUndefinedError):
            get_matrix(op)

        # if we define how the transform acts on a tape,
        # then pennylane will automatically expand the object
        # and apply the tape transform

        @get_matrix.tape_transform
        def _(tape):
            n_wires = len(tape.wires)
            unitary_matrix = np.eye(2**n_wires)

            for op in tape.operations:
                mat = qml.math.expand_matrix(get_matrix(op), op.wires, tape.wires)
                unitary_matrix = mat @ unitary_matrix

            return unitary_matrix

        res = get_matrix(op)
        assert isinstance(res, np.ndarray)
        assert res.shape == (2**3, 2**3)


with pytest.warns(qml.PennyLaneDeprecationWarning, match="Use of `op_transform`"):
    matrix = qml.op_transform(lambda op, wire_order=None: op.matrix(wire_order=wire_order))


@matrix.tape_transform
def matrix_tape(tape, wire_order=None):
    n_wires = len(wire_order)
    unitary_matrix = np.eye(2**n_wires)

    for op in tape.operations:
        unitary_matrix = matrix(op, wire_order=wire_order) @ unitary_matrix

    return unitary_matrix


class TestWireOrder:
    """Test for wire re-ordering"""

    def test_instantiated_operator(self):
        """Test that wire order can be passed to an instantiated operator"""
        op = qml.PauliZ(wires=0)
        res = matrix(op, wire_order=[1, 0])
        expected = np.kron(np.eye(2), np.diag([1, -1]))
        assert np.allclose(res, expected)

    def test_single_operator_qfunc(self, mocker):
        """Test that wire order can be passed to a quantum function"""
        spy = mocker.spy(qml.op_transform, "_make_tape")
        res = matrix(qml.PauliZ, wire_order=["a", 0])(0)
        expected = np.kron(np.eye(2), np.diag([1, -1]))
        assert np.allclose(res, expected)
        assert spy.spy_return[1].tolist() == ["a", 0]

    def test_tape(self, mocker):
        """Test that wire order can be passed to a tape"""
        spy = mocker.spy(qml.op_transform, "_make_tape")

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        res = matrix(tape, wire_order=["a", 0])
        expected = np.kron(np.eye(2), np.diag([1, -1]))
        assert np.allclose(res, expected)
        assert spy.spy_return[1].tolist() == ["a", 0]

        qs = qml.tape.QuantumScript(tape.operations)
        res_qs = matrix(qs, wire_order=["a", 0])
        assert np.allclose(res_qs, expected)

    def test_inconsistent_wires_tape(self):
        """Test that an exception is raised if the wire order and tape wires are inconsistent"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(wires=0)
            qml.PauliY(wires="b")

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.raises(
            OperationTransformError,
            match=r"Wires in circuit .+ inconsistent with those in wire\_order",
        ):
            matrix(tape, wire_order=["b", "a"])

    def test_qfunc(self, mocker):
        """Test that wire order can be passed to a qfunc"""
        spy = mocker.spy(qml.op_transform, "_make_tape")

        def qfunc():
            qml.PauliZ(wires=0)

        res = matrix(qfunc, wire_order=["a", 0])()
        expected = np.kron(np.eye(2), np.diag([1, -1]))
        assert np.allclose(res, expected)
        assert spy.spy_return[1].tolist() == ["a", 0]


def test_op_transform_is_deprecated():
    """Test that the op_transform class is deprecated."""

    def func(op):
        return op

    with pytest.warns(
        UserWarning, match="Use of `op_transform` to create a custom transform is deprecated"
    ):
        _ = qml.op_transform(func)
