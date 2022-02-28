# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.op_transforms import OperationTransformError


class TestValidation:
    def test_sphinx_build(self, monkeypatch):
        """Test that op transforms are not created during Sphinx builds"""

        @qml.op_transform
        def my_transform(op):
            return op.name

        assert isinstance(my_transform, qml.op_transform)

        monkeypatch.setenv("SPHINX_BUILD", "1")

        with pytest.warns(UserWarning, match="Operator transformations have been disabled"):

            @qml.op_transform
            def my_transform(op):
                return op.name

        assert not isinstance(my_transform, qml.batch_transform)

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

        def qfunc(x):
            return x**2

        with pytest.raises(
            OperationTransformError,
            match="Quantum function contains no quantum operations",
        ):
            my_transform(qfunc)(0.5)

    def test_empty_qfunc(self):
        """Test that an error is raised if the qfunc has no quantum operations
        (e.g., it is not a qfunc)"""


class TestUI:
    def test_instantiated_operator(self):
        """Test that a transform can be applied to an instantiated operator"""

        @qml.op_transform
        def my_transform(op):
            return op.name

        op = qml.CRX(0.5, wires=[0, 2])
        res = my_transform(op)
        assert res == "CRX"

    def test_single_operator_qfunc(self, mocker):
        """Test that a transform can be applied to a quantum function"""
        spy = mocker.spy(qml.transforms.op_transforms, "_make_tape")

        @qml.op_transform
        def my_transform(op):
            return op.name

        res = my_transform(qml.CRX)(0.5, wires=[0, "a"])
        assert res == "CRX"

        # check wire order
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

    def test_multiple_operator_tape(self):
        """Test that a transform can be applied to a quantum function
        with multiple operations as long as it is registered _how_
        the transform applies to multiple operations."""

        @qml.op_transform
        def my_transform(op):
            return op.name

        @my_transform.tape_transform
        def my_transform(tape):
            return [op.name for op in tape.operations]

        with qml.tape.QuantumTape() as tape:
            qml.RX(1.6, wires=0)
            qml.RY(0.65, wires=1)

        res = my_transform(tape)
        assert res == ["RX", "RY"]

    def test_multiple_operator_qfunc(self):
        """Test that a transform can be applied to a quantum function
        with multiple operations as long as it is registered _how_
        the transform applies to multiple operations."""

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
                qml.RY(0.65, wires=1)
            else:
                qml.RZ(x, wires=0)

        res = multi_op_qfunc(1.5)
        assert res == ["RX", "RY"]

        res = multi_op_qfunc(0.5)
        assert res == ["RZ"]

    def test_qnode(self):
        """Test that a transform can be applied to a QNode
        with multiple operations as long as it is registered _how_
        the transform applies to multiple operations."""

        @qml.op_transform
        def my_transform(op):
            return op.name

        @my_transform.tape_transform
        def my_transform(tape):
            return [op.name for op in tape.operations]

        @my_transform
        @qml.qnode(qml.device("default.qubit", wires=2))
        def multi_op_qfunc(x):
            if x > 1:
                qml.RX(x, wires=0)
                qml.RY(0.65, wires=1)
            else:
                qml.RZ(x, wires=0)

            return qml.probs(wires=0)

        res = multi_op_qfunc(1.5)
        assert res == ["RX", "RY"]

        res = multi_op_qfunc(0.5)
        assert res == ["RZ"]


class TestTransformParameters:
    def test_instantiated_operator(self):
        """Test that a transform can be applied to an instantiated operator"""

        @qml.op_transform
        def my_transform(op, lower=False):
            if lower:
                return op.name.lower()
            return op.name

        op = qml.RX(0.5, wires=0)
        res = my_transform(op, lower=True)
        assert res == "rx"

    def test_single_operator_qfunc(self):
        """Test that a transform can be applied to a quantum function"""

        @qml.op_transform
        def my_transform(op, lower=False):
            if lower:
                return op.name.lower()
            return op.name

        res = my_transform(qml.RX, lower=True)(0.5, wires=0)
        assert res == "rx"

    def test_transform_parameters_qfunc_decorator(self):
        """Test that transform parameters correctly work
        when used as a decorator"""

        @qml.op_transform
        def my_transform(op, lower=False):
            if lower:
                return op.name.lower()
            return op.name

        @my_transform.tape_transform
        def my_transform(tape, lower=False):
            if lower:
                return [op.name.lower() for op in tape.operations]
            return [op.name for op in tape.operations]

        @my_transform(lower=True)
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


@qml.op_transform
def simplify_rotation(op):
    """Simplify Rot(x, 0, 0) to RZ(x) or Rot(0,0,0) to Identity"""
    if op.name == "Rot":
        params = op.parameters
        wires = op.wires

        if qml.math.allclose(params, 0):
            return

        if qml.math.allclose(params[1:2], 0):
            return qml.RZ(params[0], wires)

    return op


@simplify_rotation.tape_transform
@qml.qfunc_transform
def simplify_rotation(tape):
    """Define how simplify rotation works on a tape"""
    for op in tape.operations + tape.measurements:
        if op.name == "Rot":
            simplify_rotation(op)
        else:
            qml.apply(op)


class TestQFuncTransformIntegration:
    """Test that @qfunc_transform seamlessly integrates
    with an operator transform."""

    def test_instantiated_operator(self):
        """Test a qfunc and operator transform applied to
        an op"""
        dev = qml.device("default.qubit", wires=2)

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
        dev = qml.device("default.qubit", wires=2)

        def ansatz(weights):
            qml.Rot(*weights, wires=0)
            qml.CRX(0.5, wires=[0, 1])

        @qml.qnode(dev)
        def circuit(weights):
            simplify_rotation(ansatz)(weights)
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
        @simplify_rotation
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

        with pytest.raises(TypeError):
            circuit(weights)
