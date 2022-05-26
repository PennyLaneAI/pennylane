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

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math.pow_class import Pow, PowOperation


class TestInheritanceMixins:
    """Test inheritance structure and mixin addition through dynamic __new__ method."""

    def test_plain_operator(self):
        """Test when base directly inherits from Operator, Adjoint only inherits
        from Adjoint and Operator."""

        class Tester(qml.operation.Operator):
            num_wires = 1
            num_params = 1

        base = Tester(1.234, wires=0)
        op = Pow(base, 1.2)

        assert isinstance(op, Pow)
        assert isinstance(op, qml.operation.Operator)
        assert not isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
        assert not isinstance(op, Pow)

        # checking we can call `dir` without problems
        assert "num_params" in dir(op)

    def test_operation(self):
        """When the operation inherits from `Operation`, the `AdjointOperation` mixin should
        be added and the Adjoint should now have Operation functionality."""

        class CustomOp(qml.operation.Operation):
            num_wires = 1
            num_params = 1

        base = CustomOp(1.234, wires=0)
        op = Pow(base, 6.5)

        assert isinstance(op, Pow)
        assert isinstance(op, qml.operation.Operator)
        assert isinstance(op, qml.operation.Operation)
        assert not isinstance(op, qml.operation.Observable)
        assert isinstance(op, PowOperation)

        # check operation-specific properties made it into the mapping
        assert "grad_recipe" in dir(op)
        assert "control_wires" in dir(op)

    def test_observable(self):
        """Test that when the base is an Observable, Adjoint will also inherit from Observable."""

        class CustomObs(qml.operation.Observable):
            num_wires = 1
            num_params = 0

        base = CustomObs(wires=0)
        ob = Pow(base, -1.2)

        assert isinstance(ob, Pow)
        assert isinstance(ob, qml.operation.Operator)
        assert not isinstance(ob, qml.operation.Operation)
        assert isinstance(ob, qml.operation.Observable)
        assert not isinstance(ob, PowOperation)

        # Check some basic observable functionality
        assert ob.compare(ob)
        assert isinstance(1.0 * ob @ ob, qml.Hamiltonian)

        # check the dir
        assert "return_type" in dir(ob)
        assert "grad_recipe" not in dir(ob)


class TestInitialization:
    """Test the initialization process and standard properties."""

    def test_nonparametric_ops(self):
        """Test pow initialization for a non parameteric operation."""
        base = qml.PauliX("a")

        op = Pow(base, -4.2, id="something")

        assert op.base is base
        assert op.z == -4.2
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == -4.2
        assert op.name == "PauliX**-4.2"
        assert op.id == "something"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == []

        assert op.wires == qml.wires.Wires("a")

    def test_parametric_ops(self):
        """Test pow initialization for a standard parametric operation."""
        params = [1.2345, 2.3456, 3.4567]
        base = qml.Rot(*params, wires="b")

        op = Pow(base, -0.766, id="id")

        assert op.base is base
        assert op.z == -0.766
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == -0.766
        assert op.name == "Rot**-0.766"
        assert op.id == "id"

        assert op.num_params == 3
        assert qml.math.allclose(params, op.parameters)
        assert qml.math.allclose(params, op.data)

        assert op.wires == qml.wires.Wires("b")

    def test_template_base(self):
        """Test adjoint initialization for a template."""
        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        params = rng.random(shape)

        base = qml.StronglyEntanglingLayers(params, wires=[0, 1])
        op = Pow(base, 2.67)

        assert op.base is base
        assert op.z == 2.67
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == 2.67
        assert op.name == "StronglyEntanglingLayers**2.67"

        assert op.num_params == 1
        assert qml.math.allclose(params, op.parameters[0])
        assert qml.math.allclose(params, op.data[0])

        assert op.wires == qml.wires.Wires((0, 1))

    def test_hamiltonian_base(self):
        """Test adjoint initialization for a hamiltonian."""
        base = 2.0 * qml.PauliX(0) @ qml.PauliY(0) + qml.PauliZ("b")

        op = Pow(base, 3.4)

        assert op.base is base
        assert op.z == 3.4
        assert op.hyperparameters["base"] is base
        assert op.hyperparameters["z"] == 3.4
        assert op.name == "Hamiltonian**3.4"

        assert op.num_params == 2
        assert qml.math.allclose(op.parameters, [2.0, 1.0])
        assert qml.math.allclose(op.data, [2.0, 1.0])

        assert op.wires == qml.wires.Wires([0, "b"])


class TestProperties:
    """Test Pow properties."""

    def test_data(self):
        """Test base data can be get and set through Pow class."""
        x = np.array(1.234)

        base = qml.RX(x, wires="a")
        op = Pow(base, 3.21)

        assert op.data == [x]

        # update parameters through pow
        x_new = np.array(2.3456)
        op.data = [x_new]
        assert base.data == [x_new]
        assert op.data == [x_new]

        # update base data updates pow data
        x_new2 = np.array(3.456)
        base.data = [x_new2]
        assert op.data == [x_new2]

    def test_has_matrix_true(self):
        """Test `has_matrix` property carries over when base op defines matrix."""
        base = qml.PauliX(0)
        op = Pow(base, -1.1)

        assert op.has_matrix

    def test_has_matrix_false(self):
        """Test has_matrix property carries over when base op does not define a matrix."""
        base = qml.QubitStateVector([1, 0], wires=0)
        op = Pow(base, 2.0)

        assert not op.has_matrix

    def test_queue_category(self):
        """Test that the queue category `"_ops"` carries over."""
        op = Pow(qml.PauliX(0), 3.5)
        assert op._queue_category == "_ops"

    def test_queue_category_None(self):
        """Test that the queue category `None` for some observables carries over."""
        op = Pow(qml.PauliX(0) @ qml.PauliY(1), -1.1)
        assert op._queue_category is None


def test_label():

    base = qml.RX(1.2, wires=0)
    op = Pow(base, -1.23456789)

    assert op.label() == "RX⁻¹⋅²³⁴⁵⁶⁷⁸⁹"
    assert op.label(decimals=2) == "RX\n(1.20)⁻¹⋅²³⁴⁵⁶⁷⁸⁹"


def test_label_matrix_param():

    base = qml.QubitUnitary(np.eye(2), wires=0)
    op = Pow(base, 6.7)

    cache = {"matrices": []}
    assert op.label(decimals=2, cache=cache) == "U(M0)⁻¹⋅²"
    assert len(cache["matrices"]) == 1


class TestDiagonalizingGates:
    def test_diagonalizing_gates_int_exist(self):

        base = qml.PauliX(0)
        op = Pow(base, 2)

        op_gates = op.diagonalizing_gates()
        base_gates = base.diagonalizing_gates()

        assert len(op_gates) == len(base_gates)

        for op1, op2 in zip(op_gates, base_gates):
            assert op1.__class__ is op2.__class__
            assert op1.wires == op2.wires

    def test_diagonalizing_gates_float(self):

        base = qml.PauliX(0)
        op = Pow(base, 0.5)

        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            op.diagonalizing_gates()

    def test_base_doesnt_define(self):

        base = qml.RX(1.2, wires=0)
        op = Pow(base, 2)

        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            op.diagonalizing_gates()


class TestQueueing:
    """Test that Pow operators queue and update base metadata"""

    def test_queueing(self):
        """Test queuing and metadata when both Pow and base defined inside a recording context."""

        with qml.tape.QuantumTape() as tape:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            op = Pow(base, 1.2)

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_queueing_base_defined_outside(self):
        """Test that base is added to queue even if it's defined outside the recording context."""

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.tape.QuantumTape() as tape:
            op = Pow(base, 3.4)

        assert tape._queue[base]["owner"] is op
        assert tape._queue[op]["owns"] is base
        assert tape.operations == [op]

    def test_do_queue_False(self):
        """Test that when `do_queue` is specified, the operation is not queued."""
        base = qml.PauliX(0)
        with qml.tape.QuantumTape() as tape:
            op = Pow(base, 4.5, do_queue=False)

        assert len(tape) == 0
