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
Unit tests for the sum function.
"""
import pytest
import pennylane as qml
from pennylane.ops.symbolic import Sum
from gate_data import CZ, H


class DummyOp(qml.operation.Operator):
    """General operator subclass."""

    @property
    def num_wires(self):
        return 3


class TestCreation:
    """Test the creation of Sum instances"""

    @pytest.mark.parametrize("summands", [
        [qml.PauliX(0), qml.PauliX(1)],
        [qml.PauliX(0), qml.PauliX(1), qml.PauliY(0)],
        [qml.QubitUnitary(CZ, wires=[0,2]), qml.QubitUnitary(H, wires=[1])],
        [DummyOp(wires=[0, 1, 2]), qml.CNOT(wires=[1,2])]
    ])
    def test_construction_for_different_inputs(self, summands):
        """Test initialization for different types and numbers of summands."""
        op = Sum(*summands)
        for s_exp, s in zip(summands, op.hyperparameters["summands"]):
            assert s_exp.name == s.name
            assert s_exp.wires == s.wires

    def test_wires_are_combined(self):
        """Test that the Sum instance's wires are the combined summand operators' wires."""
        op = Sum(qml.PauliX(0), qml.PauliX(2), qml.CNOT(wires=[0, 1]))
        assert op.wires.tolist() == [0, 2, 1]

    def test_parameters_are_combined(self):
        """Test that the Sum instance's parameters are all summand operators' parameters."""
        op = Sum(qml.RX(0.1, wires=0), qml.U2(0.2, 0.3, wires=0))
        assert op.parameters[0] == [0.1]
        assert op.parameters[1] == [0.2, 0.3]

    def test_error_if_only_one_op(self):
        """Test that an error is raised if only one operator is fed into the sum."""
        with pytest.raises(ValueError, match="Require at least two summands"):
            Sum(qml.PauliX(0))

        with pytest.raises(ValueError, match="Require at least two summands"):
            Sum()


class TestRepresentations:
    """Test that the representations of the sum operator are defined as expected"""

    @pytest.mark.parametrize("summands", [
        [qml.PauliX(0), qml.PauliX(1)],
        [qml.PauliX(0), qml.PauliX(1), qml.PauliY(0)],
        [qml.QubitUnitary(CZ, wires=[0,2]), qml.QubitUnitary(H, wires=[1])],
        [DummyOp(wires=[0, 1, 2]), qml.CNOT(wires=[1,2])]
    ])
    def test_terms_for_different_inputs(self, summands):
        """Test terms different types and numbers of summands."""
        op = Sum(*summands)
        for s_exp, s in zip(summands, op.terms()[1]):
            assert s_exp.name == s.name
            assert s_exp.wires == s.wires
        assert all(t == 1. for t in op.terms()[0])

    def test_that_decomposition_undefined(self):
        """Test that a general sum does not define the decomposition"""
        op = Sum(qml.PauliX(0), qml.PauliX(1))
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            op.decomposition()
