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

import pytest
import numpy as np

import pennylane as qml
from pennylane.wires import Wires

from pennylane.transforms.optimization import cancel_inverses, commute_controlled, merge_rotations
from pennylane import compile

from test_optimization.utils import compare_operation_lists


def build_qfunc(wires):
    def qfunc(x, y, z):
        qml.Hadamard(wires=wires[0])
        qml.RZ(z, wires=wires[2])
        qml.CNOT(wires=[wires[2], wires[1]])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RX(x, wires=wires[0])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(-z, wires=wires[2])
        qml.RX(y, wires=wires[0])
        qml.PauliY(wires=wires[2])
        qml.CY(wires=[wires[1], wires[2]])
        return qml.expval(qml.PauliZ(wires=wires[0]))

    return qfunc


class TestCompile:
    """Test that compilation pipelines work as expected."""

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_empty_pipeline(self, wires):
        """Test that an empty pipeline returns the original function."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=wires)

        qnode = qml.QNode(qfunc, dev)

        transformed_qfunc = compile(pipeline=[])(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = [op.name for op in qnode.qtape.operations]
        wires_expected = [op.wires for op in qnode.qtape.operations]

        print(names_expected)
        print(wires_expected)
        print(transformed_qnode.qtape.operations)

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_default_pipeline(self, wires):
        """Test that the default pipeline returns the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        transformed_qfunc = compile()(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["Hadamard", "CNOT", "RX", "CY", "PauliY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires([wires[1], wires[2]]),
            Wires(wires[2]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_pipeline_with_non_default_arguments(self, wires):
        """Test that using non-default arguments returns the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        pipeline = [commute_controlled(direction="left"), cancel_inverses, merge_rotations]

        transformed_qfunc = compile(pipeline=pipeline)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["Hadamard", "CNOT", "RX", "PauliY", "CY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_multiple_passes(self, wires):
        """Test that running multiple passes produces the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        # Rotation merging will not occur at all until commuting gates are
        # pushed through
        pipeline = [merge_rotations, commute_controlled(direction="left"), cancel_inverses]

        transformed_qfunc = compile(pipeline=pipeline, num_passes=2)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["Hadamard", "CNOT", "RX", "PauliY", "CY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_decompose_into_basis_gates(self, wires):
        """Test that running multiple passes produces the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        pipeline = [commute_controlled(direction="left"), cancel_inverses, merge_rotations]

        basis_set = ["CNOT", "RX", "RY", "RZ"]

        transformed_qfunc = compile(pipeline=pipeline, basis_set=basis_set)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = [
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "CNOT",
            "RX",
            "RZ",
            "RY",
            "RZ",
            "RY",
            "CNOT",
            "RY",
            "CNOT",
        ]

        wires_expected = [
            Wires(wires[0]),
            Wires(wires[0]),
            Wires(wires[0]),
            Wires(wires[2]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires(wires[1]),
            Wires(wires[2]),
            Wires(wires[2]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)
