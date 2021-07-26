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

from pennylane.transforms.optimization import *
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
        """Test that the default pipeline returns the correct results."""

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

    # @pytest.mark.parametrize(
    #     ("inputs", "pipeline", "expected_ops"),
    #     [
    #         (
    #             [0.3, 0.4, 0.5],
    #             [cancel_inverses, merge_rotations],
    #             [qml.RX(0.7, wires=0), qml.CNOT(wires=[1, 2]), qml.CRZ(0.9, wires=[2, 0])],
    #         ),
    #     ],
    # )
    # def test_full_pass(self, inputs, pipeline, expected_ops):
    #     """Test that different combinations of pipelines work as expected."""

    #     transformed_qfunc = compile(pipeline=pipeline)(qfunc)
    #     transformed_qnode = qml.QNode(transformed_qfunc, dev)
    #     transformed_result = transformed_qnode(*inputs)

    #     original_result = qnode(*inputs)
    #     assert np.allclose(original_result, transformed_result)

    #     assert len(transformed_qnode.qtape.operations) == len(expected_ops)

    #     for op_obtained, op_expected in zip(transformed_qnode.qtape.operations, expected_ops):
    #         assert op_obtained.name == op_expected.name
    #         assert op_obtained.wires == op_expected.wires
    #         assert np.allclose(op_obtained.parameters, op_expected.parameters)

    # @pytest.mark.parametrize(
    #     ("gate_type", "angles"),
    #     [
    #         (qml.RX, [0.1, 0.2, 0.3, 0.4, 0.5]),
    #         (qml.RY, [0.01]),
    #         (qml.RZ, [0.25, 0.8, 0.98]),
    #         (qml.PhaseShift, [-0.1, 0.5]),
    #     ],
    # )
    # def test_cumulative_merge_one_qubit_single_parameter_nonzero(self, gate_type, angles):
    #     """Test that multiple adjacent instances of a single-qubit gate get merged in one pass"""

    #     def qfunc_with_many_rots():
    #         for angle in angles:
    #             gate_type(angle, wires=0)

    #     compiled_qfunc = compile(pipeline=[merge_rotations], num_passes=1)(qfunc_with_many_rots)
    #     ops = qml.transforms.make_tape(compiled_qfunc)().operations

    #     assert len(ops) == 1

    #     assert ops[0].name == gate_type(0, wires=0).name
    #     assert ops[0].parameters[0] == sum(angles)

    # @pytest.mark.parametrize(
    #     ("gate_type", "angles"),
    #     [
    #         (qml.CRX, [0.1, 0.2, 0.3, 0.4, 0.5]),
    #         (qml.CRY, [0.01]),
    #         (qml.CRZ, [0.25, 0.8, 0.98]),
    #     ],
    # )
    # def test_cumulative_merge_two_qubit_single_parameter_nonzero(self, gate_type, angles):
    #     """Test that multiple adjacent instances of a two-qubit gate get merged in one pass."""

    #     def qfunc_with_many_rots():
    #         for angle in angles:
    #             gate_type(angle, wires=[0, 1])

    #     compiled_qfunc = compile(pipeline=[merge_rotations], num_passes=1)(qfunc_with_many_rots)
    #     ops = qml.transforms.make_tape(compiled_qfunc)().operations

    #     assert len(ops) == 1

    #     assert ops[0].name == gate_type(0, wires=[0, 1]).name
    #     assert ops[0].parameters[0] == sum(angles)

    # def test_basis_expansion(self):
    #     """Test that two passes of a pipeline produce the expected result"""

    #     def qfunc_with_cry():
    #         qml.RY(0.2, wires=1)
    #         qml.CRY(0.1, wires=[0, 1])
    #         qml.RZ(np.pi / 4, wires=2)
    #         qml.S(wires=2)

    #     pipeline = [cancel_inverses, merge_rotations]
    #     basis = ["CNOT", "RX", "RY", "RZ"]

    #     two_passes = compile(pipeline=[merge_rotations], basis_set=basis, num_passes=1)(
    #         qfunc_with_cry
    #     )
    #     ops_two_passes = qml.transforms.make_tape(two_passes)().operations

    #     assert len(ops_two_passes) == 5

    #     assert ops_two_passes[0].name == "RY"
    #     assert ops_two_passes[0].wires == Wires(1)
    #     assert ops_two_passes[0].parameters[0] == 0.25

    #     assert ops_two_passes[1].name == "CNOT"
    #     assert ops_two_passes[1].wires == Wires([0, 1])

    #     assert ops_two_passes[2].name == "RY"
    #     assert ops_two_passes[2].wires == Wires(1)
    #     assert ops_two_passes[2].parameters[0] == -0.05

    #     assert ops_two_passes[3].name == "CNOT"
    #     assert ops_two_passes[3].wires == Wires([0, 1])

    #     assert ops_two_passes[4].name == "RZ"
    #     assert ops_two_passes[4].wires == Wires(2)
    #     assert np.isclose(ops_two_passes[4].parameters[0], np.pi / 2 + np.pi / 4)
