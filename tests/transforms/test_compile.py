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

from pennylane.transforms import cancel_inverses, cnot_to_cz, merge_rotations
from pennylane import compile

dev = qml.device("default.qubit", wires=3)


def qfunc(x, y, z):
    qml.RX(x, wires=0)
    qml.RX(y, wires=0)
    qml.CNOT(wires=[1, 2])
    qml.RY(y, wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=2)
    qml.CRZ(z, wires=[2, 0])
    qml.RY(-y, wires=1)
    qml.CRZ(y, wires=[2, 0])
    return qml.expval(qml.PauliZ(0))


qnode = qml.QNode(qfunc, dev)


class TestCompile:
    """Test that compilation pipelines work as expected."""

    def test_empty_pipeline(self):
        """Test that an empty pipeline returns the original function."""

        transformed_qfunc = compile(pipeline=[])(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        assert len(qnode.qtape.operations) == len(transformed_qnode.qtape.operations)

        for op_old, op_new in zip(qnode.qtape.operations, transformed_qnode.qtape.operations):
            assert op_old.name == op_new.name
            assert op_old.wires == op_new.wires
            assert np.allclose(op_old.parameters, op_new.parameters)

    @pytest.mark.parametrize(
        ("inputs", "pipeline", "expected_ops"),
        [
            (
                [0.3, 0.4, 0.5],
                [cancel_inverses, merge_rotations],
                [qml.RX(0.7, wires=0), qml.CNOT(wires=[1, 2]), qml.CRZ(0.9, wires=[2, 0])],
            ),
            (
                [0.1, -0.1, 0.2],
                [cancel_inverses, cnot_to_cz],
                [
                    qml.RX(0.1, wires=0),
                    qml.RX(-0.1, wires=0),
                    qml.Hadamard(wires=2),
                    qml.CZ(wires=[1, 2]),
                    qml.Hadamard(wires=2),
                    qml.RY(-0.1, wires=1),
                    qml.CRZ(0.2, wires=[2, 0]),
                    qml.RY(0.1, wires=1),
                    qml.CRZ(-0.1, wires=[2, 0]),
                ],
            ),
            (
                [0.2, -0.3, 0.4],
                [cnot_to_cz, cancel_inverses],
                [
                    qml.RX(0.2, wires=0),
                    qml.RX(-0.3, wires=0),
                    qml.Hadamard(wires=2),
                    qml.CZ(wires=[1, 2]),
                    qml.RY(-0.3, wires=1),
                    qml.Hadamard(wires=2),
                    qml.CRZ(0.4, wires=[2, 0]),
                    qml.RY(0.3, wires=1),
                    qml.CRZ(-0.3, wires=[2, 0]),
                ],
            ),
            (
                [0.2, -0.3, 0.4],
                [cnot_to_cz, cancel_inverses, merge_rotations],
                [
                    qml.RX(-0.1, wires=0),
                    qml.Hadamard(wires=2),
                    qml.CZ(wires=[1, 2]),
                    qml.Hadamard(wires=2),
                    qml.CRZ(0.1, wires=[2, 0]),
                ],
            ),
        ],
    )
    def test_full_pass(self, inputs, pipeline, expected_ops):
        """Test that different combinations of pipelines work as expected."""

        transformed_qfunc = compile(pipeline=pipeline)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)
        transformed_result = transformed_qnode(*inputs)

        original_result = qnode(*inputs)
        assert np.allclose(original_result, transformed_result)

        assert len(transformed_qnode.qtape.operations) == len(expected_ops)

        for op_obtained, op_expected in zip(transformed_qnode.qtape.operations, expected_ops):
            assert op_obtained.name == op_expected.name
            assert op_obtained.wires == op_expected.wires
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    def test_two_pass(self):
        """Test that two passes of a pipeline produce the expected result"""

        def func_with_many_rots():
            qml.RX(0.1, wires=0)
            qml.RX(0.2, wires=0)
            qml.RX(0.3, wires=0)
            qml.RX(0.4, wires=0)
            qml.RX(0.5, wires=0)

        two_passes = compile(pipeline=[merge_rotations], num_passes=2)(func_with_many_rots)
        ops_two_passes = qml.transforms.make_tape(two_passes)().operations

        assert len(ops_two_passes) == 2

        assert ops_two_passes[0].name == "RX"
        assert ops_two_passes[0].parameters[0] == 1.0

        assert ops_two_passes[1].name == "RX"
        assert ops_two_passes[1].parameters[0] == 0.5
