# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains tests for `construct_tape`.

"""

import pytest

import pennylane as qml
from pennylane.workflow import construct_tape


class TestConstructTape:

    def test_error_is_raised(self):

        @qml.gradients.param_shift
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RX(x, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Level requested corresponds to more than one tape."):
            construct_tape(circuit, level=None)(0.5)

    def test_handle_dynamic_shots(self):

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.X(1)
            return qml.expval(qml.X(0) + qml.Y(0))

        tape = construct_tape(circuit)(shots=10)
        expected_tape = qml.tape.QuantumScript(
            [qml.X(1)], [qml.expval(qml.X(0) + qml.Y(0))], shots=10
        )

        qml.assert_equal(tape, expected_tape)

    def test_correct_output_tape(self):

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RY(x, 0)
            qml.X(1)
            return qml.expval(qml.X(0) + qml.Y(0))

        tape = construct_tape(circuit)(0.5)
        expected_tape = qml.tape.QuantumScript(
            [qml.RY(0.5, 0), qml.X(1)],
            [qml.expval(qml.X(0) + qml.Y(0))],
        )
        # FIXME: Need this b/c of L617 in qscript.py
        expected_tape.trainable_params = []

        qml.assert_equal(tape, expected_tape)
