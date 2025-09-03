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
    """Tests for the construct_tape function."""

    @pytest.mark.parametrize(
        ("level", "weights", "order", "expected_ops"),
        [
            (
                0,
                qml.math.array([[1.0, 2.0]]),
                [2, 1, 0],
                [
                    qml.RandomLayers(qml.math.array([[1.0, 2.0]]), wires=(0, 1)),
                    qml.Permute([2, 1, 0], wires=(0, 1, 2)),
                    qml.PauliX(0),
                    qml.PauliX(0),
                    qml.RX(0.1, wires=0),
                    qml.RX(-0.1, wires=0),
                ],
            ),
            (
                1,
                qml.math.array([[1.0, 2.0]]),
                [2, 1, 0],
                [
                    qml.RandomLayers(qml.math.array([[1.0, 2.0]]), wires=(0, 1)),
                    qml.Permute([2, 1, 0], wires=(0, 1, 2)),
                    # cancel inverses
                    qml.RX(0.1, wires=0),
                    qml.RX(-0.1, wires=0),
                ],
            ),
        ],
    )
    def test_level_argument(self, level, weights, order, expected_ops):
        """Tests that the level argument is correctly passed through."""

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circuit(weights, order):
            qml.RandomLayers(weights, wires=(0, 1))
            qml.Permute(order, wires=(0, 1, 2))
            qml.PauliX(0)
            qml.PauliX(0)
            qml.RX(0.1, wires=0)
            qml.RX(-0.1, wires=0)
            return qml.expval(qml.PauliX(0))

        tape = construct_tape(qml.set_shots(circuit, shots=10), level=level)(weights, order)

        trainable_params = [] if level == 0 else None
        expected_tape = qml.tape.QuantumScript(
            expected_ops, [qml.expval(qml.PauliX(0))], shots=10, trainable_params=trainable_params
        )
        qml.assert_equal(tape, expected_tape)

    def test_level_error_is_raised(self):
        """Tests that a ValueError is raised if the user requests a level that corresponds to more than one tape."""

        @qml.gradients.param_shift
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RX(x, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Level requested corresponds to more than one tape."):
            construct_tape(circuit, level="device")(0.5)

    def test_handle_qfunc_with_dynamic_shots(self):
        """Tests that dynamic shots can be handled properly."""

        with pytest.warns(UserWarning, match="Detected 'shots' as an argument"):

            @qml.qnode(qml.device("default.qubit"))
            def circuit(shots):
                for _ in range(shots):
                    qml.X(0)
                return qml.expval(qml.PauliZ(0))

        num_shots = 10
        tape = construct_tape(circuit)(shots=num_shots)
        expected_tape = qml.tape.QuantumScript(
            [qml.X(0)] * num_shots, [qml.expval(qml.PauliZ(0))], trainable_params=[]
        )

        qml.assert_equal(tape, expected_tape)

    def test_correct_tape_is_constructed(self):
        """Tests that the constructed tape is as expected."""

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RY(x, 0)
            qml.X(1)
            return qml.expval(qml.X(0) + qml.Y(0))

        tape = construct_tape(circuit)(0.5)
        expected_tape = qml.tape.QuantumScript(
            [qml.RY(0.5, 0), qml.X(1)], [qml.expval(qml.X(0) + qml.Y(0))], trainable_params=[]
        )

        qml.assert_equal(tape, expected_tape)
