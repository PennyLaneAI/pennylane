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

import pennylane as qp
from pennylane.workflow import construct_tape


class TestConstructTape:
    """Tests for the construct_tape function."""

    @pytest.mark.parametrize(
        ("level", "weights", "order", "expected_ops"),
        [
            (
                0,
                qp.math.array([[1.0, 2.0]]),
                [2, 1, 0],
                [
                    qp.RandomLayers(qp.math.array([[1.0, 2.0]]), wires=(0, 1)),
                    qp.Permute([2, 1, 0], wires=(0, 1, 2)),
                    qp.PauliX(0),
                    qp.PauliX(0),
                    qp.RX(0.1, wires=0),
                    qp.RX(-0.1, wires=0),
                ],
            ),
            (
                1,
                qp.math.array([[1.0, 2.0]]),
                [2, 1, 0],
                [
                    qp.RandomLayers(qp.math.array([[1.0, 2.0]]), wires=(0, 1)),
                    qp.Permute([2, 1, 0], wires=(0, 1, 2)),
                    # cancel inverses
                    qp.RX(0.1, wires=0),
                    qp.RX(-0.1, wires=0),
                ],
            ),
        ],
    )
    def test_level_argument(self, level, weights, order, expected_ops):
        """Tests that the level argument is correctly passed through."""

        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("default.qubit"), diff_method="parameter-shift")
        def circuit(weights, order):
            qp.RandomLayers(weights, wires=(0, 1))
            qp.Permute(order, wires=(0, 1, 2))
            qp.PauliX(0)
            qp.PauliX(0)
            qp.RX(0.1, wires=0)
            qp.RX(-0.1, wires=0)
            return qp.expval(qp.PauliX(0))

        tape = construct_tape(qp.set_shots(circuit, shots=10), level=level)(weights, order)

        trainable_params = [] if level == 0 else None
        expected_tape = qp.tape.QuantumScript(
            expected_ops, [qp.expval(qp.PauliX(0))], shots=10, trainable_params=trainable_params
        )
        qp.assert_equal(tape, expected_tape)

    def test_level_error_is_raised(self):
        """Tests that a ValueError is raised if the user requests a level that corresponds to more than one tape."""

        @qp.gradients.param_shift
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("default.qubit"))
        def circuit(x):
            qp.RX(x, 0)
            qp.RX(x, 0)
            return qp.expval(qp.PauliZ(0))

        with pytest.raises(ValueError, match="Level requested corresponds to more than one tape."):
            construct_tape(circuit, level="device")(0.5)

    def test_handle_qfunc_with_dynamic_shots(self):
        """Tests that dynamic shots can be handled properly."""

        with pytest.warns(UserWarning, match="Detected 'shots' as an argument"):

            @qp.qnode(qp.device("default.qubit"))
            def circuit(shots):
                for _ in range(shots):
                    qp.X(0)
                return qp.expval(qp.PauliZ(0))

        num_shots = 10
        tape = construct_tape(circuit)(shots=num_shots)
        expected_tape = qp.tape.QuantumScript(
            [qp.X(0)] * num_shots, [qp.expval(qp.PauliZ(0))], trainable_params=[]
        )

        qp.assert_equal(tape, expected_tape)

    def test_correct_tape_is_constructed(self):
        """Tests that the constructed tape is as expected."""

        @qp.qnode(qp.device("default.qubit"))
        def circuit(x):
            qp.RY(x, 0)
            qp.X(1)
            return qp.expval(qp.X(0) + qp.Y(0))

        tape = construct_tape(circuit)(0.5)
        expected_tape = qp.tape.QuantumScript(
            [qp.RY(0.5, 0), qp.X(1)], [qp.expval(qp.X(0) + qp.Y(0))], trainable_params=[]
        )

        qp.assert_equal(tape, expected_tape)
