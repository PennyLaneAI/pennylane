# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the BlockEncode operation.
"""
import pytest
import pennylane as qml
from pennylane import numpy as pnp


def test_testing():
    assert 1 == 1


class TestInput:
    @pytest.mark.parametrize(
        ("input_matrix", "wires"),
        [
            (1, 1),
            ([1], 1),
            (pnp.array(1), [1]),
            (pnp.array([1]), 1),
            ([[1, 0], [0, 1]], [0, 1]),
            (pnp.array([[1, 0], [0, 1]]), range(2)),
            (pnp.identity(3), ["a", "b", "c"]),
        ],
    )
    def test_accepts_various_types(self, input_matrix, wires):
        op = qml.BlockEncode(input_matrix, wires)

    @pytest.mark.parametrize(
        ("input_matrix", "wires"),
        [(1, 1), (1, 2), (1, [1]), (1, range(2)), (pnp.identity(2), ["a", "b"])],
    )
    def test_varied_wires(self, input_matrix, wires):
        op = qml.BlockEncode(input_matrix, wires)

    @pytest.mark.parametrize(
        ("input_matrix", "wires", "msg"),
        [
            (
                [[0, 1], [1, 0]],
                1,
                f"Block encoding a {2} x {2} matrix requires a hilbert space of size"
                f" at least {4} x {4}.Cannot be embedded in a {1} qubit system.",
            ),
        ],
    )
    def test_correct_error_message(self, input_matrix, wires, msg):
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            op = qml.BlockEncode(input_matrix, wires)

    # def test_scalars_work_for_all_types
