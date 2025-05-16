# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ftqc.pauli_tracker module"""

import pytest

import pennylane as qml
from pennylane.ftqc import pauli_encode_xz, pauli_prod_to_xz


class TestPauliTracker:
    """Test for the pauli tracker related functions."""

    @pytest.mark.parametrize("op", [qml.I, qml.X, qml.Y, qml.Z, qml.S])
    def test_pauli_encode_xz(self, op):
        if op not in [qml.I, qml.X, qml.Y, qml.Z]:
            with pytest.raises(NotImplementedError):
                _ = pauli_encode_xz(op)
        else:
            x, z = pauli_encode_xz(op)
            assert x in [0, 1]
            assert z in [0, 1]

            x_res = 1 if op in [qml.X, qml.Y] else 0
            z_res = 1 if op in [qml.Y, qml.Z] else 0

            assert x == x_res
            assert z == z_res

    @pytest.mark.parametrize(
        "ops, res",
        [
            ([], None),
            ([qml.I], (0, 0)),
            ([qml.X], (1, 0)),
            ([qml.Y], (1, 1)),
            ([qml.Z], (0, 1)),
            ([qml.I, qml.I], (0, 0)),
            ([qml.I, qml.X], (1, 0)),
            ([qml.I, qml.Y], (1, 1)),
            ([qml.I, qml.Z], (0, 1)),
            ([qml.X, qml.I], (1, 0)),
            ([qml.X, qml.X], (0, 0)),
            ([qml.X, qml.Y], (0, 1)),
            ([qml.X, qml.Z], (1, 1)),
            ([qml.Y, qml.I], (1, 1)),
            ([qml.Y, qml.X], (0, 1)),
            ([qml.Y, qml.Y], (0, 0)),
            ([qml.Y, qml.Z], (1, 0)),
            ([qml.Z, qml.I], (0, 1)),
            ([qml.Z, qml.X], (1, 1)),
            ([qml.Z, qml.Y], (1, 0)),
            ([qml.Z, qml.Z], (0, 0)),
            ([qml.X, qml.Y, qml.Z, qml.I, qml.Z], (0, 1)),
        ],
    )
    def test_pauli_prod_to_xz(self, ops, res):
        if len(ops) == 0:
            with pytest.raises(
                ValueError,
                match="Please ensure that a valid list of operators are passed to the method.",
            ):
                _ = pauli_prod_to_xz(ops)
        else:
            op = pauli_prod_to_xz(ops)
            assert len(op) == 2
            assert res[0] == op[0]
            assert res[1] == op[1]
