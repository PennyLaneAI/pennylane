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
from pennylane.ftqc import apply_clifford_op, pauli_encode_xz, pauli_prod_to_xz


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
            ([qml.I], qml.I),
            ([qml.X], qml.X),
            ([qml.Y], qml.Y),
            ([qml.Z], qml.Z),
            ([qml.I, qml.I], qml.I),
            ([qml.I, qml.X], qml.X),
            ([qml.I, qml.Y], qml.Y),
            ([qml.I, qml.Z], qml.Z),
            ([qml.X, qml.I], qml.X),
            ([qml.X, qml.X], qml.I),
            ([qml.X, qml.Y], qml.Z),
            ([qml.X, qml.Z], qml.Y),
            ([qml.Y, qml.I], qml.Y),
            ([qml.Y, qml.X], qml.Z),
            ([qml.Y, qml.Y], qml.I),
            ([qml.Y, qml.Z], qml.X),
            ([qml.Z, qml.I], qml.Z),
            ([qml.Z, qml.X], qml.Y),
            ([qml.Z, qml.Y], qml.X),
            ([qml.Z, qml.Z], qml.I),
            ([qml.X, qml.Y, qml.Z, qml.I, qml.Z], qml.Z),
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
            assert res == op

    @pytest.mark.parametrize(
        "clifford_op, pauli, res",
        [
            (qml.S, [qml.I], [qml.I]),
            (qml.S, [qml.X], [qml.Y]),
            (qml.S, [qml.Y], [qml.X]),
            (qml.S, [qml.Z], [qml.Z]),
            (qml.H, [qml.I], [qml.I]),
            (qml.H, [qml.X], [qml.Z]),
            (qml.H, [qml.Y], [qml.Y]),
            (qml.H, [qml.Z], [qml.X]),
            (qml.CNOT, [qml.I, qml.I], [qml.I, qml.I]),
            (qml.CNOT, [qml.X, qml.I], [qml.X, qml.X]),
            (qml.CNOT, [qml.Y, qml.I], [qml.Y, qml.X]),
            (qml.CNOT, [qml.Z, qml.I], [qml.Z, qml.I]),
            (qml.CNOT, [qml.I, qml.X], [qml.I, qml.X]),
            (qml.CNOT, [qml.X, qml.X], [qml.X, qml.I]),
            (qml.CNOT, [qml.Y, qml.X], [qml.Y, qml.I]),
            (qml.CNOT, [qml.Z, qml.X], [qml.Z, qml.X]),
            (qml.CNOT, [qml.I, qml.Y], [qml.Z, qml.Y]),
            (qml.CNOT, [qml.X, qml.Y], [qml.Y, qml.Z]),
            (qml.CNOT, [qml.Y, qml.Y], [qml.X, qml.Z]),
            (qml.CNOT, [qml.Z, qml.Y], [qml.I, qml.Y]),
            (qml.CNOT, [qml.I, qml.Z], [qml.Z, qml.Z]),
            (qml.CNOT, [qml.X, qml.Z], [qml.Y, qml.Y]),
            (qml.CNOT, [qml.Y, qml.Z], [qml.X, qml.Y]),
            (qml.CNOT, [qml.Z, qml.Z], [qml.I, qml.Z]),
        ],
    )
    def test_apply_clifford_ops(self, clifford_op, pauli, res):
        new_pauli = apply_clifford_op(clifford_op, pauli)

        assert new_pauli == res

    @pytest.mark.parametrize("clifford_op", [qml.X, qml.RZ, qml.RX, qml.T])
    @pytest.mark.parametrize("paulis", [[qml.I]])
    def test_apply_clifford_ops_not_imp(self, clifford_op, paulis):
        with pytest.raises(
            NotImplementedError, match="Only qml.H, qml.S and qml.CNOT are supported."
        ):
            _ = apply_clifford_op(clifford_op, paulis)

    @pytest.mark.parametrize(
        "clifford_op, paulis",
        [(qml.S, [qml.I, qml.I]), (qml.S, [qml.RZ]), (qml.CNOT, [qml.I])],
    )
    def test_apply_clifford_ops_val_err(self, clifford_op, paulis):
        with pytest.raises(ValueError):
            _ = apply_clifford_op(clifford_op, paulis)
