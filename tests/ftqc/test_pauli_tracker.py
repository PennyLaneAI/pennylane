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
from pennylane.ftqc.pauli_tracker import apply_clifford_op, pauli_prod, pauli_to_xz, xz_to_pauli


class TestPauliTracker:
    """Test for the pauli tracker related functions."""

    @pytest.mark.parametrize(
        "op, expected",
        [(qml.I(0), (0, 0)), (qml.X(1), (1, 0)), (qml.Y(0), (1, 1)), (qml.Z(0), (0, 1))],
    )
    def test_pauli_to_xz(self, op, expected):
        xz = pauli_to_xz(op)
        assert xz == expected

    @pytest.mark.parametrize("op", [qml.S(0), qml.CNOT(wires=[0, 1]), qml.H(2)])
    def test_unsuppored_ops_pauli_to_xz(self, op):
        with pytest.raises(NotImplementedError):
            _ = pauli_to_xz(op)

    @pytest.mark.parametrize(
        "x, z, expected", [(0, 0, qml.I), (1, 0, qml.X), (1, 1, qml.Y), (0, 1, qml.Z)]
    )
    def test_xz_to_pauli(self, x, z, expected):
        op = xz_to_pauli(x, z)
        assert op == expected

    @pytest.mark.parametrize("x, z", [(0, -1), (-1, 0), (-1, -1)])
    def test_xz_decode_pauli_unsupported_error(self, x, z):
        with pytest.raises(ValueError):
            _ = xz_to_pauli(x, z)

    @pytest.mark.parametrize(
        "ops, expected",
        [
            ([qml.I(0)], (0, 0)),
            ([qml.X(1)], (1, 0)),
            ([qml.Y(2)], (1, 1)),
            ([qml.Z(3)], (0, 1)),
            ([qml.I(0), qml.I(0)], (0, 0)),
            ([qml.I(1), qml.X(1)], (1, 0)),
            ([qml.I(2), qml.Y(2)], (1, 1)),
            ([qml.I(3), qml.Z(3)], (0, 1)),
            ([qml.X(0), qml.I(0)], (1, 0)),
            ([qml.X(1), qml.X(1)], (0, 0)),
            ([qml.X(2), qml.Y(2)], (0, 1)),
            ([qml.X(3), qml.Z(3)], (1, 1)),
            ([qml.Y(0), qml.I(0)], (1, 1)),
            ([qml.Y(1), qml.X(1)], (0, 1)),
            ([qml.Y(2), qml.Y(2)], (0, 0)),
            ([qml.Y(3), qml.Z(3)], (1, 0)),
            ([qml.Z(0), qml.I(0)], (0, 1)),
            ([qml.Z(1), qml.X(1)], (1, 1)),
            ([qml.Z(2), qml.Y(2)], (1, 0)),
            ([qml.Z(3), qml.Z(3)], (0, 0)),
            ([qml.X(4), qml.Y(4), qml.Z(4), qml.I(4), qml.Z(4)], (0, 1)),
        ],
    )
    def test_pauli_prod(self, ops, expected):
        op = pauli_prod(ops)
        assert op == expected

    @pytest.mark.parametrize("ops", [[]])
    def test_pauli_prod_to_xz_unsupported_error(self, ops):
        with pytest.raises(ValueError):
            _ = pauli_prod(ops)

    @pytest.mark.parametrize(
        "clifford_op, xz, expected_xz",
        [
            (qml.S(0), [(0, 0)], [(0, 0)]),
            (qml.S(1), [(1, 0)], [(1, 1)]),
            (qml.S(2), [(1, 1)], [(1, 0)]),
            (qml.S(3), [(0, 1)], [(0, 1)]),
            (qml.H(0), [(0, 0)], [(0, 0)]),
            (qml.H(1), [(1, 0)], [(0, 1)]),
            (qml.H(2), [(1, 1)], [(1, 1)]),
            (qml.H(3), [(0, 1)], [(1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0), (0, 0)], [(0, 0), (0, 0)]),
            (qml.CNOT(wires=[0, 1]), [(1, 0), (0, 0)], [(1, 0), (1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(1, 1), (0, 0)], [(1, 1), (1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1), (0, 0)], [(0, 1), (0, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0), (1, 0)], [(0, 0), (1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(1, 0), (1, 0)], [(1, 0), (0, 0)]),
            (qml.CNOT(wires=[0, 1]), [(1, 1), (1, 0)], [(1, 1), (0, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1), (1, 0)], [(0, 1), (1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0), (1, 1)], [(0, 1), (1, 1)]),
            (qml.CNOT(wires=[0, 1]), [(1, 0), (1, 1)], [(1, 1), (0, 1)]),
            (qml.CNOT(wires=[0, 2]), [(1, 1), (1, 1)], [(1, 0), (0, 1)]),
            (qml.CNOT(wires=[0, 2]), [(0, 1), (1, 1)], [(0, 0), (1, 1)]),
            (qml.CNOT(wires=[1, 2]), [(0, 0), (0, 1)], [(0, 1), (0, 1)]),
            (qml.CNOT(wires=[1, 2]), [(1, 0), (0, 1)], [(1, 1), (1, 1)]),
            (qml.CNOT(wires=[1, 2]), [(1, 1), (0, 1)], [(1, 0), (1, 1)]),
            (qml.CNOT(wires=[1, 2]), [(0, 1), (0, 1)], [(0, 0), (0, 1)]),
        ],
    )
    def test_apply_clifford_ops(self, clifford_op, xz, expected_xz):
        new_xz = apply_clifford_op(clifford_op, xz)

        assert new_xz == expected_xz

    @pytest.mark.parametrize(
        "clifford_op", [qml.X(0), qml.RZ(phi=0.123, wires=0), qml.RX(phi=0.123, wires=0)]
    )
    @pytest.mark.parametrize("xz", [[(0, 1)]])
    def test_apply_clifford_ops_not_imp(self, clifford_op, xz):
        with pytest.raises(
            NotImplementedError, match="Only qml.H, qml.S and qml.CNOT are supported."
        ):
            _ = apply_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 1), (1, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0), (0, 0), (0, 0)]),
        ],
    )
    def test_apply_clifford_ops_xz_size_mismatch(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure that the length of xz matches the number of wires of the clifford_op.",
        ):
            _ = apply_clifford_op(clifford_op, xz)
