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

import random

import numpy as np
import pytest

import pennylane as qml
from pennylane.ftqc import RotXZX, get_byproduct_corrections
from pennylane.ftqc.pauli_tracker import (
    _parse_mid_measurements,
    commute_clifford_op,
    pauli_prod,
    pauli_to_xz,
    xz_to_pauli,
)

_PAULIS = (qml.I, qml.X, qml.Y, qml.Z)


def generate_pauli_list(wire: int, num_ops: int):
    pauli_list = []

    for _ in range(num_ops):
        pauli_list.append(random.choice(_PAULIS)(wire))

    return pauli_list


def generate_random_state(n, seed=0):
    rng = np.random.default_rng(seed=seed)
    input_state = rng.random(2**n) + 1j * rng.random(2**n)
    return input_state / np.linalg.norm(input_state)


class TestPauliTracker:
    """Test for the pauli tracker related functions."""

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.I(0), (0, 0)),
            (qml.X(1), (1, 0)),
            (qml.Y(0), (1, 1)),
            (qml.Z(0), (0, 1)),
            (qml.I, (0, 0)),
            (qml.X, (1, 0)),
            (qml.Y, (1, 1)),
            (qml.Z, (0, 1)),
        ],
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

    @pytest.mark.parametrize("target_wire", [0, 10, 100, 1000])
    @pytest.mark.parametrize("num_ops", [1, 10, 100, 1000])
    def test_pauli_prod(self, target_wire, num_ops):
        pauli_list = generate_pauli_list(target_wire, num_ops)

        x, z = pauli_prod(pauli_list)

        op = xz_to_pauli(x, z)(wires=target_wire)

        op_simplify = qml.prod(*pauli_list).simplify()
        expected_op = op_simplify if isinstance(op_simplify, _PAULIS) else op_simplify.terms()[1][0]

        assert op == expected_op

    @pytest.mark.parametrize("ops", [[]])
    def test_pauli_prod_to_xz_unsupported_error(self, ops):
        with pytest.raises(ValueError):
            _ = pauli_prod(ops)

    @pytest.mark.parametrize("clifford_op", [qml.S, qml.H])
    @pytest.mark.parametrize("wires", [0, 1, 10, 100])
    @pytest.mark.parametrize("pauli", [qml.I, qml.X, qml.Y, qml.Z])
    def test_apply_clifford_ops_one_wire(self, clifford_op, wires, pauli):
        pauli = pauli(wires=wires)
        clifford_op = clifford_op(wires=wires)
        xz = [pauli_to_xz(pauli)]
        new_xz = commute_clifford_op(clifford_op, xz)
        new_x, new_z = new_xz[0]
        new_pauli = xz_to_pauli(new_x, new_z)(wires=wires)

        new_xz_clifford_op = qml.prod(new_pauli, clifford_op).matrix()
        clifford_op_xz = qml.prod(clifford_op, pauli).matrix()

        assert np.allclose(new_xz_clifford_op, clifford_op_xz) | np.allclose(
            new_xz_clifford_op, -clifford_op_xz
        )

    @pytest.mark.parametrize("clifford_op", [qml.CNOT])
    @pytest.mark.parametrize("wires", [[0, 1], [1, 2], [10, 100]])
    @pytest.mark.parametrize("pauli_control", [qml.I, qml.X, qml.Y, qml.Z])
    @pytest.mark.parametrize("pauli_target", [qml.I, qml.X, qml.Y, qml.Z])
    def test_apply_clifford_ops_two_wires(self, clifford_op, wires, pauli_control, pauli_target):
        pauli_control = pauli_control(wires=wires[0])
        pauli_target = pauli_target(wires=wires[1])
        clifford_op = clifford_op(wires=wires)

        xz = [pauli_to_xz(pauli_control), pauli_to_xz(pauli_target)]
        new_xz = commute_clifford_op(clifford_op, xz)

        _xc, _zc = new_xz[0]
        new_pauli_control = xz_to_pauli(_xc, _zc)(wires=wires[0])
        _xt, _zt = new_xz[1]
        new_pauli_target = xz_to_pauli(_xt, _zt)(wires=wires[1])

        new_xz_clifford_op = qml.prod(new_pauli_control, new_pauli_target, clifford_op).matrix()
        clifford_op_xz = qml.prod(clifford_op, pauli_control, pauli_target).matrix()

        assert np.allclose(new_xz_clifford_op, clifford_op_xz) | np.allclose(
            new_xz_clifford_op, -clifford_op_xz
        )

    @pytest.mark.parametrize(
        "clifford_op", [qml.X(0), qml.RZ(phi=0.123, wires=0), qml.RX(phi=0.123, wires=0)]
    )
    @pytest.mark.parametrize("xz", [[(0, 1)]])
    def test_apply_clifford_ops_not_imp(self, clifford_op, xz):
        with pytest.raises(
            NotImplementedError, match="Only qml.H, qml.S and qml.CNOT are supported."
        ):
            _ = commute_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 1), (1, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0), (0, 0), (0, 0)]),
        ],
    )
    def test_apply_clifford_ops_xz_list_size_mismatch(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure that the length of xz matches the number of wires of the clifford_op.",
        ):
            _ = commute_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 1, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1), (0, 1, 1)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0, 1), (0, 0)]),
            (qml.CNOT(wires=[0, 1]), [(0, 0, 1), (0, 0, 1)]),
        ],
    )
    def test_apply_clifford_ops_xz_tuple_size_mismatch(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure there are 2 elements instead of in each tuple in the xz list.",
        ):
            _ = commute_clifford_op(clifford_op, xz)

    @pytest.mark.parametrize(
        "clifford_op, xz",
        [
            (qml.S(0), [(0, 2)]),
            (qml.CNOT(wires=[0, 1]), [(0, 1), (0, 3)]),
            (qml.CNOT(wires=[0, 1]), [(2, 0), (0, 0)]),
        ],
    )
    def test_apply_clifford_ops_xz_value_err(self, clifford_op, xz):
        with pytest.raises(
            ValueError,
            match="Please ensure xz are either 0 or 1.",
        ):
            _ = commute_clifford_op(clifford_op, xz)


class TestOfflineCorrection:
    """Tests for byproduct operation offline corrections."""

    @pytest.mark.parametrize(
        "ops, mid_measures, expected",
        [
            (
                [qml.CNOT(wires=[0, 1]), qml.X(0), qml.X(1)],
                [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [[(0, 1), (0, 1)]],
            ),
            ([qml.S(0), qml.Y(0)], [1, 0, 0, 1], [[(1, 0)]]),
            ([qml.H(0), qml.Z(0)], [1, 1, 0, 0], [[(1, 1)]]),
            ([qml.RZ(0.1, wires=[0]), qml.X(0)], [1, 0, 1, 1], [[(1, 0)]]),
        ],
    )
    def test_parse_mid_measurements(self, ops, mid_measures, expected):
        tape = qml.tape.QuantumScript(ops=ops)
        by_ops = _parse_mid_measurements(tape, mid_measures)
        assert by_ops == expected

    @pytest.mark.parametrize("num_shots", [250])
    @pytest.mark.parametrize(
        "ops",
        [
            [qml.RZ(0.1, wires=[0]), qml.RZ(0.3, wires=[0])],
            [qml.RZ(0.1, wires=[0]), RotXZX(0.1, 0.2, 0.3, wires=[0])],
            [qml.MultiControlledX(wires=[0, 1, 2, 3], control_values=[0, 1, 0])],
        ],
    )
    def test_unsupported_tape_offline_correction(self, num_shots, ops):
        measurements = [qml.sample(wires=[0])]

        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements, shots=num_shots)

        mid_meas = [random.choice([0, 1]) for _ in range(8)]

        with pytest.raises(NotImplementedError):
            _ = get_byproduct_corrections(tape, mid_meas, [1])

    @pytest.mark.parametrize(
        "ops, mid_meas, sample_wires, measures_expected",
        [
            (
                [qml.RZ(0.1, wires=[0]), qml.RZ(0.3, wires=[1]), qml.Z(0)],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [0],
                [[1], [0]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), qml.RZ(0.3, wires=[2]), qml.Z(0)],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [0],
                [[1], [0]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), qml.S(wires=[0])],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0],
                [[1], [1]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), RotXZX(0.1, -0.1, 0.3, wires=[1])],
                [0, 1, 0, 1, 1, 0, 0, 1],
                [0],
                [[0], [0]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), RotXZX(0.1, -0.1, 0.3, wires=[1])],
                [0, 0, 1, 0, 1, 0, 0, 1],
                [0],
                [[1], [1]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), qml.RZ(0.3, wires=[1])],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [1],
                [[0], [1]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), RotXZX(0.1, 0.2, 0.3, wires=[1])],
                [0, 1, 1, 0, 1, 0, 0, 0],
                [1],
                [[0], [0]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), RotXZX(0.1, -0.1, 0.3, wires=[1])],
                [0, 1, 0, 1, 1, 0, 0, 1],
                [1],
                [[0], [1]],
            ),
            (
                [qml.RZ(0.1, wires=[0]), RotXZX(0.1, -0.1, 0.3, wires=[1])],
                [0, 1, 0, 1, 1, 1, 0, 1],
                [1],
                [[1], [1]],
            ),
            (
                [qml.RZ(-0.1, wires=[0]), qml.RZ(0.1, wires=[1]), qml.X(wires=[1])],
                [0, 1, 0, 1, 1, 1, 0, 1],
                [0, 1],
                [[1, 0], [1, 1]],
            ),
        ],
    )
    def test_non_clifford_offline_correction(self, ops, mid_meas, sample_wires, measures_expected):
        measurements = [qml.sample(wires=sample_wires)]

        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        measures, expected = measures_expected

        corrected_meas = get_byproduct_corrections(tape, mid_meas, measures)

        assert np.allclose(corrected_meas, expected)
