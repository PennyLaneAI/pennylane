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
"""Unit tests for the Clifford+T transform."""

import math
from functools import reduce

import pytest
import pennylane as qml

from pennylane.transforms.decompositions.clifford_t.clifford_t_transform import (
    check_clifford_t,
    clifford_t_decomposition,
    _rot_decompose,
    _one_qubit_decompose,
    _two_qubit_decompose,
    _CLIFFORD_T_GATES,
)

from pennylane.transforms.optimization.optimization_utils import _fuse_global_phases

_SKIP_GATES = [qml.Barrier, qml.Snapshot, qml.WireCut, qml.GlobalPhase]
_CLIFFORD_PHASE_GATES = _CLIFFORD_T_GATES + _SKIP_GATES

INVSQ2 = 1 / math.sqrt(2)
PI = math.pi


def circuit_1():
    """Circuit 1 with quantum chemistry gates"""
    qml.RZ(1.0, wires=[0])
    qml.PhaseShift(1.0, wires=[1])
    qml.SingleExcitation(2.0, wires=[1, 2])
    qml.PauliX(0)
    return qml.expval(qml.PauliZ(1))


def circuit_2():
    """Circuit 2 without chemistry gates"""
    qml.CRX(1, wires=[0, 1])
    qml.IsingXY(2, wires=[1, 2])
    qml.ISWAP(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


def circuit_3():
    """Circuit 3 with Clifford gates"""
    qml.GlobalPhase(PI)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.ISWAP(wires=[0, 1])
    qml.Hadamard(wires=[0])
    qml.WireCut(wires=[1])
    qml.RZ(PI, wires=[0])
    return qml.expval(qml.PauliZ(0))


def circuit_4():
    """Circuit 4 with a Template"""
    qml.RandomLayers(weights=qml.math.array([[0.1, -2.1, 1.4]]), wires=range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_5():
    """Circuit 5 with Qubit Unitaries"""
    matrix = qml.math.array(
        [
            [-0.728 - 0.244j, 0.297 - 0.079j, 0.262 - 0.397j, -0.129 + 0.272j],
            [-0.264 + 0.018j, -0.043 + 0.69j, -0.534 - 0.107j, -0.315 - 0.235j],
            [-0.549 + 0.113j, -0.451 - 0.067j, 0.181 + 0.651j, -0.041 - 0.139j],
            [0.033 - 0.159j, -0.297 + 0.362j, -0.1 + 0.103j, 0.346 + 0.784j],
        ]
    )
    qml.QubitUnitary(matrix, wires=[0, 1])
    qml.DiagonalQubitUnitary(
        [qml.math.exp(1j * 0.1), qml.math.exp(1j * PI), INVSQ2 * (1 + 1j), INVSQ2 * (1 - 1j)],
        wires=[0, 1],
    )
    return qml.expval(qml.PauliZ(0))


class TestCliffordCompile:
    """Unit tests for clifford compilation function."""

    @pytest.mark.parametrize(
        "op, res",
        [
            (qml.DoubleExcitation(2.0, wires=[0, 1, 2, 3]), False),
            (qml.PauliX(wires=[1]), True),
            (qml.ECR(wires=["e", "f"]), True),
            (qml.CH(wires=["a", "b"]), False),
            (qml.WireCut(0), False),
        ],
    )
    def test_clifford_checker(self, op, res):
        """Test Clifford checker operation for gate"""
        assert check_clifford_t(op) == res

    @pytest.mark.parametrize(
        ("circuit, max_depth"),
        [(circuit_1, 1), (circuit_2, 0), (circuit_3, 0), (circuit_4, 1), (circuit_5, 0)],
    )
    def test_decomposition(self, circuit, max_depth):
        """Test decomposition for the Clifford transform."""

        with qml.tape.QuantumTape() as old_tape:
            circuit()

        [new_tape], tape_fn = clifford_t_decomposition(old_tape, max_depth=max_depth, depth=3)

        assert all(
            any(
                (
                    isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
                    for gate in _CLIFFORD_PHASE_GATES
                )
            )
            for op in new_tape.operations
        )

        dev = qml.device("default.qubit")
        transform_program = dev.preprocess()[0]
        res1, res2 = qml.execute(
            [old_tape, new_tape], device=dev, transform_program=transform_program
        )
        qml.math.isclose(res1, tape_fn([res2]), atol=1e-2)

    def test_qnode_decomposition(self):
        """Test decomposition for the Clifford transform."""

        dev = qml.device("default.qubit")

        def qfunc():
            qml.PhaseShift(1.0, wires=[0])
            qml.PhaseShift(2.0, wires=[1])
            qml.ISWAP(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        original_qnode = qml.QNode(qfunc, dev)
        transfmd_qnode = qml.QNode(clifford_t_decomposition(qfunc, depth=3, basis_length=10), dev)

        res1, res2 = original_qnode(), transfmd_qnode()
        assert qml.math.isclose(res1, res2, atol=1e-2)

        assert all(
            any(
                (
                    isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
                    for gate in _CLIFFORD_PHASE_GATES
                )
            )
            for op in transfmd_qnode.tape.operations
        )

    @pytest.mark.parametrize(
        ("op"), [qml.RX(1.0, wires="a"), qml.U3(1, 2, 3, wires=[1]), qml.PhaseShift(1.0, wires=[2])]
    )
    def test_one_qubit_decomposition(self, op):
        """Test decomposition for the Clifford transform."""

        decomp_ops, global_ops = _one_qubit_decompose(op)
        decomp_ops = _fuse_global_phases(decomp_ops + [global_ops])

        assert all(
            any(
                (
                    isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
                    for gate in _CLIFFORD_PHASE_GATES + [qml.RZ]
                )
            )
            for op in decomp_ops
        )

        global_ops = decomp_ops.pop()
        assert isinstance(global_ops, qml.GlobalPhase)

        matrix_op = reduce(
            lambda x, y: x @ y, [qml.matrix(op) for op in decomp_ops][::-1]
        ) * qml.matrix(global_ops)

        # check for matrice equivalence up to global phase
        phase = qml.math.divide(
            matrix_op,
            qml.matrix(op),
            out=qml.math.zeros_like(matrix_op, dtype=complex),
            where=matrix_op != 0,
        )[qml.math.nonzero(qml.math.round(matrix_op, 10))]
        assert qml.math.allclose(
            phase / phase[0], qml.math.ones(qml.math.shape(phase)[0]), atol=1e-5
        )

    @pytest.mark.parametrize(
        ("op"),
        [
            qml.PSWAP(1.0, wires=["a", "b"]),
            qml.SingleExcitation(1, wires=[1, 2]),
            qml.IsingXX(1.0, wires=[2, 3]),
        ],
    )
    def test_two_qubit_decomposition(self, op):
        """Test decomposition for the Clifford transform."""

        decomp_ops = _fuse_global_phases(_two_qubit_decompose(op))

        assert all(
            any(
                (
                    isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
                    for gate in _CLIFFORD_PHASE_GATES + [qml.RZ]
                )
            )
            for op in decomp_ops
        )

        decomp_ops, global_ops = decomp_ops[:-1], decomp_ops[-1]
        wire_map = {wire: idx for idx, wire in enumerate(op.wires)}
        mapped_op = [qml.map_wires(op, wire_map=wire_map) for op in decomp_ops][::-1]
        matrix_op = reduce(
            lambda x, y: x @ y, [qml.matrix(op, wire_order=[0, 1]) for op in mapped_op]
        ) * qml.matrix(global_ops, wire_order=[0, 1])

        # check for matrice equivalence up to global phase
        phase = qml.math.divide(
            matrix_op,
            qml.matrix(op),
            out=qml.math.zeros_like(matrix_op, dtype=complex),
            where=matrix_op != 0,
        )[qml.math.nonzero(qml.math.round(matrix_op, 10))]
        assert qml.math.allclose(phase / phase[0], qml.math.ones(qml.math.shape(phase)[0]))

    @pytest.mark.parametrize(
        ("op"),
        [
            qml.adjoint(qml.RX(1.0, wires=["b"])),
            qml.Rot(1, 2, 3, wires=[2]),
            qml.PhaseShift(1.0, wires=[0]),
            qml.PhaseShift(3 * PI, wires=[0]),
        ],
    )
    def test_rot_decomposition(self, op):
        """Test decomposition for the Clifford transform."""

        decomp_ops = _fuse_global_phases(_rot_decompose(op))

        assert all(
            any(
                (
                    isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
                    for gate in _CLIFFORD_PHASE_GATES + [qml.RZ]
                )
            )
            for op in decomp_ops
        )

        decomp_ops, global_ops = decomp_ops[:-1], decomp_ops[-1]
        matrix_op = reduce(
            lambda x, y: x @ y, [qml.matrix(op) for op in decomp_ops][::-1]
        ) * qml.matrix(global_ops)

        # check for matrice equivalence up to global phase
        phase = qml.math.divide(
            matrix_op,
            qml.matrix(op),
            out=qml.math.zeros_like(matrix_op, dtype=complex),
            where=matrix_op != 0,
        )[qml.math.nonzero(qml.math.round(matrix_op, 10))]
        assert qml.math.allclose(phase / phase[0], qml.math.ones(qml.math.shape(phase)[0]))

    def test_raise_with_cliffordt_decomposition(self):
        """Test that exception is correctly raise when decomposing gates without any decomposition"""

        with qml.tape.QuantumTape() as tape:
            qml.QubitUnitary(qml.math.eye(8), wires=[0, 1, 2])

        with pytest.raises(ValueError, match="Cannot unroll"):
            clifford_t_decomposition(tape, max_depth=0)

    @pytest.mark.parametrize(
        ("op"),
        [
            qml.U1(1.0, wires=["b"]),
        ],
    )
    def test_raise_with_rot_decomposition(self, op):
        """Test that exception is correctly raise when decomposing parameterized gates for which we already don't have a recipie"""

        with pytest.raises(
            ValueError,
            match="qml.RX, qml.RY, qml.RZ, qml.Rot and qml.PhaseShift",
        ):
            _fuse_global_phases(_rot_decompose(op))

    def test_raise_with_decomposition_method(self):
        """Test that exception is correctly raise when using incorrect decomposing method"""

        with pytest.raises(
            NotImplementedError,
            match=r"Currently we only support Solovay-Kitaev \('sk'\) decompostion",
        ):
            dev = qml.device("default.qubit")

            def qfunc():
                qml.RX(1.0, wires=[0])
                return qml.expval(qml.PauliZ(0))

            qml.QNode(clifford_t_decomposition(qfunc, method="synth"), dev)()
