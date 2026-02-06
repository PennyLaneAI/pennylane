# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the transform ``qp.transform.rz_phase_gradient``"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.transforms.rz_phase_gradient import _binary_repr_int, _rz_phase_gradient


def prepare_phase_gradient(wires):
    ops = []
    for i, w in enumerate(wires):
        ops.append(qp.H(w))
        ops.append(qp.PhaseShift(-np.pi / 2**i, w))
    return ops


@pytest.mark.parametrize(
    "phi, p, expected",
    [
        (1 / 2 * 2 * np.pi, 2, "10"),
        (1 / 2 * 2 * np.pi, 3, "100"),
        ((1 / 2 + 1 / 8 + 1 / 16) * 2 * np.pi, 2, "11"),
        ((1 / 2 + 1 / 8 + 1 / 16 + 1 / 32) * 2 * np.pi, 3, "110"),
        ((1 / 2 + 1 / 8 + 1 / 16 + 1 / 32) * 2 * np.pi, 5, "10111"),
    ],
)
def test_binary_repr_int(phi, expected, p):
    """Test that the binary representation or approximation of the angle is correct"""

    assert expected == bin(_binary_repr_int(phi, p))[-p:]


@pytest.mark.parametrize("p", [2, 3, 4])
def test_units_rz_phase_gradient(p):
    """Test the outputs of _rz_phase_gradient"""

    phi = (1 / 2 + 1 / 4 + 1 / 8 + 1 / 16 + 1 / 32) * 2 * np.pi

    wire = "targ"
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    op = _rz_phase_gradient(
        phi,
        wire,
        angle_wires=angle_wires,
        phase_grad_wires=phase_grad_wires,
        work_wires=work_wires,
    )

    assert isinstance(op, qp.ops.op_math.ChangeOpBasis)

    operands = op.operands

    assert isinstance(operands[0], qp.ops.op_math.controlled.ControlledOp)
    assert np.allclose(operands[0].base.parameters, [0] * p)
    assert operands[0].base.wires == angle_wires

    assert isinstance(operands[1], qp.SemiAdder)
    assert operands[1].wires == angle_wires + phase_grad_wires + work_wires

    assert isinstance(operands[2], qp.ops.op_math.controlled.ControlledOp)
    assert np.allclose(operands[2].base.parameters, [0] * p)
    assert operands[2].base.wires == angle_wires


def test_global_phases():
    """Test that one single global phase is correctly returned"""

    phis = np.array([0.5, 0.3, 0.1])
    circ = qp.tape.QuantumScript([qp.RZ(phi, 0) for phi in phis])

    p = 4
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

    res, fn = qp.transforms.rz_phase_gradient(
        circ,
        angle_wires=angle_wires,
        phase_grad_wires=phase_grad_wires,
        work_wires=work_wires,
    )
    tape = fn(res)

    global_phase = tape.operations[-1]
    assert not any(isinstance(op, qp.GlobalPhase) for op in tape.operations[:-1])
    assert isinstance(global_phase, qp.GlobalPhase)
    assert np.isclose(global_phase.parameters[0], np.sum(phis / 2))


def test_wire_validation():
    """Test that an error is raised when phg wires are fewer than angle wires"""

    circ = qp.tape.QuantumScript([qp.RZ(0.5, 0)])

    angle_wires = qp.wires.Wires([f"angle_{i}" for i in range(3)])
    phase_grad_wires = qp.wires.Wires([f"phg_{i}" for i in range(2)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(2)])

    with pytest.raises(
        ValueError, match="phase_grad_wires needs to be at least as large as angle_wires"
    ):
        _ = qp.transforms.rz_phase_gradient(
            circ,
            angle_wires=angle_wires,
            phase_grad_wires=phase_grad_wires,
            work_wires=work_wires,
        )


@pytest.mark.parametrize(
    "phi",
    [
        (1 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi,
        -(1 / 2 + 1 / 4 + 1 / 8) * 2 * np.pi,
        (1 / 8) * 2 * np.pi,
        -(1 / 2) * 2 * np.pi,
    ],
)
def test_integration_rz_phase_gradient(phi):
    """Test that the transform applies the RZ gate correctly by doing an X rotation via two Hadamards"""
    precision = 3
    wire = "targ"
    angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(precision)])
    phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(precision)])
    work_wires = qp.wires.Wires([f"work_{i}" for i in range(precision - 1)])
    wire_order = [wire] + angle_wires + phase_grad_wires + work_wires

    rz_circ = qp.tape.QuantumScript(
        [
            qp.Hadamard(wire),  # prepare |+>
            *prepare_phase_gradient(phase_grad_wires),
            qp.RZ(phi, wire),
            *[qp.adjoint(op) for op in prepare_phase_gradient(phase_grad_wires)[::-1]],
            qp.Hadamard(wire),  # unprepare |+>
        ]
    )

    res, fn = qp.transforms.rz_phase_gradient(rz_circ, angle_wires, phase_grad_wires, work_wires)
    tapes = fn(res)
    output = qp.matrix(tapes, wire_order=wire_order)[:, 0]

    output_expected = qp.matrix(qp.RX(phi, 0))[:, 0]
    output_expected = np.kron(output_expected, np.eye(2 ** (len(wire_order) - 1))[0])

    assert np.allclose(output, output_expected)
