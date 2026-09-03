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
from pennylane.transforms.rz_phase_gradient import _rz_phase_gradient


def prepare_phase_gradient(wires):
    ops = []
    for i, w in enumerate(wires):
        ops.append(qp.H(w))
        ops.append(qp.PhaseShift(-np.pi / 2**i, w))
    return ops


@pytest.mark.parametrize(
    "frac, adaptive_precision, expected_width",
    [
        # all-ones angle: full width regardless of adaptive_precision
        (1 - 2.0**-4, True, 4),
        (1 - 2.0**-4, False, 4),
        # trailing zeros: adaptive truncates to the MSB-anchored width, non-adaptive keeps full width
        (1 / 2, True, 1),
        (1 / 2, False, 4),
        (1 / 4, True, 2),
        (1 / 4, False, 4),
        # rounds to zero: adaptive drops the addition (None), non-adaptive keeps full width
        (0.0, True, 0),
        (0.0, False, 4),
    ],
)
def test_units_rz_phase_gradient(frac, adaptive_precision, expected_width):
    """Test the outputs of ``_rz_phase_gradient``. With ``adaptive_precision`` a concrete angle is
    truncated to its MSB-anchored significant width (dropped entirely if it rounds to zero); with
    ``adaptive_precision=False`` the full-width adder is always emitted. The work register is kept
    in full in every case."""

    p = 4
    phi = frac * 2 * np.pi

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
        adaptive_precision=adaptive_precision,
    )

    if expected_width == 0:
        assert op is None
        return

    expected_bits = qp.math.binary_decimals(phi, p, unit=2 * np.pi)[:expected_width]
    # Positions of the set bits within the (possibly truncated) angle. The fanout is a ``MultiX``
    # (one ``X`` per set bit on the angle wires) controlled by the RZ target wire.
    set_positions = [i for i, b in enumerate(expected_bits) if int(b)]
    expected_targets = {angle_wires[i] for i in set_positions}

    assert isinstance(op, qp.ops.op_math.ChangeOpBasis)
    operands = op.operands

    # operands[0] and operands[2] are the (self-inverse) compute / uncompute controlled-MultiX fanouts.
    for fanout in (operands[0], operands[2]):
        assert isinstance(fanout.base, qp.MultiX)
        assert fanout.control_wires == qp.wires.Wires(wire)  # controlled by the RZ target wire
        assert fanout.base.wires == angle_wires[:expected_width]
        flipped = {angle_wires[i] for i, b in enumerate(fanout.base.bitstring) if int(b)}
        assert flipped == expected_targets

    assert isinstance(operands[1], qp.SemiAdder)
    assert (
        operands[1].wires
        == angle_wires[:expected_width] + phase_grad_wires[:expected_width] + work_wires
    )


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
