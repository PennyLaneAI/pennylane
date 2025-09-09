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

"""Tests for the transform ``qml.transform.rz_phase_gradient``"""
from itertools import product

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms.rz_phase_gradient import _binary_repr_int, _rz_phase_gradient


@pytest.mark.parametrize("string", list(product([0, 1], repeat=4)))
@pytest.mark.parametrize("p", [2, 3, 4])
def test_binary_repr_int(string, p):
    """Test that the binary representation or approximation of the angle is correct

    In particular, this tests that phi = (c1 2^-1 + c2 2^-2 + .. + cp 2^-p + ... + 2^-N) 2pi
    is correctly represented as (c1, c2, .., cp) for precision p
    """
    phi = np.sum([c * 2 ** (-i - 1) for i, c in enumerate(string)]) * 2 * np.pi
    string_str = "".join([str(i) for i in string])
    binary_rep_re = np.binary_repr(_binary_repr_int(phi, precision=p), width=p)
    assert binary_rep_re == string_str[:p], f"nope: {binary_rep_re}, {string_str[:p]}, {p}"


@pytest.mark.parametrize("p", [2, 3, 4])
def test_units_rz_phase_gradient(p):
    """Test the outputs of _rz_phase_gradient"""
    phi = -(1 / 2 + 1 / 4 + 1 / 8 + 1 / 16) * 2 * np.pi
    wire = "targ"
    aux_wires = qml.wires.Wires([f"aux_{i}" for i in range(p)])
    phase_grad_wires = qml.wires.Wires([f"qft_{i}" for i in range(p)])
    work_wires = qml.wires.Wires([f"work_{i}" for i in range(p - 1)])

    ops = _rz_phase_gradient(
        qml.RZ(phi, wire),
        aux_wires=aux_wires,
        phase_grad_wires=phase_grad_wires,
        work_wires=work_wires,
    )

    assert isinstance(ops[0], qml.ops.op_math.controlled.ControlledOp)
    assert np.allclose(ops[0].base.parameters, [1] * p)
    assert ops[0].base.wires == aux_wires

    assert isinstance(ops[1], qml.SemiAdder)
    assert ops[1].wires == aux_wires + phase_grad_wires + work_wires

    assert isinstance(ops[2], qml.ops.op_math.controlled.ControlledOp)
    assert np.allclose(ops[2].base.parameters, [1] * p)
    assert ops[2].base.wires == aux_wires


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
    aux_wires = qml.wires.Wires([f"aux_{i}" for i in range(precision)])
    phase_grad_wires = qml.wires.Wires([f"qft_{i}" for i in range(precision)])
    work_wires = qml.wires.Wires([f"work_{i}" for i in range(precision - 1)])
    wire_order = [wire] + aux_wires + phase_grad_wires + work_wires

    rz_circ = qml.tape.QuantumScript(
        [
            qml.Hadamard(wire),  # prepare |+>
            qml.X(phase_grad_wires[-1]),  # prepare phase gradient state
            qml.QFT(phase_grad_wires),  # prepare phase gradient state
            qml.RZ(phi, wire),
            qml.adjoint(qml.QFT)(phase_grad_wires),  # unprepare phase gradient state
            qml.X(phase_grad_wires[-1]),  # unprepare phase gradient state
            qml.Hadamard(wire),  # unprepare |+>
        ]
    )

    res, fn = qml.transforms.rz_phase_gradient(rz_circ, aux_wires, phase_grad_wires, work_wires)
    tapes = fn(res)
    output = qml.matrix(tapes, wire_order=wire_order)[:, 0]

    output_expected = qml.matrix(qml.RX(phi, 0))[:, 0]
    output_expected = np.kron(output_expected, np.eye(2 ** (len(wire_order) - 1))[0])

    assert np.allclose(output, output_expected)
