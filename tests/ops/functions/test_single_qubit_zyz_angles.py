# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for qp.single_qubit_zyz_angles."""

import pytest

import pennylane as qp
from pennylane.ops import (
    CRX,
    RX,
    RY,
    RZ,
    SX,
    U2,
    U3,
    GlobalPhase,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    Rot,
    S,
    T,
)
from pennylane.ops.functions.single_qubit_zyz_angles import single_qubit_zyz_angles

TEST_OPS = [
    RX(0.123, wires=0),
    RY(0.123, wires=0),
    RZ(0.123, wires=0),
    PhaseShift(0.123, wires=0),
    Rot(0.1, 0.2, 0.3, wires=0),
    SX(0),
    Hadamard(0),
    PauliX(0),
    PauliY(0),
    PauliZ(0),
    S(0),
    T(0),
]


def test_error_for_multi_qubit_ops():
    """Tests that a ValueError is raised for ops on multiple wires."""

    with pytest.raises(ValueError, match="not applicable to operators on more than one wire."):
        single_qubit_zyz_angles(CRX(0.5, [0, 1]))


@pytest.mark.parametrize("op", TEST_OPS)
def test_angles_correct(op):
    """Tests that single_qubit_zyz_angles are correct."""

    expected_mat = qp.matrix(op)

    phi, theta, omega, alpha = single_qubit_zyz_angles(op)
    decomp = [RZ(phi, wires=0), RY(theta, wires=0), RZ(omega, wires=0), GlobalPhase(-alpha)]
    decomp_mat = qp.matrix(decomp, wire_order=[0])

    assert qp.math.allclose(decomp_mat, expected_mat)


@pytest.mark.parametrize("op", TEST_OPS)
def test_angles_correct_adjoint(op):
    """Tests that single_qubit_zyz_angles are correct for adjoint ops"""

    expected_mat = qp.matrix(qp.adjoint(op))

    phi, theta, omega, alpha = single_qubit_zyz_angles(qp.adjoint(op))
    decomp = [RZ(phi, wires=0), RY(theta, wires=0), RZ(omega, wires=0), GlobalPhase(-alpha)]
    decomp_mat = qp.matrix(decomp, wire_order=[0])

    assert qp.math.allclose(decomp_mat, expected_mat)


@pytest.mark.parametrize("op", [U2(0.1, 0.2, wires=0), U3(0.1, 0.2, 0.3, wires=0)])
def test_computed_angles(op):
    """Tests that the computed angles are correct for non-registered ops."""

    assert type(op) not in single_qubit_zyz_angles.registry

    expected_mat = qp.matrix(op)

    phi, theta, omega, alpha = single_qubit_zyz_angles(op)
    decomp = [RZ(phi, wires=0), RY(theta, wires=0), RZ(omega, wires=0), GlobalPhase(-alpha)]
    decomp_mat = qp.matrix(decomp, wire_order=[0])

    assert qp.math.allclose(decomp_mat, expected_mat)
