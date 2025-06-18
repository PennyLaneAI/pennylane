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
"""Tests for Ross-Selinger decomposition's implementation."""

import math

import pytest

import pennylane as qml
from pennylane.ops.op_math.decompositions.rings import ZOmega
from pennylane.ops.op_math.decompositions.ross_selinger import _domain_correction, rs_decomposition


@pytest.mark.parametrize(
    ("angle", "result"),
    [
        (math.pi / 3, (-math.pi / 2, ZOmega(b=1))),
        (math.pi / 8, (0.0, ZOmega(d=1))),
        (math.pi / 2, (-math.pi / 2, ZOmega(b=1))),
        (5 * math.pi / 3, (-3 * math.pi / 2, ZOmega(b=-1))),
        (3 * math.pi / 2, (-3 * math.pi / 2, ZOmega(b=-1))),
        (math.pi, (-math.pi, ZOmega(d=-1))),
    ],
)
def test_domain_correction(angle, result):
    """Test the functionality to create domain correction"""
    shift, scale = _domain_correction(angle)
    assert qml.math.isclose(shift, result[0])
    assert scale == result[1]


@pytest.mark.parametrize(
    ("op", "epsilon"),
    [
        (qml.RZ(math.pi / 42, wires=[1]), 1e-4),
        (qml.PhaseShift(math.pi / 7, wires=["a"]), 1e-3),
        (qml.RZ(math.pi / 3, wires=[1]), 1e-5),
        (qml.PhaseShift(-math.pi / 6, wires=[0]), 1e-3),
        (qml.RZ(-math.pi / 5, wires=[0]), 1e-4),
        (qml.RZ(-math.pi / 8, wires=[2]), 1e-2),
    ],
)
def test_ross_selinger(op, epsilon):
    """Test Ross-Selinger decomposition method with specified max-depth"""
    with qml.queuing.AnnotatedQueue() as q:
        gates = rs_decomposition(op, epsilon=epsilon)
    assert q.queue == gates

    matrix_rs = qml.matrix(qml.tape.QuantumScript(gates))

    mat1 = qml.matrix(op)
    mat2 = matrix_rs
    phase = qml.math.divide(
        mat1, mat2, out=qml.math.zeros_like(mat1, dtype=complex), where=mat1 != 0
    )[qml.math.nonzero(qml.math.round(mat1, 10))]

    assert qml.math.allclose(phase / phase[0], qml.math.ones(len(phase)), atol=1e-3)
    assert qml.math.allclose(qml.matrix(op), phase[0] * matrix_rs, atol=epsilon)
    assert qml.prod(*gates, lazy=False).wires == op.wires


def test_epsilon_value_effect():
    """Test that different epsilon values create different decompositions."""
    op = qml.RZ(math.pi / 5, 0)
    decomp_with_error = rs_decomposition(op, 1e-4)
    decomp_less_error = rs_decomposition(op, 1e-2)
    assert len(decomp_with_error) > len(decomp_less_error)


def test_exception():
    """Test operation wire exception in Ross-Selinger"""
    op = qml.SingleExcitation(1.0, wires=[1, 2])

    with pytest.raises(
        ValueError,
        match=r"Operator must be a RZ or PhaseShift gate",
    ):
        rs_decomposition(op, epsilon=1e-4, max_trials=1)
