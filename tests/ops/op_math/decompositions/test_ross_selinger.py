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
from pennylane.ops.op_math.decompositions.ross_selinger import (
    _domain_correction,
    _jit_rs_decomposition,
    rs_decomposition,
)


@pytest.mark.parametrize(
    ("angle", "result"),
    [
        (math.pi / 3, (-math.pi / 2, ZOmega(b=1))),
        (math.pi / 8, ((1.0, 1), (ZOmega(b=1), ZOmega(b=1)))),
        (-3 * math.pi / 8, ((-1.0, 3), (ZOmega(b=-1), ZOmega(c=1)))),
        (math.pi / 2, (-math.pi / 2, ZOmega(b=1))),
        (5 * math.pi / 3, (-3 * math.pi / 2, ZOmega(b=-1))),
        (3 * math.pi / 2, (-3 * math.pi / 2, ZOmega(b=-1))),
        (math.pi, (-math.pi, ZOmega(d=-1))),
    ],
)
def test_domain_correction(angle, result):
    """Test the functionality to create domain correction"""
    shift, scale = _domain_correction(angle)
    assert qml.math.allclose(shift, result[0])
    assert scale == result[1]


@pytest.mark.parametrize(
    ("op", "epsilon"),
    [
        (qml.RZ(math.pi / 42, wires=[1]), 1e-4),
        (qml.RZ(5 * math.pi / 4, wires=[1]), 1e-5),
        (qml.PhaseShift(math.pi / 7, wires=["a"]), 1e-3),
        (qml.RZ(math.pi / 3, wires=[1]), 1e-5),
        (qml.RZ(-math.pi / 3, wires=[1]), 1e-3),
        (qml.RZ(-math.pi / 4, wires=[1]), 1e-4),
        (qml.PhaseShift(-math.pi / 6, wires=[0]), 1e-3),
        (qml.RZ(-math.pi / 5, wires=[0]), 1e-4),
        (qml.RZ(-math.pi / 7, wires=[1]), 1e-4),
        (qml.RZ(-math.pi / 8, wires=[2]), 1e-2),
        (qml.RZ(9 * math.pi / 4, wires=[0]), 1e-3),
        (qml.RZ(4.434079721283546, wires=[3]), 1e-3),
    ],
)
def test_ross_selinger(op, epsilon):
    """Test Ross-Selinger decomposition method with specified max-depth"""
    with qml.queuing.AnnotatedQueue() as q:
        gates = rs_decomposition(op, epsilon=epsilon)
    assert q.queue == gates

    matrix_rs = qml.matrix(qml.tape.QuantumScript(gates))

    assert qml.math.allclose(qml.matrix(op), matrix_rs, atol=epsilon)
    assert qml.prod(*gates, lazy=False).wires == op.wires


@pytest.mark.catalyst
@pytest.mark.jax
@pytest.mark.external
@pytest.mark.parametrize(
    ("op", "epsilon", "wires"),
    [
        (qml.RZ(math.pi / 42, wires=[1]), 1e-4, 2),
        (qml.RZ(5 * math.pi / 4, wires=[1]), 1e-2, 2),
        (qml.RZ(math.pi / 3, wires=[1]), 1e-3, 2),
        (qml.RZ(-math.pi / 3, wires=[1]), 1e-3, 2),
        (qml.PhaseShift(-math.pi / 6, wires=[0]), 1e-3, 1),
        (qml.RZ(-math.pi / 8, wires=[2]), 1e-2, 3),
        (qml.RZ(9 * math.pi / 4, wires=[0]), 1e-3, 1),
        (qml.RZ(4.434079721283546, wires=[3]), 1e-3, 4),
    ],
)
@pytest.mark.filterwarnings("ignore::pennylane.exceptions.PennyLaneDeprecationWarning")
def test_ross_selinger_qjit(op, epsilon, wires):
    """Test Ross-Selinger decomposition method with specified max-depth"""
    pytest.importorskip("catalyst")
    dev = qml.device("lightning.qubit", wires=wires)

    @qml.qjit(static_argnums=0)
    @qml.qnode(dev)
    def circuit(is_qjit):
        rs_decomposition(op, epsilon=epsilon, is_qjit=is_qjit)
        return qml.state()

    qjit_result = circuit(True)
    non_qjit_result = circuit(False)
    assert qml.math.allclose(qjit_result, non_qjit_result)


def test_epsilon_value_effect():
    """Test that different epsilon values create different decompositions."""
    op = qml.RZ(math.pi / 5, 0)
    decomp_with_error = rs_decomposition(op, 1e-4)
    decomp_less_error = rs_decomposition(op, 1e-2)
    assert len(decomp_with_error) > len(decomp_less_error)


def test_warm_start():
    """Test that warm start is working."""
    op = qml.RZ(math.pi / 8, 0)
    decomp_with_error = rs_decomposition(op, 1e-10, max_search_trials=100)
    decomp_less_error = rs_decomposition(op, 1e-3, max_search_trials=100)
    assert len(decomp_with_error) == len(decomp_less_error)


def test_exception():
    """Test operation wire exception in Ross-Selinger"""
    op = qml.SingleExcitation(1.0, wires=[1, 2])

    with pytest.raises(
        ValueError,
        match=r"Operator must be a RZ or PhaseShift gate",
    ):
        rs_decomposition(op, epsilon=1e-4, max_search_trials=1)


@pytest.mark.catalyst
@pytest.mark.jax
@pytest.mark.external
@pytest.mark.parametrize(
    ("decomposition_info"),
    [
        # T, (SHT, HT, SHT, SHT, SHT, HT ) , I
        [1, [1, 0, 1, 1, 1, 0], 0],
        # _ ,  (SHT, HT, SHT, SHT, SHT, HT ), S, Y
        [0, [1, 0, 1, 1, 1, 0], 12],
        # _ , , H, Sd
        [0, [], 9],
    ],
)
@pytest.mark.filterwarnings("ignore::pennylane.exceptions.PennyLaneDeprecationWarning")
def test_jit_rs_decomposition(decomposition_info):
    """Test that the qjit rs decomposition is working."""
    pytest.importorskip("catalyst")
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    # Create decomposition info using jnp
    has_leading_t = jnp.int32(decomposition_info[0])  # First element
    syllable_sequence = jnp.array(decomposition_info[1], dtype=jnp.int32)  # Second element
    clifford_op_idx = jnp.int32(decomposition_info[2])  # Third element

    decomposition_info = (has_leading_t, syllable_sequence, clifford_op_idx)

    # Get the operations from _jit_rs_decomposition
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qjit_circuit():
        _jit_rs_decomposition(0, decomposition_info)
        return qml.state()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def non_qjit_circuit():
        if int(decomposition_info[0].item()) == 1:
            qml.T(0)
        for i in decomposition_info[1]:
            if i == 0:
                qml.H(0)
                qml.T(0)
            elif i == 1:
                qml.S(0)
                qml.H(0)
                qml.T(0)
        if decomposition_info[2] == 12:
            qml.S(0)
            qml.Y(0)
        elif decomposition_info[2] == 9:
            qml.H(0)
            qml.adjoint(qml.S)(0)

        return qml.state()

    qjit_result = qml.qjit(qjit_circuit)()
    # Do not jit the reference circuit; it uses standard Python control flow
    non_qjit_result = non_qjit_circuit()

    assert qml.math.allclose(qjit_result, non_qjit_result)
