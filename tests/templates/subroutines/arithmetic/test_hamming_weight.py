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
"""Tests for the private in-place Hamming-weight subcircuit."""

from collections import Counter

import numpy as np
import pytest

import pennylane as qp
from pennylane.templates.subroutines.arithmetic.hamming_weight import _HammingWeight

# (n, ANDs, CNOTs) of the circuit for n input bits
GATE_COUNTS = [(1, 0, 0), (2, 1, 1), (3, 1, 5), (4, 3, 7), (5, 3, 11), (6, 4, 16), (7, 4, 20)]


def _hamming_weight_circuit(n):
    """Quantum function that copies the Hamming weight of ``n`` wires in a uniform superposition
    onto output wires, least significant bit first, computing and uncomputing it with
    :func:`~._HammingWeight`, and returns the probabilities of the input, work and output
    wires."""
    wires = list(range(n))
    work_wires = list(range(n, 2 * n - n.bit_count()))
    out_wires = [wires[-1], *work_wires[: n.bit_length() - 1]]
    copy_wires = list(range(2 * n - n.bit_count(), 2 * n - n.bit_count() + n.bit_length()))

    def circuit():
        for wire in wires:
            qp.H(wire)
        _HammingWeight(wires, work_wires)
        for out_wire, copy_wire in zip(out_wires, copy_wires):
            qp.CNOT([out_wire, copy_wire])
        qp.adjoint(_HammingWeight(wires, work_wires))
        return qp.probs(wires=wires + work_wires + copy_wires)

    return circuit


def _expected_probs(n):
    """Uniform probabilities of the inputs ``x``, zeroed work wires and the Hamming weight of
    ``x``, least significant bit first."""
    num_work, num_out = n - n.bit_count(), n.bit_length()
    probs = np.zeros(2 ** (n + num_work + num_out))
    for x in range(2**n):
        probs[(x << num_work + num_out) + int(f"{x.bit_count():0{num_out}b}"[::-1], 2)] = 2**-n
    return probs


@pytest.mark.parametrize("n", range(2, 8))
def test_computes_and_uncomputes_hamming_weight(n):
    """Test that the Hamming weight ends up on the last input wire and the first work wires, and
    that its uncomputation restores the inputs and the work wires, for all basis inputs at once.
    For n = 1, the compute block would be empty."""
    circuit = qp.qnode(qp.device("default.qubit"))(_hamming_weight_circuit(n))
    assert np.allclose(circuit(), _expected_probs(n))


@pytest.mark.catalyst
def test_qjit():
    """Test that the Hamming-weight circuit compiles and computes the Hamming weight with qjit."""
    n = 5
    num_wires = int(np.log2(len(_expected_probs(n))))
    dev = qp.device("lightning.qubit", wires=num_wires)
    circuit = qp.qjit(qp.qnode(dev)(_hamming_weight_circuit(n)), capture=True)
    assert np.allclose(circuit(), _expected_probs(n))


@pytest.mark.parametrize("n, num_ands, num_cnots", GATE_COUNTS)
def test_gate_counts(n, num_ands, num_cnots):
    """Test that the circuit uses n - popcount(n) ANDs, each with its own work wire as target, and
    5n - 4 * n.bit_length() - popcount(n) CNOTs, and no other gates, matching the registered
    resources."""
    work_wires = list(range(n, n + num_ands))
    op = _HammingWeight(range(n), work_wires)
    qp.testing.assert_valid_decomp_rule(op, qp.list_decomps(_HammingWeight)[0], check_matrix=False)
    ops = op.decomposition()
    counts = Counter(type(op) for op in ops)
    assert set(counts) <= {qp.TemporaryAND, qp.CNOT}
    assert (counts[qp.TemporaryAND], counts[qp.CNOT]) == (num_ands, num_cnots)
    and_targets = [op.wires[2] for op in ops if isinstance(op, qp.TemporaryAND)]
    assert sorted(and_targets) == work_wires
