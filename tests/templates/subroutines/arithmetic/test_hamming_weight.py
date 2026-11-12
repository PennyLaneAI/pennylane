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
"""Tests for the private in-place Hamming-weight helpers."""

from collections import Counter
from functools import partial

import numpy as np
import phase_gradient_helpers
import pytest

import pennylane as qp
from pennylane.templates.subroutines.arithmetic.hamming_weight import (
    _apply_hamming_weight,
    _hamming_weight_plan,
)

# (n, ANDs, CNOTs) of the plan for n input bits
GATE_COUNTS = [(1, 0, 0), (2, 1, 1), (3, 1, 5), (4, 3, 7), (5, 3, 11), (6, 4, 16), (7, 4, 20)]


@pytest.mark.parametrize("n", range(1, 8))
def test_outputs_hold_hamming_weight(n):
    """Test that the output positions of the plan hold the Hamming weight of every basis input,
    least significant bit first."""
    plan = _hamming_weight_plan(n)
    _, out_positions, num_work = plan
    num_wires = n + num_work

    qfunc = partial(_apply_hamming_weight, plan)
    ops = phase_gradient_helpers.record(qfunc, list(range(num_wires)))
    x_bits = qp.math.int_to_binary(np.arange(2**n), n)
    inputs = np.hstack([x_bits, np.zeros((2**n, num_work), dtype=int)])
    outputs = phase_gradient_helpers.basis_outputs(ops, num_wires, inputs)
    weights = outputs[:, out_positions] @ 2 ** np.arange(len(out_positions))
    assert np.array_equal(weights, x_bits.sum(axis=1))


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("n", range(2, 8))
def test_change_op_basis_uncomputes(n):
    """Test that a target applied in a ``change_op_basis`` with the Hamming-weight circuit as
    compute block (default uncompute) sees the Hamming weight, and that the inputs and the work
    wires are restored, on every basis input. For n = 1, the compute block would be empty."""
    plan = _hamming_weight_plan(n)
    _, out_positions, num_work = plan
    num_out = len(out_positions)

    def qfunc(wires):
        def copy_out():
            for j, p in enumerate(out_positions):
                qp.CNOT([wires[p], wires[n + num_work + j]])

        qp.change_op_basis(partial(_apply_hamming_weight, plan, wires), copy_out)

    num_wires = n + num_work + num_out
    ops = phase_gradient_helpers.record(qfunc, list(range(num_wires)))
    x_bits = qp.math.int_to_binary(np.arange(2**n), n)
    inputs = np.hstack([x_bits, np.zeros((2**n, num_work + num_out), dtype=int)])
    outputs = phase_gradient_helpers.basis_outputs(ops, num_wires, inputs)
    weight_bits = qp.math.int_to_binary(x_bits.sum(axis=1), num_out)[:, ::-1]
    zeros = np.zeros((2**n, num_work), dtype=int)
    assert np.array_equal(outputs, np.hstack([x_bits, zeros, weight_bits]))


@pytest.mark.parametrize("n, num_ands, num_cnots", GATE_COUNTS)
def test_gate_counts(n, num_ands, num_cnots):
    """Test that the plan uses n - popcount(n) ANDs with one work wire each and
    5n - 4 * n.bit_length() - popcount(n) CNOTs, and no other gates."""
    plan = _hamming_weight_plan(n)
    _, out_positions, num_work = plan
    assert num_work == num_ands
    assert len(out_positions) == n.bit_length()

    ops = qp.tape.make_qscript(_apply_hamming_weight)(plan, range(n + num_work)).operations
    counts = Counter(type(op) for op in ops)
    assert set(counts) <= {qp.TemporaryAND, qp.CNOT}
    assert (counts[qp.TemporaryAND], counts[qp.CNOT]) == (num_ands, num_cnots)
    and_targets = [op.wires[2] for op in ops if isinstance(op, qp.TemporaryAND)]
    assert sorted(and_targets) == list(range(n, n + num_ands))
