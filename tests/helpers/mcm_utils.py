# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pytest helper functions are defined in this module.
"""

from collections.abc import Iterable, Sequence
from functools import reduce

import numpy as np

import pennylane as qp
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.ops.op_math.condition import Conditional
from pennylane.pauli import PauliWord


def validate_counts(shots, results1, results2, batch_size=None):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_counts(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2, batch_size=None)
        return

    for key1, val1 in results1.items():
        val2 = results2[key1]
        if abs(val1 + val2) > 100:
            assert np.allclose(val1, val2, atol=20, rtol=0.2)


def validate_samples(shots, results1, results2, batch_size=None):
    """Compares two samples.

    If the results are ``Sequence``s, loop over entries.

    Fails if the results do not have the same shape, within ``20`` entries plus 20 percent.
    This is to handle cases when post-selection yields variable shapes.
    Otherwise, fails if the sums of samples differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_samples(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_samples(shots, r1, r2, batch_size=None)
        return

    sh1, sh2 = results1.shape[0], results2.shape[0]
    assert np.allclose(sh1, sh2, atol=20, rtol=0.2)
    assert results1.ndim == results2.ndim
    if results2.ndim > 1:
        assert results1.shape[1] == results2.shape[1]
    np.allclose(qp.math.sum(results1), qp.math.sum(results2), atol=20, rtol=0.2)


def validate_expval(shots, results1, results2, batch_size=None):
    """Compares two expval, probs or var.

    If the results are ``Sequence``s, validate the average of items.

    If ``shots is None``, validate using ``np.allclose``'s default parameters.
    Otherwise, fails if the results do not match within ``0.01`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        results1 = reduce(lambda x, y: x + y, results1) / len(results1)
        results2 = reduce(lambda x, y: x + y, results2) / len(results2)
        validate_expval(sum(shots), results1, results2, batch_size=batch_size)
        return

    if shots is None:
        assert np.allclose(results1, results2)
        return

    if batch_size is not None:
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_expval(shots, r1, r2, batch_size=None)

    assert np.allclose(results1, results2, atol=0.01, rtol=0.2)


def validate_measurements(func, shots, results1, results2, batch_size=None):
    """Calls the correct validation function based on measurement type."""
    if func is qp.counts:
        validate_counts(shots, results1, results2, batch_size=batch_size)
        return

    if func is qp.sample:
        validate_samples(shots, results1, results2, batch_size=batch_size)
        return

    validate_expval(shots, results1, results2, batch_size=batch_size)


def _ppm_circuit_branches(circuit, state, wire_order, idx=0, outcomes=None):
    """Yield normalized output states for each PPM branch of a circuit.

    After each Pauli measurement the state is renormalized before continuing on
    that branch, so every yielded state has unit norm.
    """
    outcomes = outcomes or {}
    if idx == len(circuit):
        yield state
        return

    operation = circuit[idx]
    if isinstance(operation, PauliMeasure):
        # Apply PPM: branch into two branches and recurse the simulation
        pauli = PauliWord(dict(zip(operation.wires, operation.pauli_word, strict=True)))
        observable = pauli.to_mat(wire_order=wire_order)
        identity = np.eye(observable.shape[0], dtype=complex)
        for outcome in (0, 1):
            projector = (identity + (1 - 2 * outcome) * observable) / 2
            projected = projector @ state
            norm = np.linalg.norm(projected)
            if norm == 0:
                continue
            projected /= norm
            branch_outcomes = {**outcomes, operation.meas_uid: outcome}
            yield from _ppm_circuit_branches(
                circuit, projected, wire_order, idx + 1, branch_outcomes
            )
        return

    if isinstance(operation, Conditional):
        # Apply conditional operation. The measurement values are stored in ``outcomes``.
        branch = tuple(outcomes[m.meas_uid] for m in operation.meas_val.measurements)
        if operation.meas_val.processing_fn(*branch):
            state = qp.matrix(operation.base, wire_order=wire_order) @ state
    else:
        # Apply standard unitary gate
        state = qp.matrix(operation, wire_order=wire_order) @ state

    yield from _ppm_circuit_branches(circuit, state, wire_order, idx + 1, outcomes)


def assert_ppm_decomposition(circuit, init_state, wire_order, expected_state, *, atol=1e-10):
    """Assert that every PPM branch matches ``expected_state``, including a global phase comparison.
    This verifies that the PPM decomposition is unitary.
    """
    branches = list(_ppm_circuit_branches(circuit, init_state, wire_order))
    assert branches, "No PPM branches had non-zero amplitude."

    for branch_state in branches:
        assert np.isclose(np.linalg.norm(branch_state), 1.0, atol=atol)
        assert np.allclose(branch_state, expected_state, atol=atol)
