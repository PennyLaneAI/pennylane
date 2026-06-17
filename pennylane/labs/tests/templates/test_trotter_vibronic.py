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
"""Contains tests for the Trotter template for vibronic Hamiltonians."""

from itertools import combinations

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.trotter_vibronic import (
    diagonalize_vibronic,
    float_to_binary,
    get_coefficients,
)
from pennylane.labs.trotter_error.realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
)


class TestDiagonalizeVibronic:
    """Tests for ``diagonalize_vibronic``."""

    @pytest.mark.parametrize(
        "n_states, elec_key, expected_support",
        [
            (2, (0, 0), []),
            (2, (0, 1), [0]),
            (3, (0, 0), []),
            (3, (0, 1), [1]),
            (3, (0, 2), [0]),
            (4, (0, 0), []),
            (4, (0, 1), [1]),
            (4, (0, 2), [0]),
            (4, (0, 3), [1, 0]),
            (5, (0, 4), [0]),
            (6, (0, 1), [2]),
            (7, (0, 2), [1]),
            (8, (0, 3), [2, 1]),
            (8, (0, 7), [2, 1, 0]),
            (128, (0, 62), [5, 4, 3, 2, 1]),
            (128, (0, 127), [6, 5, 4, 3, 2, 1, 0]),
        ],
    )
    def test_expected_circuit(self, n_states, elec_key, expected_support, seed):
        """Test that the diagonalization circuit looks as expected for some small-scale examples."""
        n_modes = 3
        n_wires = qp.math.ceil_log2(n_states)
        constant = RealspaceOperator(n_modes, (), RealspaceCoeffs(np.array(1)))
        fragment = RealspaceMatrix(n_states, n_modes, {elec_key: constant})
        wires = list(range(n_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)
        with qp.queuing.AnnotatedQueue() as q:
            return_value = diagonalize_vibronic(fragment, wires)

        if expected_support:
            c = wires[expected_support[0]]
            expected_ops = [qp.Hadamard(c)]
            expected_ops += [qp.CNOT([c, wires[idx]]) for idx in expected_support[1:]]
        else:
            expected_ops = []
        assert q.queue == expected_ops, f"{q.queue}\n{expected_ops}"

    def test_diagonalizes_correctly(self):
        """Test that the diagonalization works as expected."""


class TestFloatToBinary:
    """Tests for the helper function ``float_to_binary``."""

    def test_hard_coded(self):
        """Test that some hard-coded floats are mapped to their hard-coded expected
        bit string representations."""

    def test_correct_phase_imposed(self):
        """Test that the correct complex phase is imposed when using
        phase gradient logic and conventions."""


def _random_vibronic_elec_ids(n_states, rng):
    num_wires = qp.math.ceil_log2(n_states)
    options = [[]]
    for i in range(num_wires):
        subsets = sum((list(combinations(range(i), r=r)) for r in range(i)), start=[])
        options.extend([[qp.H(i)] + [qp.CNOT([i, w]) for w in targets] for targets in subsets])
    option = options[rng.choice(len(options))]
    if option == []:
        return [(i, i) for i in range(n_states)]
    M = qp.matrix(option, wire_order=range(num_wires))[:n_states, :n_states]
    transformed = M @ np.diag(np.arange(n_states)) @ M.conj().T
    transformed[np.tril_indices(n_states)] = 0.0
    return zip(*np.where(transformed))


def random_vibronic_fragment(n_states, n_modes, include_op_types=None, seed=None):
    if include_op_types is None:
        include_op_types = [(), ("Q",), ("Q", "Q")]
    if seed is None:
        seed = np.random.randint(251126)

    rng = np.random.default_rng(seed)
    elec_ids = _random_vibronic_elec_ids(n_states, rng)
    blocks = {}
    for elec_idx in elec_ids:
        ops = []
        for op_type in include_op_types:
            tensor = np.random.random((n_modes,) * len(op_type))
            ops.append(RealspaceOperator(n_modes, op_type, RealspaceCoeffs(tensor, "label")))
        blocks[elec_idx] = blocks[elec_idx[::-1]] = RealspaceSum(n_modes, ops)

    return RealspaceMatrix(n_states, n_modes, blocks)


class TestGetCoefficients:
    """Tests for the helper function ``get_coefficients``."""

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    def test_compare_to_dense_rep(self, n_states, n_modes, seed):
        """Compare ``get_coefficients`` to a dense matrix readout in ``fragment_to_dense`` above.
        This is essentially a self-consistency test."""
        num_wires = qp.math.ceil_log2(n_states)
        wires = list(range(num_wires))
        include_op_types = [(), ("Q",), ("Q", "Q")]
        fragment = random_vibronic_fragment(n_states, n_modes, include_op_types, seed)
        constant, linear, quadratic, bilinear = get_coefficients(fragment, "position")

        # Get basis change matrix
        M = qp.matrix(diagonalize_vibronic, wires)(fragment, wires)[:n_states, :n_states]
        print(M)
        print(constant)
        print(np.diag(constant))
        mapped_constant = M @ np.diag(constant) @ M.conj().T
        print(np.round(mapped_constant, 3))

        # Test constant part
        assert np.shape(constant) == (n_states,)
        constant_dense = fragment_to_dense(fragment, ())
        print(np.round(constant_dense, 3))
        print(np.round(mapped_constant, 3))
        assert np.allclose(constant_dense, mapped_constant)

    def test_roundtrip(self):  # Might split between position/momentum, or even between orders
        """Test that extracting coefficients with ``get_coefficients`` and converting them
        back to a nested dictionary structure yields the identity mapping."""

    def test_higher_order_position_terms_error(self):
        """Test that ``get_coefficients`` raises an error if position terms with order higher
        than two are present in the input fragment."""

    def test_wrong_momentum_terms_error(self):
        """Test that ``get_coefficients`` raises an error if non-quadratic momentum terms
        are present in the input fragment."""

    def test_momentum_electronic_interaction_error(self):
        """Test that ``get_coefficients`` raises an error if the momentum terms are not encoded
        redundantly and on the diagonal w.r.t. electronic degrees of freedom, which would lead to
        an effective electronic-momentum interaction."""
