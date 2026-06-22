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

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.trotter_vibronic import (
    diagonalize_vibronic,
    fragment_to_dense,
    get_momentum_coefficients,
    get_position_coefficients,
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
        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)
        with qp.queuing.AnnotatedQueue() as q:
            diagonalize_vibronic(elec_key, wires)

        if expected_support:
            c = wires[expected_support[0]]
            expected_ops = [qp.Hadamard(c)]
            expected_ops += [qp.CNOT([c, wires[idx]]) for idx in expected_support[1:]]
        else:
            expected_ops = []
        assert q.queue == expected_ops, f"{q.queue}\n{expected_ops}"

    @pytest.mark.parametrize(
        "n_states, col", [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (17, 9), (34, 19)]
    )
    @pytest.mark.parametrize("row", (0, 1, 2))
    def test_diagonalizes_correctly(self, n_states, col, row, seed):
        """Test that the diagonalization works as expected."""
        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)
        m = row ^ col
        matrix = np.zeros((n_states, n_states))
        for i, val in enumerate(rng.random(n_states)):
            if i < i ^ m < n_states:
                matrix[i, i ^ m] = matrix[i ^ m, i] = val
        key = (row, col)
        diag_mat = qp.matrix(diagonalize_vibronic, wires)(key, wires)
        # Just make sure it is an orthogonal matrix
        assert np.all(np.isreal(diag_mat)) and np.allclose(
            diag_mat @ diag_mat.T, np.eye(2**n_wires)
        )
        diag_mat = diag_mat[:n_states, :n_states]

        diagonalized = diag_mat.T @ matrix @ diag_mat
        # Test that the diagonalization worked
        assert np.allclose(np.diag(np.diag(diagonalized)), diagonalized)


def _random_vibronic_elec_ids(n_states, rng):
    m = rng.integers(0, n_states)
    keys = [(i, int(i ^ m)) for i in range(n_states) if i ^ m < n_states]
    return keys


def random_vibronic_fragment(n_states, n_modes, include_op_types=None, seed=None):
    """Construct a random vibronic fragment.

    Args:
        n_states (int): Number of electronic states
        n_modes (int): Number of modes
        include_op_types (list[tuple[str]]): List of operator types to include
        seed (int): Randomness seed used for numerical data and basis choice of the fragment

    Returns:
        RealspaceMatrix: The random vibronic fragment

    """
    if seed is None:
        seed = np.random.randint(251126)
    rng = np.random.default_rng(seed)

    if include_op_types is None:
        include_op_types = [(), ("Q",), ("Q", "Q")]

    # Can't mix kinetic with potential terms
    kin_op = ("P", "P")
    if kin_op in include_op_types:
        assert include_op_types == [kin_op]
        tensor = np.diag(rng.random(n_modes))
        op = RealspaceOperator(n_modes, kin_op, RealspaceCoeffs(tensor, "label"))
        blocks = {(i, i): RealspaceSum(n_modes, [op]) for i in range(n_states)}
        return RealspaceMatrix(n_states, n_modes, blocks)

    elec_ids = _random_vibronic_elec_ids(n_states, rng)
    blocks = {}
    # Iterate over pairs of electronic states
    for elec_idx in elec_ids:
        ops = []
        # Iterate over operator types such as (), ("Q",) and ("Q", "Q")
        for op_type in include_op_types:
            degree = len(op_type)
            tensor = rng.random((n_modes,) * degree)
            if degree == 2:
                # If the term is of degree two, our convention is that only the upper triangle
                # of the coefficient matrix (incl the diagonal) is populated.
                tensor[np.tril_indices(n_modes, k=-1)] = 0.0
            ops.append(RealspaceOperator(n_modes, op_type, RealspaceCoeffs(tensor, "label")))
        # The coefficients are always symmetric with respect to the electronic state indices,
        # because Hamiltonians are Hermitian, and vibronic Hamiltonians are real-valued
        blocks[elec_idx] = blocks[elec_idx[::-1]] = RealspaceSum(n_modes, ops)

    return RealspaceMatrix(n_states, n_modes, blocks)


class TestFragmentReadout:
    """Tests for helper functions that extract information from fragments."""

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    @pytest.mark.parametrize("op_type", [(), ("Q",), ("Q", "Q"), ("P", "P")])
    def test_fragment_to_dense_roundtrip(self, n_states, n_modes, op_type, seed):
        """Test that extracting coefficients with ``fragment_to_dense`` and converting them
        back to a nested dictionary structure yields the identity mapping."""
        fragment = random_vibronic_fragment(n_states, n_modes, [op_type], seed)
        print(fragment)
        dense_coeffs = fragment_to_dense(fragment, op_type)
        degree = len(op_type)
        assert isinstance(dense_coeffs, np.ndarray)
        assert dense_coeffs.shape == (n_states, n_states) + (n_modes,) * degree
        assert np.allclose(np.moveaxis(dense_coeffs, 1, 0), dense_coeffs)

        rng = np.random.default_rng(seed)
        if op_type == ("P", "P"):
            # Kinetic term must be diagonal w.r.t. electronic d.o.f.s
            expected_ids = [(i, i) for i in range(n_states)]
        else:
            expected_ids = _random_vibronic_elec_ids(n_states, rng)
        where = np.abs(dense_coeffs) > 1e-12
        for _ in range(degree):
            where = np.any(where, axis=-1)
        ids = list(zip(*np.where(where)))
        assert set(ids) == set(expected_ids)

        # Reconstruct the fragment from the dense matrix
        blocks = {}
        for idx in ids:
            idx = (int(idx[0]), int(idx[1]))
            op = RealspaceOperator(n_modes, op_type, RealspaceCoeffs(dense_coeffs[idx], "label"))
            blocks[idx] = blocks[idx[::-1]] = RealspaceSum(n_modes, [op])
        reconstructed_fragment = RealspaceMatrix(n_states, n_modes, blocks)

        # Compare to the original fragment
        assert reconstructed_fragment == fragment

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    def test_get_position_coefficients(self, n_states, n_modes, seed):
        """Test that ``get_position_coefficients`` returns the correct terms."""
        fragment = random_vibronic_fragment(n_states, n_modes, seed=seed)

        # Obtain diagonalization matrix
        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        diag_key = next(iter(k for k, v in fragment.get_coefficients().items() if v))
        M = qp.matrix(diagonalize_vibronic, wires)(diag_key, wires)[:n_states, :n_states]

        constant, linear, quadratic, bilinear = get_position_coefficients(fragment)

        # 0th order
        assert np.shape(constant) == (n_states,)
        exp_order_zero = fragment_to_dense(fragment, ())
        # Make sure the diagonalization and extraction worked by inverting the np.diag call
        assert np.allclose(M.T @ exp_order_zero @ M, np.diag(constant))

        # 1st order
        assert np.shape(linear) == (n_modes, n_states)
        exp_order_one = fragment_to_dense(fragment, ("Q",))
        # Make sure the diagonalization and extraction worked by inverting the np.diag call
        exp_order_one = np.einsum("ba,bcz,cd->zad", M, exp_order_one, M)
        reconstructed_order_one = [np.diag(sub_diag) for sub_diag in linear]
        assert np.allclose(reconstructed_order_one, exp_order_one)

        # 2nd order
        assert np.shape(quadratic) == (n_modes, n_states)
        assert np.shape(bilinear) == (n_modes, n_modes, n_states)
        exp_order_two = fragment_to_dense(fragment, ("Q", "Q"))
        # Make sure the diagonalization and extraction worked by inverting the np.diag call
        exp_order_two = np.einsum("ba,bcyz,cd->yzad", M, exp_order_two, M)

        reconstructed_order_two = np.array([np.diag(sub_diag) for sub_diag in quadratic.T]).T
        reconstructed_order_two[np.triu_indices(n_modes, k=1)] = bilinear[
            np.triu_indices(n_modes, k=1)
        ]
        reconstructed_order_two = np.array(
            [[np.diag(_sub) for _sub in sub] for sub in reconstructed_order_two]
        )

        assert np.allclose(reconstructed_order_two, exp_order_two)

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    def test_get_momentum_coefficients(self, n_states, n_modes, seed):
        """Test that ``get_momentum_coefficients`` returns the correct terms."""
        fragment = random_vibronic_fragment(
            n_states, n_modes, include_op_types=[("P", "P")], seed=seed
        )

        diag_key = next(iter(k for k, v in fragment.get_coefficients().items() if v))
        assert diag_key[0] == diag_key[1]  # Consistency check for fragment sampler

        quadratic = get_momentum_coefficients(fragment)
        assert np.shape(quadratic) == (n_modes,)

        exp_quadratic = fragment_to_dense(fragment, ("P", "P"))
        # Add redundant n_states axes and double n_modes axis
        reconstructed_quadratic = np.einsum("ab,cd->abcd", np.eye(n_states), np.diag(quadratic))
        assert np.allclose(reconstructed_quadratic, exp_quadratic)
