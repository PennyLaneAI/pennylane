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
"""Tests for the one-body qubitization walk operator."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.fermi import FermiWord, jordan_wigner
from pennylane.labs.templates import alias_sampling_wires, one_body_walk, one_body_walk_wires
from pennylane.labs.templates.alias_sampling import _build_alias_tables

# helper functions


def _discretized_weights(weights, mu_bits):
    r"""The probabilities coherent alias sampling actually prepares.

    ``one_body_walk``'s only approximation is that PREP loads a :math:`\mu`-bit
    approximation of :math:`\sqrt{|\mu_p| / \lambda}` rather than the exact value. The
    alias tables are deterministic, so that approximation is classically predictable
    and the walk can be checked to machine precision instead of to a loose
    ``L / 2**mu`` tolerance.
    """
    alt, keep = _build_alias_tables(weights, mu_bits)
    n_states, n_keep = len(alt), 2**mu_bits
    rho = np.zeros(n_states)
    for target in range(n_states):
        rho[target] += keep[target]
        for source in range(n_states):
            if alt[source] == target:
                rho[target] += n_keep - keep[source]
    return rho / (n_keep * n_states)


def _reference_block_matrix(op_matrix, system_wires, mu_bits=None):
    r"""Independent reference for the encoded block.

    Builds

    .. math::

        -\frac{1}{2} \sum_{p,\sigma} \rho_p\, \mathrm{sign}(\mu_p)\,
        \hat V^\dagger \hat Z_{p\sigma} \hat V

    The rotated Pauli comes from Jordan-Wigner mapping
    :math:`\sum_{qs} V_{qp} V_{sp} c_q^\dagger c_s` and using
    :math:`\hat Z = \hat 1 - 2 \hat n`.

    Args:
        op_matrix (array): the real symmetric one-body matrix.
        system_wires (list[int]): wires for representing the ``2 N`` system spin-orbitals
        mu_bits (int or None): alias sampling precision; if None, the exact weights are used.
    """
    norbs = qp.math.shape(op_matrix)[0]
    mu, vmat = np.linalg.eigh(op_matrix)
    weights = np.abs(mu)
    signs = np.where(weights > 0, np.sign(mu), 1.0)
    rho = weights / weights.sum() if mu_bits is None else _discretized_weights(weights, mu_bits)

    dim = 2 ** (2 * norbs)
    wire_map = {m: system_wires[m] for m in range(2 * norbs)}
    total = np.zeros((dim, dim), dtype=complex)
    for p in range(norbs):
        for sigma in (0, 1):
            fermi_op = 0
            for q in range(norbs):
                for s in range(norbs):
                    fermi_op += (
                        vmat[q, p]
                        * vmat[s, p]
                        * FermiWord({(0, sigma * norbs + q): "+", (1, sigma * norbs + s): "-"})
                    )
            number_op = qp.matrix(
                jordan_wigner(fermi_op, wire_map=wire_map), wire_order=system_wires
            )
            total += rho[p] * signs[p] * (np.eye(dim) - 2 * number_op)
    return -0.5 * total


def _apply_walk(op_matrix, mu_bits, state, n_powers=1, system_wires=None):
    r"""Helper function to apply the walk to one system state and return the :math:`|\vec 0\rangle` block."""
    norbs = qp.math.shape(op_matrix)[0]
    req = one_body_walk_wires(norbs, mu_bits)
    n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
    prep = list(range(n_prep))
    system = list(range(n_prep, n_prep + n_sys))
    work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))

    n_wires = n_prep + n_sys + n_work

    if system_wires is not None:
        system = system_wires
        n_sys = len(system)

    dev = qp.device("default.qubit", wires=n_wires)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(state, wires=system)
        for _ in range(n_powers):
            one_body_walk(op_matrix, mu_bits, prep, system, work)
        return qp.state()

    psi = np.asarray(circuit()).reshape(2**n_prep, 2**n_sys, 2**n_work)
    work_scratch = float(np.linalg.norm(psi[:, :, 1:]))
    return psi[0, :, 0], work_scratch


def _walk_block(op_matrix, mu_bits, n_powers=1):
    """The full encoded block, one column per system basis state."""
    norbs = qp.math.shape(op_matrix)[0]
    req = one_body_walk_wires(norbs, mu_bits)
    n_prep, n_sys = req["prep_wires"], req["system_wires"]
    system = list(range(n_prep, n_prep + n_sys))
    dim = 2 ** len(system)

    block = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        basis = np.zeros(dim)
        basis[column] = 1.0
        block[:, column], _ = _apply_walk(op_matrix, mu_bits, basis, n_powers=n_powers)

    return block


@pytest.mark.parametrize("norbs", [2, 3, 4, 16])
@pytest.mark.parametrize("mu_bits", [2, 4, 7])
def test_one_body_walk_wires(norbs, mu_bits):
    """Test that the wire counts returned by one_body_walk_wires match the alias sampling wires."""
    req = one_body_walk_wires(norbs, mu_bits)
    alias = alias_sampling_wires(norbs, mu_bits)

    assert req["prep_wires"] == alias["target_wires"] + 1 + alias["temp_wires"]
    assert req["system_wires"] == 2 * norbs
    assert req["work_wires"] == alias["work_wires"]


class TestOneBodyWalk:
    """Test that the walk block-encodes the intended operator.

    These run the real ``alias_sampling`` and compare against a discretization-aware
    reference, so they hold to machine precision rather than to the alias bound.
    """

    @pytest.mark.parametrize("norbs, mu_bits", [(2, 1), (2, 2), (2, 3), (3, 2), (4, 2)])
    def test_encodes_operator_on_random_state(self, norbs, mu_bits):
        """Test that the block reproduces the discretized operator on a random state."""

        rng = np.random.default_rng(10 * norbs + mu_bits)
        a = rng.standard_normal((norbs, norbs))
        op_matrix = (a + a.T) / 2

        req = one_body_walk_wires(norbs, mu_bits)
        n_prep, n_sys = req["prep_wires"], req["system_wires"]
        system = list(range(n_prep, n_prep + n_sys))

        dim = 2 ** (2 * norbs)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        block_encoded_state, work_scratch = _apply_walk(op_matrix, mu_bits, state)

        expected = _reference_block_matrix(op_matrix, system, mu_bits) @ state
        assert np.allclose(block_encoded_state, expected, atol=1e-8)
        assert work_scratch < 1e-10

        # the discretized block also stays within the alias bound of the exact operator
        exact = _reference_block_matrix(op_matrix, system) @ state
        assert np.allclose(block_encoded_state, exact, atol=np.linalg.norm(state) / 2**mu_bits)

    def test_encodes_operator_full_block(self):
        """Test that the full block matches column by column, not just on one vector."""

        rng = np.random.default_rng(1000)
        a = rng.standard_normal((2, 2))
        op_matrix = (a + a.T) / 2
        req = one_body_walk_wires(2, 2)
        n_prep, n_sys = req["prep_wires"], req["system_wires"]
        system = list(range(n_prep, n_prep + n_sys))

        dim = 2**n_sys

        block = np.zeros((dim, dim), dtype=complex)
        for column in range(dim):
            basis = np.zeros(dim)
            basis[column] = 1.0
            block[:, column], _ = _apply_walk(op_matrix, 2, basis)

        assert np.allclose(block, _reference_block_matrix(op_matrix, system, 2), atol=1e-8)

    @pytest.mark.parametrize("norbs", [2, 3])
    def test_negative_definite_spectrum(self, norbs):
        """Test a negative spectrum: signs phase the index register, magnitude in PREP."""
        rng = np.random.default_rng(1000 * norbs)
        a = rng.standard_normal((norbs, norbs))
        op_matrix = (a + a.T) / 2
        op_matrix -= (np.linalg.eigvalsh(op_matrix).max() + 0.5) * np.eye(norbs)
        assert np.all(np.linalg.eigvalsh(op_matrix) < 0)

        req = one_body_walk_wires(norbs, 2)
        n_prep, n_sys = req["prep_wires"], req["system_wires"]
        system = list(range(n_prep, n_prep + n_sys))
        dim = 2 ** (2 * norbs)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        got, _ = _apply_walk(op_matrix, 2, state)

        assert np.allclose(got, _reference_block_matrix(op_matrix, system, 2) @ state, atol=1e-8)

    def test_singular_matrix(self):
        """Test that a zero eigenvalue is encoded correctly: zero weight and no sign phase."""
        op_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

        req = one_body_walk_wires(2, 2)
        n_prep, n_sys = req["prep_wires"], req["system_wires"]
        system = list(range(n_prep, n_prep + n_sys))
        dim = 2 ** (2 * 2)
        rng = np.random.default_rng(1000)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        got, _ = _apply_walk(op_matrix, 2, state)

        assert np.allclose(got, _reference_block_matrix(op_matrix, system, 2) @ state, atol=1e-8)

    @pytest.mark.parametrize("mu_bits", [1, 2, 3])
    def test_within_precision_bound(self, mu_bits):
        """Test that the gap to the ideal operator stays inside the O(L / 2**mu) alias bound."""
        rng = np.random.default_rng(1000 * mu_bits)
        a = rng.standard_normal((2, 2))
        op_matrix = (a + a.T) / 2

        req = one_body_walk_wires(2, mu_bits)
        n_prep, n_sys = req["prep_wires"], req["system_wires"]
        system = list(range(n_prep, n_prep + n_sys))

        block = _walk_block(op_matrix, mu_bits)

        error = np.abs(block - _reference_block_matrix(op_matrix, system_wires=system)).max()
        assert error <= 1 / 2**mu_bits

    @pytest.mark.parametrize("n_powers", [2, 3])
    def test_chebyshev_recursion(self, n_powers):
        r"""Test that applying the walk n times block-encodes the :math:`n^{th}` Chebyshev polynomial of the operator it encodes."""
        rng = np.random.default_rng(n_powers)
        a = rng.standard_normal((2, 2))
        op_matrix = (a + a.T) / 2
        block_1 = _walk_block(op_matrix, 2, n_powers=1)
        block_n = _walk_block(op_matrix, 2, n_powers=n_powers)

        eigvals, eigvecs = np.linalg.eigh(block_1)
        chebyshev = np.polynomial.chebyshev.Chebyshev.basis(n_powers)
        expected = eigvecs @ np.diag(chebyshev(eigvals)) @ eigvecs.conj().T

        assert np.allclose(block_n, expected, atol=1e-8)

    def test_non_square_raises(self):
        """Test that a non-square op_matrix is rejected."""
        req = one_body_walk_wires(2, 2)
        n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
        prep = list(range(n_prep))
        system = list(range(n_prep, n_prep + n_sys))
        work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))
        with pytest.raises(ValueError, match="must be square"):
            one_body_walk(np.zeros((2, 3)), 2, prep, system, work)

    def test_complex_raises(self):
        """Test that a complex op_matrix is rejected."""
        req = one_body_walk_wires(2, 2)
        n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
        prep = list(range(n_prep))
        system = list(range(n_prep, n_prep + n_sys))
        work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))
        with pytest.raises(ValueError, match="must be real"):
            one_body_walk(np.eye(2, dtype=complex) * 1j, 2, prep, system, work)

    def test_non_symmetric_raises(self):
        """Test that a non-symmetric op_matrix is rejected."""
        req = one_body_walk_wires(2, 2)
        n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
        prep = list(range(n_prep))
        system = list(range(n_prep, n_prep + n_sys))
        work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))
        with pytest.raises(ValueError, match="must be symmetric"):
            one_body_walk(np.array([[1.0, 2.0], [0.0, 1.0]]), 2, prep, system, work)

    @pytest.mark.parametrize("register", ["prep_wires", "system_wires", "work_wires"])
    def test_wrong_register_size_raises(self, register):
        """Test that each register must have the size reported by one_body_walk_wires."""
        req = one_body_walk_wires(2, 2)
        n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
        prep = list(range(n_prep))
        system = list(range(n_prep, n_prep + n_sys))
        work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))
        registers = {"prep_wires": prep, "system_wires": system, "work_wires": work}
        registers[register] = registers[register][:-1]
        with pytest.raises(ValueError, match=f"{register} must have"):
            one_body_walk(
                np.eye(2),
                2,
                registers["prep_wires"],
                registers["system_wires"],
                registers["work_wires"],
            )

    def test_zero_matrix_raises(self):
        """Test that an all-zero op_matrix has lambda = 0 and cannot be normalized."""
        req = one_body_walk_wires(2, 2)
        n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
        prep = list(range(n_prep))
        system = list(range(n_prep, n_prep + n_sys))
        work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))
        with pytest.raises(ValueError, match="positive value"):
            one_body_walk(np.zeros((2, 2)), 2, prep, system, work)
