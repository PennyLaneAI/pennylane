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
"""Tests for the tensor hypercontraction SELECT oracle."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import assert_valid
from pennylane.templates.subroutines import SelectTHC, select_thc_wires
from pennylane.templates.subroutines.select_thc import (
    _angle_batches,
    _apply_loaded_rotation,
    _build_qrom_givens_data,
    _cascade_angles,
)
from pennylane.typing import AbstractWires


def _static_matrix(matrix):
    """Return a matrix as hashable compilable data."""
    return tuple(tuple(float(entry) for entry in row) for row in np.asarray(matrix))


def _reference_V(leaf, N, sector):
    r"""Reference :math:`V = I - 2 c^\dagger c`, built from fermionic operators.

    This is independent of the circuit, and uses Jordan-Wigner on :math:`c = \sum_p
    \mathrm{leaf}_p a_{p,\sigma}`, with no Givens rotations, ``QROM`` or discretization.
    """
    leaf = np.asarray(leaf, dtype=float)
    leaf = leaf / np.linalg.norm(leaf)
    off = 0 if sector == 0 else N // 2
    cdag = sum(float(leaf[p]) * qp.FermiC(off + p) for p in range(len(leaf)))
    ann = sum(float(leaf[p]) * qp.FermiA(off + p) for p in range(len(leaf)))
    n_op = qp.jordan_wigner(cdag * ann, wire_map={w: w for w in range(N)})
    return np.eye(2**N) - 2.0 * qp.matrix(n_op, wire_order=list(range(N)))


def _layout(M, N, beth, extra_work=0, num_batches=1):
    """Wire layout: system | mu | nu | flags | phase gradient | work."""
    sizes = select_thc_wires(M, N, beth, num_batches)
    n = sizes["index_wires"] // 2
    system = list(range(N))
    index = list(range(N, N + 2 * n))
    flags = list(range(N + 2 * n, N + 2 * n + 5))
    gradient = list(range(flags[-1] + 1, flags[-1] + 2 + beth))
    n_work = sizes["work_wires"] + extra_work
    work = list(range(gradient[-1] + 1, gradient[-1] + 1 + n_work))
    return system, index, flags, gradient, work, gradient[-1] + 1 + n_work


def _prep_gradient(wires):
    """Prepare the shared phase gradient state, which is a product state."""
    beth = len(wires)
    for j, wire in enumerate(wires):
        qp.Hadamard(wire)
        qp.PhaseShift(-2 * np.pi * 2 ** (beth - 1 - j) / 2**beth, wires=wire)


class TestCascadeAngles:
    """Tests for the classical Givens cascade."""

    @pytest.mark.parametrize("n_half", [2, 3, 4, 7])
    def test_row_zero_is_the_leaf(self, n_half):
        """Test that the cascade builds a U whose row 0 is the normalized leaf, so that
        U^dagger Z_1 U is the reflection about that leaf."""
        leaf = np.random.default_rng(n_half).standard_normal(n_half)
        thetas = _cascade_angles(leaf)

        unitary = np.eye(n_half)
        for p in reversed(range(n_half - 1)):
            givens = np.eye(n_half)
            cos, sin = np.cos(thetas[p]), np.sin(thetas[p])
            givens[p, p], givens[p, p + 1] = cos, sin
            givens[p + 1, p], givens[p + 1, p + 1] = -sin, cos
            unitary = givens @ unitary

        expected = leaf / np.linalg.norm(leaf)
        assert np.allclose(unitary[0, :], expected)
        assert np.allclose(unitary @ expected, np.eye(n_half)[0])
        assert np.allclose(unitary @ unitary.T, np.eye(n_half))

    def test_single_orbital(self):
        """Test that N/2 = 1 needs no rotations at all."""
        assert _cascade_angles([1.0]).size == 0

    def test_zero_vector_raises(self):
        """Test that a zero vector raises a ValueError."""
        with pytest.raises(ValueError, match="zero vector"):
            _cascade_angles([0.0, 0.0])


class TestQROMTable:
    """Tests for the loaded rotation table: the angle quantization, the one-body block and
    the incremental loading of each batch as a difference from the previous one."""

    @pytest.mark.parametrize("M, n_half", [(3, 4), (2, 3), (4, 2), (8, 4)])
    @pytest.mark.parametrize("one_body", [False, True])
    def test_shape_and_padding(self, M, n_half, one_body):
        """Test that the QROM table has the correct shape and padding."""
        beth = 5
        rng = np.random.default_rng(0)
        rows = _build_qrom_givens_data(
            rng.standard_normal((M, n_half)),
            np.eye(n_half),
            beth,
            one_body,
            _angle_batches(n_half, 1)[0],
        )[0]

        block = 1 << qp.math.ceil_log2(max(M, n_half))
        assert len(rows) == (block + n_half if one_body else M)
        assert all(len(r) == (n_half - 1) * beth for r in rows)
        assert all(set(r) <= {0, 1} for r in rows)

        if one_body:
            assert all(all(b == 0 for b in rows[a]) for a in range(M, block))

    @pytest.mark.parametrize("beth", [3, 6])
    def test_grid_angles_are_exact(self, beth):
        """Test that a leaf with cascade angle ``theta = 2 pi m / 2**beth`` packs to exactly ``m``.

        The table stores ``2 * theta`` quantized over ``[0, 4 pi)``, the period of the
        Givens rotation. This also covers that wrap: for ``m >= 2**(beth - 1)`` the
        cascade angle comes back negative out of ``arctan2`` and must still encode to
        ``m``, which it would not if the encoding used ``[0, 2 pi)``.
        """
        levels = 1 << beth
        theta = np.pi * (2.0 * np.arange(levels) + 1.0) / levels
        leaves = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        rows = _build_qrom_givens_data(leaves, np.eye(2), beth, False, [[0]])[0]
        for m in range(1 << beth):
            assert rows[m] == list(qp.math.int_to_binary(levels - 1 - m, beth))

    def test_one_body_block_placement(self):
        """Test that the one-body rotations sit at address edge=1, index=ell."""
        M, n_half, beth = 2, 3, 4
        pairs = _angle_batches(n_half, 1)[0]
        tev = np.linalg.qr(np.random.default_rng(1).standard_normal((n_half, n_half)))[0]
        rows = _build_qrom_givens_data(np.eye(M, n_half), tev, beth, True, pairs)[0]
        expected = _build_qrom_givens_data(tev.T, np.eye(n_half), beth, False, pairs)[0]
        block = 1 << qp.math.ceil_log2(M + 1)
        for ell in range(n_half):
            assert rows[block + ell] == expected[ell]

    def test_empty_for_one_orbital(self):
        """Test that the QROM data is empty for a single orbital."""
        assert _build_qrom_givens_data(np.ones((3, 1)), np.eye(1), 4, True, []) == []

    def test_qrom_is_an_xor_load(self):
        """Test that ``qp.QROM(clean=True)`` maps
        ``|i>|d>`` to ``|i>|d XOR b_i>`` for *any* target state, not just ``|0>``."""
        table = [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]]
        ctrl, targ, work = [0, 1], [2, 3, 4], [5, 6, 7]

        @qp.qnode(qp.device("default.qubit", wires=8))
        def circuit(index, data):
            qp.BasisState(index, wires=ctrl)
            qp.BasisState(data, wires=targ)
            qp.QROM(table, control_wires=ctrl, target_wires=targ, work_wires=work, clean=True)
            return qp.probs(wires=ctrl + targ + work)

        for i, d in ((0, 0), (1, 5), (3, 7)):
            index = list(qp.math.int_to_binary(i, 2))
            data = list(qp.math.int_to_binary(d, 3))
            expected_data = [a ^ b for a, b in zip(data, table[i])]
            # work wires must come back clean, so the outcome is a single basis state
            expected_state = int("".join(map(str, index + expected_data)) + "000", 2)
            assert np.isclose(np.asarray(circuit(index, data))[expected_state], 1.0, atol=1e-9)

    @pytest.mark.parametrize("n_half, num_batches", [(8, 3), (5, 2)])
    def test_running_xor_rebuilds_each_batch(self, n_half, num_batches):
        """Test that XOR-ing the increments up to batch b give exactly batch b's own table."""
        beth = 4
        batches, width = _angle_batches(n_half, num_batches)
        chi = np.random.default_rng(0).standard_normal((3, n_half))
        args = (chi, np.eye(n_half), beth, False)
        tables = [_build_qrom_givens_data(*args, [b])[0] for b in batches]
        loads = _build_qrom_givens_data(*args, batches)
        assert len(loads) == len(tables)

        register = [[0] * (width * beth) for _ in tables[0]]
        for b, table in enumerate(tables):
            register = [[x ^ y for x, y in zip(r, l)] for r, l in zip(register, loads[b])]
            # only the last batch can be short, and it is zero-padded to the register
            padded = [row + [0] * (width * beth - len(row)) for row in table]
            assert register == padded
        # walking the increments back down and re-loading the first clears the register
        for b in reversed(range(len(tables))):
            register = [[x ^ y for x, y in zip(r, l)] for r, l in zip(register, loads[b])]
        assert all(all(bit == 0 for bit in row) for row in register)


@pytest.mark.parametrize(
    "M, N, beth, expected",
    [
        (1, 2, 4, {"system": 2, "index": 2, "flag": 5, "gradient": 5, "work": 0}),
        (3, 4, 4, {"system": 4, "index": 4, "flag": 5, "gradient": 5, "work": 8}),
        (3, 6, 3, {"system": 6, "index": 4, "flag": 5, "gradient": 4, "work": 9}),
        (5, 8, 5, {"system": 8, "index": 6, "flag": 5, "gradient": 6, "work": 20}),
    ],
)
def test_select_thc_wires(M, N, beth, expected):
    """Test that select_thc_wires returns the correct wire mapping."""
    assert select_thc_wires(M, N, beth) == {k + "_wires": v for k, v in expected.items()}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"M": 0, "N": 4, "beth": 4}, "M must be a positive integer"),
        ({"M": 3, "N": 0, "beth": 4}, "N must be a positive integer"),
        ({"M": 3, "N": 4, "beth": 0}, "beth must be a positive integer"),
        ({"M": 3, "N": 4, "beth": True}, "beth must be a positive integer"),
        ({"M": 3, "N": 4, "beth": 4, "num_batches": 0}, "num_batches must be a positive integer"),
        ({"M": 1, "N": 8, "beth": 4}, "do not fit the index register"),
    ],
)
def test_select_thc_wires_raises(kwargs, match):
    """Test that select_thc_wires raises a ValueError for invalid input."""
    with pytest.raises(ValueError, match=match):
        select_thc_wires(**kwargs)


@pytest.mark.parametrize(
    "num_batches, expected_work", [(1, 72 + 8), (2, 40 + 8), (3, 24 + 8), (9, 8 + 8)]
)
def test_select_thc_wires_batched(num_batches, expected_work):
    """Test that batching shrinks only the angle part of the scratch."""
    sizes = select_thc_wires(M=20, N=20, beth=8, num_batches=num_batches)
    assert sizes["work_wires"] == expected_work
    assert {k: v for k, v in sizes.items() if k != "work_wires"} == {
        "system_wires": 20,
        "index_wires": 10,
        "flag_wires": 5,
        "gradient_wires": 9,
    }


@pytest.mark.parametrize(
    "n_index, n_flag, n_grad, match",
    [
        (3, 5, 3, "index_wires must have"),
        (4, 4, 3, "flag_wires must have"),
        (4, 5, 2, "gradient_wires must have"),
    ],
)
def test_select_thc_register_sizes(n_index, n_flag, n_grad, match):
    """Test that the registers are kept disjoint so a wrong size surfaces as a ValueError"""
    sizes = [4, n_index, n_flag, n_grad, 6]
    edges = np.cumsum([0] + sizes)
    regs = [range(int(lo), int(hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    with pytest.raises(ValueError, match=match):
        SelectTHC(_static_matrix(np.ones((3, 2))), _static_matrix(np.eye(2)), 3, *regs)


class TestAngleBatches:
    """Tests for the classical batch partition of the Givens rotations."""

    @pytest.mark.parametrize("n_half", [1, 2, 5, 10, 21])
    @pytest.mark.parametrize("num_batches", [1, 2, 3, 4, 7])
    def test_partition(self, n_half, num_batches):
        """Test that the batches concatenate back to the full set of rotations."""
        batches, width = _angle_batches(n_half, num_batches)
        assert [p for batch in batches for p in batch] == list(reversed(range(n_half - 1)))
        assert all(1 <= len(batch) <= width for batch in batches)
        assert len(batches) <= num_batches

    @pytest.mark.parametrize(
        "n_half, num_batches, expected_width", [(10, 1, 9), (10, 3, 3), (10, 4, 3), (21, 4, 5)]
    )
    def test_correct_width(self, n_half, num_batches, expected_width):
        """Test that the register width is ceil((N/2 - 1) / num_batches)."""
        assert _angle_batches(n_half, num_batches)[1] == expected_width


class TestSelectTHCOperator:
    """Checks the block that ``SelectTHC`` applies against a Jordan-Wigner reference.

    ``beth`` is kept as small as the grid allows: the phase gradient register is simulated
    too, so every extra bit of precision doubles the statevector. beth = 2 is the coarsest
    grid whose exact leaves are not all axis aligned.
    """

    M, N, beth = 2, 4, 2
    # leaves at the centers of the angle grid cells, theta = (2 m + 1) pi / 2**beth, which
    # the beth-bit floor quantization represents exactly, so the circuit carries no
    # discretization error. Integer multiples of the step 2 pi / 2**beth are the worst case.
    theta = np.pi * np.array([1, 5]) / (1 << beth)
    chi = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    tev = np.eye(2)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_assert_valid_and_decomposition(self):
        """Test Operator2 validity and decomposition with and without capture."""
        system, index, flags, gradient, work, _ = _layout(2, 4, 1)
        op = qp.SelectTHC(
            ((1.0, 1.0), (1.0, -1.0)),
            ((1.0, 0.0), (0.0, 1.0)),
            1,
            system,
            index,
            flags,
            gradient,
            work,
        )

        assert_valid(op, skip_differentiation=True)
        assert op.wires == qp.wires.Wires(system + index + flags + gradient + work)

    def test_abstract_wires_keep_static_data(self):
        """Test that abstract construction stores compilable data as given."""
        sizes = qp.select_thc_wires(1, 2, 1)
        chi = ((1.0,),)
        t_eigenvectors = ((1.0,),)
        op = qp.SelectTHC(
            chi,
            t_eigenvectors,
            1,
            AbstractWires(sizes["system_wires"]),
            AbstractWires(sizes["index_wires"]),
            AbstractWires(sizes["flag_wires"]),
            AbstractWires(sizes["gradient_wires"]),
            AbstractWires(sizes["work_wires"]),
        )

        assert op.chi == chi
        assert op.t_eigenvectors == t_eigenvectors
        assert qp.equal(op, op)

    @pytest.mark.parametrize(
        ("chi", "t_eigenvectors", "match"),
        [
            (np.ones((2, 2)), ((1.0, 0.0), (0.0, 1.0)), "chi must be a tuple of tuples"),
            (([1.0, 1.0], [1.0, -1.0]), ((1.0, 0.0), (0.0, 1.0)), "chi must be"),
            (((1.0, 1.0), (1.0, -1.0)), np.eye(2), "t_eigenvectors must be"),
            (((1.0, 1.0), (1.0, -1.0)), ([1.0, 0.0], [0.0, 1.0]), "t_eigenvectors must be"),
        ],
    )
    def test_static_data_must_be_nested_tuples(self, chi, t_eigenvectors, match):
        """Test that arrays and nested lists are rejected as compilable data."""
        system, index, flags, gradient, work, _ = _layout(2, 4, 1)
        with pytest.raises(ValueError, match=match):
            qp.SelectTHC(chi, t_eigenvectors, 1, system, index, flags, gradient, work)

    def _run(self, psi, prep, out, data=None):
        """Run ``SelectTHC`` on system state ``psi`` and slice out the system amplitudes."""
        chi, tev = data or (self.chi, self.tev)
        beth = self.beth
        system, index, flags, gradient, work, ntot = _layout(len(chi), self.N, beth)

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def circuit():
            qp.StatePrep(psi, wires=system)
            for wire, value in prep.items():
                if value:
                    qp.X(wire)
            _prep_gradient(gradient)
            SelectTHC(
                _static_matrix(chi),
                _static_matrix(tev),
                beth,
                system,
                index,
                flags,
                gradient,
                work,
            )
            qp.adjoint(_prep_gradient)(gradient)
            return qp.state()

        sel = [slice(None)] * ntot
        for wire, value in {**prep, **out, **{w: 0 for w in gradient + work}}.items():
            sel[wire] = value
        amps = np.asarray(np.asarray(circuit()).reshape([2] * ntot)[tuple(sel)]).reshape(-1)
        assert amps.size == 2 ** len(system)  # every ancilla was pinned
        return amps

    @pytest.mark.parametrize("swap, spin1, spin2", [(0, 0, 0), (1, 0, 1)])
    def test_branch_operator(self, swap, spin1, spin2):
        """Test that with the swap flag classical, the block is a bare product of two V's."""
        M, N, beth, chi = self.M, self.N, self.beth, self.chi
        _, index, flags, _, _, _ = _layout(M, N, beth)
        n = len(index) // 2
        psi = np.random.default_rng(0).standard_normal((2, 2**N)).T @ [1, 1j]
        psi /= np.linalg.norm(psi)

        prep = dict(zip(index[:n], qp.math.int_to_binary(0, n)))  # mu = 0
        prep.update(zip(index[n:], qp.math.int_to_binary(1, n)))  # nu = 1
        prep.update({flags[0]: 1, flags[1]: 0, flags[2]: swap, flags[3]: spin1, flags[4]: spin2})

        out = dict(zip(index[:n], qp.math.int_to_binary(1, n)))  # the closing index exchange
        out.update(zip(index[n:], qp.math.int_to_binary(0, n)))
        out[flags[2]] = 1 - swap  # the X on the symmetrization flag
        out[flags[3]], out[flags[4]] = spin2, spin1  # the spin exchange

        result = self._run(psi, prep, out)
        expected = _reference_V(chi[1], N, spin2) @ _reference_V(chi[0], N, spin1) @ psi
        assert np.allclose(result, expected, atol=1e-8)

    def test_symmetrized_block_on_a_prepare_state(self):
        """Test that when post-selected on PREPARE's symmetrized output the block is the Hermitian
        (V_mu V_nu + V_nu V_mu)/2, which is what the walk operator block-encodes.
        """
        M, N, beth, chi = self.M, self.N, self.beth, self.chi
        system, index, flags, gradient, work, ntot = _layout(M, N, beth)
        n = len(index) // 2
        branch = index + [flags[2]]

        prep_state = np.zeros(2 ** len(branch), dtype=complex)
        for mu, nu, sym in [(0, 1, 0), (1, 0, 1)]:
            bits = [*qp.math.int_to_binary(mu, n), *qp.math.int_to_binary(nu, n), sym]
            prep_state[int("".join(str(b) for b in bits), 2)] = 1 / np.sqrt(2)
        psi = np.random.default_rng(1).standard_normal((2, 2**N)).T @ [1, 1j]
        psi /= np.linalg.norm(psi)

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def circuit():
            qp.StatePrep(psi, wires=system)
            qp.StatePrep(prep_state, wires=branch)
            qp.X(flags[0])  # success
            for w in flags[3:]:  # the two spin flags
                qp.Hadamard(w)
            _prep_gradient(gradient)
            SelectTHC(
                _static_matrix(chi),
                _static_matrix(self.tev),
                beth,
                system,
                index,
                flags,
                gradient,
                work,
            )
            qp.adjoint(_prep_gradient)(gradient)
            for w in flags[3:]:
                qp.Hadamard(w)
            return qp.state()

        state = np.asarray(circuit()).reshape([2] * ntot)
        sel = [slice(None)] * ntot
        sel[flags[0]], sel[flags[1]], sel[flags[3]], sel[flags[4]] = 1, 0, 0, 0
        for w in gradient + work:
            sel[w] = 0
        block = np.asarray(state[tuple(sel)]).reshape(2**N, 2 ** len(branch))
        result = block @ prep_state.conj()
        expected = np.zeros(2**N, dtype=complex)
        for alpha in (0, 1):
            for beta in (0, 1):
                v_mu_a, v_nu_b = _reference_V(chi[0], N, alpha), _reference_V(chi[1], N, beta)
                v_nu_a, v_mu_b = _reference_V(chi[1], N, alpha), _reference_V(chi[0], N, beta)
                expected += 0.25 * 0.5 * (v_nu_b @ v_mu_a + v_mu_b @ v_nu_a) @ psi
        assert np.allclose(result, expected, atol=1e-8)

    def test_one_body_branch(self):
        """Test that when the sentinel flag is set the oracle applies a single one-body V from the
        t_eigenvectors table."""
        ell, spin = 0, 0
        M, N, beth = self.M, self.N, self.beth
        theta = np.pi * np.array([3, 1]) / (1 << beth)  # grid leaves, as columns
        tev = np.stack([np.cos(theta), np.sin(theta)])
        _, index, flags, _, _, _ = _layout(M, N, beth)
        n = len(index) // 2
        psi = np.random.default_rng(2).standard_normal((2, 2**N)).T @ [1, 1j]
        psi /= np.linalg.norm(psi)

        prep = dict(zip(index[:n], qp.math.int_to_binary(ell, n)))
        prep.update(zip(index[n:], qp.math.int_to_binary(0, n)))
        prep.update({flags[0]: 1, flags[1]: 1, flags[2]: 0, flags[3]: spin, flags[4]: 1 - spin})

        # the indices and the spins are NOT exchanged on edge=1
        result = self._run(psi, prep, {flags[2]: 1}, (self.chi, tev))
        assert np.allclose(result, _reference_V(tev[:, ell], N, spin) @ psi, atol=1e-8)

    def test_discretization_bound(self):
        """Test that for a generic chi the oracle stays unitary and the error stays within
        one grid step of the reference.
        """
        M, N, beth = 2, 4, 2
        chi = np.random.default_rng(4).standard_normal((M, 2))
        _, index, flags, _, _, _ = _layout(M, N, beth)
        n = len(index) // 2
        psi = np.zeros(2**N, dtype=complex)
        psi[0b0101] = 1.0  # a two-electron basis state

        prep = dict(zip(index[:n], qp.math.int_to_binary(0, n)))
        prep.update(zip(index[n:], qp.math.int_to_binary(1, n)))
        prep.update({flags[0]: 1, flags[1]: 0, flags[2]: 0, flags[3]: 0, flags[4]: 1})

        out = dict(zip(index[:n], qp.math.int_to_binary(1, n)))  # the closing exchange
        out.update(zip(index[n:], qp.math.int_to_binary(0, n)))
        out[flags[2]], out[flags[3]], out[flags[4]] = 1, 1, 0

        result = self._run(psi, prep, out, (chi, np.eye(2)))
        exact = _reference_V(chi[1], N, 1) @ _reference_V(chi[0], N, 0) @ psi
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-8)
        # coarse sanity bound: one full step of the beth-bit angle grid, since the floor
        # quantization can put a single angle a whole step off. The sharp check is the
        # unitarity assertion above; measured slack here is about 5x.
        assert np.abs(result - exact).max() <= 2 * np.pi / 2**self.beth


class TestPhaseGradientRotation:  # pylint: disable=too-few-public-methods
    """Checks the phase-gradient compilation of a single Givens rotation."""

    @staticmethod
    def _block(beth, k, adjoint):
        """The 4x4 system block produced by one loaded rotation, with every ancilla
        post-selected on |0>. Raises if any amplitude leaks out of that subspace."""
        ntot = 3 * beth + 3
        system, angle = [0, 1], list(range(2, 2 + beth))
        gradient = list(range(2 + beth, 3 + 2 * beth))
        adder_work = list(range(3 + 2 * beth, ntot))
        bits = qp.math.int_to_binary(k, beth)

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def circuit(column):
            qp.BasisState(bits, wires=angle)
            qp.BasisState(qp.math.int_to_binary(column, 2), wires=system)
            _prep_gradient(gradient)
            fn = qp.adjoint(_apply_loaded_rotation) if adjoint else _apply_loaded_rotation
            fn(system, angle, beth, [0], gradient, adder_work)
            qp.adjoint(_prep_gradient)(gradient)
            for bit, wire in zip(bits, angle):  # undo the angle load
                if bit:
                    qp.X(wire)
            return qp.state()

        columns = []
        for column in range(4):
            state = np.asarray(circuit(column)).reshape([4] + [2] * (ntot - 2))
            out = np.asarray(state[(slice(None),) + (0,) * (ntot - 2)]).reshape(-1)
            assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-8)
            columns.append(out)
        return np.stack(columns, axis=1)

    @pytest.mark.parametrize("adjoint", [False, True])
    def test_equals_single_excitation(self, adjoint):
        """Test that the two sandwiched additions of the loaded value k reproduce
        SingleExcitation at theta = -2 pi (2 k + 1) / 2**beth.

        The half-step offset and the reflection are the signature of the CNOT sandwich: it
        phases *both* branches, so the realised angle is their difference. The leftover
        common-mode phase is the same for every k, exp(i pi / 2**beth), because the two
        additions sit in opposite Clifford frames and their k-dependent parts cancel.
        """
        sign = -1.0 if adjoint else 1.0
        beth = 3
        for k in (0, 3, (1 << beth) - 1):
            theta = -sign * 2.0 * np.pi * (2 * k + 1) / (1 << beth)
            expected = np.exp(sign * 1j * np.pi / (1 << beth)) * qp.matrix(
                qp.SingleExcitation(theta, wires=[0, 1])
            )
            assert np.allclose(self._block(beth, k, adjoint), expected, atol=1e-8)


class TestSelectTHCInvariants:
    """Properties that hold for any chi, checkable without knowing the target operator."""

    @pytest.mark.parametrize("M, N, beth", [(2, 4, 2), (1, 2, 2)])
    def test_self_inverse(self, M, N, beth):
        """Test that Select is self-inverse."""
        n_half = N // 2
        theta = np.pi * (2.0 * np.arange(1, M + 1) + 1) / (1 << beth)  # distinct grid leaves
        leaves = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        chi = np.ones((M, 1)) if n_half == 1 else leaves
        system, index, flags, gradient, work, ntot = _layout(M, N, beth)
        base = ntot - len(work)
        psi = np.random.default_rng(5).standard_normal((2, 2**base)).T @ [1, 1j]
        psi /= np.linalg.norm(psi)

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def circuit():
            qp.StatePrep(psi, wires=range(base))
            for _ in range(2):
                SelectTHC(
                    _static_matrix(chi),
                    _static_matrix(np.eye(n_half)),
                    beth,
                    system,
                    index,
                    flags,
                    gradient,
                    work,
                )
            return qp.state()

        state = np.asarray(circuit()).reshape([2] * ntot)
        result = np.asarray(state[(slice(None),) * base + (0,) * len(work)]).reshape(-1)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-8)
        assert np.allclose(result, psi, atol=1e-8)

    @pytest.mark.parametrize("M, N, beth", [(2, 4, 2)])
    def test_work_wires_restored(self, M, N, beth):
        """Test that the clean scratch comes back to |0> for every input, so it can be reused."""
        chi = np.random.default_rng(M).standard_normal((M, N // 2))
        system, index, flags, gradient, work, ntot = _layout(M, N, beth)

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def circuit():
            for w in system + index + flags:
                qp.Hadamard(w)
            _prep_gradient(gradient)
            SelectTHC(
                _static_matrix(chi),
                _static_matrix(np.eye(N // 2)),
                beth,
                system,
                index,
                flags,
                gradient,
                work,
            )
            return qp.probs(wires=work)

        assert np.isclose(np.asarray(circuit())[0], 1.0, atol=1e-8)

    def test_extra_work_wires_do_not_change_the_unitary(self):
        """Test that extra work wires switch QROM to a SelectSwap decomposition."""
        extra = 1
        M, N, beth = 2, 4, 2
        chi = np.random.default_rng(6).standard_normal((M, N // 2))
        base = _layout(M, N, beth)[5]
        psi = np.random.default_rng(7).standard_normal((2, 2**base)).T @ [1, 1j]
        psi /= np.linalg.norm(psi)

        def run(e):
            system, index, flags, gradient, work, ntot = _layout(M, N, beth, extra_work=e)

            @qp.qnode(qp.device("default.qubit", wires=ntot))
            def circuit():
                qp.StatePrep(psi, wires=range(base))
                SelectTHC(
                    _static_matrix(chi),
                    _static_matrix(np.eye(N // 2)),
                    beth,
                    system,
                    index,
                    flags,
                    gradient,
                    work,
                )
                return qp.state()

            state = np.asarray(circuit()).reshape([2] * ntot)
            return np.asarray(state[(slice(None),) * base + (0,) * e]).reshape(-1)

        outs = [run(0), run(extra)]
        assert np.isclose(np.linalg.norm(outs[1]), 1.0, atol=1e-8)
        assert np.allclose(outs[0], outs[1], atol=1e-8)


class TestControlledSelectTHC:
    """A control on ``SelectTHC`` should reach only the two ``Z_1`` reflections."""

    @staticmethod
    def _case(beth):
        M, N = 3, 4
        rng = np.random.default_rng(0)
        chi = _static_matrix(rng.standard_normal((M, N // 2)))
        tev = _static_matrix(np.linalg.qr(rng.standard_normal((N // 2, N // 2)))[0])
        sizes = select_thc_wires(M, N, beth)
        wires = qp.registers(sizes)
        args = (
            chi,
            tev,
            beth,
            wires["system_wires"],
            wires["index_wires"],
            wires["flag_wires"],
            wires["gradient_wires"],
            wires["work_wires"],
        )
        return sum(sizes.values()), wires, args

    def test_control_reaches_only_the_reflection(self):
        """The QROM loads and the phase-gradient additions stay uncontrolled."""
        ntot, _, args = self._case(3)
        control = ntot

        with qp.queuing.AnnotatedQueue() as q:
            qp.ctrl(SelectTHC, control=control)(*args)
        tape = qp.tape.QuantumScript.from_queue(q)

        with qp.decomposition.toggle_graph_ctx(True):
            # one ChangeOpBasis per V sandwich, and the control lands on the outside of it
            tape = qp.transforms.decompose(tape, max_expansion=1)[0][0]
            names = [op.name for op in tape.operations]
            assert sum(name.startswith("C(ChangeOpBasis") for name in names) == 2

            tape = qp.transforms.decompose(tape, max_expansion=1)[0][0]
            tape = qp.transforms.decompose(tape, max_expansion=1)[0][0]

        loads = [op for op in tape.operations if "QROM" in op.name or "SemiAdder" in op.name]
        assert len(loads) == 12
        assert all(control not in op.wires for op in loads)

    def test_controlled_select_acts_as_the_control(self):
        """The two branches are the identity and ``SELECT``."""
        beth = 2
        ntot, wires, args = self._case(beth)
        control = ntot

        def prep():
            qp.X(wires["flag_wires"][0])
            for w in wires["flag_wires"][3:]:
                qp.Hadamard(w)
            grad = wires["gradient_wires"]
            for j, w in enumerate(grad):
                qp.Hadamard(w)
                qp.PhaseShift(-2 * np.pi * 2 ** (len(grad) - 1 - j) / 2 ** len(grad), wires=w)
            for w in wires["index_wires"]:
                qp.Hadamard(w)
            qp.Hadamard(wires["system_wires"][0])

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def off():
            prep()
            return qp.state()

        @qp.qnode(qp.device("default.qubit", wires=ntot))
        def on():
            prep()
            SelectTHC(*args)
            return qp.state()

        @qp.qnode(qp.device("default.qubit", wires=ntot + 1))
        def branches():
            qp.Hadamard(control)
            prep()
            qp.ctrl(SelectTHC, control=control)(*args)
            return qp.state()

        got = np.asarray(branches()).reshape(2**ntot, 2) * np.sqrt(2)
        assert np.allclose(got[:, 0], off(), atol=1e-8)
        assert np.allclose(got[:, 1], on(), atol=1e-8)


@pytest.mark.parametrize(
    "overlap, match",
    [
        ("index_wires", "system_wires and index_wires must not overlap"),
        ("flag_wires", "system_wires and flag_wires must not overlap"),
        ("gradient_wires", "system_wires and gradient_wires must not overlap"),
        ("work_wires", "system_wires and work_wires must not overlap"),
    ],
)
def test_select_thc_register_overlap(overlap, match):
    """Test that a register sharing a wire with another one is rejected by name."""
    names = ["system_wires", "index_wires", "flag_wires", "gradient_wires", "work_wires"]
    edges = np.cumsum([0, 4, 4, 5, 4, 6])
    regs = {
        name: list(range(int(lo), int(hi))) for name, lo, hi in zip(names, edges[:-1], edges[1:])
    }
    regs[overlap] = [0] + regs[overlap][1:]
    with pytest.raises(ValueError, match=match):
        SelectTHC(_static_matrix(np.ones((3, 2))), _static_matrix(np.eye(2)), 3, *regs.values())
