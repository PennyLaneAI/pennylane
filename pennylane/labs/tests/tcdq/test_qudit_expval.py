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
# pylint: disable=too-many-arguments,too-few-public-methods,unbalanced-tuple-unpacking
"""Reference and regression tests for the qudit IQP expectation estimator."""

import itertools
from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm

import pennylane as qp
from pennylane.labs.tcdq.qudit_expval_functions import (
    QuditCircuitConfig,
    _build_character_expansion,
    _character_amplitudes,
    _control_variate_expected_value,
    _control_variate_integrand,
    _dims_to_numpy,
    _parse_qudit_generator_dict,
    build_qudit_expval_func,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

NUM_SAMPLES = 10000


def _build_qudit_expval_func_exact(config):
    """Build a brute-force reference evaluator by summing over all basis states."""
    generators, param_map = _parse_qudit_generator_dict(config.gates, config.n_qudits)

    dims = _dims_to_numpy(config.d, config.n_qudits)  # (n_qudits,)
    dims_f = jnp.asarray(dims, dtype=jnp.float32)  # (n_qudits,)

    all_states = jnp.array(
        list(itertools.product(*(range(int(d_j)) for d_j in dims))),
        dtype=jnp.int32,
    )  # (prod(dims), n_qudits)

    l_vecs = jnp.array(config.observables[0], dtype=jnp.int32)
    m_vecs = jnp.array(config.observables[1], dtype=jnp.int32)

    d_n = int(np.prod(dims))
    n_obs = l_vecs.shape[0]
    d_col = dims_f[jnp.newaxis, :, jnp.newaxis]

    g_f = generators.astype(jnp.float32)
    s_f = all_states.astype(jnp.float32)
    outer = g_f[:, :, jnp.newaxis] * s_f.T[jnp.newaxis, :, :]
    per_qudit_vals = jnp.sqrt(2.0) * jnp.cos(2 * jnp.pi * outer / d_col + jnp.pi / 4)
    val_k = jnp.prod(per_qudit_vals, axis=1)  # (n_gates, prod(dims))

    def qudit_expval(gates_params):
        expanded_params = jnp.asarray(gates_params)[param_map]

        def single_obs(l, m):
            k_shifted = (all_states - l[jnp.newaxis, :]) % dims[jnp.newaxis, :]
            ks_f = k_shifted.astype(jnp.float32)
            m_f = m.astype(jnp.float32)
            l_f = l.astype(jnp.float32)
            obs_phase_scalar = jnp.exp(1j * jnp.pi * jnp.sum(m_f * l_f / dims_f))
            obs_state_phase = jnp.exp(1j * 2 * jnp.pi * (ks_f @ (m_f / dims_f)))
            obs_phase = obs_phase_scalar * obs_state_phase
            outer_shifted = g_f[:, :, jnp.newaxis] * ks_f.T[jnp.newaxis, :, :]
            val_k_shifted = jnp.prod(
                jnp.sqrt(2.0) * jnp.cos(2 * jnp.pi * outer_shifted / d_col + jnp.pi / 4),
                axis=1,
            )
            gate_phase_sum = jnp.sum(
                expanded_params[:, jnp.newaxis] * (val_k - val_k_shifted), axis=0
            )
            gate_phase = jnp.exp(1j * gate_phase_sum)
            return jnp.sum(obs_phase * gate_phase) / d_n

        expvals = jax.vmap(single_obs)(l_vecs, m_vecs)
        return expvals, jnp.zeros(n_obs), jnp.zeros(n_obs)

    return qudit_expval


def _shift_operator(d):
    """Return the single-qudit shift operator used by the dense reference code."""
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    return X


def _clock_operator(d):
    """Return the single-qudit clock operator used by the dense reference code."""
    return np.diag([np.exp(2j * np.pi * j / d) for j in range(d)])


def _displacement_operator(l, m, d):
    """Return one dense Heisenberg-Weyl displacement operator."""
    Z = _clock_operator(d)
    X = _shift_operator(d)
    Z_l = np.linalg.matrix_power(Z, int(l % d))
    X_m = np.linalg.matrix_power(X, int(m % d))
    phase = np.exp(-1j * np.pi * l * m / d)
    return phase * (Z_l @ X_m)


def _hermitian_observable(l, m, d):
    """Return the Hermitian observable used by the dense reference circuit."""
    chi = (1 + 1j) / 2
    O = _displacement_operator(l, m, d)
    return chi * O + np.conj(chi) * O.conj().T


def _dft_matrix(d):
    """Return the single-qudit discrete Fourier transform matrix."""
    j = np.arange(d)
    return np.exp(2j * np.pi * np.outer(j, j) / d) / np.sqrt(d)


def _kron_n(mats):
    """Return the Kronecker product of a sequence of matrices."""
    return reduce(np.kron, mats)


def qudit_expectation_brute_force(
    n, d, gates, thetas, l_vec, m_vec, init_state_elems=None, init_state_amps=None, phase_diag=None
):
    """Compute one exact expectation value with a dense-matrix reference path.

    This helper mirrors the mathematical definition of the circuit and is used
    only in tests where full enumeration is still feasible.

    Args:
        n: Number of qudits.
        d: Local dimension(s). Either a scalar broadcast to all qudits or a
            per-qudit sequence of length ``n`` for non-uniform dimensions.
        gates: Sequence of gate vectors.
        thetas: Sequence of gate angles.
        l_vec: Observable frequency indices.
        m_vec: Observable shift indices.
        init_state_elems: Optional sparse support of the input state.
        init_state_amps: Optional amplitudes for ``init_state_elems``.
        phase_diag: Optional phase layer for ``phase_fn``.

    Returns:
        complex: Exact expectation value of the requested observable.
    """
    dims = _dims_to_numpy(d, n)  # (n,)
    dims = [int(d_j) for d_j in dims]
    dim = int(np.prod(dims))

    # F^{otimes n}, each factor sized by its qudit's dimension.
    F_n = _kron_n([_dft_matrix(dims[i]) for i in range(n)])

    # D(theta) = prod_g exp(i theta_g Q_g)  [eqn 43]
    D = np.eye(dim, dtype=complex)
    for g, theta in zip(gates, thetas):
        Q_g = _kron_n([_hermitian_observable(g[i], 0, dims[i]) for i in range(n)])
        D = expm(1j * theta * Q_g) @ D

    if phase_diag is not None:
        D = np.diag(np.exp(1j * np.asarray(phase_diag, dtype=complex))) @ D

    # U(theta) = (F^{otimes n})^dag  D(theta)  F^{otimes n}  [eqn 44]
    U = F_n.conj().T @ D @ F_n

    # O(l, m) = bigotimes_i O(l_i, m_i)  [eqn 36, 46]
    O = _kron_n([_displacement_operator(l_vec[i], m_vec[i], dims[i]) for i in range(n)])

    if init_state_elems is None or init_state_amps is None:
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0
    else:
        psi0 = np.zeros(dim, dtype=complex)
        for elem, amp in zip(init_state_elems, init_state_amps):
            idx = 0
            for i, e in enumerate(elem):
                idx = idx * dims[i] + int(e)
            psi0[idx] += amp

    # <psi_in| U^dag O U |psi_in>
    U_psi = U @ psi0
    return U_psi.conj() @ O @ U_psi


def _pennylane_qubit_expval(generators_list, thetas_list, l_vec, m_vec):
    """Use ``default.qubit`` as a ``d=2`` reference implementation."""
    n = len(l_vec)

    def pauli_map(l, m, n):
        if (l, m) == (0, 1):
            return qp.X(n)

        if (l, m) == (1, 0):
            return qp.Z(n)

        if (l, m) == (1, 1):
            return qp.Y(n)

        return qp.I(n)

    obs_list = []
    for j in range(n):
        l, m = int(l_vec[j]), int(m_vec[j])
        obs_list.append(pauli_map(l, m, j))

    obs = qp.prod(*obs_list) if len(obs_list) > 1 else obs_list[0]
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def circuit():
        for i in range(n):
            qp.Hadamard(i)

        for theta, gen in zip(thetas_list, generators_list):
            active = [i for i, g in enumerate(gen) if g == 1]
            if active:
                qp.MultiRZ(2 * -theta, wires=active)

        for i in range(n):
            qp.Hadamard(i)

        return qp.expval(obs)

    return float(circuit())


def _make_config_one_param_per_gate(
    d, n, generators_array, thetas, l_vecs, m_vecs, n_samples=NUM_SAMPLES, key=None
):
    """Build a QuditCircuitConfig with one unique parameter per gate."""
    if key is None:
        key = jax.random.PRNGKey(0)
    gates = {i: [list(gen)] for i, gen in enumerate(generators_array)}
    return QuditCircuitConfig(
        d=d,
        n_qudits=n,
        gates=gates,
        observables=(np.array(l_vecs), np.array(m_vecs)),
        n_samples=n_samples,
        key=key,
    ), np.array(thetas)


class TestQuditExpvalVsPennyLane:
    """For d=2 the qudit framework must match the qubit PennyLane simulation."""

    @pytest.mark.parametrize(
        "n, generators, thetas, l_vecs, m_vecs",
        [
            # Single qubit, X observable
            (1, [[1]], [0.37], [[1]], [[0]]),
            # Single qubit, Z observable
            (1, [[1]], [0.7], [[0]], [[1]]),
            # Single qubit, Y observable (D(1,1) = Y)
            (1, [[1]], [0.5], [[1]], [[1]]),
            # Single qubit, identity observable
            (1, [[1]], [0.9], [[0]], [[0]]),
            # Two qubits, X0 X1
            (2, [[1, 0], [1, 1]], [0.5, 0.2], [[1, 1]], [[0, 0]]),
            # Two qubits, X0 Z1
            (2, [[1, 0], [0, 1]], [0.3, 0.6], [[1, 0]], [[0, 1]]),
            # Two qubits, Y0 Y1 (sign = +1 since (-1)^2 = +1)
            (2, [[1, 1]], [0.4], [[1, 1]], [[1, 1]]),
            # Two qubits, Z0 Z1; two-body gate
            (2, [[1, 1]], [0.2], [[0, 0]], [[1, 1]]),
            # Three qubits, batch of observables
            (
                3,
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0.1, 0.2, 0.3],
                [[1, 0, 0], [0, 1, 0], [1, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ),
        ],
    )
    def test_matches_pennylane(self, n, generators, thetas, l_vecs, m_vecs):
        """Qudit expectation values must match exact PennyLane simulation for d=2."""
        generators_arr = np.array(generators)
        thetas_arr = np.array(thetas)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)

        config, params = _make_config_one_param_per_gate(
            2, n, generators_arr, thetas_arr, l_arr, m_arr
        )
        expval_fn = _build_qudit_expval_func_exact(config)
        our_vals, *_ = expval_fn(jnp.array(params))

        for i, (l, m) in enumerate(zip(l_arr, m_arr)):
            pl_val = _pennylane_qubit_expval(generators_arr.tolist(), thetas_arr.tolist(), l, m)
            assert np.isclose(our_vals[i], pl_val, atol=1e-6), (
                f"Observable {i} (l={l}, m={m}): got {our_vals[i]:.8f}, "
                f"PennyLane gives {pl_val:.8f}"
            )


@pytest.mark.parametrize(
    "circuit_def, n_qudits, expected_generators, expected_param_map",
    [
        ({0: [[1, 0, 2]]}, 3, [[1, 0, 2]], [0]),
        ({0: [[1, 0]], 1: [[0, 2], [1, 1]]}, 2, [[1, 0], [0, 2], [1, 1]], [0, 1, 1]),
        ({}, 2, np.zeros((0, 2), dtype=int), []),
        ({3: [[0, 1]], 0: [[2, 0]]}, 2, [[2, 0], [0, 1]], [0, 3]),
    ],
)
def test_parse_qudit_generator_dict(circuit_def, n_qudits, expected_generators, expected_param_map):
    """_parse_qudit_generator_dict should produce the correct generator matrix and param map."""
    generators, param_map = _parse_qudit_generator_dict(circuit_def, n_qudits)

    assert isinstance(generators, jnp.ndarray)
    assert isinstance(param_map, jnp.ndarray)

    expected_generators = np.array(expected_generators)
    expected_param_map = np.array(expected_param_map)

    assert generators.shape == expected_generators.shape
    assert param_map.shape == expected_param_map.shape
    assert np.allclose(generators, expected_generators)
    assert np.allclose(param_map, expected_param_map)


def test_parse_qudit_generator_dict_wrong_length():
    """Generator with wrong length should raise ValueError."""
    with pytest.raises(ValueError, match="length"):
        _parse_qudit_generator_dict({0: [[1, 2]]}, n_qudits=3)


@pytest.mark.parametrize(
    "n, thetas, generators, l, m",
    [
        # Single qubit, Z observable
        (1, [0.7], [[1]], [0], [1]),
        # Single qubit, X observable
        (1, [0.37], [[1]], [1], [0]),
        # Two qubits, two gates, X0 Z1
        (2, [0.3, 0.6], [[1, 0], [0, 1]], [1, 0], [0, 1]),
        # Two qubits, entangling gate, Y0 Y1
        (2, [0.4], [[1, 1]], [1, 1], [1, 1]),
        # Three qubits, three gates, Z0 I1 X2
        (3, [0.1, 0.2, 0.3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 1], [1, 0, 0]),
    ],
)
def test_qudit_expval_exact_matches_pennylane(n, thetas, generators, l, m):
    """Test _build_qudit_expval_func_exact against PennyLane for d=2."""
    generators = np.array(generators)
    thetas = np.array(thetas)
    l_vecs = np.array([l])
    m_vecs = np.array([m])

    config, gate_params = _make_config_one_param_per_gate(2, n, generators, thetas, l_vecs, m_vecs)
    expval_fn = _build_qudit_expval_func_exact(config)
    our_vals, *_ = expval_fn(jnp.array(gate_params))

    pl_val = _pennylane_qubit_expval(generators.tolist(), thetas.tolist(), l, m)

    assert np.isclose(
        our_vals[0], pl_val, atol=1e-5
    ), f"Mismatch: _build_qudit_expval_func_exact={our_vals[0]}, PennyLane={pl_val}"


class TestQuditExpvalBatchedVsExact:
    """Test that the batched Monte Carlo version converges to the exact result."""

    @pytest.mark.parametrize(
        "d, n, generators, thetas, l_vecs, m_vecs",
        [
            (2, 1, [[1]], [0.5], [[1]], [[0]]),
            (2, 1, [[1]], [0.3], [[0]], [[1]]),
            (2, 2, [[1, 0], [1, 1]], [0.5, 0.2], [[1, 1]], [[0, 0]]),
            (
                2,
                2,
                [[1, 0], [0, 1]],
                [0.4, 0.6],
                [[1, 0], [0, 1], [1, 1]],
                [[0, 0], [0, 0], [0, 0]],
            ),
            (3, 1, [[1]], [0.42], [[1]], [[0]]),
            (3, 1, [[2]], [0.3], [[1]], [[1]]),
            (3, 2, [[1, 0], [0, 2]], [0.5, 0.2], [[1, 1]], [[0, 1]]),
            (4, 2, [[1, 2], [3, 1]], [0.3, 0.7], [[0, 0]], [[0, 0]]),
        ],
    )
    def test_matches_exact(self, d, n, generators, thetas, l_vecs, m_vecs):
        """Batched Monte Carlo must agree with the exact qudit expval within sampling noise."""
        generators_arr = np.array(generators)
        thetas_arr = np.array(thetas)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators_arr,
            thetas_arr,
            l_arr,
            m_arr,
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(42),
        )
        exact_fn = _build_qudit_expval_func_exact(config)
        exact_vals, *_ = exact_fn(jnp.array(params))

        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(jnp.array(params))

        assert mc_vals.shape == exact_vals.shape
        assert mc_cov.shape == exact_vals.shape + (2, 2)

        np.testing.assert_allclose(mc_vals, exact_vals, atol=3.5 / np.sqrt(NUM_SAMPLES))


class TestQuditExpvalBatchedVsMatrix:
    """Test that the batched Monte Carlo version matches the brute-force matrix reference."""

    @pytest.mark.parametrize(
        "d, n, generators, thetas, l_vecs, m_vecs",
        [
            (2, 2, [[1, 0], [0, 1]], [0.3, 0.6], [[1, 0]], [[0, 1]]),
            (3, 2, [[1, 0], [0, 2]], [0.5, 0.2], [[1, 1]], [[2, 1]]),
            (3, 2, [[1, 2], [0, 2]], [0.1, 0.8], [[1, 1]], [[1, 2]]),
            (3, 1, [[2]], [0.3], [[1]], [[1]]),
        ],
    )
    def test_matches_matrix_reference(self, d, n, generators, thetas, l_vecs, m_vecs):
        """Batched Monte Carlo must agree with the dense matrix reference."""
        generators_arr = np.array(generators)
        thetas_arr = np.array(thetas)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators_arr,
            thetas_arr,
            l_arr,
            m_arr,
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(123),
        )
        batched_fn = build_qudit_expval_func(config)
        mc_vals, *_ = batched_fn(jnp.array(params))

        for i, (l, m) in enumerate(zip(l_arr, m_arr)):
            ref = qudit_expectation_brute_force(n, d, generators_arr, thetas_arr, l, m)
            assert np.isclose(
                mc_vals[i], ref, atol=3.5 / np.sqrt(NUM_SAMPLES)
            ), f"Observable {i} (l={l}, m={m}): got {mc_vals[i]}, expected {ref}"


class TestQuditExpvalNonUniformDims:
    """Systems of qudits with non-uniform local dimensions.

    ``d`` is passed as a per-qudit sequence, and every generator / observable
    entry stays within its own qudit's ``{0, ..., d_j - 1}`` range.
    """

    @pytest.mark.parametrize(
        "dims, n, generators, thetas, l_vecs, m_vecs",
        [
            # qubit x qutrit
            (
                [2, 3],
                2,
                [[1, 0], [0, 2], [1, 1]],
                [0.5, 0.2, 0.3],
                [[1, 1], [0, 2]],
                [[0, 1], [1, 0]],
            ),
            # qutrit x ququart
            (
                [3, 4],
                2,
                [[1, 0], [0, 3], [2, 1]],
                [0.4, 0.7, 0.1],
                [[2, 3], [1, 0]],
                [[1, 2], [0, 1]],
            ),
            # qubit x qutrit x ququart, batch of observables
            (
                [2, 3, 4],
                3,
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]],
                [0.2, 0.5, 0.3, 0.15],
                [[1, 0, 0], [0, 2, 3], [1, 1, 1]],
                [[0, 0, 0], [1, 1, 2], [0, 2, 1]],
            ),
        ],
    )
    def test_matches_exact_and_matrix(self, dims, n, generators, thetas, l_vecs, m_vecs):
        """Batched MC must agree with both the exact and dense-matrix references."""
        generators_arr = np.array(generators)
        thetas_arr = np.array(thetas)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)

        config, params = _make_config_one_param_per_gate(
            dims,
            n,
            generators_arr,
            thetas_arr,
            l_arr,
            m_arr,
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(2024),
        )
        exact_fn = _build_qudit_expval_func_exact(config)
        exact_vals, *_ = exact_fn(jnp.array(params))

        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(jnp.array(params))

        assert mc_vals.shape == exact_vals.shape
        assert mc_cov.shape == exact_vals.shape + (2, 2)
        np.testing.assert_allclose(mc_vals, exact_vals, atol=3.5 / np.sqrt(NUM_SAMPLES))

        for i, (l, m) in enumerate(zip(l_arr, m_arr)):
            ref = qudit_expectation_brute_force(n, dims, generators_arr, thetas_arr, l, m)
            assert np.isclose(
                mc_vals[i], ref, atol=3.5 / np.sqrt(NUM_SAMPLES)
            ), f"Observable {i} (l={l}, m={m}): got {mc_vals[i]}, expected {ref}"

    def test_scalar_and_broadcast_sequence_agree(self):
        """A scalar ``d`` and the equivalent constant sequence must match exactly."""
        n = 2
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        thetas = np.array([0.5, 0.2, 0.3])
        l_vecs = np.array([[1, 2], [0, 1]])
        m_vecs = np.array([[0, 1], [2, 0]])

        config_scalar, params = _make_config_one_param_per_gate(
            3, n, generators, thetas, l_vecs, m_vecs, key=jax.random.PRNGKey(5)
        )
        config_seq, _ = _make_config_one_param_per_gate(
            [3, 3], n, generators, thetas, l_vecs, m_vecs, key=jax.random.PRNGKey(5)
        )

        vals_scalar, *_ = build_qudit_expval_func(config_scalar)(jnp.array(params))
        vals_seq, *_ = build_qudit_expval_func(config_seq)(jnp.array(params))
        np.testing.assert_allclose(vals_scalar, vals_seq, atol=1e-6)

    def test_non_uniform_dims_with_init_state(self):
        """Non-uniform dims combined with a sparse custom initial state."""
        dims, n = [2, 3], 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 2]])
        m_vecs = np.array([[0, 1]])
        elems = np.array([[0, 0], [1, 2]])
        amps = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)

        config, params = _make_config_one_param_per_gate(
            dims,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(
            jnp.array(params),
            init_state_elems=jnp.array(elems),
            init_state_amps=jnp.array(amps),
        )
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        ref = qudit_expectation_brute_force(
            n,
            dims,
            generators,
            thetas,
            l_vecs[0],
            m_vecs[0],
            init_state_elems=elems,
            init_state_amps=amps,
        )
        tol = max(3.5 * float(mc_err_re[0]), 3.5 * float(mc_err_im[0]), 1e-5)
        assert np.isclose(mc_vals[0], ref, atol=tol)


class TestQuditExpvalBatchedEdgeCases:
    """Edge cases and structural tests for the batched Monte Carlo function."""

    def test_identity_observable_gives_one(self):
        """D(0, 0) = identity, so its expectation value is always 1."""
        d, n = 3, 2
        generators = np.array([[1, 2], [0, 1]])
        thetas = np.array([0.8, 0.3])
        l_vecs = np.array([[0, 0]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(0),
        )
        batched_fn = build_qudit_expval_func(config)
        mc_vals, *_ = batched_fn(jnp.array(params))

        assert np.isclose(mc_vals[0], 1.0, atol=3.5 / np.sqrt(NUM_SAMPLES))

    def test_zero_params_matches_exact(self):
        """All-zero parameters reduce gates to identity."""
        d, n = 2, 2
        generators = np.array([[1, 0], [1, 1]])
        thetas = np.zeros(2)
        l_vecs = np.array([[1, 1]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(7),
        )
        exact_fn = _build_qudit_expval_func_exact(config)
        exact_vals, *_ = exact_fn(jnp.array(params))

        batched_fn = build_qudit_expval_func(config)
        mc_vals, *_ = batched_fn(jnp.array(params))

        np.testing.assert_allclose(mc_vals, exact_vals, atol=3.5 / np.sqrt(NUM_SAMPLES))

    def test_empty_gates(self):
        """Circuit with no gates should give the free-evolution expval."""
        d, n = 2, 2
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates={},
            observables=(l_vecs, m_vecs),
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(99),
        )
        exact_fn = _build_qudit_expval_func_exact(config)
        exact_vals, *_ = exact_fn(jnp.array([]))

        batched_fn = build_qudit_expval_func(config)
        mc_vals, *_ = batched_fn(jnp.array([]))

        np.testing.assert_allclose(mc_vals, exact_vals, atol=3.5 / np.sqrt(NUM_SAMPLES))

    def test_parameter_broadcasting(self):
        """Multiple gates sharing a parameter index should all use the same theta."""
        d, n = 2, 3
        gates = {0: [[1, 0, 0], [0, 1, 0]], 1: [[0, 0, 1]]}
        thetas_unique = np.array([0.5, 0.3])
        l_vecs = np.array([[1, 0, 0]])
        m_vecs = np.array([[0, 0, 0]])

        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(55),
        )
        exact_fn = _build_qudit_expval_func_exact(config)
        exact_vals, *_ = exact_fn(jnp.array(thetas_unique))

        batched_fn = build_qudit_expval_func(config)
        mc_vals, *_ = batched_fn(jnp.array(thetas_unique))

        np.testing.assert_allclose(mc_vals, exact_vals, atol=3.5 / np.sqrt(NUM_SAMPLES))

    def test_covariance_is_valid(self):
        """Covariance matrices returned by the batched function must be valid."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.5, 0.2])
        l_vecs = np.array([[1, 1], [0, 1]])
        m_vecs = np.array([[0, 1], [1, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=5000,
            key=jax.random.PRNGKey(11),
        )
        batched_fn = build_qudit_expval_func(config)
        _, mc_cov, mean_y_sq = batched_fn(jnp.array(params), return_mean_y_sq=True)

        # Symmetric.
        np.testing.assert_allclose(mc_cov, np.swapaxes(mc_cov, -1, -2), atol=1e-7)
        # Unit-modulus default-state integrands give mean |y_r|^2 = 1.
        np.testing.assert_allclose(mean_y_sq, np.ones_like(mean_y_sq), atol=1e-7)
        # Non-negative variances on the diagonal.
        assert np.all(mc_cov[:, 0, 0] >= 0)
        assert np.all(mc_cov[:, 1, 1] >= 0)
        # Positive semi-definite: non-negative determinant.
        dets = mc_cov[:, 0, 0] * mc_cov[:, 1, 1] - mc_cov[:, 0, 1] * mc_cov[:, 1, 0]
        assert np.all(dets >= -1e-12)

    def test_covariance_decreases_with_more_samples(self):
        """Covariance of the mean should decrease when we use more samples."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 1]])

        config, params = _make_config_one_param_per_gate(d, n, generators, thetas, l_vecs, m_vecs)
        batched_fn = build_qudit_expval_func(config)
        _, cov_lo = batched_fn(jnp.array(params), n_samples=1_000, key=jax.random.PRNGKey(0))
        _, cov_hi = batched_fn(jnp.array(params), n_samples=100_000, key=jax.random.PRNGKey(0))

        assert np.all(cov_hi[:, 0, 0] < cov_lo[:, 0, 0])
        assert np.all(cov_hi[:, 1, 1] < cov_lo[:, 1, 1])

    def test_jit_compatible(self):
        """The batched function should be JIT-compilable."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=10_000,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)
        jitted = jax.jit(batched_fn)
        mc_vals, mc_cov = jitted(jnp.array(params))
        mc_vals_nojit, mc_cov_nojit = batched_fn(jnp.array(params))

        np.testing.assert_allclose(mc_vals, mc_vals_nojit, atol=1e-6)
        np.testing.assert_allclose(mc_cov, mc_cov_nojit, atol=1e-6)

    def test_observables_override_matches_exact(self):
        """Overridden observables should match the exact brute-force computation."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])

        default_l = np.array([[0, 0]])
        default_m = np.array([[0, 0]])

        override_l = np.array([[1, 0], [0, 1], [1, 1]])
        override_m = np.array([[0, 0], [0, 0], [0, 0]])

        key = jax.random.PRNGKey(7)

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            default_l,
            default_m,
            n_samples=NUM_SAMPLES,
            key=key,
        )
        batched_fn = build_qudit_expval_func(config)
        mc_vals, _ = batched_fn(
            jnp.array(params),
            observables=(override_l, override_m),
        )

        exact_config, _ = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            override_l,
            override_m,
            n_samples=NUM_SAMPLES,
            key=key,
        )
        exact_fn = _build_qudit_expval_func_exact(exact_config)
        exact_vals, *_ = exact_fn(jnp.array(params))

        np.testing.assert_allclose(mc_vals, exact_vals, atol=3.5 / np.sqrt(NUM_SAMPLES))

    def test_differentiable(self):
        """The batched function should be differentiable via JAX grad."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=10_000,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)

        def loss(p):
            vals, *_ = batched_fn(p)
            return jnp.real(jnp.sum(vals))

        grad_fn = jax.grad(loss)
        grads = grad_fn(jnp.array(params))
        assert grads.shape == (len(thetas),)
        assert np.all(np.isfinite(grads))


@pytest.mark.parametrize(
    "d, n, thetas, generators, l, m",
    [
        # d=2, single qubit, single gate
        (2, 1, [0.5], [[1]], [1], [0]),
        # d=2, two qubits, two-body gate
        (2, 2, [0.4], [[1, 1]], [1, 1], [0, 0]),
        # d=3, single qutrit, single gate
        (3, 1, [0.7], [[2]], [1], [2]),
        # d=3, two qutrits, two gates
        (3, 2, [0.3, 0.6], [[1, 0], [0, 2]], [2, 1], [1, 0]),
        # d=4, single ququart
        (4, 1, [1.1], [[3]], [2], [1]),
    ],
)
def test_qudit_expval_batched_matches_exact(d, n, thetas, generators, l, m):
    """Test build_qudit_expval_func against the exact version for various dimensions."""
    generators = np.array(generators)
    thetas = np.array(thetas)
    l_vecs = np.array([l])
    m_vecs = np.array([m])

    config, gate_params = _make_config_one_param_per_gate(
        d,
        n,
        generators,
        thetas,
        l_vecs,
        m_vecs,
        n_samples=NUM_SAMPLES,
        key=jax.random.PRNGKey(0),
    )
    exact_fn = _build_qudit_expval_func_exact(config)
    exact_vals, *_ = exact_fn(jnp.array(gate_params))

    batched_fn = build_qudit_expval_func(config)
    mc_vals, mc_cov = batched_fn(jnp.array(gate_params))

    assert np.isclose(
        mc_vals[0], exact_vals[0], atol=3.5 / np.sqrt(NUM_SAMPLES)
    ), f"Mismatch: batched={mc_vals[0]}, exact={exact_vals[0]}"
    assert mc_cov[0, 0, 0] >= 0
    assert mc_cov[0, 1, 1] >= 0


NUM_SAMPLES_INIT_STATE = 50000


class TestQuditExpvalBatchedWithInitState:
    """Test batched Monte Carlo with general initial states against brute-force reference."""

    @pytest.mark.parametrize(
        "d, n, generators, thetas, l_vecs, m_vecs, state_elems, state_amps",
        [
            # d=2, single computational basis state |1>
            (2, 1, [[1]], [0.5], [[1]], [[0]], [[1]], [1.0]),
            # d=2, n=2, computational basis state |10>
            (2, 2, [[1, 0], [0, 1]], [0.4, 0.6], [[1, 0]], [[0, 0]], [[1, 0]], [1.0]),
            # d=2, n=2, equal superposition (|00> + |11>)/sqrt(2), multiple observables
            (
                2,
                2,
                [[1, 0], [0, 1]],
                [0.3, 0.5],
                [[1, 0], [0, 1], [1, 1]],
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [1, 1]],
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
            ),
            # d=2, n=2, m != 0, multi-element state
            (
                2,
                2,
                [[1, 0], [1, 1]],
                [0.4, 0.2],
                [[1, 1]],
                [[1, 0]],
                [[0, 0], [1, 0]],
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
            ),
            # d=3, single qutrit, basis state |2>
            (3, 1, [[1]], [0.42], [[1]], [[0]], [[2]], [1.0]),
            # d=3, n=2, basis state |1, 2>
            (3, 2, [[1, 0], [0, 2]], [0.5, 0.2], [[1, 1]], [[0, 1]], [[1, 2]], [1.0]),
            # d=3, n=2, equal superposition (|00> + |12> + |21>)/sqrt(3)
            (
                3,
                2,
                [[1, 0], [0, 1]],
                [0.3, 0.4],
                [[1, 0], [0, 1]],
                [[0, 0], [0, 0]],
                [[0, 0], [1, 2], [2, 1]],
                [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            ),
            # d=3, n=2, complex amplitudes
            (
                3,
                2,
                [[1, 0]],
                [0.5],
                [[1, 0]],
                [[0, 0]],
                [[0, 0], [1, 1]],
                [1 / np.sqrt(2), 1j / np.sqrt(2)],
            ),
            # d=2, n=2, displacement observable (m != 0), superposition
            (
                2,
                2,
                [[1, 0], [0, 1]],
                [0.3, 0.5],
                [[1, 1]],
                [[1, 1]],
                [[0, 0], [1, 1]],
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
            ),
            # d=3, n=2, m != 0, multi-element state
            (
                3,
                2,
                [[1, 0], [0, 1]],
                [0.3, 0.4],
                [[1, 2], [0, 1]],
                [[2, 1], [0, 0]],
                [[0, 0], [1, 2]],
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
            ),
            # d=4, n=2, basis state
            (4, 2, [[1, 2], [3, 1]], [0.3, 0.7], [[1, 0]], [[0, 0]], [[0, 1]], [1.0]),
            # d=4, n=1, complex amplitudes with m != 0
            (4, 1, [[2]], [0.5], [[1]], [[2]], [[0], [3]], [1 / np.sqrt(2), 1j / np.sqrt(2)]),
        ],
    )
    def test_matches_matrix_reference(
        self, d, n, generators, thetas, l_vecs, m_vecs, state_elems, state_amps
    ):
        """Batched Monte Carlo with init state must agree with dense matrix reference."""
        generators_arr = np.array(generators)
        thetas_arr = np.array(thetas)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)
        elems_arr = np.array(state_elems)
        amps_arr = np.array(state_amps, dtype=complex)

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators_arr,
            thetas_arr,
            l_arr,
            m_arr,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(
            jnp.array(params),
            init_state_elems=jnp.array(elems_arr),
            init_state_amps=jnp.array(amps_arr),
        )
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        for i, (l, m) in enumerate(zip(l_arr, m_arr)):
            ref = qudit_expectation_brute_force(
                n,
                d,
                generators_arr,
                thetas_arr,
                l,
                m,
                init_state_elems=elems_arr,
                init_state_amps=amps_arr,
            )
            tol = max(3.5 * float(mc_err_re[i]), 3.5 * float(mc_err_im[i]), 1e-5)
            assert np.isclose(mc_vals[i], ref, atol=tol), (
                f"Observable {i} (l={l}, m={m}): got {mc_vals[i]}, "
                f"expected {ref}, tol={tol:.2e}"
            )
        assert np.all(mc_err_re >= 0)
        assert np.all(mc_err_im >= 0)

    def test_default_state_matches_no_state(self):
        """When |0> is explicitly passed, results must match the default (no state) exactly."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.5, 0.2])
        l_vecs = np.array([[1, 1]])
        m_vecs = np.array([[0, 1]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(99),
        )
        batched_fn = build_qudit_expval_func(config)

        vals_default, *_ = batched_fn(jnp.array(params))
        vals_explicit, *_ = batched_fn(
            jnp.array(params),
            init_state_elems=jnp.array([[0, 0]]),
            init_state_amps=jnp.array([1.0 + 0j]),
        )

        # Both calls use the same fixed PRNG key; H = 1 exactly for |0> because
        # all X=0 makes all omega phases trivially 1. Results agree to float precision.
        np.testing.assert_allclose(vals_explicit, vals_default, atol=1e-5)

    def test_config_init_state(self):
        """Init state set in config should be used without runtime overrides."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])
        elems = np.array([[0, 0], [1, 1]])
        amps = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)

        gates = {i: [list(gen)] for i, gen in enumerate(generators)}
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            init_state_elems=elems,
            init_state_amps=amps,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )

        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(jnp.array(thetas))
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        ref = qudit_expectation_brute_force(
            n,
            d,
            generators,
            thetas,
            l_vecs[0],
            m_vecs[0],
            init_state_elems=elems,
            init_state_amps=amps,
        )
        tol = max(3.5 * float(mc_err_re[0]), 3.5 * float(mc_err_im[0]), 1e-5)
        assert np.isclose(mc_vals[0], ref, atol=tol)

    def test_runtime_override_takes_precedence(self):
        """Runtime init state should override the config init state."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.3, 0.5])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])

        config_elems = np.array([[0, 0]])
        config_amps = np.array([1.0 + 0j])
        runtime_elems = np.array([[1, 1]])
        runtime_amps = np.array([1.0 + 0j])

        gates = {i: [list(gen)] for i, gen in enumerate(generators)}
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            init_state_elems=config_elems,
            init_state_amps=config_amps,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )

        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(
            jnp.array(thetas),
            init_state_elems=jnp.array(runtime_elems),
            init_state_amps=jnp.array(runtime_amps),
        )
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        runtime_ref = qudit_expectation_brute_force(
            n,
            d,
            generators,
            thetas,
            l_vecs[0],
            m_vecs[0],
            init_state_elems=runtime_elems,
            init_state_amps=runtime_amps,
        )
        config_ref = qudit_expectation_brute_force(
            n,
            d,
            generators,
            thetas,
            l_vecs[0],
            m_vecs[0],
            init_state_elems=config_elems,
            init_state_amps=config_amps,
        )

        # Ensure the two states give different expectation values (non-vacuous test).
        assert not np.isclose(
            runtime_ref, config_ref, atol=1e-4
        ), "Config and runtime refs are too close; choose states with different expvals"

        tol = max(3.5 * float(mc_err_re[0]), 3.5 * float(mc_err_im[0]), 1e-5)
        assert np.isclose(mc_vals[0], runtime_ref, atol=tol)

    def test_jit_compatible_with_init_state(self):
        """The batched function should be JIT-compilable with an init state."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=10_000,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)

        elems = jnp.array([[0, 0], [1, 1]])
        amps = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)

        def fn_with_state(p):
            return batched_fn(p, init_state_elems=elems, init_state_amps=amps)

        jitted = jax.jit(fn_with_state)
        mc_vals, mc_cov = jitted(jnp.array(params))
        mc_vals_nojit, mc_cov_nojit = fn_with_state(jnp.array(params))

        np.testing.assert_allclose(mc_vals, mc_vals_nojit, atol=1e-6)
        np.testing.assert_allclose(mc_cov, mc_cov_nojit, atol=1e-6)

    def test_complex_expval_with_init_state(self):
        """Non-Hermitian observable with complex-amplitude state gives complex expval."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 2]])
        m_vecs = np.array([[2, 1]])
        elems = np.array([[0, 0], [1, 2]])
        amps = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(
            jnp.array(params),
            init_state_elems=jnp.array(elems),
            init_state_amps=jnp.array(amps),
        )
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        ref = qudit_expectation_brute_force(
            n,
            d,
            generators,
            thetas,
            l_vecs[0],
            m_vecs[0],
            init_state_elems=elems,
            init_state_amps=amps,
        )

        assert np.abs(np.imag(ref)) > 1e-3, (
            f"Reference imaginary part too small ({np.imag(ref):.6f}); "
            "choose parameters that produce a genuinely complex expectation"
        )
        assert mc_err_re[0] > 0
        assert mc_err_im[0] > 0

        tol = max(3.5 * float(mc_err_re[0]), 3.5 * float(mc_err_im[0]), 1e-5)
        assert np.isclose(
            mc_vals[0], ref, atol=tol
        ), f"got {mc_vals[0]}, expected {ref}, tol={tol:.2e}"

    def test_unnormalized_state_scales_quadratically(self):
        """Scaling state amplitudes by c scales expectation values by |c|^2."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)

        elems = jnp.array([[0, 0], [1, 1]])
        amps_norm = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        scale = 3.0
        amps_unnorm = amps_norm * scale

        vals_norm, *_ = batched_fn(
            jnp.array(params), init_state_elems=elems, init_state_amps=amps_norm
        )
        vals_unnorm, *_ = batched_fn(
            jnp.array(params), init_state_elems=elems, init_state_amps=amps_unnorm
        )

        # Both calls use the same fixed PRNG key, so the H ∝ |c|^2 scaling
        # holds to float precision without any Monte Carlo noise.
        np.testing.assert_allclose(vals_unnorm, scale**2 * vals_norm, atol=1e-5)

    def test_differentiable_with_init_state(self):
        """JAX gradients must match finite differences for circuits with an init state."""
        d, n = 2, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 0]])
        m_vecs = np.array([[0, 0]])

        config, params = _make_config_one_param_per_gate(
            d,
            n,
            generators,
            thetas,
            l_vecs,
            m_vecs,
            n_samples=100_000,
            key=jax.random.PRNGKey(42),
        )
        batched_fn = build_qudit_expval_func(config)

        elems = jnp.array([[0, 0], [1, 1]])
        amps = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)

        def loss(p):
            vals, *_ = batched_fn(p, init_state_elems=elems, init_state_amps=amps)
            return jnp.real(jnp.sum(vals))

        grad_fn = jax.grad(loss)
        grads = grad_fn(jnp.array(params))
        assert grads.shape == (len(thetas),)
        assert np.all(np.isfinite(grads))

        # Finite-difference check: same PRNG key means same samples, so Monte Carlo noise
        # cancels and only O(eps^2) truncation error remains.
        eps = 1e-3
        p = np.array(params, dtype=float)
        fd_grads = np.zeros_like(p)
        for k in range(len(p)):
            p_plus = p.copy()
            p_plus[k] += eps
            p_minus = p.copy()
            p_minus[k] -= eps
            fd_grads[k] = (loss(jnp.array(p_plus)) - loss(jnp.array(p_minus))) / (2 * eps)

        np.testing.assert_allclose(np.array(grads), fd_grads, atol=1e-3)


@pytest.mark.parametrize(
    "d, n, thetas, generators, l, m, state_elems, state_amps",
    [
        # d=2, single qubit, computational basis |1>
        (2, 1, [0.5], [[1]], [1], [0], [[1]], [1.0]),
        # d=2, two qubits, |10>
        (2, 2, [0.4, 0.6], [[1, 0], [0, 1]], [1, 0], [0, 0], [[1, 0]], [1.0]),
        # d=2, two qubits, superposition (|00> + |11>)/sqrt(2)
        (
            2,
            2,
            [0.3, 0.5],
            [[1, 0], [0, 1]],
            [1, 0],
            [0, 0],
            [[0, 0], [1, 1]],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ),
        # d=3, single qutrit, |2> with complex amplitude
        (3, 1, [0.7], [[2]], [1], [2], [[2]], [1.0 + 0j]),
        # d=3, two qutrits, superposition (|01> + i|20>)/sqrt(2)
        (
            3,
            2,
            [0.3],
            [[1, 2]],
            [2, 1],
            [1, 0],
            [[0, 1], [2, 0]],
            [1 / np.sqrt(2), 1j / np.sqrt(2)],
        ),
    ],
)
def test_qudit_expval_batched_init_state_matches_brute_force(
    d, n, thetas, generators, l, m, state_elems, state_amps
):
    """Test build_qudit_expval_func with init_state against the dense matrix reference."""
    generators = np.array(generators)
    thetas = np.array(thetas)
    state_elems = np.array(state_elems)
    state_amps = np.array(state_amps, dtype=complex)

    l_vecs = np.array([l])
    m_vecs = np.array([m])

    config, gate_params = _make_config_one_param_per_gate(
        d, n, generators, thetas, l_vecs, m_vecs, n_samples=20_000, key=jax.random.PRNGKey(0)
    )
    batched_fn = build_qudit_expval_func(config)
    mc_vals, mc_cov = batched_fn(
        jnp.array(gate_params),
        init_state_elems=jnp.array(state_elems),
        init_state_amps=jnp.array(state_amps),
    )
    mc_err_re = np.sqrt(mc_cov[:, 0, 0])
    mc_err_im = np.sqrt(mc_cov[:, 1, 1])

    ref = qudit_expectation_brute_force(
        n,
        d,
        generators,
        thetas,
        l,
        m,
        init_state_elems=state_elems,
        init_state_amps=state_amps,
    )
    tol = max(3.5 * float(mc_err_re[0]), 3.5 * float(mc_err_im[0]), 1e-4)
    assert np.isclose(
        mc_vals[0], ref, atol=tol
    ), f"Mismatch: batched={mc_vals[0]}, matrix={ref}, tol={tol:.2e}"


class TestQuditExpvalWithPhaseLayer:
    """Test batched Monte Carlo with a custom phase layer against brute-force reference."""

    @staticmethod
    def _phase_fn(params, z):
        """Polynomial in normalised mean of z: f(params, z) = sum_t params[t] * (mean(z)/d)^t."""
        x = jnp.mean(z.astype(jnp.float32)) / 3.0
        powers = jnp.array([x**t for t in range(len(params))])
        return jnp.sum(params * powers)

    @staticmethod
    def _build_phase_diag(phase_fn, phase_params, d, n):
        """Evaluate phase_fn at every z in Z_d^n to build the diagonal vector."""
        all_states = list(itertools.product(range(d), repeat=n))
        return np.array([float(phase_fn(phase_params, jnp.array(z))) for z in all_states])

    def test_phase_layer_default_state(self):
        """Phase layer with default |0> input, d=3, nonzero l, nontrivial params."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2], [1, 2]])
        thetas = np.array([0.5, 0.3, 0.7])
        l_vecs = np.array([[2, 1], [0, 1], [1, 0]])
        m_vecs = np.array([[0, 0], [0, 0], [0, 0]])
        phase_params = jnp.array([0.1, 0.5, 2.0, 1.0])
        n_samples = 80000

        gates = {i: [list(gen)] for i, gen in enumerate(generators)}
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            n_samples=n_samples,
            key=jax.random.PRNGKey(42),
            phase_fn=self._phase_fn,
        )

        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(jnp.array(thetas), phase_params)
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        phase_diag = self._build_phase_diag(self._phase_fn, phase_params, d, n)

        for i, (l, m) in enumerate(zip(l_vecs, m_vecs)):
            ref = qudit_expectation_brute_force(
                n,
                d,
                generators,
                thetas,
                l,
                m,
                phase_diag=phase_diag,
            )
            tol = max(3.5 * float(mc_err_re[i]), 3.5 * float(mc_err_im[i]), 1e-5)
            assert np.isclose(mc_vals[i], ref, atol=tol), (
                f"Observable {i} (l={l}, m={m}): got {mc_vals[i]}, "
                f"expected {ref}, tol={tol:.2e}"
            )

    def test_phase_layer_with_init_state(self):
        """Phase layer combined with a sparse initial state."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.4, 0.6])
        l_vecs = np.array([[1, 2], [2, 0]])
        m_vecs = np.array([[0, 0], [0, 0]])
        phase_params = jnp.array([0.2, 1.5, -0.3])
        state_elems = np.array([[0, 0], [1, 2], [2, 1]])
        state_amps = np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], dtype=complex)
        n_samples = 80000

        gates = {i: [list(gen)] for i, gen in enumerate(generators)}
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            n_samples=n_samples,
            key=jax.random.PRNGKey(99),
            phase_fn=self._phase_fn,
        )

        batched_fn = build_qudit_expval_func(config)
        mc_vals, mc_cov = batched_fn(
            jnp.array(thetas),
            phase_params,
            init_state_elems=jnp.array(state_elems),
            init_state_amps=jnp.array(state_amps),
        )
        mc_err_re = np.sqrt(mc_cov[:, 0, 0])
        mc_err_im = np.sqrt(mc_cov[:, 1, 1])

        phase_diag = self._build_phase_diag(self._phase_fn, phase_params, d, n)

        for i, (l, m) in enumerate(zip(l_vecs, m_vecs)):
            ref = qudit_expectation_brute_force(
                n,
                d,
                generators,
                thetas,
                l,
                m,
                init_state_elems=state_elems,
                init_state_amps=state_amps,
                phase_diag=phase_diag,
            )
            tol = max(3.5 * float(mc_err_re[i]), 3.5 * float(mc_err_im[i]), 1e-5)
            assert np.isclose(mc_vals[i], ref, atol=tol), (
                f"Observable {i} (l={l}, m={m}): got {mc_vals[i]}, "
                f"expected {ref}, tol={tol:.2e}"
            )

    def test_phase_layer_grad(self):
        """Verify gradients flow through phase_fn_params."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.5, 0.3])
        l_vecs = np.array([[2, 1]])
        m_vecs = np.array([[0, 0]])
        phase_params = jnp.array([0.1, 0.5, 2.0])

        gates = {i: [list(gen)] for i, gen in enumerate(generators)}
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            n_samples=5000,
            key=jax.random.PRNGKey(0),
            phase_fn=self._phase_fn,
        )

        batched_fn = build_qudit_expval_func(config)

        def loss(p_params):
            vals, _ = batched_fn(jnp.array(thetas), p_params)
            return jnp.sum(jnp.real(vals) ** 2)

        grad_val = jax.grad(loss)(phase_params)
        assert grad_val.shape == phase_params.shape
        assert not jnp.allclose(grad_val, 0.0)


# ---------------------------------------------------------------------------
# Order-2 Taylor control variate
# ---------------------------------------------------------------------------

CHI = (1 + 1j) / 2


def _Q_g(g, z, dims):
    """Q_g(z) = prod_{k in supp(g)} sqrt(2) cos(2 pi g_k z_k / d_k + pi/4)."""
    out = 1.0
    for k in np.nonzero(g)[0]:
        out *= np.sqrt(2) * np.cos(2 * np.pi * g[k] * z[k] / dims[k] + np.pi / 4)
    return out


def _phase_difference(thetas, generators, l, z, dims):
    """Delta_l(z) = sum_g theta_g [Q_g(z) - Q_g(z - l)]."""
    z_shifted = (z - l) % dims
    return sum(
        t * (_Q_g(g, z, dims) - _Q_g(g, z_shifted, dims)) for t, g in zip(thetas, generators)
    )


def _observable_phase(l, m, z, dims):
    """J_l(z) = exp(i pi sum_k m_k (2 z_k - l_k) / d_k)."""
    return np.exp(1j * np.pi * np.sum(m * (2 * z - l) / dims))


def _state_factor(l, z, dims, elems, amps):
    """H_l(z) = sum_{a,b} Psi_a conj(Psi_b) omega^{l.x_b} omega^{z.(x_a - x_b)}."""
    if elems is None or amps is None:
        return 1.0 + 0j
    total = 0j
    for a, x_a in enumerate(elems):
        for b, x_b in enumerate(elems):
            phase = np.exp(2j * np.pi * (np.sum(l * x_b / dims) + np.sum(z * (x_a - x_b) / dims)))
            total += amps[a] * np.conj(amps[b]) * phase
    return total


def control_variate_integrand_brute_force(thetas, generators, l, m, z, dims, elems, amps):
    """Per-sample order-2 control variate y~_l(z) = J (1 + i D - D^2/2) H."""
    D = _phase_difference(thetas, generators, l, z, dims)
    return (
        _observable_phase(l, m, z, dims)
        * (1 + 1j * D - 0.5 * D**2)
        * _state_factor(l, z, dims, elems, amps)
    )


def control_variate_mean_brute_force(thetas, generators, l, m, dims, elems=None, amps=None):
    """Exact control mean by full enumeration of E_z[y~_l(z)] over all dit-strings.

    This is an independent path to the closed form: it never uses the character
    expansion, it just averages the order-2 integrand over every z in the group.
    """
    all_z = np.array(list(itertools.product(*(range(int(d_j)) for d_j in dims))), dtype=int)
    vals = [
        control_variate_integrand_brute_force(thetas, generators, l, m, z, dims, elems, amps)
        for z in all_z
    ]
    return np.mean(vals)


def _make_cv_config(
    d, n, generators, l_vecs, m_vecs, n_samples=NUM_SAMPLES, key=None, control_variate=True, **kw
):
    """Build a QuditCircuitConfig with one parameter per gate and a CV toggle."""
    if key is None:
        key = jax.random.PRNGKey(0)
    gates = {i: [list(gen)] for i, gen in enumerate(generators)}
    return QuditCircuitConfig(
        d=d,
        n_qudits=n,
        gates=gates,
        observables=(np.array(l_vecs), np.array(m_vecs)),
        n_samples=n_samples,
        key=key,
        control_variate=control_variate,
        **kw,
    )


class TestControlVariateClosedForm:
    """The closed-form control mean must match brute-force enumeration exactly.

    This is the correctness-critical property: the control mean enters the
    estimator as a deterministic offset, so any error in it biases the result.
    """

    @pytest.mark.parametrize(
        "d, n, generators, thetas, l_vecs, m_vecs",
        [
            # d=3, weight-1 and weight-2 generators
            (
                3,
                2,
                [[1, 0], [0, 2], [1, 1]],
                [0.4, -0.3, 0.25],
                [[1, 0], [0, 1], [2, 1]],
                [[0, 0], [1, 2], [1, 0]],
            ),
            # d=2 (qubit limit)
            (
                2,
                3,
                [[1, 1, 0], [0, 1, 1]],
                [0.3, -0.2],
                [[1, 1, 0], [1, 0, 1]],
                [[0, 0, 0], [1, 1, 0]],
            ),
            # d=4, includes a weight-0 generator that must drop out
            (
                4,
                2,
                [[2, 0], [0, 0], [3, 1]],
                [0.5, 0.9, -0.4],
                [[1, 3], [2, 2], [0, 1]],
                [[3, 1], [0, 0], [1, 1]],
            ),
            # mixed local dimensions
            (
                [2, 3, 4],
                3,
                [[1, 0, 0], [0, 1, 2], [1, 2, 3]],
                [0.3, 0.2, -0.15],
                [[1, 1, 2], [0, 2, 3]],
                [[0, 1, 1], [1, 0, 2]],
            ),
            # l = 0 row: Delta vanishes identically, so the control is exact
            (3, 2, [[1, 0], [1, 1]], [0.35, -0.45], [[0, 0], [1, 2]], [[1, 0], [0, 0]]),
        ],
    )
    def test_closed_form_matches_enumeration_default_state(
        self, d, n, generators, thetas, l_vecs, m_vecs
    ):
        """Closed-form tau_l must equal E_z[y~] enumerated over the whole group."""
        dims = _dims_to_numpy(d, n)
        gen_arr = np.array(generators)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)

        char_data = _build_character_expansion(gen_arr, np.arange(len(gen_arr)), dims)
        got = _control_variate_expected_value(
            jnp.array(thetas),
            char_data,
            jnp.array(l_arr, dtype=jnp.float32),
            jnp.array(m_arr, dtype=jnp.float32),
            dims,
            None,
            None,
        )

        for i, (l, m) in enumerate(zip(l_arr, m_arr)):
            ref = control_variate_mean_brute_force(np.array(thetas), gen_arr, l, m, dims)
            assert np.isclose(
                got[i], ref, atol=1e-6
            ), f"Observable {i} (l={l}, m={m}): closed form {got[i]}, enumeration {ref}"

    @pytest.mark.parametrize(
        "d, n, generators, thetas, l_vecs, m_vecs, state_elems, state_amps",
        [
            # real amplitudes
            (
                3,
                2,
                [[1, 0], [0, 2]],
                [0.4, -0.3],
                [[1, 1], [0, 2]],
                [[0, 1], [1, 0]],
                [[0, 0], [1, 2]],
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
            ),
            # complex amplitudes
            (
                3,
                2,
                [[1, 0], [1, 1]],
                [0.25, 0.5],
                [[1, 2]],
                [[2, 1]],
                [[0, 0], [1, 2], [2, 1]],
                [1 / np.sqrt(3), 1j / np.sqrt(3), -1 / np.sqrt(3)],
            ),
            # qubit limit with a sparse state
            (
                2,
                2,
                [[1, 0], [1, 1]],
                [0.3, -0.2],
                [[1, 1], [1, 0]],
                [[1, 0], [0, 0]],
                [[0, 0], [1, 1]],
                [1 / np.sqrt(2), 1j / np.sqrt(2)],
            ),
            # mixed dimensions with a sparse state
            (
                [2, 3],
                2,
                [[1, 0], [0, 1], [1, 2]],
                [0.3, 0.2, -0.4],
                [[1, 2]],
                [[0, 1]],
                [[0, 0], [1, 2]],
                [1 / np.sqrt(2), 1j / np.sqrt(2)],
            ),
            # unnormalised amplitudes must still be handled exactly
            (4, 1, [[2]], [0.45], [[1]], [[2]], [[0], [3]], [0.7, 1.3j]),
        ],
    )
    def test_closed_form_matches_enumeration_sparse_state(
        self, d, n, generators, thetas, l_vecs, m_vecs, state_elems, state_amps
    ):
        """Closed-form tau_l must match enumeration for general sparse input states."""
        dims = _dims_to_numpy(d, n)
        gen_arr = np.array(generators)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)
        elems = np.array(state_elems)
        amps = np.array(state_amps, dtype=complex)

        char_data = _build_character_expansion(gen_arr, np.arange(len(gen_arr)), dims)
        got = _control_variate_expected_value(
            jnp.array(thetas),
            char_data,
            jnp.array(l_arr, dtype=jnp.float32),
            jnp.array(m_arr, dtype=jnp.float32),
            dims,
            jnp.array(elems),
            jnp.array(amps),
        )

        for i, (l, m) in enumerate(zip(l_arr, m_arr)):
            ref = control_variate_mean_brute_force(
                np.array(thetas), gen_arr, l, m, dims, elems, amps
            )
            assert np.isclose(
                got[i], ref, atol=1e-6
            ), f"Observable {i} (l={l}, m={m}): closed form {got[i]}, enumeration {ref}"

    def test_zero_params_gives_observable_delta(self):
        """At theta = 0 all amplitudes vanish, so tau_l = P_lm delta_{m, 0}."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        dims = _dims_to_numpy(d, n)
        # First row has m = 0 (expect 1), second has m != 0 (expect 0).
        l_arr = np.array([[1, 2], [1, 2]])
        m_arr = np.array([[0, 0], [1, 0]])

        char_data = _build_character_expansion(generators, np.arange(len(generators)), dims)
        got = _control_variate_expected_value(
            jnp.zeros(len(generators)),
            char_data,
            jnp.array(l_arr, dtype=jnp.float32),
            jnp.array(m_arr, dtype=jnp.float32),
            dims,
            None,
            None,
        )
        np.testing.assert_allclose(np.array(got), np.array([1.0, 0.0]), atol=1e-6)

    def test_shift_blind_characters_have_zero_amplitude(self):
        """l = 0 makes every A_{l,t} vanish, since 1 - omega^{-f.l} = 0."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        dims = _dims_to_numpy(d, n)
        char_data = _build_character_expansion(generators, np.arange(len(generators)), dims)

        amps = _character_amplitudes(
            jnp.array([0.4, -0.3, 0.25]),
            char_data,
            jnp.zeros((1, n), dtype=jnp.float32),
            dims,
        )
        np.testing.assert_allclose(np.array(amps), 0.0, atol=1e-7)

    def test_weight_zero_generators_dropped(self):
        """Generators with empty support contribute Q_g = 1 and must be excluded."""
        d, n = 3, 2
        dims = _dims_to_numpy(d, n)
        with_zero = np.array([[1, 0], [0, 0], [0, 2]])
        without_zero = np.array([[1, 0], [0, 2]])

        cd_with = _build_character_expansion(with_zero, np.arange(3), dims)
        cd_without = _build_character_expansion(without_zero, np.arange(2), dims)

        # Each weight-1 generator contributes 2 terms; the weight-0 one contributes none.
        assert cd_with.freqs.shape[0] == 4
        assert cd_without.freqs.shape[0] == 4
        # The surviving parameter indices skip the dropped generator.
        assert set(np.array(cd_with.param_indices).tolist()) == {0, 2}

    def test_expansion_term_count(self):
        """The expansion must have T = sum_g 2^{|supp(g)|} terms."""
        d, n = 3, 3
        dims = _dims_to_numpy(d, n)
        generators = np.array([[1, 0, 0], [1, 2, 0], [1, 1, 2]])  # weights 1, 2, 3
        char_data = _build_character_expansion(generators, np.arange(3), dims)
        assert char_data.freqs.shape[0] == 2**1 + 2**2 + 2**3

    def test_phase_difference_expansion_reproduces_delta(self):
        """sum_t A_{l,t} chi_{f_t}(z) must reproduce Delta_l(z) and be real."""
        d, n = 3, 2
        dims = _dims_to_numpy(d, n)
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        thetas = np.array([0.4, -0.3, 0.25])
        l = np.array([1, 2])

        char_data = _build_character_expansion(generators, np.arange(3), dims)
        amps = np.array(
            _character_amplitudes(
                jnp.array(thetas), char_data, jnp.array([l], dtype=jnp.float32), dims
            )
        )[0]
        freqs = np.array(char_data.freqs)

        for z in itertools.product(*(range(int(d_j)) for d_j in dims)):
            z = np.array(z)
            chars = np.exp(2j * np.pi * (freqs @ (z / dims)))
            reconstructed = np.sum(amps * chars)
            expected = _phase_difference(thetas, generators, l, z, dims)
            assert np.isclose(reconstructed.imag, 0.0, atol=1e-6), "Delta must be real"
            assert np.isclose(reconstructed.real, expected, atol=1e-6)


class TestControlVariateEstimator:
    """End-to-end behaviour of the control-variate estimator."""

    @pytest.mark.parametrize(
        "d, n, generators, thetas, l_vecs, m_vecs",
        [
            (3, 2, [[1, 0], [0, 2]], [0.2, -0.15], [[1, 1], [0, 1]], [[0, 1], [1, 0]]),
            (2, 2, [[1, 0], [1, 1]], [0.25, 0.1], [[1, 0], [1, 1]], [[0, 1], [0, 0]]),
            (4, 2, [[1, 2], [3, 1]], [0.15, 0.3], [[1, 0]], [[0, 0]]),
            ([2, 3], 2, [[1, 0], [0, 2], [1, 1]], [0.2, 0.1, -0.25], [[1, 1]], [[0, 1]]),
        ],
    )
    def test_unbiased_vs_exact(self, d, n, generators, thetas, l_vecs, m_vecs):
        """The CV estimator must remain unbiased: it agrees with the exact expval."""
        gen_arr = np.array(generators)
        l_arr = np.array(l_vecs)
        m_arr = np.array(m_vecs)

        config_plain = _make_cv_config(
            d, n, gen_arr, l_arr, m_arr, key=jax.random.PRNGKey(4), control_variate=False
        )
        exact_vals, *_ = _build_qudit_expval_func_exact(config_plain)(jnp.array(thetas))

        config_cv = _make_cv_config(
            d, n, gen_arr, l_arr, m_arr, key=jax.random.PRNGKey(4), control_variate=True
        )
        cv_vals, cv_cov = build_qudit_expval_func(config_cv)(jnp.array(thetas))

        assert cv_vals.shape == exact_vals.shape
        assert cv_cov.shape == exact_vals.shape + (2, 2)

        # The CV standard error is tiny at these angles, so use it to set the
        # tolerance rather than the plain 1/sqrt(s) scale.
        err_re = np.sqrt(np.array(cv_cov)[:, 0, 0])
        err_im = np.sqrt(np.array(cv_cov)[:, 1, 1])
        for i in range(len(l_arr)):
            tol = max(5.0 * float(err_re[i]), 5.0 * float(err_im[i]), 1e-5)
            assert np.isclose(
                cv_vals[i], exact_vals[i], atol=tol
            ), f"Observable {i}: cv={cv_vals[i]}, exact={exact_vals[i]}, tol={tol:.2e}"

    def test_unbiased_vs_matrix_reference_with_init_state(self):
        """CV with a sparse input state must match the dense matrix reference."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 1]])
        thetas = np.array([0.2, 0.15])
        l_vecs = np.array([[1, 2]])
        m_vecs = np.array([[2, 1]])
        elems = np.array([[0, 0], [1, 2]])
        amps = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)

        config = _make_cv_config(
            d,
            n,
            generators,
            l_vecs,
            m_vecs,
            n_samples=NUM_SAMPLES_INIT_STATE,
            key=jax.random.PRNGKey(42),
        )
        cv_vals, cv_cov = build_qudit_expval_func(config)(
            jnp.array(thetas),
            init_state_elems=jnp.array(elems),
            init_state_amps=jnp.array(amps),
        )

        ref = qudit_expectation_brute_force(
            n,
            d,
            generators,
            thetas,
            l_vecs[0],
            m_vecs[0],
            init_state_elems=elems,
            init_state_amps=amps,
        )
        tol = max(
            5.0 * float(np.sqrt(cv_cov[0, 0, 0])), 5.0 * float(np.sqrt(cv_cov[0, 1, 1])), 1e-4
        )
        assert np.isclose(
            cv_vals[0], ref, atol=tol
        ), f"cv={cv_vals[0]}, matrix reference={ref}, tol={tol:.2e}"

    def test_reduces_variance_at_small_angles(self):
        """At small angles the CV standard error must be far below the plain one."""
        d, n = 3, 4
        generators = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [1, 1, 0, 0], [0, 0, 1, 2]])
        thetas = np.array([0.05, -0.04, 0.03, 0.02])
        l_vecs = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]])
        m_vecs = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])

        errs = {}
        for cv_flag in (False, True):
            config = _make_cv_config(
                d,
                n,
                generators,
                l_vecs,
                m_vecs,
                n_samples=20000,
                key=jax.random.PRNGKey(8),
                control_variate=cv_flag,
            )
            _, cov = build_qudit_expval_func(config)(jnp.array(thetas))
            errs[cv_flag] = np.sqrt(np.array(cov)[:, 0, 0] + np.array(cov)[:, 1, 1])

        # An order-2 control at ||theta|| ~ 0.05 should buy at least an order of magnitude.
        assert np.all(
            errs[True] < errs[False] / 10.0
        ), f"plain se={errs[False]}, cv se={errs[True]}"

    def test_variance_scaling_is_higher_order(self):
        """CV standard error must fall faster than the plain estimator as theta -> 0.

        The residual y - y~ is O(||theta||^3), so shrinking theta by 10x should
        shrink the CV standard error by far more than the (constant) plain one.
        """
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        base = np.array([0.9, -1.1, 0.7])
        l_vecs = np.array([[1, 1]])
        m_vecs = np.array([[0, 1]])

        def se(scale, cv_flag):
            config = _make_cv_config(
                d,
                n,
                generators,
                l_vecs,
                m_vecs,
                n_samples=20000,
                key=jax.random.PRNGKey(5),
                control_variate=cv_flag,
            )
            _, cov = build_qudit_expval_func(config)(jnp.array(base * scale))
            return float(np.sqrt(np.array(cov)[0, 0, 0] + np.array(cov)[0, 1, 1]))

        cv_ratio = se(0.1, True) / se(0.01, True)
        plain_ratio = se(0.1, False) / se(0.01, False)

        # Plain se is essentially theta-independent; CV se shrinks by ~10^3.
        assert plain_ratio < 5.0, f"plain se should be flat in theta, ratio={plain_ratio}"
        assert cv_ratio > 100.0, f"cv se should fall steeply, ratio={cv_ratio}"

    def test_exact_at_zero_shift(self):
        """For l = 0 the control equals the integrand, so the variance collapses."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        thetas = np.array([0.4, -0.3, 0.25])
        l_vecs = np.zeros((1, n), dtype=int)
        m_vecs = np.zeros((1, n), dtype=int)

        config = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, n_samples=5000, key=jax.random.PRNGKey(1)
        )
        vals, cov = build_qudit_expval_func(config)(jnp.array(thetas))

        # l = m = 0 is the identity observable, whose expectation is exactly 1.
        assert np.isclose(vals[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(np.array(cov)[0], 0.0, atol=1e-12)

    def test_integrand_matches_brute_force(self):
        """The per-sample CV integrand must match its brute-force definition."""
        d, n = 3, 2
        dims = _dims_to_numpy(d, n)
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        thetas = np.array([0.4, -0.3, 0.25])
        l_arr = np.array([[1, 2], [0, 1]])
        m_arr = np.array([[0, 1], [2, 0]])

        # Enumerate the whole group as the "sample" set so every z is covered.
        all_z = np.array(list(itertools.product(*(range(int(d_j)) for d_j in dims))), dtype=int)

        obs_pm = np.array(
            [[_observable_phase(l, m, z, dims) for z in all_z] for l, m in zip(l_arr, m_arr)]
        )
        deltas = np.array(
            [[_phase_difference(thetas, generators, l, z, dims) for z in all_z] for l in l_arr]
        )

        got = _control_variate_integrand(jnp.array(obs_pm), jnp.array(deltas), None)
        want = np.array(
            [
                [
                    control_variate_integrand_brute_force(
                        thetas, generators, l, m, z, dims, None, None
                    )
                    for z in all_z
                ]
                for l, m in zip(l_arr, m_arr)
            ]
        )
        np.testing.assert_allclose(np.array(got), want, atol=1e-6)

    def test_control_variate_off_matches_original(self):
        """control_variate=False must leave the plain estimator untouched."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.5, 0.2])
        l_vecs = np.array([[1, 1], [0, 1]])
        m_vecs = np.array([[0, 1], [1, 0]])

        config_default, params = _make_config_one_param_per_gate(
            d, n, generators, thetas, l_vecs, m_vecs, key=jax.random.PRNGKey(3)
        )
        config_explicit = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, key=jax.random.PRNGKey(3), control_variate=False
        )

        vals_a, cov_a = build_qudit_expval_func(config_default)(jnp.array(params))
        vals_b, cov_b = build_qudit_expval_func(config_explicit)(jnp.array(params))

        np.testing.assert_array_equal(np.array(vals_a), np.array(vals_b))
        np.testing.assert_array_equal(np.array(cov_a), np.array(cov_b))

    def test_covariance_is_valid(self):
        """The CV covariance must stay symmetric and positive semi-definite."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.3, 0.2])
        l_vecs = np.array([[1, 1], [0, 1]])
        m_vecs = np.array([[0, 1], [1, 0]])

        config = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, n_samples=5000, key=jax.random.PRNGKey(11)
        )
        _, cov, mean_y_sq = build_qudit_expval_func(config)(
            jnp.array(thetas), return_mean_y_sq=True
        )

        np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-9)
        assert np.all(np.array(cov)[:, 0, 0] >= 0)
        assert np.all(np.array(cov)[:, 1, 1] >= 0)
        dets = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        assert np.all(np.array(dets) >= -1e-12)
        # mean_y_sq is taken from the raw integrand, so unit-modulus still holds.
        np.testing.assert_allclose(mean_y_sq, np.ones_like(mean_y_sq), atol=1e-6)

    def test_observables_override(self):
        """Runtime observable overrides must work and stay unbiased under CV."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.2, 0.15])
        override_l = np.array([[1, 0], [0, 1], [1, 1]])
        override_m = np.array([[0, 0], [1, 0], [0, 1]])

        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates={i: [list(g)] for i, g in enumerate(generators)},
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(7),
            control_variate=True,
        )
        cv_vals, cv_cov = build_qudit_expval_func(config)(
            jnp.array(thetas), observables=(override_l, override_m)
        )

        exact_config = _make_cv_config(
            d, n, generators, override_l, override_m, key=jax.random.PRNGKey(7)
        )
        exact_vals, *_ = _build_qudit_expval_func_exact(exact_config)(jnp.array(thetas))

        err_re = np.sqrt(np.array(cv_cov)[:, 0, 0])
        err_im = np.sqrt(np.array(cv_cov)[:, 1, 1])
        for i in range(len(override_l)):
            tol = max(5.0 * float(err_re[i]), 5.0 * float(err_im[i]), 1e-5)
            assert np.isclose(cv_vals[i], exact_vals[i], atol=tol)

    def test_key_and_n_samples_override(self):
        """Sampling overrides must be honoured on the control-variate path."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.2, 0.15])
        l_vecs = np.array([[1, 1]])
        m_vecs = np.array([[0, 1]])

        config = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, n_samples=2000, key=jax.random.PRNGKey(0)
        )
        fn = build_qudit_expval_func(config)
        _, cov_lo = fn(jnp.array(thetas), n_samples=1000, key=jax.random.PRNGKey(1))
        _, cov_hi = fn(jnp.array(thetas), n_samples=100_000, key=jax.random.PRNGKey(1))

        assert np.all(np.array(cov_hi)[:, 0, 0] < np.array(cov_lo)[:, 0, 0])
        assert np.all(np.array(cov_hi)[:, 1, 1] < np.array(cov_lo)[:, 1, 1])

    def test_empty_gates(self):
        """A gate set with no generators yields an empty expansion, not a crash."""
        d, n = 3, 2
        # Mix m = 0 and m != 0 rows so the check is not vacuous.
        l_vecs = np.array([[1, 0], [1, 2]])
        m_vecs = np.array([[0, 0], [1, 0]])
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates={},
            observables=(l_vecs, m_vecs),
            n_samples=2000,
            key=jax.random.PRNGKey(99),
            control_variate=True,
        )
        vals, cov = build_qudit_expval_func(config)(jnp.array([]))

        # With no gates Delta = 0, so the control is exact and the estimator must
        # reproduce the exact expectation with (numerically) zero variance.
        exact_vals, *_ = _build_qudit_expval_func_exact(config)(jnp.array([]))
        np.testing.assert_allclose(np.array(vals), np.array(exact_vals), atol=1e-6)
        np.testing.assert_allclose(np.array(cov), 0.0, atol=1e-12)

    def test_all_weight_zero_gates(self):
        """Generators with empty support leave the expansion empty but valid."""
        d, n = 3, 2
        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates={0: [[0, 0]]},
            observables=(np.array([[1, 0]]), np.array([[0, 0]])),
            n_samples=2000,
            key=jax.random.PRNGKey(0),
            control_variate=True,
        )
        vals, cov = build_qudit_expval_func(config)(jnp.array([0.3]))
        assert np.all(np.isfinite(np.array(vals)))
        assert np.all(np.isfinite(np.array(cov)))

    def test_parameter_broadcasting(self):
        """Gates sharing a parameter index must share theta in the expansion too."""
        d, n = 3, 3
        gates = {0: [[1, 0, 0], [0, 1, 0]], 1: [[0, 0, 2]]}
        thetas = np.array([0.2, 0.15])
        l_vecs = np.array([[1, 1, 0]])
        m_vecs = np.array([[0, 0, 1]])

        config = QuditCircuitConfig(
            d=d,
            n_qudits=n,
            gates=gates,
            observables=(l_vecs, m_vecs),
            n_samples=NUM_SAMPLES,
            key=jax.random.PRNGKey(55),
            control_variate=True,
        )
        exact_vals, *_ = _build_qudit_expval_func_exact(config)(jnp.array(thetas))
        cv_vals, cv_cov = build_qudit_expval_func(config)(jnp.array(thetas))

        tol = max(
            5.0 * float(np.sqrt(cv_cov[0, 0, 0])), 5.0 * float(np.sqrt(cv_cov[0, 1, 1])), 1e-5
        )
        assert np.isclose(cv_vals[0], exact_vals[0], atol=tol)

    def test_jit_compatible(self):
        """The CV estimator must be JIT-compilable and agree with eager mode."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.2, 0.15])
        l_vecs = np.array([[1, 1], [0, 1]])
        m_vecs = np.array([[0, 1], [1, 0]])

        config = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, n_samples=5000, key=jax.random.PRNGKey(42)
        )
        fn = build_qudit_expval_func(config)
        vals_jit, cov_jit = jax.jit(fn)(jnp.array(thetas))
        vals_eager, cov_eager = fn(jnp.array(thetas))

        np.testing.assert_allclose(vals_jit, vals_eager, atol=1e-6)
        np.testing.assert_allclose(cov_jit, cov_eager, atol=1e-6)

    def test_differentiable(self):
        """Gradients through the CV estimator must be finite and match finite differences."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        thetas = np.array([0.2, 0.15])
        l_vecs = np.array([[1, 1]])
        m_vecs = np.array([[0, 1]])

        config = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, n_samples=20000, key=jax.random.PRNGKey(42)
        )
        fn = build_qudit_expval_func(config)

        def loss(p):
            vals, *_ = fn(p)
            return jnp.real(jnp.sum(vals))

        grads = jax.grad(loss)(jnp.array(thetas))
        assert grads.shape == (len(thetas),)
        assert np.all(np.isfinite(np.array(grads)))

        # Same PRNG key means the sampling noise cancels in the difference.
        eps = 1e-3
        p = np.array(thetas, dtype=float)
        fd = np.zeros_like(p)
        for k in range(len(p)):
            p_plus, p_minus = p.copy(), p.copy()
            p_plus[k] += eps
            p_minus[k] -= eps
            fd[k] = (loss(jnp.array(p_plus)) - loss(jnp.array(p_minus))) / (2 * eps)

        np.testing.assert_allclose(np.array(grads), fd, atol=1e-4)

    def test_gradient_finite_when_control_is_degenerate(self):
        """l = 0 makes the control constant; gradients must stay finite.

        A naive jnp.where(var > 0, -cov / var, 0) still evaluates the division on
        the untaken branch under reverse-mode AD and returns NaN here, so this is
        a regression guard for that specific failure.
        """
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2], [1, 1]])
        thetas = np.array([0.2, -0.15, 0.1])
        # First row is the degenerate l = 0 observable, second is ordinary.
        l_vecs = np.array([[0, 0], [1, 2]])
        m_vecs = np.array([[0, 0], [0, 1]])

        config = _make_cv_config(
            d, n, generators, l_vecs, m_vecs, n_samples=2000, key=jax.random.PRNGKey(2)
        )
        fn = build_qudit_expval_func(config)

        def loss(p):
            vals, *_ = fn(p)
            return jnp.sum(jnp.abs(vals) ** 2)

        grads = jax.grad(loss)(jnp.array(thetas))
        assert np.all(np.isfinite(np.array(grads))), f"non-finite gradients: {grads}"

    def test_only_zero_shift_observable_gradient(self):
        """An all-degenerate observable batch must still produce finite gradients."""
        d, n = 3, 2
        generators = np.array([[1, 0], [0, 2]])
        config = _make_cv_config(
            d,
            n,
            generators,
            np.zeros((1, n), dtype=int),
            np.zeros((1, n), dtype=int),
            n_samples=1000,
            key=jax.random.PRNGKey(2),
        )
        fn = build_qudit_expval_func(config)
        grads = jax.grad(lambda p: jnp.sum(jnp.abs(fn(p)[0]) ** 2))(jnp.array([0.2, 0.1]))
        assert np.all(np.isfinite(np.array(grads)))

    def test_phase_fn_incompatible(self):
        """Phase layers and control variates must not be combined silently."""
        with pytest.raises(ValueError, match="not compatible"):
            build_qudit_expval_func(
                QuditCircuitConfig(
                    d=3,
                    n_qudits=2,
                    gates={0: [[1, 0]]},
                    observables=(np.array([[1, 0]]), np.array([[0, 0]])),
                    n_samples=100,
                    key=jax.random.PRNGKey(0),
                    control_variate=True,
                    phase_fn=lambda params, z: jnp.sum(params * z.astype(jnp.float32)),
                )
            )

    def test_missing_observables_raises(self):
        """Omitting observables entirely must still raise on the CV path."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
            control_variate=True,
        )
        with pytest.raises(ValueError, match="No observables specified"):
            build_qudit_expval_func(config)(jnp.array([0.3]))
