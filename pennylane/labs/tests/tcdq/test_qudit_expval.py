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
"""
Tests for the qudit IQP expectation value function.
"""

import itertools
from functools import reduce

import numpy as np
import pytest
from scipy.linalg import expm

import pennylane as qp
from pennylane.labs.tcdq.expval_functions import (
    QuditCircuitConfig,
    _parse_qudit_generator_dict,
    build_qudit_expval_func,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

NUM_SAMPLES = 10000


def _build_qudit_expval_func_exact(config):
    """Exact (brute-force sum over all basis states) qudit IQP expectation value factory."""
    generators, param_map = _parse_qudit_generator_dict(config.gates, config.n_qudits)

    all_states = jnp.array(
        list(itertools.product(range(config.d), repeat=config.n_qudits)),
        dtype=jnp.int32,
    )  # (d^n, n_qudits)

    l_vecs = jnp.array(config.observables[0], dtype=jnp.int32)
    m_vecs = jnp.array(config.observables[1], dtype=jnp.int32)

    d = config.d
    d_n = d**config.n_qudits
    n_obs = l_vecs.shape[0]

    g_f = generators.astype(jnp.float32)
    s_f = all_states.astype(jnp.float32)
    outer = g_f[:, :, jnp.newaxis] * s_f.T[jnp.newaxis, :, :]
    per_qudit_vals = jnp.sqrt(2.0) * jnp.cos(2 * jnp.pi * outer / d + jnp.pi / 4)
    val_k = jnp.prod(per_qudit_vals, axis=1)  # (n_gates, d^n)

    def qudit_expval(gates_params):
        expanded_params = jnp.asarray(gates_params)[param_map]

        def single_obs(l, m):
            k_shifted = (all_states - l[jnp.newaxis, :]) % d
            ks_f = k_shifted.astype(jnp.float32)
            m_f = m.astype(jnp.float32)
            l_f = l.astype(jnp.float32)
            obs_phase_scalar = jnp.exp(1j * jnp.pi * jnp.dot(m_f, l_f) / d)
            obs_state_phase = jnp.exp(1j * 2 * jnp.pi * (ks_f @ m_f) / d)
            obs_phase = obs_phase_scalar * obs_state_phase
            outer_shifted = g_f[:, :, jnp.newaxis] * ks_f.T[jnp.newaxis, :, :]
            val_k_shifted = jnp.prod(
                jnp.sqrt(2.0) * jnp.cos(2 * jnp.pi * outer_shifted / d + jnp.pi / 4),
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
    """X|j> = |j+1 mod d>"""
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    return X


def _clock_operator(d):
    """Z|j> = exp(2*pi*i*j/d)|j>"""
    return np.diag([np.exp(2j * np.pi * j / d) for j in range(d)])


def _displacement_operator(l, m, d):
    """O(l, m) = Z^l X^m exp(-i*pi*l*m/d)  [eqn 36]"""
    Z = _clock_operator(d)
    X = _shift_operator(d)
    Z_l = np.linalg.matrix_power(Z, int(l % d))
    X_m = np.linalg.matrix_power(X, int(m % d))
    phase = np.exp(-1j * np.pi * l * m / d)
    return phase * (Z_l @ X_m)


def _hermitian_observable(l, m, d):
    """Q(l, m) = chi * O(l, m) + chi^* O^dag(l, m), with chi = (1+i)/2"""
    chi = (1 + 1j) / 2
    O = _displacement_operator(l, m, d)
    return chi * O + np.conj(chi) * O.conj().T


def _dft_matrix(d):
    """Discrete Fourier transform matrix for Z_d."""
    j = np.arange(d)
    return np.exp(2j * np.pi * np.outer(j, j) / d) / np.sqrt(d)


def _kron_n(mats):
    """Tensor product of a list of matrices."""
    return reduce(np.kron, mats)


def qudit_expectation_brute_force(
    n, d, gates, thetas, l_vec, m_vec, init_state_elems=None, init_state_amps=None
):
    """Brute-force computation of qudit IQP-type expectation values.

    Computes  <psi_in| U^dag(theta) O(l, m) U(theta) |psi_in>   where

        U(theta) = (F^{x n})^dag  D(theta)  F^{x n}           [eqn 44]
        D(theta) = prod_g exp(i theta_g Q_g)                   [eqn 43]
        O(l, m)  = bigotimes_i O(l_i, m_i)                     [eqn 46]
        O(l, m)  = Z^l X^m exp(-i pi l m / d)                  [eqn 36]

    When init_state_elems and init_state_amps are None, the initial state
    is |0>.

    Args:
        n:      Number of qudits.
        d:      Local dimension (cyclic group Z_d).
        gates:  Sequence of gate vectors g in Z_d^n.  Each g is a
                length-n sequence of integers in {0, ..., d-1}.
        thetas: Sequence of rotation angles, one per gate.
        l_vec:  Length-n sequence specifying l_i for the observable.
        m_vec:  Length-n sequence specifying m_i for the observable.
        init_state_elems: Optional array of shape (N, n) with dit-string
                support elements in {0, ..., d-1}.
        init_state_amps:  Optional array of shape (N,) with complex
                amplitudes for each support element.

    Returns:
        Complex expectation value <O(l, m)>.
    """
    dim = d**n

    # F^{otimes n}
    F1 = _dft_matrix(d)
    F_n = _kron_n([F1] * n)

    # D(theta) = prod_g exp(i theta_g Q_g)  [eqn 43]
    D = np.eye(dim, dtype=complex)
    for g, theta in zip(gates, thetas):
        Q_g = _kron_n([_hermitian_observable(g[i], 0, d) for i in range(n)])
        D = expm(1j * theta * Q_g) @ D

    # U(theta) = (F^{otimes n})^dag  D(theta)  F^{otimes n}  [eqn 44]
    U = F_n.conj().T @ D @ F_n

    # O(l, m) = bigotimes_i O(l_i, m_i)  [eqn 36, 46]
    O = _kron_n([_displacement_operator(l_vec[i], m_vec[i], d) for i in range(n)])

    if init_state_elems is None or init_state_amps is None:
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0
    else:
        psi0 = np.zeros(dim, dtype=complex)
        for elem, amp in zip(init_state_elems, init_state_amps):
            idx = sum(int(e) * d ** (n - 1 - i) for i, e in enumerate(elem))
            psi0[idx] += amp

    # <psi_in| U^dag O U |psi_in>
    U_psi = U @ psi0
    return U_psi.conj() @ O @ U_psi


def _pennylane_qubit_expval(generators_list, thetas_list, l_vec, m_vec):
    """
    Compute <D(l, m)> via PennyLane for d=2 (qubit) circuits.

    Returns the real expectation value.
    """
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
                "PennyLane gives {pl_val:.8f}"
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
        """Batched MC must agree with the exact qudit expval within sampling noise."""
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
    """Test that the batched MC version matches the brute-force matrix reference."""

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
        """Batched MC must agree with the dense matrix reference."""
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


class TestQuditExpvalBatchedEdgeCases:
    """Edge cases and structural tests for the batched MC function."""

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
    """Test batched MC with general initial states against brute-force reference."""

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
        """Batched MC with init state must agree with dense matrix reference."""
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
        # holds to float precision without any MC noise.
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

        # Finite-difference check: same PRNG key means same samples, so MC noise
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
