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
# pylint: disable=too-many-arguments,too-few-public-methods
"""Reference and regression tests for the qudit MMD loss."""

import itertools

import numpy as np
import pytest

from pennylane.labs.tcdq.qudit_expval_functions import QuditCircuitConfig
from pennylane.labs.tcdq.qudit_mmd_loss import (
    QuditMMDConfig,
    _complete_marginal_probs,
    _cycle_marginal_probs,
    _sample_fourier_indices,
    _unbiased_mmd_squared,
    build_qudit_mmd_loss,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)


def _call_qudit_mmd_loss(
    params,
    circuit_config,
    mmd_config,
    target_data,
    key=None,
):
    """Construct the loss function and evaluate it once."""
    loss_fn = build_qudit_mmd_loss(circuit_config, mmd_config)
    return loss_fn(params, target_data, key)


def _qudit_phi_g_z(gen, z, d):
    """Reference implementation of the per-gate eigenvalue factor."""
    n = len(z)
    val = 1.0
    for j in range(n):
        val *= np.sqrt(2.0) * np.cos(2.0 * np.pi * z[j] * gen[j] / d + np.pi / 4.0)
    return val


def _flatten_gates(gates, n_qudits):
    """Flatten the gate dictionary into generators plus parameter indices."""
    flat_gens, flat_pidx = [], []
    for pidx in sorted(gates.keys()):
        for g in gates[pidx]:
            assert len(g) == n_qudits
            flat_gens.append(np.array(g, dtype=int))
            flat_pidx.append(pidx)
    return flat_gens, flat_pidx


def _qudit_circuit_probs(gates, params, n_qudits, d):
    """Return the exact circuit output distribution by exhaustive enumeration."""
    params = np.asarray(params, dtype=float)
    flat_gens, flat_pidx = _flatten_gates(gates, n_qudits)
    omega = np.exp(2j * np.pi / d)
    all_states = list(itertools.product(range(d), repeat=n_qudits))
    n_states = d**n_qudits

    gammas = np.zeros(n_states, dtype=complex)
    for k, z in enumerate(all_states):
        z_arr = np.array(z)
        phase = sum(
            params[pidx] * _qudit_phi_g_z(g, z_arr, d) for g, pidx in zip(flat_gens, flat_pidx)
        )
        gammas[k] = np.exp(1j * phase)

    probs = np.zeros(n_states)
    for i, x in enumerate(all_states):
        x_arr = np.array(x)
        amp = sum(
            gammas[k] * omega ** (-np.dot(np.array(all_states[k]), x_arr)) for k in range(n_states)
        )
        amp /= n_states
        probs[i] = np.abs(amp) ** 2

    return probs, all_states


def _qudit_expval_exact(gates, params, l_vec, n_qudits, d):
    """Return one exact observable moment by summing over all basis states."""
    params = np.asarray(params, dtype=float)
    flat_gens, flat_pidx = _flatten_gates(gates, n_qudits)
    all_states = list(itertools.product(range(d), repeat=n_qudits))
    n_states = d**n_qudits
    l_arr = np.array(l_vec)

    total = 0.0 + 0j
    for z_tuple in all_states:
        z = np.array(z_tuple)
        z_minus_l = (z - l_arr) % d
        delta = sum(
            params[pidx] * (_qudit_phi_g_z(g, z, d) - _qudit_phi_g_z(g, z_minus_l, d))
            for g, pidx in zip(flat_gens, flat_pidx)
        )
        total += np.exp(1j * delta)
    return total / n_states


def _qudit_kernel_1d(delta, marginal, d):
    """Return the one-site kernel value used by the exact MMD reference."""
    omega = np.exp(2j * np.pi / d)
    # index is k in the notes
    return np.real(sum(marginal[index] * omega ** (index * delta) for index in range(d)))


def _qudit_kernel_matrix(X1, X2, marginal, d):
    """Build the product kernel matrix used by the exact MMD reference."""
    n1, n2 = len(X1), len(X2)
    K = np.ones((n1, n2))
    for i in range(n1):
        for j in range(n2):
            for k in range(len(X1[i])):
                K[i, j] *= _qudit_kernel_1d((X1[i][k] - X2[j][k]) % d, marginal, d)
    return K


def _exact_qudit_mmd2_kernel(model_probs, data, d, bandwidth, graph_type, n_qudits):
    """Compute exact MMD² from dense kernel matrices for small systems."""
    all_states = np.array(list(itertools.product(range(d), repeat=n_qudits)))
    marginal = (
        _cycle_marginal_probs(d, bandwidth)
        if graph_type == "cycle"
        else _complete_marginal_probs(d, bandwidth)
    )
    dq = np.asarray(data, dtype=float)
    m = len(dq)

    K_pp = _qudit_kernel_matrix(all_states, all_states, marginal, d)
    pp = model_probs @ K_pp @ model_probs

    K_pq = _qudit_kernel_matrix(all_states, dq, marginal, d)
    pq = 2.0 * np.sum(model_probs[:, None] * K_pq) / m

    K_qq = _qudit_kernel_matrix(dq, dq, marginal, d)
    qq = (np.sum(K_qq) - np.trace(K_qq)) / (m * (m - 1))

    return float(pp - pq + qq)


def _exact_qudit_mmd2_operators(gates, params, data, d, bandwidth, graph_type, n_qudits):
    """Compute exact MMD² by summing the observable expansion explicitly."""
    marginal = (
        _cycle_marginal_probs(d, bandwidth)
        if graph_type == "cycle"
        else _complete_marginal_probs(d, bandwidth)
    )
    omega = np.exp(2j * np.pi / d)
    all_l = list(itertools.product(range(d), repeat=n_qudits))
    m = len(data)

    total = 0.0
    for l_tuple in all_l:
        l_arr = np.array(l_tuple)
        w = np.prod([marginal[li] for li in l_arr])

        mu_q = _qudit_expval_exact(gates, params, l_arr, n_qudits, d)
        mu_p = np.mean([omega ** np.dot(l_arr, data[i]) for i in range(m)])

        pp_l = (m * np.abs(mu_p) ** 2 - 1.0) / (m - 1)
        total += w * (np.abs(mu_q) ** 2 - 2.0 * np.real(np.conj(mu_p) * mu_q) + pp_l)

    return float(np.real(total))


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


class TestMarginalProbs:
    """Verify graph-spectral marginal distributions."""

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_cycle_sums_to_one(self, d):
        """Cycle marginal probabilities must sum to 1."""
        p = _cycle_marginal_probs(d, t=1.0)
        assert np.isclose(p.sum(), 1.0)

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_complete_sums_to_one(self, d):
        """Complete-graph marginal probabilities must sum to 1."""
        p = _complete_marginal_probs(d, t=1.0)
        assert np.isclose(p.sum(), 1.0)

    def test_cycle_symmetric(self):
        """P_1(k) = P_1(d - k) for the cycle graph."""
        d = 5
        p = _cycle_marginal_probs(d, t=0.8)
        for k in range(d):
            assert np.isclose(p[k], p[(d - k) % d])

    def test_complete_uniform_nonzero(self):
        """K_d assigns equal weight to all k != 0."""
        d = 4
        p = _complete_marginal_probs(d, t=0.5)
        assert np.allclose(p[1:], p[1])

    def test_large_bandwidth_concentrates_on_zero(self):
        """Large t should make P_1(0) dominate."""
        p_cycle = _cycle_marginal_probs(5, t=100.0)
        p_comp = _complete_marginal_probs(5, t=100.0)
        assert p_cycle[0] > 0.99
        assert p_comp[0] > 0.99

    def test_small_bandwidth_approaches_uniform(self):
        """t -> 0 should make the distribution approach uniform."""
        p_cycle = _cycle_marginal_probs(4, t=1e-6)
        p_comp = _complete_marginal_probs(4, t=1e-6)
        assert np.allclose(p_cycle, 0.25, atol=1e-4)
        assert np.allclose(p_comp, 0.25, atol=1e-4)


class TestObservableSampling:
    """Verify observable sampling mechanics."""

    def test_values_in_range(self):
        """Sampled Fourier indices must lie in [0, d)."""
        obs = _sample_fourier_indices(jax.random.PRNGKey(0), 50, 3, 5, 1.0, "cycle", (0, 1, 2))
        assert obs.shape == (50, 3)
        assert jnp.all(obs >= 0) and jnp.all(obs < 5)

    def test_non_visible_wires_are_zero(self):
        """Wires outside the visible set must remain zero."""
        obs = _sample_fourier_indices(jax.random.PRNGKey(1), 100, 4, 3, 1.0, "cycle", (0, 2))
        assert jnp.all(obs[:, 1] == 0)
        assert jnp.all(obs[:, 3] == 0)

    def test_invalid_graph_type_raises(self):
        """An unrecognised graph_type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown graph_type"):
            _sample_fourier_indices(jax.random.PRNGKey(0), 10, 2, 3, 1.0, "unknown_graph", (0, 1))

    @pytest.mark.parametrize("graph_type", ["cycle", "complete"])
    def test_empirical_marginal_close_to_exact(self, graph_type):
        """With many samples the empirical site marginal should match theory."""
        d, t, n_samples = 4, 0.5, 50_000
        obs = _sample_fourier_indices(jax.random.PRNGKey(42), n_samples, 1, d, t, graph_type, (0,))
        counts = np.bincount(np.array(obs[:, 0]), minlength=d)
        empirical = counts / n_samples
        if graph_type == "cycle":
            expected = _cycle_marginal_probs(d, t)
        else:
            expected = _complete_marginal_probs(d, t)
        assert np.allclose(empirical, expected, atol=0.02)


class TestUnbiasedMmdSquared:
    """Tests for the low-level unbiased MMD² estimator."""

    def test_matching_moments_near_zero(self):
        """When model and data moments agree the loss is ≈ 0."""
        rng = np.random.default_rng(0)
        n_ops, n_q, d = 30, 2, 3
        m = 50
        n_samples = 1000
        data = jnp.array(rng.integers(0, d, (m, n_q)))
        l_obs = jnp.array(rng.integers(0, d, (n_ops, n_q)))

        inner = l_obs.astype(jnp.float64) @ data.astype(jnp.float64).T
        data_moments = jnp.mean(jnp.exp(2j * jnp.pi * inner / d), axis=1)

        result = _unbiased_mmd_squared(
            data_moments,
            jnp.ones(n_ops),
            data,
            l_obs,
            d,
            n_samples=n_samples,
            sqrt_loss=False,
        )
        assert abs(float(jnp.real(result))) < 0.05

    def test_sqrt_loss_flag(self):
        """sqrt_loss=True returns sqrt(|result|)."""
        rng = np.random.default_rng(1)
        n_ops, n_q, d = 20, 2, 3
        n_samples = 1000
        data = jnp.array(rng.integers(0, d, (40, n_q)))
        l_obs = jnp.array(rng.integers(0, d, (n_ops, n_q)))
        fake_model = jnp.array(rng.normal(0, 0.3, n_ops) + 0.1j * rng.normal(0, 0.3, n_ops))
        mean_y_sq = jnp.ones(n_ops)

        val = _unbiased_mmd_squared(fake_model, mean_y_sq, data, l_obs, d, n_samples, False)
        sqr = _unbiased_mmd_squared(fake_model, mean_y_sq, data, l_obs, d, n_samples, True)
        assert np.isclose(float(sqr), np.sqrt(abs(float(val))), atol=1e-7)

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        rng = np.random.default_rng(3)
        n_ops, n_q, d = 15, 2, 3
        n_samples = 500
        data = jnp.array(rng.integers(0, d, (30, n_q)))
        l_obs = jnp.array(rng.integers(0, d, (n_ops, n_q)))
        model = jnp.array(rng.normal(0, 0.4, n_ops) + 0.1j * rng.normal(0, 0.4, n_ops))
        mean_y_sq = jnp.ones(n_ops)

        r1 = _unbiased_mmd_squared(model, mean_y_sq, data, l_obs, d, n_samples, False)
        r2 = _unbiased_mmd_squared(model, mean_y_sq, data, l_obs, d, n_samples, False)
        assert np.isclose(float(r1), float(r2), atol=1e-10)


class TestExactQuditMMDConsistency:
    """Validate kernel-matrix and operator-decomposition exact MMD^2 agree."""

    @pytest.mark.parametrize(
        "gates, params, n_qudits, d, biases, graph_type",
        [
            (
                {0: [[1, 0]], 1: [[0, 1]], 2: [[1, 1]]},
                [0.3, 0.5, 0.2],
                2,
                3,
                [0.5, 0.5],
                "cycle",
            ),
            (
                {0: [[1, 0]], 1: [[0, 2]]},
                [0.4, 0.7],
                2,
                3,
                [0.3, 0.7],
                "complete",
            ),
            (
                {0: [[1, 0]], 1: [[0, 1]], 2: [[2, 1]]},
                [0.1, 0.6, 0.9],
                2,
                4,
                [0.4, 0.6],
                "cycle",
            ),
        ],
    )
    def test_kernel_vs_operator(self, gates, params, n_qudits, d, biases, graph_type):
        """Kernel-matrix MMD^2 should equal operator-decomposition MMD^2."""
        rng = np.random.default_rng(42)
        data = np.stack([rng.integers(0, d, 50) for _ in biases], axis=1)
        bandwidth = 0.5

        probs, _ = _qudit_circuit_probs(gates, params, n_qudits, d)
        mmd_k = _exact_qudit_mmd2_kernel(probs, data, d, bandwidth, graph_type, n_qudits)
        mmd_o = _exact_qudit_mmd2_operators(gates, params, data, d, bandwidth, graph_type, n_qudits)

        assert np.isclose(mmd_k, mmd_o, atol=1e-8), f"kernel={mmd_k:.10f}, operator={mmd_o:.10f}"


class TestQuditMMDLossAPI:
    """API-level tests for one-shot qudit MMD loss evaluation."""

    def test_raises_n_samples_le_one(self):
        """n_samples <= 1 should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=1,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5)
        with pytest.raises(ValueError, match="n_samples must be greater than 1"):
            _call_qudit_mmd_loss(jnp.array([0.1]), config, mmd_cfg, jnp.array([[0, 0]]))

    def test_raises_target_data_too_few_samples(self):
        """Target data with fewer than 2 samples should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5)
        with pytest.raises(ValueError, match="target_data must have at least 2 samples"):
            _call_qudit_mmd_loss(jnp.array([0.1]), config, mmd_cfg, jnp.array([[0, 0]]))

    def test_raises_n_ops_zero(self):
        """n_ops=0 should raise ValueError at build time."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=0)
        with pytest.raises(ValueError, match="n_ops must be at least 1"):
            build_qudit_mmd_loss(config, mmd_cfg)

    def test_raises_empty_bandwidth(self):
        """Empty bandwidth list should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=[], n_ops=5)
        with pytest.raises(ValueError, match="bandwidth must not be empty"):
            build_qudit_mmd_loss(config, mmd_cfg)

    def test_raises_wire_out_of_range(self):
        """Wire index beyond n_qudits should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5, wires=[0, 5])
        with pytest.raises(ValueError, match="Wire index 5 out of range"):
            build_qudit_mmd_loss(config, mmd_cfg)

    def test_raises_duplicate_wires(self):
        """Duplicate wire indices should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=3,
            gates={0: [[1, 0, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5, wires=[0, 0])
        with pytest.raises(ValueError, match="wires must not contain duplicates"):
            build_qudit_mmd_loss(config, mmd_cfg)

    def test_raises_target_data_wrong_ndim(self):
        """Non-2D target data should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5)
        with pytest.raises(ValueError, match="target_data must be 2-D"):
            _call_qudit_mmd_loss(jnp.array([0.1]), config, mmd_cfg, jnp.array([0, 1, 2]))

    def test_raises_target_data_wrong_columns(self):
        """Target data with wrong number of columns should raise ValueError."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=3,
            gates={0: [[1, 0, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5, wires=[0, 2])
        data = jnp.array([[0, 1, 2], [1, 0, 2]])  # 3 cols, but 2 visible wires
        with pytest.raises(ValueError, match="columns but expected 2"):
            _call_qudit_mmd_loss(jnp.array([0.1]), config, mmd_cfg, data)

    def test_return_per_bandwidth_type_and_length(self):
        """return_per_bandwidth=True returns a list of length len(bandwidth)."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=[0.5, 1.0], n_ops=30, return_per_bandwidth=True)
        data = jnp.array([[0, 1], [1, 0], [0, 0]])
        res = _call_qudit_mmd_loss(jnp.array([0.1, 0.2]), config, mmd_cfg, data)
        assert isinstance(res, list) and len(res) == 2

    def test_multi_bandwidth_mean(self):
        """Scalar output equals the mean of per-bandwidth outputs."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]]},
            n_samples=200,
            key=jax.random.PRNGKey(42),
        )
        data = jnp.array([[0, 1], [1, 2], [2, 0], [1, 1]])
        bws = [0.5, 1.0, 2.0]
        params = jnp.array([0.1, 0.2])

        per = _call_qudit_mmd_loss(
            params,
            config,
            QuditMMDConfig(bandwidth=bws, n_ops=100, return_per_bandwidth=True),
            data,
        )
        avg = _call_qudit_mmd_loss(
            params,
            config,
            QuditMMDConfig(bandwidth=bws, n_ops=100, return_per_bandwidth=False),
            data,
        )
        expected = np.mean([float(v) for v in per])
        assert np.isclose(float(avg), expected, atol=1e-6)

    def test_single_bandwidth_returns_scalar(self):
        """A single float bandwidth produces a scalar output."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=20)
        data = jnp.array([[0, 1], [1, 0]])
        res = _call_qudit_mmd_loss(jnp.array([0.5]), config, mmd_cfg, data)
        assert res.shape == ()

    def test_deterministic_same_key(self):
        """Identical keys must yield bit-identical results."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]], 2: [[1, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(99),
        )
        kwargs = {
            "params": jnp.array([0.3, 0.5, 0.1]),
            "circuit_config": config,
            "mmd_config": QuditMMDConfig(bandwidth=1.0, n_ops=20),
            "target_data": jnp.array([[0, 1], [1, 0], [2, 2], [1, 1]]),
        }
        assert float(_call_qudit_mmd_loss(**kwargs)) == float(_call_qudit_mmd_loss(**kwargs))

    def test_different_keys_give_different_results(self):
        """Different PRNG keys should give different losses."""
        gates = {0: [[1, 0]], 1: [[0, 1]]}
        data = jnp.array([[0, 1], [1, 0], [2, 2], [1, 1]])
        params = jnp.array([0.3, 0.7])
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=20)

        r1 = _call_qudit_mmd_loss(
            params,
            QuditCircuitConfig(
                d=3,
                n_qudits=2,
                gates=gates,
                n_samples=100,
                key=jax.random.PRNGKey(0),
            ),
            mmd_cfg,
            data,
        )
        r2 = _call_qudit_mmd_loss(
            params,
            QuditCircuitConfig(
                d=3,
                n_qudits=2,
                gates=gates,
                n_samples=100,
                key=jax.random.PRNGKey(999),
            ),
            mmd_cfg,
            data,
        )
        assert float(r1) != float(r2)

    def test_mmd_loss_with_custom_init_state(self):
        """Verify one-shot evaluation handles custom initial states without error."""
        state_elems = jnp.array([[0, 0], [1, 1]])
        state_amps = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)])

        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(42),
            init_state_elems=state_elems,
            init_state_amps=state_amps,
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=10)
        data = jnp.array([[0, 0], [1, 1]])
        res = _call_qudit_mmd_loss(jnp.array([0.5]), config, mmd_cfg, data)
        assert res.shape == () and np.isfinite(float(res))


class TestQuditMMDLossStatistical:
    """Statistical Z-test validation of the qudit MMD estimator.

    For each parametrised circuit / dataset pair:
    1. Compute exact MMD^2 from brute-force probabilities and data.
    2. Collect many stochastic tcdq estimates using different PRNG keys.
    3. Verify |exact - mean_est| / SE < Z_THRESHOLD.
    """

    Z_THRESHOLD = 4.0
    N_TRIALS = 80
    N_OPS = 30
    N_SAMPLES = 200

    @pytest.mark.parametrize(
        "gates, params, n_qudits, d, n_data, graph_type",
        [
            # 2-qutrit, 3 gates, cycle
            (
                {0: [[1, 0]], 1: [[0, 1]], 2: [[1, 1]]},
                [0.37, 0.95, 0.73],
                2,
                3,
                80,
                "cycle",
            ),
            # 2-qutrit, 2 gates, complete
            (
                {0: [[1, 0]], 1: [[0, 2]]},
                [0.5, 0.3],
                2,
                3,
                60,
                "complete",
            ),
            # 2-qudit d=4, 3 gates, cycle
            (
                {0: [[1, 0]], 1: [[0, 1]], 2: [[2, 1]]},
                [0.2, 0.8, 0.4],
                2,
                4,
                80,
                "cycle",
            ),
        ],
    )
    def test_unbiased_z_test(self, gates, params, n_qudits, d, n_data, graph_type):
        """Statistical Z-test: mean MMD estimate must be close to exact value."""
        probs, _ = _qudit_circuit_probs(gates, params, n_qudits, d)
        rng = np.random.default_rng(42)
        X = np.stack([rng.integers(0, d, n_data) for _ in range(n_qudits)], axis=1)
        bandwidth = 0.5

        exact = _exact_qudit_mmd2_kernel(probs, X, d, bandwidth, graph_type, n_qudits)

        batch = min(40, n_data)
        mmd_cfg = QuditMMDConfig(bandwidth=bandwidth, n_ops=self.N_OPS, graph_type=graph_type)
        params_jnp = jnp.array(params)
        X_jnp = jnp.array(X)

        config = QuditCircuitConfig(
            d=d,
            n_qudits=n_qudits,
            gates=gates,
            n_samples=self.N_SAMPLES,
            key=jax.random.PRNGKey(0),
        )

        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)

        estimates = []
        master_key = jax.random.PRNGKey(42)
        for _ in range(self.N_TRIALS):
            master_key, loss_key, sample_key = jax.random.split(master_key, 3)
            idx = jax.random.choice(sample_key, n_data, shape=(batch,), replace=False)
            est = loss_fn(
                params=params_jnp,
                target_data=X_jnp[idx],
                key=loss_key,
            )
            estimates.append(float(est))

        estimates = np.array(estimates)
        mean_est = np.mean(estimates)
        se = np.std(estimates, ddof=1) / np.sqrt(self.N_TRIALS)

        z = abs(exact - mean_est) / se if se > 1e-15 else 0.0
        assert z < self.Z_THRESHOLD, (
            f"Z-test FAILED: z={z:.2f}, exact={exact:.6f}, " f"mean={mean_est:.6f}, se={se:.6f}"
        )

    def test_wires_subset_executes(self):
        """One-shot evaluation with a wires subset should run without error."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=3,
            gates={0: [[1, 0, 0]], 1: [[0, 1, 0]], 2: [[0, 0, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        rng = np.random.default_rng(0)
        data = jnp.array(rng.integers(0, 3, (30, 2)))
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=20, wires=[0, 2])
        res = _call_qudit_mmd_loss(jnp.array([0.1, 0.2, 0.3]), config, mmd_cfg, data)
        assert res.shape == () and np.isfinite(float(res))

    def test_sqrt_loss_positive(self):
        """sqrt_loss=True should produce a non-negative scalar."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(77),
        )
        rng = np.random.default_rng(0)
        data = jnp.array(rng.integers(0, 3, (50, 2)))
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=30, sqrt_loss=True)
        res = _call_qudit_mmd_loss(jnp.array([0.5, 0.3]), config, mmd_cfg, data)
        assert res.shape == () and float(res) >= 0.0

    def test_key_override_provides_new_randomness(self):
        """Passing an explicit key should override the config key."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(0),
        )
        data = jnp.array([[0, 1], [1, 0], [2, 2], [1, 1]])
        params = jnp.array([0.3, 0.7])
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=100)

        r_default = _call_qudit_mmd_loss(params, config, mmd_cfg, data)
        r_override = _call_qudit_mmd_loss(
            params, config, mmd_cfg, data, key=jax.random.PRNGKey(12345)
        )
        assert float(r_default) != float(r_override)

    @pytest.mark.parametrize("graph_type", ["cycle", "complete"])
    def test_both_graph_types_finite(self, graph_type):
        """Both cycle and complete graph types produce finite results."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]]},
            n_samples=200,
            key=jax.random.PRNGKey(0),
        )
        data = jnp.array([[0, 1], [1, 2], [2, 0]])
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=50, graph_type=graph_type)
        res = _call_qudit_mmd_loss(jnp.array([0.5, 0.3]), config, mmd_cfg, data)
        assert np.isfinite(float(res))

    def test_gradient_flows(self):
        """jax.grad through one-shot loss evaluation should produce finite gradients."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]]},
            n_samples=200,
            key=jax.random.PRNGKey(0),
        )
        data = jnp.array([[0, 1], [1, 2], [2, 0]])
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=50)
        params = jnp.array([0.5, 0.3])

        grad = jax.grad(_call_qudit_mmd_loss)(params, config, mmd_cfg, data)
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))


class TestBuildQuditMMDLoss:
    """Tests for the build_qudit_mmd_loss factory pattern."""

    def _make_config_and_data(self):
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]], 1: [[0, 1]], 2: [[1, 1]]},
            n_samples=200,
            key=jax.random.PRNGKey(42),
        )
        data = jnp.array([[0, 1], [1, 0], [2, 2], [1, 1], [0, 2]])
        params = jnp.array([0.3, 0.5, 0.1])
        return config, data, params

    def test_factory_matches_direct_call(self):
        """build_qudit_mmd_loss matches one-shot evaluation."""
        config, data, params = self._make_config_and_data()
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=50)

        direct = _call_qudit_mmd_loss(params, config, mmd_cfg, data)
        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)
        factory = loss_fn(params, data)

        assert np.isclose(float(direct), float(factory), atol=1e-10)

    def test_jit_compatible(self):
        """The returned loss_fn can be JIT-compiled."""
        config, data, params = self._make_config_and_data()
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=50)
        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)

        eager = loss_fn(params, data, key=jax.random.PRNGKey(7))
        jitted = jax.jit(loss_fn)(params, data, key=jax.random.PRNGKey(7))

        assert np.isclose(float(eager), float(jitted), atol=1e-10)

    def test_grad_compatible(self):
        """jax.grad through the factory loss produces finite gradients."""
        config, data, params = self._make_config_and_data()
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=50)
        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)

        grad = jax.grad(loss_fn)(params, data, key=jax.random.PRNGKey(0))
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_reuse_factory(self):
        """Calling loss_fn with different params gives different results."""
        config, data, params = self._make_config_and_data()
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=50)
        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)

        r1 = loss_fn(params, data)
        r2 = loss_fn(params * 2.0, data)
        assert float(r1) != float(r2)

    def test_multi_bandwidth_jit(self):
        """Factory with multiple bandwidths works under JIT."""
        config, data, params = self._make_config_and_data()
        mmd_cfg = QuditMMDConfig(bandwidth=[0.5, 1.0], n_ops=50)
        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)

        eager = loss_fn(params, data, key=jax.random.PRNGKey(3))
        jitted = jax.jit(loss_fn)(params, data, key=jax.random.PRNGKey(3))

        assert np.isclose(float(eager), float(jitted), atol=1e-10)

    def test_with_init_state(self):
        """Factory works with custom init state in circuit_config."""
        state_elems = jnp.array([[0, 0], [1, 1]])
        state_amps = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)])

        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 1]]},
            n_samples=100,
            key=jax.random.PRNGKey(42),
            init_state_elems=state_elems,
            init_state_amps=state_amps,
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=10)
        data = jnp.array([[0, 0], [1, 1]])
        params = jnp.array([0.5])

        loss_fn = build_qudit_mmd_loss(config, mmd_cfg)
        res = loss_fn(params, data)
        assert res.shape == () and np.isfinite(float(res))

    def test_raises_n_samples_le_one(self):
        """Factory raises ValueError at build time for n_samples <= 1."""
        config = QuditCircuitConfig(
            d=3,
            n_qudits=2,
            gates={0: [[1, 0]]},
            n_samples=1,
            key=jax.random.PRNGKey(0),
        )
        mmd_cfg = QuditMMDConfig(bandwidth=1.0, n_ops=5)
        with pytest.raises(ValueError, match="n_samples must be greater than 1"):
            build_qudit_mmd_loss(config, mmd_cfg)
