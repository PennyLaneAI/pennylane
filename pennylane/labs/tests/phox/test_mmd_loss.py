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
"""Tests for the phox MMD loss function."""

import itertools

import numpy as np
import pytest

import pennylane as qp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

try:
    from pennylane.labs.phox.expval_functions import CircuitConfig
    from pennylane.labs.phox.mmd_loss import (
        MMDConfig,
        _binary_ops_to_pauli_int,
        _compute_single_mmd,
        median_heuristic,
        mmd_loss,
    )

    # Alias kept for backward-compat in exact-MMD helpers
    # (these helpers use 'sigma' as the mathematical symbol).
except ImportError:
    pytest.skip("pennylane.labs.phox not found", allow_module_level=True)


def _iqp_probs_pennylane(generators, params, n_qubits):
    """Return the exact output probability vector of an IQP circuit."""
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            qp.Hadamard(i)
        for param, gen in zip(params, generators):
            qp.MultiRZ(2.0 * -param, wires=gen)
        for i in range(n_qubits):
            qp.Hadamard(i)
        return qp.probs()

    return np.asarray(circuit(), dtype=np.float64)


def _iqp_expval_pennylane(generators, params, obs_ints, n_qubits):
    """Return the exact expectation value <O> for an IQP circuit.

    ``obs_ints`` uses the convention I=0, X=1, Y=2, Z=3 per qubit.
    """
    pauli_map = {0: qp.Identity, 1: qp.X, 2: qp.Y, 3: qp.Z}
    ops = [pauli_map[o](i) for i, o in enumerate(obs_ints)]
    observable = qp.prod(*ops) if len(ops) > 1 else ops[0]

    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            qp.Hadamard(i)
        for param, gen in zip(params, generators):
            qp.MultiRZ(2.0 * -param, wires=gen)
        for i in range(n_qubits):
            qp.Hadamard(i)
        return qp.expval(observable)

    return float(circuit())


def _gaussian_kernel_matrix(a, b, sigma):
    """``K[i,j] = exp(-||a_i - b_j||^2 / (2 sigma^2))``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-sq / (2.0 * sigma**2))


def _exact_mmd2_kernel(probs_p, data_q, sigma, n_qubits):
    """Exact MMD^2(P, Q_emp) via Gaussian kernel matrices.

    * PP = sum_{x,x'} P(x)P(x') K(x,x')  (V-statistic, includes diagonal)
    * PQ = (2/m) sum_x P(x) sum_j K(x, X_j)
    * QQ = (1/(m(m-1))) sum_{j!=k} K(X_j, X_k)  (U-statistic)
    """
    bs = np.array(list(itertools.product([0, 1], repeat=n_qubits)), dtype=np.float64)
    dq = np.asarray(data_q, dtype=np.float64)
    m = len(dq)

    K_pp = _gaussian_kernel_matrix(bs, bs, sigma)
    pp = probs_p @ K_pp @ probs_p

    K_pq = _gaussian_kernel_matrix(bs, dq, sigma)
    pq = 2.0 * np.sum(probs_p[:, None] * K_pq) / m

    K_qq = _gaussian_kernel_matrix(dq, dq, sigma)
    qq = (np.sum(K_qq) - np.trace(K_qq)) / (m * (m - 1))

    return float(pp - pq + qq)


def _exact_mmd2_operators(generators, params, data_q, sigma, n_qubits):
    """Exact MMD^2 via the operator (Fourier) decomposition of the Gaussian kernel.

    The Gaussian kernel factorises as:

        K(x, y) = sum_z  w(z) (-1)^{z.x} (-1)^{z.y}

    where ``w(z) = prod_i p^{z_i} (1-p)^{1-z_i}`` and
    ``p = (1 - exp(-1/(2 sigma^2))) / 2``.

    Therefore::

        MMD^2 = sum_z w(z) [tr_P(z)^2 - 2 tr_P(z) tr_Q(z) + U_QQ(z)]

    This function enumerates all 2^n operators (feasible for small n) and
    computes each term exactly using PennyLane.
    """
    p = (1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0
    dq = np.asarray(data_q, dtype=np.float64)
    m = len(dq)

    total = 0.0
    for bits in itertools.product([0, 1], repeat=n_qubits):
        z = np.array(bits, dtype=np.float64)
        hw = int(np.sum(z))
        w = (p**hw) * ((1.0 - p) ** (n_qubits - hw))

        obs = [3 if b else 0 for b in bits]
        tr_p = _iqp_expval_pennylane(generators, params, obs, n_qubits)

        parities = (dq @ z) % 2
        signs = 1.0 - 2.0 * parities
        tr_q = float(np.mean(signs))

        u_qq = (tr_q**2 * m - 1.0) / (m - 1.0)

        total += w * (tr_p**2 - 2.0 * tr_p * tr_q + u_qq)

    return total


class TestMedianHeuristic:
    """Unit tests for :func:`median_heuristic`."""

    def test_raises_fewer_than_two_samples(self):
        """Should raise ValueError with fewer than 2 samples."""
        with pytest.raises(ValueError, match="at least two"):
            median_heuristic(np.array([[1.0, 0.0]]))

    def test_identical_samples_returns_one(self):
        """All-zero pairwise distances fall back to 1.0."""
        data = np.ones((5, 3))
        assert median_heuristic(data) == 1.0

    def test_two_distinct_samples(self):
        """With two samples the only nonzero distance is the inter-sample distance."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert np.isclose(median_heuristic(data), np.sqrt(2.0))

    def test_positive_on_random_binary_data(self):
        """Median heuristic on distinct binary data should be positive."""
        rng = np.random.default_rng(0)
        data = rng.binomial(1, 0.5, (40, 4)).astype(float)
        assert median_heuristic(data) > 0


class TestBinaryOpsToPauliInt:
    """Unit tests for :func:`_binary_ops_to_pauli_int`."""

    def test_one_maps_to_three_zero_stays_zero(self):
        """Binary one entries map to Pauli-Z integer code 3; zero stays identity 0."""
        inp = jnp.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]])
        expected = jnp.array([[3, 0, 3], [0, 0, 0], [3, 3, 3]])
        assert jnp.array_equal(_binary_ops_to_pauli_int(inp), expected)

    def test_preserves_shape(self):
        """The transformed operator array should keep the input shape."""
        inp = jnp.array([[1, 0], [0, 1], [1, 1]])
        assert _binary_ops_to_pauli_int(inp).shape == inp.shape


class TestComputeSingleMmd:
    """Focused tests for the low-level :func:`_compute_single_mmd` kernel."""

    def test_matching_traces_near_zero(self):
        """If circuit traces match data traces the loss is approximately zero."""
        rng = np.random.default_rng(1)
        n_ops, n_q, m = 20, 3, 80
        vis = jnp.array(rng.binomial(1, 0.3, (n_ops, n_q)), dtype=jnp.float64)
        gt = jnp.array(rng.binomial(1, 0.5, (m, n_q)), dtype=jnp.float64)
        tr_train = jnp.mean(1 - 2 * ((gt @ vis.T) % 2), axis=0)

        result = _compute_single_mmd(tr_train, jnp.zeros(n_ops), gt, vis, 10_000, sqrt_loss=False)
        assert abs(float(result)) < 0.05

    def test_sqrt_loss_flag(self):
        """``sqrt_loss=True`` returns ``sqrt(abs(result))``."""
        rng = np.random.default_rng(2)
        n_ops, n_q = 15, 2
        vis = jnp.array(rng.binomial(1, 0.5, (n_ops, n_q)), dtype=jnp.float64)
        gt = jnp.array(rng.binomial(1, 0.5, (40, n_q)), dtype=jnp.float64)
        tr_iqp = jnp.array(rng.normal(0, 0.3, n_ops))
        se = jnp.zeros(n_ops)

        val = _compute_single_mmd(tr_iqp, se, gt, vis, 10_000, False)
        sqr = _compute_single_mmd(tr_iqp, se, gt, vis, 10_000, True)

        assert np.isclose(float(sqr), np.sqrt(abs(float(val))), atol=1e-7)

    def test_correction_removes_bias(self):
        """The U-statistic correction should make the PP term unbiased.

        When ``std_err = 0`` (exact circuit traces), the correction simplifies
        to ``tr_iqp^2 / n_samples`` and the result should reduce to:
        ``tr_iqp^2 - 2*tr_iqp*tr_train + (tr_train^2*m - 1)/(m-1)``
        independently of ``n_samples``.
        """
        rng = np.random.default_rng(3)
        n_ops, n_q, m = 10, 2, 50
        vis = jnp.array(rng.binomial(1, 0.5, (n_ops, n_q)), dtype=jnp.float64)
        gt = jnp.array(rng.binomial(1, 0.5, (m, n_q)), dtype=jnp.float64)
        tr_iqp = jnp.array(rng.normal(0, 0.5, n_ops))
        zero_se = jnp.zeros(n_ops)

        res_100 = _compute_single_mmd(tr_iqp, zero_se, gt, vis, 100, False)
        res_10000 = _compute_single_mmd(tr_iqp, zero_se, gt, vis, 10_000, False)

        assert np.isclose(float(res_100), float(res_10000), atol=1e-10)


class TestExactMMDConsistency:
    """Validate that kernel-based and operator-decomposition-based
    exact MMD^2 computations agree numerically.

    This cross-check confirms the Fourier expansion of the Gaussian kernel
    (Eq. 2 of arXiv:2501.04776) and the U-statistic decomposition used
    by the phox estimator.
    """

    @pytest.mark.parametrize(
        "generators, params, n_qubits, biases",
        [
            ([[0], [1], [0, 1, 2]], [0.37, 0.95, 0.73], 3, [0.3, 0.7, 0.5]),
            ([[0, 1], [1, 2]], [0.5, 0.3], 3, [0.1, 0.9, 0.5]),
            ([[0], [1], [0, 1]], [0.2, 0.8, 0.4], 2, [0.4, 0.6]),
        ],
    )
    def test_kernel_vs_operator(self, generators, params, n_qubits, biases):
        """Kernel-matrix MMD^2 should equal operator-decomposition MMD^2."""
        probs = _iqp_probs_pennylane(generators, params, n_qubits)
        rng = np.random.default_rng(42)
        data = np.stack([rng.binomial(1, b, 50) for b in biases], axis=1)
        sigma = float(median_heuristic(data))

        mmd_k = _exact_mmd2_kernel(probs, data, sigma, n_qubits)
        mmd_o = _exact_mmd2_operators(generators, params, data, sigma, n_qubits)

        assert np.isclose(mmd_k, mmd_o, atol=1e-8), f"kernel={mmd_k:.10f}, operator={mmd_o:.10f}"

    def test_operator_expression_returns_finite_value(self):
        """The exact operator-decomposition MMD^2 should be finite."""
        generators = [[0], [1], [0, 1]]
        params = [0.2, 0.8, 0.4]
        n_qubits = 2
        rng = np.random.default_rng(123)
        data = np.stack([rng.binomial(1, 0.4, 40), rng.binomial(1, 0.6, 40)], axis=1)
        sigma = float(median_heuristic(data))

        mmd_o = _exact_mmd2_operators(generators, params, data, sigma, n_qubits)

        assert np.isfinite(mmd_o)


class TestMMDLossAPI:
    """Tests for :func:`mmd_loss`."""

    def test_raises_n_samples_le_one(self):
        """``n_samples <= 1`` should raise ``ValueError``."""
        config = CircuitConfig(
            gates={0: [[0]]},
            observables=[[3]],
            n_samples=1,
            key=jax.random.PRNGKey(0),
            n_qubits=1,
        )
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=5)

        with pytest.raises(ValueError, match="n_samples must be greater than 1"):
            mmd_loss(
                jnp.array([0.1]),
                config,
                mmd_cfg,
                jnp.array([[0]]),
            )

    def test_return_per_bandwidth_type_and_length(self):
        """``return_per_bandwidth=True`` returns a list of length ``len(bandwidth)``."""
        config = CircuitConfig(
            gates={0: [[0]], 1: [[1]]},
            observables=[[3, 0]],
            n_samples=100,
            key=jax.random.PRNGKey(0),
            n_qubits=2,
        )
        mmd_cfg = MMDConfig(bandwidth=[0.5, 1.0], n_ops=30, return_per_bandwidth=True)
        data = jnp.array([[0, 1], [1, 0], [0, 0]])

        res = mmd_loss(
            jnp.array([0.1, 0.2]),
            config,
            mmd_cfg,
            data,
        )
        assert isinstance(res, list)
        assert len(res) == 2

    def test_multi_bandwidth_mean(self):
        """The scalar output should equal the mean of per-bandwidth outputs.

        Both calls start from the same ``config.key`` and iterate through
        bandwidths identically, so per-bandwidth values are deterministic.
        """
        config = CircuitConfig(
            gates={0: [[0]], 1: [[1]]},
            observables=[[3, 0]],
            n_samples=200,
            key=jax.random.PRNGKey(42),
            n_qubits=2,
        )
        data = jnp.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        bandwidths = [0.5, 1.0, 2.0]

        cfg_per = MMDConfig(bandwidth=bandwidths, n_ops=100, return_per_bandwidth=True)
        cfg_avg = MMDConfig(bandwidth=bandwidths, n_ops=100, return_per_bandwidth=False)

        per = mmd_loss(jnp.array([0.1, 0.2]), config, cfg_per, data)
        avg = mmd_loss(jnp.array([0.1, 0.2]), config, cfg_avg, data)

        expected = np.mean([float(v) for v in per])
        assert np.isclose(float(avg), expected, atol=1e-6)

    def test_single_bandwidth_returns_scalar(self):
        """A single float bandwidth should produce a scalar output."""
        config = CircuitConfig(
            gates={0: [[0]]},
            observables=[[3]],
            n_samples=100,
            key=jax.random.PRNGKey(0),
            n_qubits=1,
        )
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=20)
        data = jnp.array([[0], [1]])

        res = mmd_loss(jnp.array([0.5]), config, mmd_cfg, data)
        assert res.shape == ()

    def test_deterministic_same_key(self):
        """Identical keys must yield bit-identical results."""
        config = CircuitConfig(
            gates={0: [[0]], 1: [[1]], 2: [[0, 1]]},
            observables=[[3, 0]],
            n_samples=500,
            key=jax.random.PRNGKey(99),
            n_qubits=2,
        )
        kwargs = {
            "params": jnp.array([0.3, 0.5, 0.1]),
            "circuit_config": config,
            "mmd_config": MMDConfig(bandwidth=1.0, n_ops=100),
            "target_data": jnp.array([[0, 1], [1, 0], [0, 0], [1, 1]]),
        }
        assert float(mmd_loss(**kwargs)) == float(mmd_loss(**kwargs))

    def test_different_keys_give_different_results(self):
        """Different PRNG keys should (almost surely) give different losses."""
        gates = {0: [[0]], 1: [[1]]}
        data = jnp.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        params = jnp.array([0.3, 0.7])
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=100)

        r1 = mmd_loss(
            params,
            CircuitConfig(
                gates=gates,
                observables=[[3, 0]],
                n_samples=200,
                key=jax.random.PRNGKey(0),
                n_qubits=2,
            ),
            mmd_cfg,
            data,
        )
        r2 = mmd_loss(
            params,
            CircuitConfig(
                gates=gates,
                observables=[[3, 0]],
                n_samples=200,
                key=jax.random.PRNGKey(999),
                n_qubits=2,
            ),
            mmd_cfg,
            data,
        )
        assert float(r1) != float(r2)

    def test_mmd_loss_with_custom_init_state(self):
        """Ensure mmd_loss properly passes down separated init_state_elems and init_state_amps."""
        jax_state_elems = jnp.array([[0, 0], [1, 1]])
        jax_state_amps = jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2)])

        config = CircuitConfig(
            gates={0: [[0, 1]]},
            observables=[[3, 3]],
            n_samples=100,
            key=jax.random.PRNGKey(42),
            n_qubits=2,
            init_state_elems=jax_state_elems,
            init_state_amps=jax_state_amps,
        )
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=10)
        data = jnp.array([[0, 0], [1, 1]])
        params = jnp.array([0.5])

        # This should execute and return a finite scalar without any JAX tracer/shape errors
        res = mmd_loss(params, config, mmd_cfg, data)
        assert res.shape == () and np.isfinite(float(res))

    def test_mmd_loss_default_phase_fn_params_is_none(self):
        """Unspecified ``phase_fn_params`` is forwarded as ``None`` (no phase fn)."""
        config = CircuitConfig(
            gates={0: [[0, 1]]},
            observables=[[3, 3]],
            n_samples=100,
            key=jax.random.PRNGKey(0),
            n_qubits=2,
        )
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=10)
        data = jnp.array([[0, 0], [1, 1]])
        params = jnp.array([0.5])

        res_default = mmd_loss(params, config, mmd_cfg, data)
        res_explicit_none = mmd_loss(params, config, mmd_cfg, data, phase_fn_params=None)
        assert np.isclose(float(res_default), float(res_explicit_none), atol=1e-12)

    def test_mmd_loss_with_phase_fn_params(self):
        """``phase_fn_params`` should flow through to the phase function and be differentiable."""

        def phase_fn(params, z):
            hamming = jnp.sum(z.astype(jnp.float32))
            return jnp.sum(params * jnp.array([1.0, hamming, hamming**2]))

        n_qubits = 2
        config = CircuitConfig(
            gates={0: [[0, 1]]},
            observables=[[3, 3]],
            n_samples=200,
            key=jax.random.PRNGKey(7),
            n_qubits=n_qubits,
            phase_fn=phase_fn,
        )
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=20)
        data = jnp.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        params = jnp.array([0.3])
        zero_phase = jnp.zeros(3)
        phase_params = jnp.array([0.1, 0.2, 0.05])

        # A zero phase is differentiable w.r.t. the phase params and yields a
        # finite scalar loss.
        res_zero = mmd_loss(params, config, mmd_cfg, data, phase_fn_params=zero_phase)
        assert res_zero.shape == () and np.isfinite(float(res_zero))

        # Non-trivial phase parameters change the loss value.
        res_phase = mmd_loss(params, config, mmd_cfg, data, phase_fn_params=phase_params)
        assert res_phase.shape == () and np.isfinite(float(res_phase))
        assert not np.isclose(float(res_zero), float(res_phase), atol=1e-3)

        # Gradient w.r.t. ``phase_fn_params`` is finite and non-zero.
        grad_phase = jax.grad(
            lambda p: mmd_loss(params, config, mmd_cfg, data, phase_fn_params=p)
        )(phase_params)
        assert grad_phase.shape == phase_params.shape
        assert np.all(np.isfinite(np.asarray(grad_phase)))
        assert float(jnp.linalg.norm(grad_phase)) > 0.0


class TestMMDLossStatistical:
    """Statistical (Z-test) validation of the phox MMD estimator.

    For each parametrized circuit / dataset pair we:

    1. Compute the *exact* MMD^2 from PennyLane probabilities and data.
    2. Collect many stochastic phox estimates using different PRNG keys
       and random data sub-samples.
    3. Verify the sample mean is consistent with the exact value via a
       two-sided Z-test:  ``|exact - mean_est| / SE < z_threshold``.
    """

    Z_THRESHOLD = 4.0
    N_TRIALS = 300
    N_OPS = 40
    N_SAMPLES = 300

    @pytest.mark.parametrize(
        "generators, params, biases, n_data",
        [
            # 3-qubit, 3-gate circuit, moderate data bias
            (
                [[0], [1], [0, 1, 2]],
                [0.37, 0.95, 0.73],
                [0.3, 0.7, 0.5],
                200,
            ),
            # 3-qubit, 2-gate circuit, skewed data
            (
                [[0, 1], [1, 2]],
                [0.5, 0.3],
                [0.1, 0.9, 0.5],
                150,
            ),
            # 2-qubit, 3-gate circuit
            (
                [[0], [1], [0, 1]],
                [0.2, 0.8, 0.4],
                [0.4, 0.6],
                100,
            ),
        ],
    )
    def test_unbiased_z_test(self, generators, params, biases, n_data):
        """Mean of phox estimates should be consistent with exact MMD^2."""
        n_qubits = len(biases)

        probs_p = _iqp_probs_pennylane(generators, params, n_qubits)

        rng = np.random.default_rng(42)
        X = np.stack([rng.binomial(1, b, n_data) for b in biases], axis=1)

        sigma = float(median_heuristic(X))
        exact = _exact_mmd2_kernel(probs_p, X, sigma, n_qubits)

        gates = {i: [w] for i, w in enumerate(generators)}
        batch = min(80, n_data)

        mmd_cfg = MMDConfig(bandwidth=sigma, n_ops=self.N_OPS)

        params_jnp = jnp.array(params)
        X_jnp = jnp.array(X)

        config = CircuitConfig(
            gates=gates,
            observables=[[0] * n_qubits],
            n_samples=self.N_SAMPLES,
            key=jax.random.PRNGKey(0),
            n_qubits=n_qubits,
        )

        def evaluate_single_trial(key):
            loss_key, sample_key = jax.random.split(key)

            idx = jax.random.choice(sample_key, n_data, shape=(batch,), replace=False)

            return mmd_loss(
                params=params_jnp,
                circuit_config=config,
                mmd_config=mmd_cfg,
                target_data=X_jnp[idx],
                key=loss_key,
            )

        vmapped_eval = jax.vmap(evaluate_single_trial)

        master_key = jax.random.PRNGKey(42)
        trial_keys = jax.random.split(master_key, self.N_TRIALS)

        estimates_jnp = vmapped_eval(trial_keys)

        estimates = np.array(estimates_jnp)
        mean_est = np.mean(estimates)
        se = np.std(estimates, ddof=1) / np.sqrt(self.N_TRIALS)

        z = abs(exact - mean_est) / se if se > 1e-15 else 0.0
        assert z < self.Z_THRESHOLD, (
            f"Z-test FAILED: z={z:.2f}, exact={exact:.6f}, " f"mean={mean_est:.6f}, se={se:.6f}"
        )

    def test_wires_subset_executes(self):
        """``mmd_loss`` with a ``wires`` subset should run without error."""
        config = CircuitConfig(
            gates={0: [[0]], 1: [[1]], 2: [[2]], 3: [[0, 1]]},
            observables=[[0] * 3],
            n_samples=500,
            key=jax.random.PRNGKey(0),
            n_qubits=3,
        )
        rng = np.random.default_rng(0)
        data = jnp.array(rng.binomial(1, 0.5, (30, 2)))
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=50, wires=[0, 2])

        res = mmd_loss(
            jnp.array([0.1, 0.2, 0.3, 0.15]),
            config,
            mmd_cfg,
            data,
        )
        assert res.shape == () and np.isfinite(float(res))

    def test_sqrt_loss_positive(self):
        """``sqrt_loss=True`` should return a non-negative scalar."""
        config = CircuitConfig(
            gates={0: [[0]], 1: [[1]]},
            observables=[[3, 0]],
            n_samples=500,
            key=jax.random.PRNGKey(77),
            n_qubits=2,
        )
        rng = np.random.default_rng(0)
        data = jnp.array(rng.binomial(1, 0.3, (50, 2)))
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=200, sqrt_loss=True)

        res = mmd_loss(
            jnp.array([0.5, 0.3]),
            config,
            mmd_cfg,
            data,
        )
        assert res.shape == ()
        assert float(res) >= 0.0

    def test_key_override_provides_new_randomness(self):
        """Passing ``key`` to ``mmd_loss`` should override ``config.key``."""
        config = CircuitConfig(
            gates={0: [[0]], 1: [[1]]},
            observables=[[3, 0]],
            n_samples=500,
            key=jax.random.PRNGKey(0),
            n_qubits=2,
        )
        data = jnp.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        params = jnp.array([0.3, 0.7])
        mmd_cfg = MMDConfig(bandwidth=1.0, n_ops=100)

        r_default = mmd_loss(params, config, mmd_cfg, data)
        r_override = mmd_loss(
            params,
            config,
            mmd_cfg,
            data,
            key=jax.random.PRNGKey(12345),
        )
        assert float(r_default) != float(r_override)
