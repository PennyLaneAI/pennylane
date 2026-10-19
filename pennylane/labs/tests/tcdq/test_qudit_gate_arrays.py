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
# pylint: disable=too-many-arguments,too-few-public-methods,too-many-locals
"""Tests for the array-based gate description of the qudit IQP estimator."""

import itertools

import numpy as np
import pytest
from scipy.stats import chisquare

from pennylane.labs.tcdq import (
    QuditCircuitConfig,
    QuditGateArrays,
    build_qudit_expval_func,
    create_random_qudit_gates,
    create_random_qudit_gates_jax,
)
from pennylane.labs.tcdq.qudit_expval_functions import (
    _BYTES_PER_GB,
    _dims_to_numpy,
    _gate_arrays_to_dict,
    _parse_qudit_gate_arrays,
    _qudit_phase_bytes_per_gate,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _dict_to_arrays(gates, max_weight):
    """Hand conversion of a gate dictionary into padded arrays with explicit parameter indices."""
    supports, powers, pidx = [], [], []
    for k in sorted(gates):
        for gate in gates[k]:
            items = list(gate.items())
            supports.append([q for q, _ in items] + [0] * (max_weight - len(items)))
            powers.append([p for _, p in items] + [0] * (max_weight - len(items)))
            pidx.append(k)
    return QuditGateArrays(np.array(supports), np.array(powers), np.array(pidx))


def _assert_same_estimator(fn_ref, fn_new, params, phase_params=None, **call_kwargs):
    """Values, covariances and gradients of two estimators must agree."""
    args = (params,) if phase_params is None else (params, phase_params)
    ev_r, cov_r = fn_ref(*args, **call_kwargs)
    ev_n, cov_n = fn_new(*args, **call_kwargs)
    np.testing.assert_allclose(ev_n, ev_r, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(cov_n, cov_r, atol=1e-6, rtol=1e-3)

    def loss(fn):
        def inner(*a):
            ev, cov = fn(*a, **call_kwargs)
            return jnp.real(jnp.sum(ev)) + jnp.sum(cov)

        return inner

    argnums = 0 if phase_params is None else (0, 1)
    g_r = jax.grad(loss(fn_ref), argnums=argnums)(*args)
    g_n = jax.grad(loss(fn_new), argnums=argnums)(*args)
    for a, b in zip(jax.tree.leaves(g_r), jax.tree.leaves(g_n)):
        np.testing.assert_allclose(b, a, atol=2e-5, rtol=2e-5)


def _tiny_budget(n_samples, n_obs):
    return 2.5 * _qudit_phase_bytes_per_gate(1, n_samples, n_obs) / _BYTES_PER_GB


class TestParseGateArrays:
    """Parsing of array descriptions into weight groups."""

    def test_canonical_arrays_are_used_in_place(self):
        """Canonical arrays are grouped by row offsets without copying."""
        gates = create_random_qudit_gates_jax(jax.random.PRNGKey(0), 50, [3] * 6, 1, 3)
        groups, n_params = _parse_qudit_gate_arrays(gates, 6, _dims_to_numpy(3, 6))
        assert n_params == 50
        assert [g.omega for g in groups] == [1, 2, 3]
        assert sum(g.n_gates for g in groups) == 50
        offsets = [g.row_offset for g in groups]
        assert offsets == [0] + list(np.cumsum([g.n_gates for g in groups])[:-1])
        for g in groups:
            assert g.arrays.supports is gates.supports
            assert g.arrays.param_indices is None

    def test_non_canonical_arrays_are_normalised(self):
        """Negative indices, unreduced powers, padding first and shuffled rows are accepted."""
        supports = np.array([[-1, 0], [0, 2], [0, -3], [0, 1]])
        powers = np.array([[1, 0], [0, 4], [0, 3], [5, 1]])  # 4 mod 3 == 1; 3 mod 3 == 0 (identity)
        groups, n_params = _parse_qudit_gate_arrays(
            QuditGateArrays(supports, powers, np.array([3, 0, 1, 1])), 3, _dims_to_numpy(3, 3)
        )
        assert n_params == 4
        assert [g.omega for g in groups] == [1, 2]
        w1, w2 = groups
        np.testing.assert_array_equal(np.asarray(w1.arrays.supports), [[2], [2]])
        np.testing.assert_array_equal(np.asarray(w1.arrays.powers), [[1], [1]])
        np.testing.assert_array_equal(np.asarray(w1.arrays.param_indices), [3, 0])
        np.testing.assert_array_equal(np.asarray(w2.arrays.supports), [[0, 1]])
        np.testing.assert_array_equal(np.asarray(w2.arrays.powers), [[2, 1]])
        np.testing.assert_array_equal(np.asarray(w2.arrays.param_indices), [1])

    @pytest.mark.parametrize(
        "gates, error, match",
        [
            (QuditGateArrays(np.array([[3, 0]]), np.array([[1, 0]])), IndexError, "out of range"),
            (QuditGateArrays(np.array([[-4, 0]]), np.array([[1, 0]])), IndexError, "out of range"),
            (QuditGateArrays(np.array([[1, 1]]), np.array([[1, 2]])), ValueError, "more than once"),
            (QuditGateArrays(np.array([[1, 2]]), np.array([[1]])), ValueError, "same shape"),
            (
                QuditGateArrays(np.array([[1, 2]]), np.array([[1, 1]]), np.array([-1])),
                ValueError,
                "non-negative",
            ),
            (QuditGateArrays(np.array([[1.0, 2.0]]), np.array([[1, 1]])), TypeError, "integer"),
            (QuditGateArrays(np.zeros((2, 4), int), np.ones((2, 4), int)), ValueError, "exceeds"),
        ],
    )
    def test_invalid_arrays(self, gates, error, match):
        """Malformed arrays raise informative errors."""
        with pytest.raises(error, match=match):
            _parse_qudit_gate_arrays(gates, 3, _dims_to_numpy(3, 3))

    def test_inactive_columns_are_ignored(self):
        """Entries of columns with zero power are never validated."""
        groups, _ = _parse_qudit_gate_arrays(
            QuditGateArrays(np.array([[1, -99]]), np.array([[1, 0]])), 3, _dims_to_numpy(3, 3)
        )
        assert [g.omega for g in groups] == [1]


class TestArraysMatchDict:
    """The array description must reproduce the dictionary estimator exactly."""

    @pytest.mark.parametrize("blocked", [False, True])
    def test_generator_output(self, blocked):
        """Canonical generator output, uniform dimension, with runtime overrides."""
        d, n_qudits, n_gates, n_samples, n_obs = 3, 6, 40, 512, 7
        rng = np.random.default_rng(0)
        dims_np = _dims_to_numpy(d, n_qudits)
        gates = create_random_qudit_gates_jax(jax.random.PRNGKey(0), n_gates, [d] * n_qudits, 1, 3)
        gates_dict = _gate_arrays_to_dict(gates, n_qudits, dims_np)
        l_vecs = np.stack([rng.integers(0, dims_np) for _ in range(n_obs)])
        m_vecs = np.stack([rng.integers(0, dims_np) for _ in range(n_obs)])
        common = {
            "dims": d,
            "n_qudits": n_qudits,
            "observables": (l_vecs, m_vecs),
            "n_samples": n_samples,
            "key": jax.random.PRNGKey(7),
        }
        max_memory = _tiny_budget(n_samples, n_obs) if blocked else 1.0
        fn_dict = build_qudit_expval_func(QuditCircuitConfig(gates=gates_dict, **common))
        fn_arr = build_qudit_expval_func(
            QuditCircuitConfig(gates=gates, max_memory=max_memory, **common)
        )
        params = jnp.array(rng.normal(size=n_gates))
        _assert_same_estimator(fn_dict, fn_arr, params)
        _assert_same_estimator(fn_dict, fn_arr, params, key=jax.random.PRNGKey(3), n_samples=128)
        _assert_same_estimator(fn_dict, fn_arr, params, observables=(l_vecs[:2], m_vecs[:2]))

    @pytest.mark.parametrize("blocked", [False, True])
    def test_non_canonical_mixed_dims(self, blocked):
        """Shared parameters, identity gates, phase layer and initial state with messy arrays."""
        dims = [2, 3, 4, 5, 3, 2]
        n_qudits = len(dims)
        rng = np.random.default_rng(1)
        dims_np = _dims_to_numpy(dims, n_qudits)
        gates = {}
        for i in range(23):
            weight = int(rng.integers(1, 4))
            support = sorted(int(q) for q in rng.choice(n_qudits, size=weight, replace=False))
            gates[i] = [{q: int(rng.integers(1, dims[q])) for q in support}]
        gates[0].append({0: 1, 1: 1})  # shared parameter
        gates[5].append({3: 2})
        gates[7].append({2: 4})  # identity gate (4 mod 4 == 0)
        arrays = _dict_to_arrays(gates, 3)
        perm = rng.permutation(arrays.supports.shape[0])
        sup = arrays.supports[perm]
        pw = arrays.powers[perm]
        pidx = arrays.param_indices[perm]
        sup = np.where((pw != 0) & (rng.random(sup.shape) < 0.5), sup - n_qudits, sup)
        pw = np.where(pw != 0, pw + dims_np[sup % n_qudits] * rng.integers(0, 3, sup.shape), pw)
        sup, pw = sup[:, ::-1].copy(), pw[:, ::-1].copy()
        sup = np.where(pw == 0, -99, sup)
        messy = QuditGateArrays(jnp.asarray(sup), jnp.asarray(pw), jnp.asarray(pidx))

        n_samples, n_obs = 256, 5
        l_vecs = np.stack([rng.integers(0, dims_np) for _ in range(n_obs)])
        m_vecs = np.stack([rng.integers(0, dims_np) for _ in range(n_obs)])
        elems = np.stack([rng.integers(0, dims_np) for _ in range(3)])
        amps = rng.normal(size=3) + 1j * rng.normal(size=3)

        def phase_fn(p, z):
            return jnp.sum(p[:n_qudits] * z.astype(jnp.float32)) + p[-1]

        common = {
            "dims": dims,
            "n_qudits": n_qudits,
            "observables": (l_vecs, m_vecs),
            "n_samples": n_samples,
            "key": jax.random.PRNGKey(11),
            "init_state_elems": elems,
            "init_state_amps": amps / np.linalg.norm(amps),
            "phase_fn": phase_fn,
        }
        max_memory = _tiny_budget(n_samples, n_obs) if blocked else 1.0
        fn_dict = build_qudit_expval_func(QuditCircuitConfig(gates=gates, **common))
        fn_arr = build_qudit_expval_func(
            QuditCircuitConfig(gates=messy, max_memory=max_memory, **common)
        )
        params = jnp.array(rng.normal(size=23))
        phase_params = jnp.array(rng.normal(size=n_qudits + 1))
        _assert_same_estimator(fn_dict, fn_arr, params, phase_params)
        _assert_same_estimator(
            fn_dict, fn_arr, params, phase_params, key=jax.random.PRNGKey(3), n_samples=128
        )

    def test_tuple_input_and_single_weight(self):
        """A plain ``(supports, powers)`` tuple and single-weight circuits are accepted."""
        d, n_qudits = 5, 4
        rng = np.random.default_rng(2)
        dims_np = _dims_to_numpy(d, n_qudits)
        l_vecs = np.stack([rng.integers(0, dims_np) for _ in range(4)])
        m_vecs = np.stack([rng.integers(0, dims_np) for _ in range(4)])
        common = {
            "dims": d,
            "n_qudits": n_qudits,
            "observables": (l_vecs, m_vecs),
            "n_samples": 300,
            "key": jax.random.PRNGKey(2),
        }
        for gates in [
            QuditGateArrays(
                np.array([[0], [1], [2], [3], [1], [0]]), np.array([[1], [4], [2], [3], [1], [3]])
            ),
            (
                np.array([[0, 1], [2, 3], [1, 3], [0, 2]]),
                np.array([[1, 2], [3, 4], [2, 2], [1, 1]]),
            ),
        ]:
            arrays = gates if isinstance(gates, QuditGateArrays) else QuditGateArrays(*gates)
            fn_dict = build_qudit_expval_func(
                QuditCircuitConfig(gates=_gate_arrays_to_dict(arrays, n_qudits, dims_np), **common)
            )
            fn_arr = build_qudit_expval_func(QuditCircuitConfig(gates=gates, **common))
            fn_blk = build_qudit_expval_func(
                QuditCircuitConfig(gates=gates, max_memory=_tiny_budget(300, 4), **common)
            )
            params = jnp.array(rng.normal(size=arrays.supports.shape[0]))
            _assert_same_estimator(fn_dict, fn_arr, params)
            _assert_same_estimator(fn_dict, fn_blk, params)

    @pytest.mark.parametrize("blocked", [False, True])
    def test_traced_observables_under_jit(self, blocked):
        """Observables and keys may be tracers of an outer ``jax.jit``."""
        d, n_qudits = 3, 5
        rng = np.random.default_rng(3)
        dims_np = _dims_to_numpy(d, n_qudits)
        gates = create_random_qudit_gates_jax(jax.random.PRNGKey(5), 30, [d] * n_qudits, 1, 2)
        gates_dict = _gate_arrays_to_dict(gates, n_qudits, dims_np)
        l_vecs = jnp.asarray(np.stack([rng.integers(0, dims_np) for _ in range(6)]))
        m_vecs = jnp.zeros_like(l_vecs)
        common = {"dims": d, "n_qudits": n_qudits, "n_samples": 400, "key": jax.random.PRNGKey(9)}
        max_memory = _tiny_budget(400, 6) if blocked else 1.0
        fn_dict = build_qudit_expval_func(QuditCircuitConfig(gates=gates_dict, **common))
        fn_arr = build_qudit_expval_func(
            QuditCircuitConfig(gates=gates, max_memory=max_memory, **common)
        )
        params = jnp.array(rng.normal(size=30))
        key = jax.random.PRNGKey(4)

        def ref(p):
            ev, cov = fn_dict(p, observables=(l_vecs, m_vecs), key=key, n_samples=400)
            return jnp.real(jnp.sum(ev)) + jnp.sum(cov)

        @jax.jit
        def jitted(p, l, k):
            ev, cov = fn_arr(p, observables=(l, jnp.zeros_like(l)), key=k, n_samples=400)
            return jnp.real(jnp.sum(ev)) + jnp.sum(cov)

        np.testing.assert_allclose(jitted(params, l_vecs, key), ref(params), atol=2e-5, rtol=2e-5)
        g_ref = jax.grad(ref)(params)
        g_new = jax.jit(jax.grad(jitted))(params, l_vecs, key)
        np.testing.assert_allclose(g_new, g_ref, atol=2e-5, rtol=2e-5)


class TestRandomQuditGatesJax:
    """The pure-JAX generator reproduces the distribution of the NumPy generator."""

    def test_layout_and_dtypes(self):
        """Output is canonical: weight-sorted rows, sorted active prefix, reduced powers."""
        dims = [3, 4, 2, 3, 5]
        gates = create_random_qudit_gates_jax(
            jax.random.PRNGKey(1), 5000, dims, 1, 3, chunk_size=777
        )
        assert gates.supports.dtype == jnp.int8 and gates.powers.dtype == jnp.int8
        assert gates.supports.shape == gates.powers.shape == (5000, 3)
        assert gates.param_indices is None
        sup = np.asarray(gates.supports).astype(int)
        pw = np.asarray(gates.powers).astype(int)
        active = pw != 0
        weight = active.sum(axis=1)
        assert np.all(np.diff(weight) >= 0)
        assert np.all(active[:, 1:] <= active[:, :-1])
        assert np.all(~active[:, 1:] | (sup[:, 1:] > sup[:, :-1]))
        assert np.all(~active | ((pw >= 1) & (pw < np.array(dims)[sup])))
        wide = create_random_qudit_gates_jax(jax.random.PRNGKey(1), 10, [3] * 200, 1, 2)
        assert wide.supports.dtype == jnp.int16
        empty = create_random_qudit_gates_jax(jax.random.PRNGKey(1), 0, [3] * 4, 1, 2)
        assert empty.supports.shape == (0, 2)

    def test_reproducible(self):
        """Same key and chunk size give the same gates."""
        a = create_random_qudit_gates_jax(
            jax.random.PRNGKey(3), 1000, [3] * 7, 1, 2, chunk_size=300
        )
        b = create_random_qudit_gates_jax(
            jax.random.PRNGKey(3), 1000, [3] * 7, 1, 2, chunk_size=300
        )
        np.testing.assert_array_equal(np.asarray(a.supports), np.asarray(b.supports))
        np.testing.assert_array_equal(np.asarray(a.powers), np.asarray(b.powers))

    def test_distribution_matches_numpy_generator(self):
        """Weights, supports and powers are uniform and agree with the NumPy generator."""
        dims = [3, 4, 2, 3, 5]
        n_qudits, n_gates = len(dims), 60_000
        gates = create_random_qudit_gates_jax(jax.random.PRNGKey(123), n_gates, dims, 1, 3)
        sup = np.asarray(gates.supports).astype(int)
        pw = np.asarray(gates.powers).astype(int)
        active = pw != 0
        weight = active.sum(axis=1)
        assert chisquare(np.bincount(weight, minlength=4)[1:]).pvalue > 1e-3

        ref = create_random_qudit_gates(n_gates, dims, 1, 3, seed=321)
        sup_ref = np.zeros((n_gates, 3), int)
        pw_ref = np.zeros((n_gates, 3), int)
        for i, (gate,) in ref.items():
            for j, (q, p) in enumerate(sorted(gate.items())):
                sup_ref[i, j], pw_ref[i, j] = q, p
        weight_ref = (pw_ref != 0).sum(axis=1)

        for w in range(1, 4):
            combos = list(itertools.combinations(range(n_qudits), w))
            hist = np.array(
                [np.sum(np.all(sup[weight == w][:, :w] == np.array(c), axis=1)) for c in combos]
            )
            hist_ref = np.array(
                [
                    np.sum(np.all(sup_ref[weight_ref == w][:, :w] == np.array(c), axis=1))
                    for c in combos
                ]
            )
            assert chisquare(hist).pvalue > 1e-3
            # two-sample homogeneity between the JAX and the NumPy generator
            obs = np.array([hist, hist_ref])
            expected = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
            assert chisquare(obs.ravel(), expected.ravel(), ddof=len(combos)).pvalue > 1e-3
        for q in range(n_qudits):
            mask = active & (sup == q)
            if dims[q] == 2:
                assert np.all(pw[mask] == 1)
            else:
                assert chisquare(np.bincount(pw[mask], minlength=dims[q])[1:]).pvalue > 1e-3

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_gates": -1}, "non-negative"),
            ({"dims": [1, 3]}, "at least 2"),
            ({"min_weight": 0}, "Weights must satisfy"),
            ({"max_weight": 5}, "Weights must satisfy"),
            ({"chunk_size": 0}, "chunk_size"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        """Argument validation mirrors the NumPy generator."""
        base = {"n_gates": 4, "dims": [3, 3, 3], "min_weight": 1, "max_weight": 2}
        with pytest.raises(ValueError, match=match):
            create_random_qudit_gates_jax(jax.random.PRNGKey(0), **{**base, **kwargs})
