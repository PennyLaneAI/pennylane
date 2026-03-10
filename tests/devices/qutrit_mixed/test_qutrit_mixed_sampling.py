# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for sampling states in devices/qutrit_mixed."""

# pylint: disable=unused-argument,too-many-arguments

import numpy as np
import pytest
from flaky import flaky

import pennylane as qml
from pennylane import math
from pennylane.devices.qutrit_mixed import (
    apply_operation,
    create_initial_state,
    measure_with_samples,
    sample_state,
)
from pennylane.devices.qutrit_mixed.sampling import (
    _sample_probs_jax,
    _sample_state_jax,
    sample_probs,
)
from pennylane.measurements import Shots

APPROX_ATOL = 0.05
QUDIT_DIM = 3
ONE_QUTRIT = 1
TWO_QUTRITS = 2
THREE_QUTRITS = 3

MISMATCH_ERROR = "a and p must have same size"

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]

shots_and_copies = (
    [
        [(100,), 1],
        [((100, 1),), 1],
        [((100, 2),), 2],
        [(100, 100), 2],
        [(100, 200), 2],
        [(100, 100, 200), 3],
        [(200, (100, 2)), 3],
    ],
)


def get_dm_of_state(state_vector, num_qudits, normalization=1):
    state = np.outer(state_vector, np.conj(state_vector)) / normalization
    return state.reshape((QUDIT_DIM,) * num_qudits * 2)


@pytest.fixture(name="two_qutrit_pure_state")
def fixture_two_qutrit_pure_state():
    state_vector = np.array([0, 0, 1.0j, -1.0, 0, 0, 0, 1.0, 0], dtype=np.complex128)
    return get_dm_of_state(state_vector, TWO_QUTRITS, 3)


@pytest.fixture(name="batched_two_qutrit_pure_state")
def fixture_batched_two_qutrit_pure_state():
    state_vectors = [
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]) / np.sqrt(2),
        np.array([0, 1, 0, 1, 0, 0, 0, 1, 1]) / 2,
    ]
    states = [get_dm_of_state(state_vector, TWO_QUTRITS) for state_vector in state_vectors]
    return math.stack(states)


def samples_to_probs(samples, num_wires):
    """Converts samples to probs"""
    samples_decimal = [np.ravel_multi_index(sample, [QUDIT_DIM] * num_wires) for sample in samples]
    counts = [0] * (QUDIT_DIM**num_wires)

    for sample in samples_decimal:
        counts[sample] += 1

    return np.array(counts, dtype=np.float64) / len(samples)


def assert_correct_sampled_two_qutrit_pure_state(samples):
    """Asserts that the returned samples of the two qutrit pure state only contains expected states"""
    for sample in samples:
        assert (
            math.allequal(sample, [0, 2])
            or math.allequal(sample, [1, 0])
            or math.allequal(sample, [2, 1])
        )


def assert_correct_sampled_batched_two_qutrit_pure_state(samples):
    """Asserts that the returned samples of the two qutrit batched state only contains expected states"""
    # first batch of samples is always |11>
    assert np.all(samples[0] == 1)

    # second batch of samples is either |00> or |12>
    assert np.all(np.logical_or(samples[1] == [0, 0], samples[1] == [1, 2]))

    # third batch of samples can be any of |01>, |10>, |21>, or |22>
    third_batch_bool_1 = np.logical_or(samples[2] == [0, 1], samples[2] == [1, 0])
    third_batch_bool_2 = np.logical_or(samples[2] == [2, 1], samples[2] == [2, 2])
    assert np.all(np.logical_or(third_batch_bool_1, third_batch_bool_2))


class TestSampleState:
    """Test that the sample_state function works as expected"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_sample_state_basic(self, interface, two_qutrit_pure_state):
        """Tests that the returned samples are as expected."""
        state = math.array(two_qutrit_pure_state, like=interface)
        samples = sample_state(state, 10)
        assert samples.shape == (10, 2)
        assert samples.dtype == np.int64
        assert_correct_sampled_two_qutrit_pure_state(samples)

    @pytest.mark.jax
    def test_prng_key_as_seed_uses_sample_state_jax(self, mocker, two_qutrit_state):
        """Tests that sample_state calls _sample_state_jax if the seed is a JAX PRNG key"""
        import jax

        spy = mocker.spy(qml.devices.qutrit_mixed.sampling, "_sample_state_jax")
        state = qml.math.array(two_qutrit_state, like="jax")

        # prng_key specified, should call _sample_state_jax
        _ = sample_state(state, 10, prng_key=jax.random.PRNGKey(15))
        # prng_key defaults to None, should NOT call _sample_state_jax
        _ = sample_state(state, 10, rng=15)

        spy.assert_called_once()

    @pytest.mark.jax
    def test_sample_state_jax(self, two_qutrit_pure_state, seed):
        """Tests that the returned samples are as expected when explicitly calling _sample_state_jax."""
        import jax

        state = qml.math.array(two_qutrit_pure_state, like="jax")

        samples = _sample_state_jax(state, 10, prng_key=jax.random.PRNGKey(seed))

        assert samples.shape == (10, 2)
        assert samples.dtype == np.int64
        assert_correct_sampled_two_qutrit_pure_state(samples)

    @pytest.mark.jax
    def test_prng_key_determines_sample_state_jax_results(self, two_qutrit_pure_state):
        """Test that setting the seed as a JAX PRNG key determines the results for _sample_state_jax"""
        import jax

        state = qml.math.array(two_qutrit_pure_state, like="jax")

        samples = _sample_state_jax(state, shots=10, prng_key=jax.random.PRNGKey(12))
        samples2 = _sample_state_jax(state, shots=10, prng_key=jax.random.PRNGKey(12))
        samples3 = _sample_state_jax(state, shots=10, prng_key=jax.random.PRNGKey(13))

        assert np.all(samples == samples2)
        assert not np.allclose(samples, samples3)

    def test_sample_state_custom_rng(self, two_qutrit_state):
        """Tests that a custom RNG can be used with sample_state."""
        custom_rng = np.random.default_rng(12345)
        samples = sample_state(two_qutrit_state, 4, rng=custom_rng)
        expected = [[0, 2], [1, 0], [2, 1], [1, 2]]
        assert qml.math.allequal(samples, expected)

    def test_entangled_qutrit_samples_always_match(self, seed):
        """Tests that entangled qutrits are always in the same state."""
        num_samples = 10000

        bell_state_vector = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
        bell_state = get_dm_of_state(bell_state_vector, 2, 3)

        samples = sample_state(bell_state, num_samples, rng=seed)
        assert samples.shape == (num_samples, 2)
        assert not any(samples[:, 0] ^ samples[:, 1])  # all samples are entangled

        # all samples are approximately equivalently sampled
        assert np.isclose(
            math.count_nonzero(samples[:, 0] == 0) / num_samples,
            1 / 3,
            atol=APPROX_ATOL,
        )
        assert np.isclose(
            math.count_nonzero(samples[:, 0] == 1) / num_samples,
            1 / 3,
            atol=APPROX_ATOL,
        )
        assert np.isclose(
            math.count_nonzero(samples[:, 0] == 2) / num_samples,
            1 / 3,
            atol=APPROX_ATOL,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "num_wires,shuffled_wires",
        (
            (8, [7, 5, 3, 1, 6, 0, 4, 2]),
            (9, [4, 0, 2, 1, 5, 7, 6, 8, 3]),
        ),
    )
    def test_sample_state_many_wires(self, num_wires, shuffled_wires):
        """Tests that sample_state works as expected with many wires, and with re-ordering."""
        shots = 10000
        probs_shape = (QUDIT_DIM,) * num_wires
        state_shape = probs_shape * 2
        diagonal = np.arange(1, QUDIT_DIM**num_wires + 1, dtype=np.float64)
        diagonal = diagonal / np.sum(diagonal)
        state = np.diag(diagonal).reshape(state_shape)

        ordered_samples = sample_state(state, shots)
        ordered_probs = samples_to_probs(ordered_samples, num_wires)
        assert np.allclose(ordered_probs, diagonal, atol=APPROX_ATOL)

        random_samples = sample_state(state, shots, wires=shuffled_wires)
        random_probs = samples_to_probs(random_samples, num_wires)

        reordered_probs = ordered_probs.reshape(probs_shape).transpose(shuffled_wires).flatten()
        assert np.allclose(reordered_probs, random_probs, atol=APPROX_ATOL)


class TestMeasureWithSamples:
    """Test that the measure_with_samples function works as expected"""

    def test_sample_measure(self, two_qutrit_pure_state):
        """Test that a sample measurement works as expected"""
        shots = qml.measurements.Shots(100)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples(mp, two_qutrit_pure_state, shots=shots)

        assert result.shape == (shots.total_shots, 2)
        assert result.dtype == np.int64
        assert_correct_sampled_two_qutrit_pure_state(result)

    def test_sample_measure_single_wire(self):
        """Test that a sample measurement on a single wire works as expected"""
        state_vector = np.array([1, -1j, 1, 0, 0, 0, 0, 0, 0])
        state = get_dm_of_state(state_vector, 2, 3)
        shots = qml.measurements.Shots(100)

        mp0 = qml.sample(wires=0)
        mp1 = qml.sample(wires=1)

        result0 = measure_with_samples(mp0, state, shots=shots)
        result1 = measure_with_samples(mp1, state, shots=shots)

        assert result0.shape == (shots.total_shots, 1)
        assert result0.dtype == np.int64
        assert np.all(result0 == 0)

        assert result1.shape == (shots.total_shots, 1)
        assert result1.dtype == np.int64
        assert len(np.unique(result1)) == 3

    def test_approximate_sample_measure(self, two_qutrit_pure_state, seed):
        """Test that a sample measurement returns approximately the correct distribution"""
        shots = qml.measurements.Shots(10000)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples(mp, two_qutrit_pure_state, shots=shots, rng=seed)

        one_or_two_prob = np.count_nonzero(result[:, 0]) / result.shape[0]
        one_prob = np.count_nonzero(result[:, 0] == 1) / result.shape[0]
        assert np.allclose(one_or_two_prob, 2 / 3, atol=APPROX_ATOL)
        assert np.allclose(one_prob, 1 / 3, atol=APPROX_ATOL)

    def test_approximate_expval_measure(self, two_qutrit_state, seed):
        """Test that an expval measurement works as expected"""
        state = qml.math.array(two_qutrit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.expval(qml.GellMann(0, 1) @ qml.GellMann(1, 1))

        result = measure_with_samples(mp, state, shots=shots, rng=seed)

        gellmann_1_matrix = qml.GellMann.compute_matrix(1)
        observable_matrix = np.kron(gellmann_1_matrix, gellmann_1_matrix)
        expected = np.trace(observable_matrix @ state.reshape((9, 9)))

        assert isinstance(result, np.float64)
        assert np.allclose(result, expected, atol=APPROX_ATOL)

    def test_approximate_var_measure(self, two_qutrit_state, seed):
        """Test that a variance measurement works as expected"""
        state = qml.math.array(two_qutrit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.var(qml.GellMann(0, 1) @ qml.GellMann(1, 1))

        result = measure_with_samples(mp, state, shots=shots, rng=seed)

        gellmann_1_matrix = qml.GellMann.compute_matrix(1)
        obs_mat = np.kron(gellmann_1_matrix, gellmann_1_matrix)
        reshaped_state = state.reshape((9, 9))
        obs_squared = np.linalg.matrix_power(obs_mat, 2)
        expected = np.trace(obs_squared @ reshaped_state) - np.trace(obs_mat @ reshaped_state) ** 2

        assert isinstance(result, np.float64)
        assert np.allclose(result, expected, atol=APPROX_ATOL)

    @flaky
    def test_counts_measure(self, two_qutrit_pure_state):
        """Test that a counts measurement works as expected"""
        num_shots = 10000
        shots = qml.measurements.Shots(num_shots)
        mp = qml.counts()

        result = measure_with_samples(mp, two_qutrit_pure_state, shots=shots)

        assert isinstance(result, dict)
        assert sorted(result.keys()) == ["02", "10", "21"]
        assert np.isclose(result["02"] / num_shots, 1 / 3, atol=APPROX_ATOL)
        assert np.isclose(result["10"] / num_shots, 1 / 3, atol=APPROX_ATOL)
        assert np.isclose(result["21"] / num_shots, 1 / 3, atol=APPROX_ATOL)

    @flaky
    def test_counts_measure_single_wire(self):
        """Test that a counts measurement on a single wire works as expected"""
        state_vector = np.sqrt(np.array([5, -2j, 1, 0, 0, 0, 0, 0, 0]) / 8)
        state = get_dm_of_state(state_vector, 2)

        num_shots = 10000
        shots = qml.measurements.Shots(num_shots)

        mp0 = qml.counts(wires=0)
        mp1 = qml.counts(wires=1)

        result0 = measure_with_samples(mp0, state, shots=shots)
        result1 = measure_with_samples(mp1, state, shots=shots)

        assert isinstance(result1, dict)
        assert result0 == {"0": num_shots}

        assert isinstance(result1, dict)
        assert sorted(result1.keys()) == ["0", "1", "2"]
        assert np.isclose(result1["0"] / num_shots, 5 / 8, atol=APPROX_ATOL)
        assert np.isclose(result1["1"] / num_shots, 1 / 4, atol=APPROX_ATOL)
        assert np.isclose(result1["2"] / num_shots, 1 / 8, atol=APPROX_ATOL)

    def test_sample_observables(self):
        """Test that counts measurements properly counts samples of an observable"""
        state_vector = np.sqrt(np.array([0, 0, 0, 0, 2, 0, 1, 0, 1]) / 4)
        state = get_dm_of_state(state_vector, 2)
        num_shots = 100
        shots = qml.measurements.Shots(num_shots)

        results_gel_3 = measure_with_samples(qml.sample(qml.GellMann(0, 3)), state, shots=shots)
        assert results_gel_3.shape == (shots.total_shots,)
        assert results_gel_3.dtype == np.int64
        assert sorted(np.unique(results_gel_3)) == [-1, 0]

        results_gel_1s = measure_with_samples(
            qml.sample(qml.GellMann(0, 1) @ qml.GellMann(1, 1)), state, shots=shots
        )
        assert results_gel_1s.shape == (shots.total_shots,)
        assert results_gel_1s.dtype == np.float64
        assert sorted(np.unique(results_gel_1s)) == [-1, 0, 1]

    @flaky
    def test_counts_observables(self):
        """Test that a set of sample and counts measurements works as expected"""
        state_vector = np.sqrt(np.array([0, 0, 0, 0, 3, 0, 1, 0, 1]) / 5)
        state = get_dm_of_state(state_vector, 2)
        num_shots = 10000
        shots = qml.measurements.Shots(num_shots)

        results_gel_3 = measure_with_samples(qml.counts(qml.GellMann(0, 3)), state, shots=shots)

        assert isinstance(results_gel_3, dict)
        assert sorted(results_gel_3.keys()) == [-1, 0]
        assert np.isclose(results_gel_3[-1] / num_shots, 3 / 5, atol=APPROX_ATOL)
        assert np.isclose(results_gel_3[0] / num_shots, 2 / 5, atol=APPROX_ATOL)

        results_gel_1s = measure_with_samples(
            qml.counts(qml.GellMann(0, 1) @ qml.GellMann(1, 1)), state, shots=shots
        )
        assert isinstance(results_gel_1s, dict)
        assert sorted(results_gel_1s.keys()) == [-1, 0, 1]
        assert np.isclose(results_gel_1s[-1] / num_shots, 0.3, atol=APPROX_ATOL)
        assert np.isclose(results_gel_1s[0] / num_shots, 0.4, atol=APPROX_ATOL)
        assert np.isclose(results_gel_1s[1] / num_shots, 0.3, atol=APPROX_ATOL)


class TestInvalidSampling:
    """Tests for non-expected states and inputs."""

    @pytest.mark.parametrize("shots", [10, [10, 10]])
    def test_only_catch_nan_errors(self, shots):
        """Test that when probabilities don't add to 1 Error is thrown."""
        state = np.zeros((3,) * QUDIT_DIM * 2).astype(np.complex128)
        mp = qml.sample(wires=range(2))
        _shots = Shots(shots)

        with pytest.raises(ValueError, match=r"(?i)probabilities do not sum to 1"):
            _ = measure_with_samples(mp, state, _shots)

    @pytest.mark.parametrize("mp", [qml.probs(0), qml.probs(op=qml.GellMann(0, 1))])
    def test_currently_unsupported_observable(self, mp, two_qutrit_state):
        """Test sample measurements that are not sample, counts, expval,
        or var raise a NotImplementedError."""
        shots = qml.measurements.Shots(1)
        with pytest.raises(NotImplementedError):
            _ = measure_with_samples(mp, two_qutrit_state, shots)


shots_to_test_samples = [
    ((100, 2),),
    (100, 100),
    (100, 100),
    (100, 100, 200),
    (200, (100, 2)),
]

shots_to_test_nonsamples = [
    ((10000, 2),),
    (10000, 10000),
    (10000, 20000),
    (10000, 10000, 20000),
    (20000, (10000, 2)),
]


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    def test_sample_measure(self, batched_two_qutrit_pure_state, seed):
        """Test that broadcasting works for qml.sample and single shots"""
        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(100)
        state = batched_two_qutrit_pure_state

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(measurement, state, shots, is_state_batched=True, rng=rng)

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

    def test_counts_measure(self, batched_two_qutrit_pure_state, seed):
        """Test that broadcasting works for qml.sample and single shots"""
        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(100)
        state = batched_two_qutrit_pure_state

        measurement = qml.counts(wires=[0, 1])
        res = measure_with_samples(measurement, state, shots, is_state_batched=True, rng=rng)

        assert len(res) == 3
        assert list(res[0].keys()) == ["11"]
        assert list(res[1].keys()) == ["00", "12"]
        assert list(res[2].keys()) == ["01", "10", "21", "22"]

    @pytest.mark.parametrize("shots", shots_to_test_samples)
    def test_sample_measure_shot_vector(self, shots, batched_two_qutrit_pure_state, seed):
        """Test that broadcasting works for qml.sample and shot vectors"""
        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            measurement,
            batched_two_qutrit_pure_state,
            shots,
            is_state_batched=True,
            rng=rng,
        )

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert r.shape == (3, s, 2)
            assert r.dtype == np.int64

            assert_correct_sampled_batched_two_qutrit_pure_state(r)

    # pylint:disable = too-many-arguments
    @pytest.mark.parametrize(
        "shots",
        shots_to_test_nonsamples,
    )
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (qml.expval(qml.GellMann(1, 3)), np.array([1 / 2, 1, 0])),
            (qml.var(qml.GellMann(1, 3)), np.array([1 / 4, 0, 1])),
        ],
    )
    def test_nonsample_measure_shot_vector(self, shots, measurement, expected, seed):
        """Test that broadcasting works for the other sample measurements and shot vectors"""

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        state = [
            get_dm_of_state(np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0]]), 2, 2),
            get_dm_of_state(np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]), 2, 3),
            get_dm_of_state(np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0]]), 2, 4),
        ]

        state = np.stack(state)

        res = measure_with_samples(measurement, state, shots, is_state_batched=True, rng=rng)

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for r in res:
            assert r.shape == expected.shape
            assert np.allclose(r, expected, atol=APPROX_ATOL)


@pytest.mark.jax
class TestBroadcastingPRNG:
    """Test that measurements work and use _sample_state_jax when the state has a batch dim
    and a PRNG key is provided"""

    def test_sample_measure(self, mocker, batched_two_qutrit_pure_state, seed):
        """Test that broadcasting works for qml.sample and single shots"""
        import jax

        spy = mocker.spy(qml.devices.qutrit_mixed.sampling, "_sample_state_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(100)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            measurement,
            batched_two_qutrit_pure_state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )

        spy.assert_called()

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

        # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
        res = [np.array(r) for r in res]

        assert_correct_sampled_batched_two_qutrit_pure_state(res)

    @pytest.mark.parametrize("shots", shots_to_test_samples)
    def test_sample_measure_shot_vector(self, mocker, shots, batched_two_qutrit_pure_state, seed):
        """Test that broadcasting works for qml.sample and shot vectors"""
        import jax

        jax.config.update("jax_enable_x64", True)

        spy = mocker.spy(qml.devices.qutrit_mixed.sampling, "_sample_state_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            measurement,
            batched_two_qutrit_pure_state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )

        spy.assert_called()

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert r.shape == (3, s, 2)
            assert res[0][0].dtype == np.int64

            # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
            r = [np.array(i) for i in r]

            assert_correct_sampled_batched_two_qutrit_pure_state(r)

    # pylint:disable = too-many-arguments
    @pytest.mark.parametrize("shots", shots_to_test_nonsamples)
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (qml.expval(qml.GellMann(1, 3)), np.array([1 / 2, 1, 0])),
            (qml.var(qml.GellMann(1, 3)), np.array([1 / 4, 0, 1])),
        ],
    )
    def test_nonsample_measure_shot_vector(self, mocker, shots, measurement, expected, seed):
        """Test that broadcasting works for the other sample measurements and shot vectors"""
        import jax

        spy = mocker.spy(qml.devices.qutrit_mixed.sampling, "_sample_state_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        state = [
            get_dm_of_state(np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0]]), 2, 2),
            get_dm_of_state(np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]), 2, 3),
            get_dm_of_state(np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0]]), 2, 4),
        ]
        state = np.stack(state)

        res = measure_with_samples(
            measurement,
            state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )

        spy.assert_called()

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for r in res:
            assert r.shape == expected.shape
            assert np.allclose(r, expected, atol=0.03)


@pytest.mark.parametrize(
    "obs",
    [
        qml.Hamiltonian([0.8, 0.5], [qml.GellMann(0, 3), qml.GellMann(0, 1)]),
        qml.s_prod(0.8, qml.GellMann(0, 3)) + qml.s_prod(0.5, qml.GellMann(0, 1)),
    ],
)
class TestHamiltonianSamples:
    """Test that the measure_with_samples function works as expected for
    Hamiltonian and Sum observables"""

    def test_hamiltonian_expval(self, obs, seed):
        """Test that sampling works well for Hamiltonian and Sum observables"""

        shots = qml.measurements.Shots(10000)
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.TRY(x, wires=0), qml.TRZ(y, wires=0)]
        state = create_initial_state((0,))
        for op in ops:
            state = apply_operation(op, state)

        res = measure_with_samples(qml.expval(obs), state, shots=shots, rng=seed)

        expected = 0.8 * np.cos(x) + 0.5 * np.cos(y) * np.sin(x)
        assert isinstance(res, np.float64)
        assert np.allclose(res, expected, atol=APPROX_ATOL)

    def test_hamiltonian_expval_shot_vector(self, obs, seed):
        """Test that sampling works well for Hamiltonian and Sum observables with a shot vector"""

        shots = qml.measurements.Shots((10000, 100000))
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.TRY(x, wires=0), qml.TRZ(y, wires=0)]
        state = create_initial_state((0,))
        for op in ops:
            state = apply_operation(op, state)

        res = measure_with_samples(qml.expval(obs), state, shots=shots, rng=seed)

        expected = 0.8 * np.cos(x) + 0.5 * np.cos(y) * np.sin(x)

        assert len(res) == 2
        assert isinstance(res, tuple)
        assert np.allclose(res[0], expected, atol=APPROX_ATOL)
        assert np.allclose(res[1], expected, atol=APPROX_ATOL)


class TestSampleProbs:
    # pylint: disable=attribute-defined-outside-init
    @pytest.fixture(autouse=True)
    def setup(self, request):
        seed = request.getfixturevalue("seed")
        self.rng = np.random.default_rng(seed)
        self.shots = 1000

    def test_sample_probs_basic(self, seed):
        probs = np.array([0.2, 0.3, 0.5])
        num_wires = 1
        is_state_batched = False

        result = sample_probs(probs, self.shots, num_wires, is_state_batched, self.rng)

        assert result.shape == (self.shots, num_wires)
        assert np.all(result >= 0) and np.all(result < QUDIT_DIM)

        _, counts = np.unique(result, return_counts=True)
        observed_probs = counts / self.shots
        np.testing.assert_allclose(observed_probs, probs, atol=0.05)

    def test_sample_probs_multi_wire(self, seed):
        probs = np.array(
            [0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]
        )  # 3^2 = 9 probabilities for 2 wires
        num_wires = 2
        is_state_batched = False

        result = sample_probs(probs, self.shots, num_wires, is_state_batched, self.rng)

        assert result.shape == (self.shots, num_wires)
        assert np.all(result >= 0) and np.all(result < QUDIT_DIM)

    def test_sample_probs_batched(self, seed):
        probs = np.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])
        num_wires = 1
        is_state_batched = True

        result = sample_probs(probs, self.shots, num_wires, is_state_batched, self.rng)

        assert result.shape == (2, self.shots, num_wires)
        assert np.all(result >= 0) and np.all(result < QUDIT_DIM)

    @pytest.mark.parametrize(
        "probs,num_wires,is_state_batched,expected_shape",
        [
            (np.array([0.2, 0.3, 0.5]), 1, False, (1000, 1)),
            (np.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]), 2, False, (1000, 2)),
            (np.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]), 1, True, (2, 1000, 1)),
        ],
    )
    def test_sample_probs_shapes(self, probs, num_wires, is_state_batched, expected_shape, seed):
        result = sample_probs(probs, self.shots, num_wires, is_state_batched, self.rng)
        assert result.shape == expected_shape

    def test_invalid_probs(self, seed):
        probs = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )  # 4 probabilities, which is invalid for qutrit system
        num_wires = 2
        is_state_batched = False

        with pytest.raises(ValueError, match=MISMATCH_ERROR):
            sample_probs(probs, self.shots, num_wires, is_state_batched, self.rng)


class TestSampleProbsJax:
    # pylint: disable=attribute-defined-outside-init
    @pytest.fixture(autouse=True)
    def setup(self, request):
        import jax

        seed = request.getfixturevalue("seed")
        self.jax_key = jax.random.PRNGKey(seed)
        self.shots = 1000

    @pytest.mark.jax
    def test_sample_probs_jax_basic(self, seed):
        probs = np.array([0.2, 0.3, 0.5])
        num_wires = 1
        is_state_batched = False
        state_len = 1

        result = _sample_probs_jax(
            probs, self.shots, num_wires, is_state_batched, self.jax_key, state_len
        )

        assert result.shape == (self.shots, num_wires)
        assert np.all(result >= 0) and qml.math.all(result < QUDIT_DIM)

        _, counts = qml.math.unique(result, return_counts=True)
        observed_probs = counts / self.shots
        np.testing.assert_allclose(observed_probs, probs, atol=0.05)

    @pytest.mark.jax
    def test_sample_probs_jax_multi_wire(self, seed):
        probs = qml.math.array(
            [0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]
        )  # 3^2 = 9 probabilities for 2 wires
        num_wires = 2
        is_state_batched = False
        state_len = 1

        result = _sample_probs_jax(
            probs, self.shots, num_wires, is_state_batched, self.jax_key, state_len
        )

        assert result.shape == (self.shots, num_wires)
        assert qml.math.all(result >= 0) and qml.math.all(result < QUDIT_DIM)

    @pytest.mark.jax
    def test_sample_probs_jax_batched(self, seed):
        probs = qml.math.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])
        num_wires = 1
        is_state_batched = True
        state_len = 2

        result = _sample_probs_jax(
            probs, self.shots, num_wires, is_state_batched, self.jax_key, state_len
        )

        assert result.shape == (2, self.shots, num_wires)
        assert qml.math.all(result >= 0) and qml.math.all(result < QUDIT_DIM)

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "probs,num_wires,is_state_batched,expected_shape,state_len",
        [
            (qml.math.array([0.2, 0.3, 0.5]), 1, False, (1000, 1), 1),
            (
                qml.math.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]),
                2,
                False,
                (1000, 2),
                1,
            ),
            (qml.math.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]), 1, True, (2, 1000, 1), 2),
        ],
    )
    @pytest.mark.jax
    def test_sample_probs_jax_shapes(
        self, probs, num_wires, is_state_batched, expected_shape, state_len, seed
    ):
        result = _sample_probs_jax(
            probs, self.shots, num_wires, is_state_batched, self.jax_key, state_len
        )
        assert result.shape == expected_shape

    @pytest.mark.jax
    def test_invalid_probs_jax(self, seed):
        probs = qml.math.array(
            [0.1, 0.2, 0.3, 0.4]
        )  # 4 probabilities, which is invalid for qutrit system
        num_wires = 2
        is_state_batched = False
        state_len = 1

        with pytest.raises(ValueError):  # error msg determined by jax, not us
            _sample_probs_jax(
                probs, self.shots, num_wires, is_state_batched, self.jax_key, state_len
            )
