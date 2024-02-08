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
import pytest
import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit.simulate import _FlexShots  # TODO, should I switch this
from pennylane.devices.qutrit_mixed.sampling import (
    sample_state,
    _sample_state_jax,
    measure_with_samples,
)
from pennylane.measurements import Shots

APPROX_ATOL = 0.05
QUDIT_DIM = 3
ONE_QUTRIT = 1
TWO_QUTRITS = 2
THREE_QUTRITS = 3

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


def get_dm_of_state(state_vector, num_qudits, normalization=1):
    state = np.outer(np.conj(state_vector), state_vector) / normalization
    return state.reshape((QUDIT_DIM,) * num_qudits * 2)


@pytest.fixture(name="two_qutrit_pure_state")
def fixture_two_qutrit_pure_state():
    state_vector = np.array([0, 0, 1.0j, -1.0, 0, 0, 0, 1.0, 0], dtype=np.complex128)
    return get_dm_of_state(state_vector, TWO_QUTRITS, 3)


@pytest.fixture(name="batched_qutrit_pure_state")
def fixture_batched_qutrit_pure_state():
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
        if not (
            math.allequal(sample, [0, 2])
            or math.allequal(sample, [1, 0])
            or math.allequal(sample, [2, 1])
        ):
            assert (
                math.allequal(sample, [0, 2])
                or math.allequal(sample, [1, 0])
                or math.allequal(sample, [2, 1])
            )
            return


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
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
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
    def test_sample_state_jax(self, two_qutrit_state):
        """Tests that the returned samples are as expected when explicitly calling _sample_state_jax."""
        import jax

        state = qml.math.array(two_qutrit_state, like="jax")

        samples = _sample_state_jax(state, 10, prng_key=jax.random.PRNGKey(84))

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

    @pytest.mark.skip(reason="Broken due to measure")
    @pytest.mark.parametrize("wire_order", [[2], [2, 0], [0, 2, 1]])
    def test_marginal_sample_state(self, wire_order):
        """Tests that marginal states can be sampled as expected."""
        state = np.zeros((QUDIT_DIM,) * THREE_QUTRITS * 2)
        state[..., 1] = 0.5  # third wire is always 1
        alltrue_axis = wire_order.index(2)

        samples = sample_state(state, 20, wires=wire_order)
        assert all(samples[:, alltrue_axis] == 1)

    def test_sample_state_custom_rng(self, two_qutrit_state):
        """Tests that a custom RNG can be used with sample_state."""
        custom_rng = np.random.default_rng(12345)
        samples = sample_state(two_qutrit_state, 4, rng=custom_rng)
        expected = [[0, 2], [1, 0], [2, 1], [1, 2]]
        assert qml.math.allequal(samples, expected)

    def test_approximate_probs_from_samples(self, three_qutrit_state):
        """Tests that the generated samples are approximately as expected."""
        shots = 20000
        state = three_qutrit_state

        flat_state = state.reshape((QUDIT_DIM**THREE_QUTRITS,) * 2)
        expected_probs = np.abs(np.diag(flat_state))

        samples = sample_state(state, shots)
        approx_probs = samples_to_probs(samples, THREE_QUTRITS)
        assert np.allclose(approx_probs, expected_probs, atol=APPROX_ATOL)

    def test_entangled_qutrit_samples_always_match(self):
        """Tests that entangled qutrits are always in the same state."""
        num_samples = 10000

        bell_state_vector = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
        bell_state = np.outer(np.conj(bell_state_vector), bell_state_vector) / 3
        bell_state = math.reshape(bell_state, (QUDIT_DIM,) * TWO_QUTRITS * 2)

        samples = sample_state(bell_state, num_samples)
        assert samples.shape == (num_samples, 2)
        assert not any(samples[:, 0] ^ samples[:, 1])  # all samples are entangled

        # all samples are approximately equivalently sampled
        assert np.isclose(
            math.count_nonzero(samples[:, 0] == 0) / num_samples, 1 / 3, atol=APPROX_ATOL
        )
        assert np.isclose(
            math.count_nonzero(samples[:, 0] == 1) / num_samples, 1 / 3, atol=APPROX_ATOL
        )
        assert np.isclose(
            math.count_nonzero(samples[:, 0] == 2) / num_samples, 1 / 3, atol=APPROX_ATOL
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


class TestMeasureSamples:
    """Test that the measure_with_samples function works as expected"""

    def test_sample_measure(self, two_qutrit_pure_state):
        """Test that a sample measurement works as expected"""
        shots = qml.measurements.Shots(100)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], two_qutrit_pure_state, shots=shots)[0]

        assert result.shape == (shots.total_shots, 2)
        assert result.dtype == np.int64
        assert_correct_sampled_two_qutrit_pure_state(result)

    def test_sample_measure_single_wire(self):
        """Test that a sample measurement on a single wire works as expected"""
        state_vector = np.array([1, -1j, 1, 0, 0, 0, 0, 0, 0])
        state = np.outer(np.conj(state_vector), state_vector) / 3
        state = np.reshape(state, (QUDIT_DIM,) * TWO_QUTRITS * 2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.sample(wires=0)
        mp1 = qml.sample(wires=1)

        result0 = measure_with_samples([mp0], state, shots=shots)[0]
        result1 = measure_with_samples([mp1], state, shots=shots)[0]

        assert result0.shape == (shots.total_shots,)
        assert result0.dtype == np.int64
        assert np.all(result0 == 0)

        assert result1.shape == (shots.total_shots,)
        assert result1.dtype == np.int64
        assert len(np.unique(result1)) == 3

    def test_approximate_sample_measure(self, two_qutrit_pure_state):
        """Test that a sample measurement returns approximately the correct distribution"""
        shots = qml.measurements.Shots(10000)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], two_qutrit_pure_state, shots=shots, rng=123)[0]

        one_or_two_prob = np.count_nonzero(result[:, 0]) / result.shape[0]
        one_prob = np.count_nonzero(result[:, 0] == 1) / result.shape[0]
        assert np.allclose(one_or_two_prob, 2 / 3, atol=APPROX_ATOL)
        assert np.allclose(one_prob, 1 / 3, atol=APPROX_ATOL)

    # TODO: add 2 sample mps
    # TODO: add counts test
    # TODO: add mixed counts and sample test


class TestInvalidStateSamples:
    """Tests for mixed state matrices containing nan values or shot vectors with zero shots."""

    @pytest.mark.parametrize("shots", [10, [10, 10]])
    def test_only_catch_nan_errors(self, shots):
        """Test that errors are only caught if they are raised due to nan values in the state."""
        state = np.zeros((3,) * QUDIT_DIM * 2).astype(np.complex128)
        mp = qml.sample(wires=range(2))
        _shots = Shots(shots)

        with pytest.raises(ValueError, match="probabilities do not sum to 1"):
            _ = measure_with_samples([mp], state, _shots)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "mp", [qml.sample(wires=0), qml.sample(op=qml.GellMann(0, 1)), qml.sample(wires=[0, 1])]
    )
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "tensorflow", "jax"])
    @pytest.mark.parametrize("shots", [0, [0, 0]])
    def test_nan_samples(self, mp, interface, shots):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((3,) * QUDIT_DIM * 2, np.NaN, like=interface)
        res = measure_with_samples((mp,), state, _FlexShots(shots), is_state_batched=False)

        if not isinstance(shots, list):
            assert isinstance(res, tuple)
            res = res[0]
            assert qml.math.shape(res) == (shots,) if len(mp.wires) == 1 else (shots, len(mp.wires))

        else:
            assert isinstance(res, tuple)
            assert len(res) == 2
            for i, r in enumerate(res):
                assert isinstance(r, tuple)
                r = r[0]
                assert (
                    qml.math.shape(r) == (shots[i],)
                    if len(mp.wires) == 1
                    else (shots[i], len(mp.wires))
                )


shots_to_test = [
    ((100, 2),),
    (100, 100),
    (100, 100),
    (100, 100, 200),
    (200, (100, 2)),
]


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    def test_sample_measure(self, batched_qutrit_pure_state):
        """Test that broadcasting works for qml.sample and single shots"""
        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(100)
        state = batched_qutrit_pure_state

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=rng)[0]

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

    @pytest.mark.parametrize("shots", shots_to_test)
    def test_sample_measure_shot_vector(self, shots, batched_qutrit_pure_state):
        """Test that broadcasting works for qml.sample and shot vectors"""
        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(shots)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement], batched_qutrit_pure_state, shots, is_state_batched=True, rng=rng
        )

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == (3, s, 2)
            assert r.dtype == np.int64

            assert_correct_sampled_batched_two_qutrit_pure_state(r)


@pytest.mark.jax
class TestBroadcastingPRNG:
    """Test that measurements work and use _sample_state_jax when the state has a batch dim
    and a PRNG key is provided"""

    def test_sample_measure(self, mocker, batched_qutrit_pure_state):
        """Test that broadcasting works for qml.sample and single shots"""
        import jax

        spy = mocker.spy(qml.devices.qutrit_mixed.sampling, "_sample_state_jax")

        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(100)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement],
            batched_qutrit_pure_state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(184),
        )[0]

        spy.assert_called()

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

        # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
        res = [np.array(r) for r in res]

        assert_correct_sampled_batched_two_qutrit_pure_state(res)

    @pytest.mark.parametrize("shots", shots_to_test)
    def test_sample_measure_shot_vector(self, mocker, shots, batched_qutrit_pure_state):
        """Test that broadcasting works for qml.sample and shot vectors"""
        import jax
        jax.config.update("jax_enable_x64", True)

        spy = mocker.spy(qml.devices.qutrit_mixed.sampling, "_sample_state_jax")

        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(shots)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement],
            batched_qutrit_pure_state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(184),
        )

        spy.assert_called()

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == (3, s, 2)
            assert res[0][0].dtype == np.int64

            # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
            r = [np.array(i) for i in r]

            assert_correct_sampled_batched_two_qutrit_pure_state(r)
