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
from random import shuffle
import pytest
import numpy as np

import pennylane as qml
from pennylane.devices.qutrit_mixed.sampling import (
    sample_state,
    _sample_state_jax,
    measure_with_samples,
)

two_qutrit_pure_state = #TODO np.array([[0, 1j], [-1, 0]], dtype=np.complex128) / np.sqrt(2)  # TODO change
APPROX_ATOL = 0.01


@pytest.fixture(name="init_state")
def fixture_init_state():
    """Generates a random initial state"""

    def _init_state(n):
        """random initial state"""
        state = np.random.random([1 << n]) + np.random.random([1 << n]) * 1j  # TODO change
        state /= np.linalg.norm(state)
        return state.reshape((3,) * n * 2)

    return _init_state


def _valid_flex_int(s):
    """Returns True if s is a non-negative integer."""
    return isinstance(s, int) and s >= 0


def _valid_flex_tuple(s):
    """Returns True if s is a tuple of the form (shots, copies)."""
    return (
        isinstance(s, tuple)
        and len(s) == 2
        and _valid_flex_int(s[0])
        and isinstance(s[1], int)
        and s[1] > 0
    )


def samples_to_probs(samples, num_wires):
    """Converts samples to probs"""
    samples_decimal = [np.ravel_multi_index(sample, [3] * num_wires) for sample in samples]
    counts = [0] * (3**num_wires)

    for sample in samples_decimal:
        counts[sample] += 1

    return np.array(counts, dtype=np.float64) / len(samples)


class TestSampleState:
    """Test that the sample_state function works as expected"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_sample_state_basic(self, interface):
        """Tests that the returned samples are as expected."""
        state = qml.math.array(two_qutrit_pure_state, like=interface)  # TODO
        samples = sample_state(state, 10)
        assert samples.shape == (10, 2)
        assert samples.dtype == np.int64
        # TODO assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)

    @pytest.mark.jax
    def test_prng_key_as_seed_uses_sample_state_jax(self, mocker):
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
    def test_sample_state_jax(self):
        """Tests that the returned samples are as expected when explicitly calling _sample_state_jax."""
        import jax

        state = qml.math.array(two_qutrit_state, like="jax")

        samples = _sample_state_jax(state, 10, prng_key=jax.random.PRNGKey(84))

        assert samples.shape == (10, 2)
        assert samples.dtype == np.int64
        # TODO assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)

    @pytest.mark.jax
    def test_prng_key_determines_sample_state_jax_results(self, two_qutrit_state):
        """Test that setting the seed as a JAX PRNG key determines the results for _sample_state_jax"""
        import jax

        state = qml.math.array(two_qutrit_state, like="jax")

        samples = _sample_state_jax(state, shots=10, prng_key=jax.random.PRNGKey(12))
        samples2 = _sample_state_jax(state, shots=10, prng_key=jax.random.PRNGKey(12))
        samples3 = _sample_state_jax(state, shots=10, prng_key=jax.random.PRNGKey(13))

        assert np.all(samples == samples2)
        assert not np.allclose(samples, samples3)

    @pytest.mark.parametrize("wire_order", [[2], [2, 0], [0, 2, 1]])
    def test_marginal_sample_state(self, wire_order):  # TODO
        """Tests that marginal states can be sampled as expected."""
        state = np.zeros((2, 2, 2))
        state[:, :, 1] = 0.5  # third wire is always 1
        alltrue_axis = wire_order.index(2)

        samples = sample_state(state, 20, wires=wire_order)
        assert all(samples[:, alltrue_axis])

    def test_sample_state_custom_rng(self, two_qutrit_state):
        """Tests that a custom RNG can be used with sample_state."""
        custom_rng = np.random.default_rng(12345)
        samples = sample_state(two_qutrit_state, 4, rng=custom_rng)
        expected = [[0, 1], [0, 1], [1, 0], [1, 0]]  # TODO change to fit new RNG
        assert qml.math.allequal(samples, expected)

    def test_approximate_probs_from_samples(self, init_state):
        """Tests that the generated samples are approximately as expected."""
        n = 4
        shots = 20000
        state = init_state(n)

        flat_state = state.flatten()
        expected_probs = np.real(flat_state) ** 2 + np.imag(flat_state) ** 2

        samples = sample_state(state, shots)
        approx_probs = samples_to_probs(samples, n)
        assert np.allclose(approx_probs, expected_probs, atol=APPROX_ATOL)

    def test_entangled_qubit_samples_always_match(self):
        """Tests that entangled qubits are always in the same state."""
        bell_state = np.array([[1, 0], [0, 1]]) / np.sqrt(
            2
        )  # TODO change to qutrit_mixed BellState
        samples = sample_state(bell_state, 1000)
        assert samples.shape == (1000, 2)
        assert not any(samples[:, 0] ^ samples[:, 1])  # all samples are entangled
        assert not all(samples[:, 0])  # some samples are |00>
        assert any(samples[:, 0])  # ...and some are |11>!

    @pytest.mark.slow
    @pytest.mark.parametrize("num_wires", [13, 14, 15, 16])
    def test_sample_state_many_wires(self, num_wires):
        """Tests that sample_state works as expected with many wires, and with re-ordering."""
        shots = 10000
        shape = (3,) * num_wires
        flat_state = np.arange(1, 2**num_wires + 1, dtype=np.float64)
        original_norm = np.linalg.norm(flat_state)
        flat_state /= original_norm
        state = flat_state.reshape(shape)
        expected_probs = np.real(flat_state) ** 2 + np.imag(flat_state) ** 2

        ordered_samples = sample_state(state, shots)
        ordered_probs = samples_to_probs(ordered_samples, num_wires)
        assert np.allclose(ordered_probs, expected_probs, atol=APPROX_ATOL)

        random_wires = list(range(num_wires))
        shuffle(random_wires)
        random_samples = sample_state(state, shots, wires=random_wires)
        random_probs = samples_to_probs(random_samples, num_wires)

        reordered_probs = ordered_probs.reshape(shape).transpose(random_wires).flatten()
        assert np.allclose(reordered_probs, random_probs, atol=APPROX_ATOL)


class TestMeasureSamples:
    """Test that the measure_with_samples function works as expected"""

    def test_sample_measure(self):
        """Test that a sample measurement works as expected"""
        state = qml.math.array(two_qutrit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots)[0]

        assert result.shape == (shots.total_shots, 2)
        assert result.dtype == np.int64
        assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in result)

    def test_sample_measure_single_wire(self):
        """Test that a sample measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
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
        assert len(np.unique(result1)) == 2

    def test_approximate_sample_measure(self):
        """Test that a sample measurement returns approximately the correct distribution"""
        state = qml.math.array(two_qutrit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots, rng=123)[0]

        one_prob = np.count_nonzero(result[:, 0]) / result.shape[0]
        assert np.allclose(one_prob, 0.5, atol=0.05)

    def test_approximate_prob_measure(self):
        """Test that a probability measurement works as expected"""
        state = qml.math.array(two_qutrit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots, rng=123)[0]

        assert np.allclose(result[1], 0.5, atol=0.05)
        assert np.allclose(result[2], 0.5, atol=0.05)
        assert result[1] + result[2] == 1


class TestInvalidStateSamples:
    """Tests for mixed state matrices containing nan values or shot vectors with zero shots."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "mp", [qml.sample(wires=0), qml.sample(op=qml.PauliZ(0)), qml.sample(wires=[0, 1])]
    )
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "tensorflow", "jax"])
    @pytest.mark.parametrize("shots", [0, [0, 0]])
    def test_nan_samples(self, mp, interface, shots):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.NaN, like=interface)
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


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    def test_sample_measure(self):
        """Test that broadcasting works for qml.sample and single shots"""
        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(100)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=rng)[0]

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

        # first batch of samples is always |11>
        assert np.all(res[0] == 1)

        # second batch of samples is either |00> or |10>
        assert np.all(np.logical_or(res[1] == [0, 0], res[1] == [1, 0]))

        # third batch of samples can be any of |00>, |01>, |10>, or |11>
        assert np.all(np.logical_or(res[2] == 0, res[2] == 1))

    def test_sample_measure_shot_vector(self, shots):
        """Test that broadcasting works for qml.sample and shot vectors"""
        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(shots)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=rng)

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == (3, s, 2)
            assert r.dtype == np.int64

            # first batch of samples is always |11>
            assert np.all(r[0] == 1)

            # second batch of samples is either |00> or |10>
            assert np.all(np.logical_or(r[1] == [0, 0], r[1] == [1, 0]))

            # third batch of samples can be any of |00>, |01>, |10>, or |11>
            assert np.all(np.logical_or(r[2] == 0, r[2] == 1))


@pytest.mark.jax
class TestBroadcastingPRNG:
    """Test that measurements work and use _sample_state_jax when the state has a batch dim
    and a PRNG key is provided"""

    def test_sample_measure(self, mocker):
        """Test that broadcasting works for qml.sample and single shots"""
        import jax

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_state_jax")

        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(100)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement],
            state,
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

        # first batch of samples is always |11>
        assert np.all(res[0] == 1)

        # second batch of samples is either |00> or |10>
        assert np.all(np.logical_or(res[1] == [0, 0], res[1] == [1, 0]))

        # third batch of samples can be any of |00>, |01>, |10>, or |11>
        assert np.all(np.logical_or(res[2] == 0, res[2] == 1))

    def test_sample_measure_shot_vector(self, mocker, shots):
        """Test that broadcasting works for qml.sample and shot vectors"""
        import jax

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_state_jax")

        rng = np.random.default_rng(123)
        shots = qml.measurements.Shots(shots)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement],
            state,
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
            # this is has started randomly failing do to r.dtype being int32 instead of int64.
            # Not sure why they are getting returned as 32 instead, but maybe this will fix it?
            assert res[0][0].dtype in [np.int32, np.int64]

            # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
            r = [np.array(i) for i in r]

            # first batch of samples is always |11>
            assert np.all(r[0] == 1)

            # second batch of samples is either |00> or |10>
            assert np.all(np.logical_or(r[1] == [0, 0], r[1] == [1, 0]))

            # third batch of samples can be any of |00>, |01>, |10>, or |11>
            assert np.all(np.logical_or(r[2] == 0, r[2] == 1))
