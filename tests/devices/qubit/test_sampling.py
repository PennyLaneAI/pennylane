# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for sample_state in devices/qubit."""

from random import shuffle
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qubit import sample_state, measure_with_samples

two_qubit_state = np.array([[0, 1j], [-1, 0]]) / np.sqrt(2)
APPROX_ATOL = 0.01


@pytest.fixture(name="init_state")
def fixture_init_state():
    """Generates a random initial state"""

    def _init_state(n):
        """random initial state"""
        state = np.random.random([1 << n]) + np.random.random([1 << n]) * 1j
        state /= np.linalg.norm(state)
        return state.reshape((2,) * n)

    return _init_state


def samples_to_probs(samples, num_wires):
    """Converts samples to probs"""
    samples_decimal = [np.ravel_multi_index(sample, [2] * num_wires) for sample in samples]
    counts = [0] * (2**num_wires)

    for sample in samples_decimal:
        counts[sample] += 1

    return np.array(counts, dtype=np.float64) / len(samples)


class TestSampleState:
    """Test that the sample_state function works as expected"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_sample_state_basic(self, interface):
        """Tests that the returned samples are as expected."""
        state = qml.math.array(two_qubit_state, like=interface)
        samples = sample_state(state, 10)
        assert samples.shape == (10, 2)
        assert samples.dtype == np.bool8
        assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)

    @pytest.mark.parametrize("wire_order", [[2], [2, 0], [0, 2, 1]])
    def test_marginal_sample_state(self, wire_order):
        """Tests that marginal states can be sampled as expected."""
        state = np.zeros((2, 2, 2))
        state[:, :, 1] = 0.5  # third wire is always 1
        alltrue_axis = wire_order.index(2)

        samples = sample_state(state, 20, wires=wire_order)
        assert all(samples[:, alltrue_axis])

    def test_sample_state_custom_rng(self):
        """Tests that a custom RNG can be used with sample_state."""
        custom_rng = np.random.default_rng(12345)
        samples = sample_state(two_qubit_state, 4, rng=custom_rng)
        expected = [[0, 1], [0, 1], [1, 0], [1, 0]]
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
        bell_state = np.array([[1, 0], [0, 1]]) / np.sqrt(2)
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
        shape = (2,) * num_wires
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
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples(mp, state, shots=shots)

        assert result.shape == (shots.total_shots, 2)
        assert result.dtype == np.bool8
        assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in result)

    def test_sample_measure_single_wire(self):
        """Test that a sample measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.sample(wires=0)
        mp1 = qml.sample(wires=1)

        result0 = measure_with_samples(mp0, state, shots=shots)
        result1 = measure_with_samples(mp1, state, shots=shots)

        assert result0.shape == (shots.total_shots, 1)
        assert result0.dtype == np.bool8
        assert np.all(result0 == 0)

        assert result1.shape == (shots.total_shots, 1)
        assert result1.dtype == np.bool8
        assert len(np.unique(result1)) == 2

    def test_prob_measure(self):
        """Test that a probability measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples(mp, state, shots=shots)

        assert result.shape == (4,)
        assert result[0] == 0
        assert result[3] == 0
        assert result[1] + result[2] == 1

    def test_prob_measure_single_wire(self):
        """Test that a probability measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.probs(wires=0)
        mp1 = qml.probs(wires=1)

        result0 = measure_with_samples(mp0, state, shots=shots)
        result1 = measure_with_samples(mp1, state, shots=shots)

        assert result0.shape == (2,)
        assert result1.shape == (2,)

        assert result0[0] == 1
        assert result0[1] == 0
        assert result1[0] != 0
        assert result1[1] != 0
        assert result1[0] + result1[1] == 1

    def test_expval_measure(self):
        """Test that an expval measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples(mp, state, shots=shots)

        assert result.shape == ()
        assert result == -1

    def test_expval_measure_single_wire(self):
        """Test that an expval measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.expval(qml.PauliZ(0))
        mp1 = qml.expval(qml.PauliY(1))

        result0 = measure_with_samples(mp0, state, shots=shots)
        result1 = measure_with_samples(mp1, state, shots=shots)

        assert result0.shape == ()
        assert result1.shape == ()
        assert result0 == 1
        assert result1 == -1

    def test_var_measure(self):
        """Test that a variance measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.var(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples(mp, state, shots=shots)

        assert result.shape == ()
        assert result == 0

    def test_var_measure_single_wire(self):
        """Test that a variance measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.var(qml.PauliZ(0))
        mp1 = qml.var(qml.PauliY(1))

        result0 = measure_with_samples(mp0, state, shots=shots)
        result1 = measure_with_samples(mp1, state, shots=shots)

        assert result0.shape == ()
        assert result1.shape == ()
        assert result0 == 0
        assert result1 == 0

    def test_approximate_sample_measure(self):
        """Test that a sample measurement returns approximately the correct distribution"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples(mp, state, shots=shots, rng=123)

        one_prob = np.count_nonzero(result[:, 0]) / result.shape[0]
        assert np.allclose(one_prob, 0.5, atol=0.05)

    def test_approximate_prob_measure(self):
        """Test that a probability measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples(mp, state, shots=shots, rng=123)

        assert np.allclose(result[1], 0.5, atol=0.05)
        assert np.allclose(result[2], 0.5, atol=0.05)
        assert result[1] + result[2] == 1

    def test_approximate_expval_measure(self):
        """Test that an expval measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.expval(qml.prod(qml.PauliX(0), qml.PauliX(1)))

        result = measure_with_samples(mp, state, shots=shots, rng=123)

        assert result != 0
        assert np.allclose(result, 0, atol=0.05)

<<<<<<< HEAD
    def test_approximate_var_measure(self):
        """Test that a variance measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.var(qml.prod(qml.PauliX(0), qml.PauliX(1)))

        result = measure_with_samples(mp, state, shots=shots, rng=123)

        assert result != 1
        assert np.allclose(result, 1, atol=0.05)

=======
>>>>>>> e4fd474b7 (Add more tests)
    @pytest.mark.parametrize(
        "shots, total_copies",
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
    def test_sample_measure_shot_vector(self, shots, total_copies):
        """Test that a sample measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples(mp, state, shots=shots)

        if total_copies == 1:
            assert isinstance(result, np.ndarray)
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res, sh in zip(result, shots):
            assert res.shape == (sh, 2)
            assert res.dtype == np.bool8
            assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in res)

    @pytest.mark.parametrize(
        "shots, total_copies",
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
    def test_prob_measure_shot_vector(self, shots, total_copies):
        """Test that a probability measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples(mp, state, shots=shots)

        if total_copies == 1:
            assert isinstance(result, np.ndarray)
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res in result:
            assert res.shape == (4,)
            assert res[0] == 0
            assert res[3] == 0
            assert res[1] + res[2] == 1

    @pytest.mark.parametrize(
        "shots, total_copies",
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
    def test_expval_measure_shot_vector(self, shots, total_copies):
        """Test that an expval measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples(mp, state, shots=shots)

        if total_copies == 1:
            assert isinstance(result, np.float64)
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res in result:
            assert res.shape == ()
            assert res == -1

    @pytest.mark.parametrize(
        "shots, total_copies",
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
    def test_var_measure_shot_vector(self, shots, total_copies):
        """Test that a variance measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.var(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples(mp, state, shots=shots)

        if total_copies == 1:
            assert isinstance(result, np.float64)
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res in result:
            assert res.shape == ()
            assert res == 0
