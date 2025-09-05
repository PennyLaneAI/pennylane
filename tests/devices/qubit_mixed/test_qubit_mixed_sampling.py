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
"""Unit tests for sampling states in devices/qubit_mixed."""

# pylint: disable=unused-argument,too-many-arguments, import-outside-toplevel

import numpy as np
import pytest

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit_mixed import create_initial_state, measure_with_samples, sample_state
from pennylane.measurements import Shots

# Tolerance for approximate equality checks
APPROX_ATOL = 0.05

# List of ML frameworks for parameterization
ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


@pytest.fixture(name="two_qubit_pure_state")
def fixture_two_qubit_pure_state():
    """Returns a two-qubit pure state density matrix."""
    state_vector = np.array([0, 1, 1, 1], dtype=np.complex128) / np.sqrt(3)
    return get_dm_of_state(state_vector, 2)


@pytest.fixture(name="batched_two_qubit_pure_state")
def fixture_batched_two_qubit_pure_state():
    """Returns a batch of two-qubit pure state density matrices."""
    state_vectors = [
        np.array([0, 0, 0, 1]),  # |11>
        np.array([1, 0, 1, 0]) / np.sqrt(2),  # (|00> + |10>)/√2
        np.array([0, 1, 1, 1]) / np.sqrt(3),  # (|01> + |10> + |11>)/√3
    ]
    states = [get_dm_of_state(state_vector, 2) for state_vector in state_vectors]
    return math.stack(states)


def get_dm_of_state(state_vector, num_qubits):
    """Creates a density matrix from a state vector.

    Args:
        state_vector (array): Input quantum state vector
        num_qubits (int): Number of qubits
    Returns:
        array: Density matrix reshaped for num_qubits qubits
    """
    state = np.outer(state_vector, np.conj(state_vector))
    return state.reshape((2,) * num_qubits * 2)


def samples_to_probs(samples, num_wires):
    """Converts samples to probability distribution.

    Args:
        samples (array): Measurement samples
        num_wires (int): Number of wires/qubits
    Returns:
        array: Array of probabilities
    """
    samples_decimal = [np.ravel_multi_index(sample, [2] * num_wires) for sample in samples]
    counts = np.bincount(samples_decimal, minlength=2**num_wires)
    return counts / len(samples)


def assert_correct_sampled_two_qubit_pure_state(samples):
    """Asserts that samples only contain valid qubit states for the given pure state."""
    valid_states = [np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    for sample in samples:
        assert any(
            np.array_equal(sample, state) for state in valid_states
        ), f"Invalid sample: {sample}"


def assert_correct_sampled_batched_two_qubit_pure_state(samples):
    """Asserts that batched samples contain valid qubit states."""
    # First batch: samples should be [1, 1]
    assert np.all(samples[0] == 1), "First batch samples are not all [1, 1]"

    # Second batch: samples are either [0, 0] or [1, 0]
    second_batch_valid = np.all(
        np.logical_or(
            np.all(samples[1] == [0, 0], axis=1),
            np.all(samples[1] == [1, 0], axis=1),
        )
    )
    assert second_batch_valid, "Second batch samples are invalid"

    # Third batch: samples can be [0,1], [1,0], or [1,1]
    valid_states = [np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    for sample in samples[2]:
        assert any(
            np.array_equal(sample, state) for state in valid_states
        ), f"Invalid sample in third batch: {sample}"


class TestSampleState:
    """Test core sampling functionality"""

    @pytest.mark.parametrize("interface", ml_frameworks_list)
    def test_basic_sampling(self, interface, two_qubit_pure_state):
        """Test sampling across different ML interfaces."""
        state = qml.math.array(two_qubit_pure_state, like=interface)
        samples = sample_state(state, 10)
        assert samples.shape == (10, 2)
        assert samples.dtype.kind == "i"
        assert_correct_sampled_two_qubit_pure_state(samples)

    @pytest.mark.parametrize(
        "state_vector, expected_ratio",
        [
            (np.array([1, 0, 0, 1]) / np.sqrt(2), 1.0),  # Bell state |00> + |11>
            (np.array([1, 1, 0, 0]) / np.sqrt(2), 0.5),  # Prod state |00> + |01>
            (np.array([1, 0, 1, 0]) / np.sqrt(2), 0.5),  # Prod state |00> + |10>
        ],
    )
    def test_entangled_states(self, state_vector, expected_ratio):
        """Test sampling from various entangled/separable states."""
        state = get_dm_of_state(state_vector, 2)
        samples = sample_state(state, 10000)
        ratio = np.mean([s[0] == s[1] for s in samples])
        assert np.isclose(
            ratio, expected_ratio, atol=APPROX_ATOL
        ), f"Ratio {ratio} deviates from expected {expected_ratio}"

    @pytest.mark.parametrize("num_shots", [100, 1000])
    @pytest.mark.parametrize("seed", [42, 123, 987])
    def test_reproducibility(self, num_shots, seed, two_qubit_pure_state):
        """Test reproducibility with different shots and seeds."""
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        samples1 = sample_state(two_qubit_pure_state, num_shots, rng=rng1)
        samples2 = sample_state(two_qubit_pure_state, num_shots, rng=rng2)
        assert np.array_equal(samples1, samples2), "Samples with the same seed are not equal"

    @pytest.mark.parametrize("num_shots", [100, 1000])
    @pytest.mark.parametrize("seed1, seed2", [(42, 43), (123, 124), (987, 988)])
    def test_different_seeds_produce_different_samples(
        self, num_shots, seed1, seed2, two_qubit_pure_state
    ):
        """Test that different seeds produce different samples."""
        rng1 = np.random.default_rng(seed1)
        rng2 = np.random.default_rng(seed2)
        samples1 = sample_state(two_qubit_pure_state, num_shots, rng=rng1)
        samples2 = sample_state(two_qubit_pure_state, num_shots, rng=rng2)
        assert not np.array_equal(samples1, samples2), "Samples with different seeds are equal"

    def test_invalid_state(self):
        """Test error handling for invalid states."""
        invalid_state = np.zeros((2, 2, 2, 2))  # Zero state is invalid
        import re

        with pytest.raises(
            ValueError, match=re.compile(r"probabilities.*do not sum to 1", re.IGNORECASE)
        ):
            sample_state(invalid_state, 10)


class TestMeasurements:
    """Test different measurement types"""

    @pytest.mark.parametrize("num_shots", [100, 1000])
    @pytest.mark.parametrize("wires", [(0,), (1,), (0, 1)])
    def test_sample_measurement(self, num_shots, wires, two_qubit_pure_state):
        """Test sample measurements with different shots and wire configurations."""
        shots = Shots(num_shots)
        result = measure_with_samples([qml.sample(wires=wires)], two_qubit_pure_state, shots)[0]
        expected_shape = (num_shots, len(wires))
        assert result.shape == expected_shape, f"Result shape mismatch: {result.shape}"
        # Additional assertions to check the validity of the samples
        valid_values = [0, 1]
        assert np.all(np.isin(result, valid_values)), "Samples contain invalid values"

    @pytest.mark.parametrize("num_shots", [1000, 5000])
    def test_counts_measurement(self, num_shots, two_qubit_pure_state):
        """Test counts measurement."""
        shots = Shots(num_shots)
        result = measure_with_samples([qml.counts()], two_qubit_pure_state, shots)[0]
        assert isinstance(result, dict), "Result is not a dictionary"
        total_counts = sum(result.values())
        assert (
            total_counts == num_shots
        ), f"Total counts {total_counts} do not match shots {num_shots}"
        # Check that keys represent valid states
        valid_states = {"01", "10", "11"}
        assert set(result.keys()).issubset(
            valid_states
        ), f"Invalid states in counts: {result.keys()}"

    @pytest.mark.parametrize("num_shots", [100, 500])
    def test_counts_measurement_all_outcomes(self, num_shots, two_qubit_pure_state):
        """Test counts measurement with all_outcomes=True."""
        shots = Shots(num_shots)
        result = measure_with_samples([qml.counts(all_outcomes=True)], two_qubit_pure_state, shots)[
            0
        ]

        assert isinstance(result, dict), "Result is not a dictionary"
        total_counts = sum(result.values())
        assert (
            total_counts == num_shots
        ), f"Total counts {total_counts} do not match shots {num_shots}"

        # Check that all possible 2-qubit states are present in the keys
        all_possible_states = {"00", "01", "10", "11"}
        assert (
            set(result.keys()) == all_possible_states
        ), f"Missing states in counts: {all_possible_states - set(result.keys())}"

        # Check that only '01', '10', '11' have non-zero counts (based on two_qubit_pure_state fixture)
        assert result["00"] == 0, "State '00' should have zero counts"
        assert (
            sum(result[state] > 0 for state in ["01", "10", "11"]) == 3
        ), "Expected non-zero counts for '01', '10', and '11'"

    @pytest.mark.parametrize(
        "observable",
        [
            qml.PauliX(0),
            qml.PauliY(0),
            qml.PauliZ(0),
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
        ],
    )
    @pytest.mark.parametrize("measurement", [qml.expval, qml.var])
    def test_observable_measurements(self, observable, measurement, two_qubit_pure_state):
        """Test different observables with expectation and variance."""
        shots = Shots(10000)
        result = measure_with_samples([measurement(observable)], two_qubit_pure_state, shots)[0]
        if measurement is qml.expval:
            assert -1 <= result <= 1, f"Expectation value {result} out of bounds"
        else:
            assert 0 <= result <= 1, f"Variance {result} out of bounds"

    @pytest.mark.parametrize(
        "coeffs, obs",
        [
            ([1.0, 0.5], [qml.PauliX(0), qml.PauliZ(0)]),
            ([0.3, 0.7], [qml.PauliY(0), qml.PauliX(0)]),
            ([0.5, 0.5], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]),
        ],
    )
    def test_hamiltonian_measurement(self, coeffs, obs):
        """Test measuring a Hamiltonian observable."""
        # Determine the set of wires used in the Hamiltonian
        hamiltonian_wires = set()
        for o in obs:
            hamiltonian_wires.update(o.wires)

        # Create the initial state with the required number of qubits
        num_wires = max(hamiltonian_wires) + 1
        state = create_initial_state(range(num_wires))  # Adjusted to include all wires

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        shots = Shots(10000)

        result = measure_with_samples(
            [qml.expval(hamiltonian)],
            state,
            shots,
        )[0]
        assert isinstance(result, (float, np.floating)), "Result is not a floating point number"

    def test_measure_sum_with_samples_partitioned_shots(self):
        """Test measuring a Sum observable with partitioned shots."""
        # Create a simple state
        state = create_initial_state((0, 1))

        # Define a Sum observable
        obs = qml.PauliZ(0) + qml.PauliX(1)

        # Wrap it in an expectation measurement process
        mp = qml.expval(obs)

        # Define partitioned shots
        shots = Shots((100, 200))

        # Perform measurement
        result = measure_with_samples(
            [mp],
            state,
            shots,
        )

        # Check that result is a tuple of results
        assert isinstance(result, tuple), "Result is not a tuple for partitioned shots"
        assert len(result) == 2, f"Result length {len(result)} does not match expected length 2"


class TestBatchedOperations:
    """Test batched state handling"""

    @pytest.mark.parametrize("num_shots", [100, 500])
    @pytest.mark.parametrize("batch_size", [2, 3, 4])
    def test_batched_sampling(self, num_shots, batch_size):
        """Test sampling with different batch sizes."""
        # Create batch of random normalized states
        states = [np.random.rand(4) + 1j * np.random.rand(4) for _ in range(batch_size)]
        states = [state / np.linalg.norm(state) for state in states]
        batched_states = math.stack([get_dm_of_state(state, 2) for state in states])
        samples = sample_state(batched_states, num_shots, is_state_batched=True)
        assert samples.shape == (
            batch_size,
            num_shots,
            2,
        ), f"Samples shape mismatch: {samples.shape}"

    @pytest.mark.parametrize(
        "shots",
        [
            Shots(100),
            Shots((100, 200)),
            Shots((100, 200, 300)),
            Shots((200, (100, 2))),
        ],
    )
    def test_batched_measurements_shots(self, shots, batched_two_qubit_pure_state):
        """Test measurements with different shot configurations."""
        result = measure_with_samples(
            [qml.sample(wires=[0, 1])], batched_two_qubit_pure_state, shots, is_state_batched=True
        )
        batch_size = len(batched_two_qubit_pure_state)
        if shots.has_partitioned_shots:
            assert isinstance(result, tuple), "Result is not a tuple for partitioned shots"
            assert (
                len(result) == shots.num_copies
            ), f"Result length {len(result)} does not match number of shot copies {shots.num_copies}"
            for res, shot in zip(result, shots.shot_vector):
                assert res[0].shape == (
                    batch_size,
                    shot.shots,
                    2,
                ), f"Result shape {res[0].shape} does not match expected shape"
        else:
            assert result[0].shape == (
                batch_size,
                shots.total_shots,
                2,
            ), f"Result shape {result.shape} does not match expected shape"

    @pytest.mark.parametrize("shots", [Shots(1000)])
    def test_batched_expectation_measurement(self, shots):
        """Test expectation value measurements on batched states."""
        # Create batched states
        state_vectors = [
            np.array([1, 0]),  # |0⟩
            np.array([0, 1]),  # |1⟩
        ]
        states = [get_dm_of_state(state_vector, 1) for state_vector in state_vectors]
        batched_states = math.stack(states)

        # Define an observable
        obs = qml.PauliZ(0)

        # Perform measurement
        result = measure_with_samples(
            [qml.expval(obs)],
            batched_states,
            shots,
            is_state_batched=True,
        )[0]

        # Check the results
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        expected_results = np.array([1.0, -1.0])
        assert np.allclose(
            result, expected_results, atol=APPROX_ATOL
        ), f"Results {result} differ from expected {expected_results}"


class TestJaxSampling:
    """Test sampling functionality using JAX."""

    @pytest.mark.jax
    def test_sample_state_jax(self):
        """Test sampling from a quantum state using JAX."""
        import jax
        import jax.numpy as jnp

        # Define a simple state vector for a single qubit |0>
        state_vector = jnp.array([1, 0], dtype=jnp.complex64)
        state = qml.math.reshape(jnp.outer(state_vector, jnp.conj(state_vector)), (2, 2))

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(0)

        # Sample from the state
        samples = sample_state(state, shots=10, prng_key=prng_key)

        # The samples should all be 0 for this state
        assert samples.shape == (10, 1)
        assert jnp.all(samples == 0), f"Samples contain non-zero values: {samples}"

    @pytest.mark.jax
    def test_sample_state_jax_entangled_state(self):
        """Test sampling from an entangled state using JAX."""
        import jax
        import jax.numpy as jnp

        # Define a Bell state |00> + |11>
        state_vector = jnp.array([1, 0, 0, 1], dtype=jnp.complex64) / jnp.sqrt(2)
        state = qml.math.reshape(jnp.outer(state_vector, jnp.conj(state_vector)), (2, 2, 2, 2))

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(42)

        # Sample from the state
        samples = sample_state(state, shots=1000, prng_key=prng_key)

        # Samples should show that qubits are correlated
        # Count how many times qubits are equal
        equal_counts = jnp.sum(samples[:, 0] == samples[:, 1])
        ratio = equal_counts / 1000
        assert jnp.isclose(
            ratio, 1.0, atol=APPROX_ATOL
        ), f"Ratio {ratio} deviates from expected 1.0"

    @pytest.mark.jax
    def test_sample_state_jax_reproducibility(self):
        """Test that sampling with the same PRNG key produces the same samples."""
        import jax
        import jax.numpy as jnp

        # Define a simple state vector for a single qubit |+>
        state_vector = jnp.array([1, 1], dtype=jnp.complex64) / jnp.sqrt(2)
        state = qml.math.reshape(jnp.outer(state_vector, jnp.conj(state_vector)), (2, 2))

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(0)

        # Sample from the state twice with the same key
        samples1 = sample_state(state, shots=100, prng_key=prng_key)
        samples2 = sample_state(state, shots=100, prng_key=prng_key)

        # The samples should be the same
        assert jnp.array_equal(samples1, samples2), "Samples with the same PRNG key are not equal"

    @pytest.mark.jax
    def test_sample_state_jax_different_keys(self):
        """Test that sampling with different PRNG keys produces different samples."""
        import jax
        import jax.numpy as jnp

        # Define a simple state vector for a single qubit |+>
        state_vector = jnp.array([1, 1], dtype=jnp.complex64) / jnp.sqrt(2)
        state = qml.math.reshape(jnp.outer(state_vector, jnp.conj(state_vector)), (2, 2))

        # Set different PRNG keys
        prng_key1 = jax.random.PRNGKey(0)
        prng_key2 = jax.random.PRNGKey(1)

        # Sample from the state with different keys
        samples1 = sample_state(state, shots=100, prng_key=prng_key1)
        samples2 = sample_state(state, shots=100, prng_key=prng_key2)

        # The samples should be different
        assert not jnp.array_equal(samples1, samples2), "Samples with different PRNG keys are equal"

    @pytest.mark.jax
    def test_measure_with_samples_jax(self):
        """Test measure_with_samples using JAX."""
        import jax
        import jax.numpy as jnp

        # Define a simple state vector for a single qubit |0>
        state_vector = jnp.array([1, 0], dtype=jnp.complex64)
        state = qml.math.reshape(jnp.outer(state_vector, jnp.conj(state_vector)), (2, 2))

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(0)

        # Define a measurement process
        mp = qml.sample(wires=0)

        # Perform measurement
        shots = Shots(10)
        result = measure_with_samples([mp], state, shots, prng_key=prng_key)[0]

        # The result should be zeros
        assert result.shape == (10, 1)
        assert jnp.all(result == 0), f"Measurement results contain non-zero values: {result}"

    @pytest.mark.jax
    def test_measure_with_samples_jax_entangled_state(self):
        """Test measure_with_samples with an entangled state using JAX."""
        import jax
        import jax.numpy as jnp

        # Define a Bell state |00> + |11>
        state_vector = jnp.array([1, 0, 0, 1], dtype=jnp.complex64) / jnp.sqrt(2)
        state = qml.math.reshape(jnp.outer(state_vector, jnp.conj(state_vector)), (2, 2, 2, 2))

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(42)

        # Define a measurement process
        mp = qml.sample(wires=[0, 1])

        # Perform measurement
        shots = Shots(1000)
        result = measure_with_samples([mp], state, shots, prng_key=prng_key)[0]

        # Samples should show that qubits are correlated
        # Count how many times qubits are equal
        equal_counts = jnp.sum(result[:, 0] == result[:, 1])
        ratio = equal_counts / 1000
        assert jnp.isclose(
            ratio, 1.0, atol=APPROX_ATOL
        ), f"Ratio {ratio} deviates from expected 1.0"

    @pytest.mark.jax
    def test_sample_state_jax_batched(self):
        """Test sampling from a batched state using JAX."""
        import jax
        import jax.numpy as jnp

        # Define two state vectors for single qubits |0> and |1>
        state_vectors = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
        # Convert to density matrices and batch them
        states = jnp.array(
            [
                qml.math.reshape(jnp.outer(state_vectors[0], jnp.conj(state_vectors[0])), (2, 2)),
                qml.math.reshape(jnp.outer(state_vectors[1], jnp.conj(state_vectors[1])), (2, 2)),
            ]
        )

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(0)

        # Sample from the batched state
        samples = sample_state(states, shots=10, is_state_batched=True, prng_key=prng_key)

        # The samples should be [0] for first state and [1] for second state
        assert samples.shape == (2, 10, 1)
        assert jnp.all(samples[0] == 0), f"First batch samples are not all zero: {samples[0]}"
        assert jnp.all(samples[1] == 1), f"Second batch samples are not all one: {samples[1]}"

    @pytest.mark.jax
    def test_measure_with_samples_jax_batched(self):
        """Test measure_with_samples with a batched state using JAX."""
        import jax
        import jax.numpy as jnp

        # Define two state vectors for single qubits |+> and |->
        state_vectors = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        # Convert to density matrices and batch them
        states = jnp.array(
            [
                qml.math.reshape(jnp.outer(state_vectors[0], jnp.conj(state_vectors[0])), (2, 2)),
                qml.math.reshape(jnp.outer(state_vectors[1], jnp.conj(state_vectors[1])), (2, 2)),
            ]
        )

        # Set the PRNG key
        prng_key = jax.random.PRNGKey(0)

        # Define a measurement process (PauliX measurement)
        mp = qml.sample(qml.PauliX(0))

        # Perform measurement
        shots = Shots(1000)
        result = measure_with_samples(
            [mp], states, shots, is_state_batched=True, prng_key=prng_key
        )[0]

        # The first batch should have all +1 eigenvalues, the second all -1
        assert result.shape == (2, 1000)
        assert jnp.all(result[0] == 1), f"First batch measurements are not all +1: {result[0]}"
        assert jnp.all(result[1] == -1), f"Second batch measurements are not all -1: {result[1]}"
