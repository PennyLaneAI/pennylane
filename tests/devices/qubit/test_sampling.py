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
from pennylane.devices.qubit import sample_state

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
    samples_decimal = [np.ravel_multi_index(sample, [2] * num_wires) for sample in samples]
    counts = [0] * (2**num_wires)

    for sample in samples_decimal:
        counts[sample] += 1

    return np.array(counts, dtype=np.float64) / len(samples)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
def test_sample_state_basic(interface):
    """Tests that the returned samples are as expected."""
    state = qml.math.array(two_qubit_state, like=interface)
    samples = sample_state(state, 10)
    assert samples.shape == (10, 2)
    assert samples.dtype == np.bool8
    assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)


@pytest.mark.parametrize("wire_order", [[2], [2, 0], [0, 2, 1]])
def test_marginal_sample_state(wire_order):
    """Tests that marginal states can be sampled as expected."""
    state = np.zeros((2, 2, 2))
    state[:, :, 1] = 0.5  # third wire is always 1
    alltrue_axis = wire_order.index(2)

    samples = sample_state(state, 20, wires=wire_order)
    assert all(samples[:, alltrue_axis])


def test_sample_state_custom_rng():
    """Tests that a custom RNG can be used with sample_state."""
    custom_rng = np.random.default_rng(12345)
    samples = sample_state(two_qubit_state, 4, rng=custom_rng)
    expected = [[0, 1], [0, 1], [1, 0], [1, 0]]
    assert qml.math.allequal(samples, expected)


def test_approximate_probs_from_samples(init_state):
    """Tests that the generated samples are approximately as expected."""
    n = 4
    shots = 20000
    state = init_state(n)

    flat_state = state.flatten()
    expected_probs = np.real(flat_state) ** 2 + np.imag(flat_state) ** 2

    samples = sample_state(state, shots)
    approx_probs = samples_to_probs(samples, n)
    assert np.allclose(approx_probs, expected_probs, atol=APPROX_ATOL)


def test_entangled_qubit_samples_always_match():
    """Tests that entangled qubits are always in the same state."""
    bell_state = np.array([[1, 0], [0, 1]]) / np.sqrt(2)
    samples = sample_state(bell_state, 1000)
    assert samples.shape == (1000, 2)
    assert not any(samples[:, 0] ^ samples[:, 1])  # all samples are entangled
    assert not all(samples[:, 0])  # some samples are |00>
    assert any(samples[:, 0])  # ...and some are |11>!


@pytest.mark.slow
@pytest.mark.parametrize("num_wires", [13, 14, 15, 16])
def test_sample_state_many_wires(num_wires):
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
