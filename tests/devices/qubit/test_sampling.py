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

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qubit import sample_state

two_qubit_state = np.array([[0, 1j], [-1, 0]]) / np.sqrt(2)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
def test_sample_state_basic(interface):
    """Tests that the returned samples are as expected."""
    state = qml.math.array(two_qubit_state, like=interface)
    samples = sample_state(state, 10)
    assert samples.shape == (10, 2)
    assert samples.dtype == np.bool8
    assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)


def test_sample_state_custom_rng():
    """Tests that a custom RNG can be used with sample_state."""
    custom_rng = np.random.default_rng(12345)
    samples = sample_state(two_qubit_state, 4, rng=custom_rng)
    expected = [[0, 1], [0, 1], [1, 0], [1, 0]]
    assert qml.math.allequal(samples, expected)
