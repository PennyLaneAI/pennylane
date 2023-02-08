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
"""Functions to sample a state."""

import pennylane as qml
from pennylane import numpy as np


def sample_state(state, shots: int, rng=None) -> np.ndarray:
    """
    Returns a series of samples of a state.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used

    Returns:
        ndarray: Sample values of the shape (shots, num_wires)
    """
    rng = np.random.default_rng(rng)
    num_wires = len(state.shape)
    basis_states = np.arange(2**num_wires)

    flat_state = qml.math.flatten(state)
    real_state = qml.math.real(flat_state)
    imag_state = qml.math.imag(flat_state)
    probs = real_state**2 + imag_state**2

    samples = rng.choice(basis_states, shots, p=probs)
    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(np.int64)[..., ::-1]
