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


def sample_state(state, shots: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Returns a series of samples of a state.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        rng (Optional[np.random.Generator]): A random number generator to create samples.
            If no RNG is provided, a default one will be used

    Returns:
        ndarray: Sample values of the shape (shots, num_wires)
    """
    rng = np.random.default_rng(rng)

    flat_state = qml.math.flatten(state)
    real_state = qml.math.real(flat_state)
    imag_state = qml.math.imag(flat_state)
    rotated_probs = real_state**2 + imag_state**2

    num_wires = len(state.shape)
    basis_states = np.arange(2**num_wires)
    samples = rng.choice(basis_states, shots, p=rotated_probs)
    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(np.int64)[..., ::-1]
