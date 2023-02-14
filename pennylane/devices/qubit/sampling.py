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


def sample_state(state, shots: int, wires=None, rng=None) -> np.ndarray:
    """
    Returns a series of samples of a state.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        wires (Sequence[int]): The wires to sample
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used

    Returns:
        ndarray[bool]: Sample values of the shape (shots, num_wires)
    """
    rng = np.random.default_rng(rng)
    state_wires = qml.wires.Wires(range(len(state.shape)))
    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(2**num_wires)

    probs = qml.probs(wires=wires_to_sample).process_state(state, state_wires)
    samples = rng.choice(basis_states, shots, p=probs)
    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    return (samples[..., None] & powers_of_two).astype(np.bool8)
