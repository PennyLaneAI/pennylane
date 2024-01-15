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
"""
Code relevant for sampling a qutrit mixed state.
"""
from typing import List, Union, Tuple

import numpy as np
import pennylane as qml


# pylint:disable = too-many-arguments
def measure_with_samples(
    mps: List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
) -> List[TensorLike]:
    """
    Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        mp (List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]]):
            The sample measurements to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        List[TensorLike[Any]]: Sample measurement results
    """
    pass  # TODO: decide what needs to be implemented, probably just a call to sample state


def sample_state(
    state,
    shots: int,
    is_state_batched: bool = False,
    wires=None,
    rng=None,
    prng_key=None,
) -> np.ndarray:
    """
    Returns a series of samples of a state.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        wires (Sequence[int]): The wires to sample
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """
    # if prng_key is not None:
    #     return _sample_state_jax(
    #         state, shots, prng_key, is_state_batched=is_state_batched, wires=wires
    #     )
    #
    # rng = np.random.default_rng(rng)
    #
    # total_indices = len(state.shape) - is_state_batched
    # state_wires = qml.wires.Wires(range(total_indices))
    #
    # wires_to_sample = wires or state_wires
    # num_wires = len(wires_to_sample)
    # basis_states = np.arange(2**num_wires)
    #
    # flat_state = flatten_state(state, total_indices)
    # with qml.queuing.QueuingManager.stop_recording():
    #     probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)
    #
    # if is_state_batched:
    #     # rng.choice doesn't support broadcasting
    #     samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    # else:
    #     samples = rng.choice(basis_states, shots, p=probs)
    #
    # powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    # states_sampled_base_ten = samples[..., None] & powers_of_two
    # return (states_sampled_base_ten > 0).astype(np.int64)
    pass


# pylint:disable = unused-argument
def _sample_state_jax(
    state,
    shots: int,
    prng_key,
    is_state_batched: bool = False,
    wires=None,
) -> np.ndarray:
    """
    Returns a series of samples of a state for the JAX interface based on the PRNG.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        prng_key (jax.random.PRNGKey): A``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator.
        is_state_batched (bool): whether the state is batched or not
        wires (Sequence[int]): The wires to sample

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """
    # pylint: disable=import-outside-toplevel
    # import jax
    # import jax.numpy as jnp
    #
    # key = prng_key
    #
    # total_indices = len(state.shape) - is_state_batched
    # state_wires = qml.wires.Wires(range(total_indices))
    #
    # wires_to_sample = wires or state_wires
    # num_wires = len(wires_to_sample)
    # basis_states = np.arange(2**num_wires)
    #
    # flat_state = flatten_state(state, total_indices)
    # with qml.queuing.QueuingManager.stop_recording():
    #     probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)
    #
    # if is_state_batched:
    #     # Produce separate keys for each of the probabilities along the broadcasted axis
    #     keys = []
    #     for _ in state:
    #         key, subkey = jax.random.split(key)
    #         keys.append(subkey)
    #     samples = jnp.array(
    #         [
    #             jax.random.choice(_key, basis_states, shape=(shots,), p=prob)
    #             for _key, prob in zip(keys, probs)
    #         ]
    #     )
    # else:
    #     samples = jax.random.choice(key, basis_states, shape=(shots,), p=probs)
    #
    # powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    # states_sampled_base_ten = samples[..., None] & powers_of_two
    # return (states_sampled_base_ten > 0).astype(np.int64)
    pass
