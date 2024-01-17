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
from typing import List, Union

import numpy as np
import pennylane as qml
from pennylane.measurements import (
    SampleMeasurement,
    Shots,
    SampleMP,
    CountsMP,
)
from pennylane.typing import TensorLike

from .utils import QUDIT_DIM, get_num_wires
from .measure import measure


# pylint:disable = too-many-arguments
def measure_with_samples(
    mps: List[SampleMeasurement],  # todo FIX
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
        mps (List[SampleMP]):
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
    for mp in mps:
        if mp.obs is not None:
            raise NotImplementedError

    total_indices = len(state.shape) - is_state_batched
    wires = qml.wires.Wires(range(total_indices))

    def _process_single_shot(samples):
        processed = []
        for mp in mps:
            if isinstance(mp, SampleMP):
                res = mp.process_samples(samples, wires)
                res = qml.math.squeeze(res)
            elif isinstance(mp, CountsMP):
                res = mp.process_samples(samples, wires)
            else:
                raise NotImplementedError
            processed.append(res)
        return tuple(processed)

    # if there is a shot vector, build a list containing results for each shot entry
    if shots.has_partitioned_shots:
        processed_samples = []
        for s in shots:
            # currently we call sample_state for each shot entry, but it may be
            # better to call sample_state just once with total_shots, then use
            # the shot_range keyword argument
            try:
                samples = sample_state(
                    state,
                    shots=s,
                    is_state_batched=is_state_batched,
                    wires=wires,
                    rng=rng,
                    prng_key=prng_key,
                )
            except ValueError as e:
                if str(e) != "probabilities contain NaN":
                    raise e
                samples = qml.math.full((s, len(wires)), 0)

            processed_samples.append(_process_single_shot(samples))

        return processed_samples  # TODO: figure out what is right

    try:
        samples = sample_state(
            state,
            shots=shots.total_shots,
            is_state_batched=is_state_batched,
            wires=wires,
            rng=rng,
            prng_key=prng_key,
        )
    except ValueError as e:
        if str(e) != "probabilities contain NaN":
            raise e
        samples = qml.math.full((shots.total_shots, len(wires)), 0)

    return _process_single_shot(samples)


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
    if prng_key is not None:
        return _sample_state_jax(
            state, shots, prng_key, is_state_batched=is_state_batched, wires=wires
        )

    rng = np.random.default_rng(rng)

    total_indices = get_num_wires(state, is_state_batched)
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(QUDIT_DIM**num_wires)

    with qml.queuing.QueuingManager.stop_recording():  # TODO: stops probs from being added to queue
        probs = measure(qml.probs(wires=wires_to_sample), state, is_state_batched)

    if is_state_batched:
        # rng.choice doesn't support broadcasting
        samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    else:
        samples = rng.choice(basis_states, shots, p=probs)

    return _transform_samples_to_trinary(samples, num_wires)


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
    import jax
    import jax.numpy as jnp

    key = prng_key

    total_indices = get_num_wires(state, is_state_batched)
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(QUDIT_DIM**num_wires)

    with qml.queuing.QueuingManager.stop_recording():  # TODO: stops probs from being added to queue
        probs = measure(qml.probs(wires=wires_to_sample), state, is_state_batched)

    if is_state_batched:
        # Produce separate keys for each of the probabilities along the broadcasted axis
        keys = []
        for _ in state:
            key, subkey = jax.random.split(key)
            keys.append(subkey)
        samples = jnp.array(
            [
                jax.random.choice(_key, basis_states, shape=(shots,), p=prob)
                for _key, prob in zip(keys, probs)
            ]
        )
    else:
        samples = jax.random.choice(key, basis_states, shape=(shots,), p=probs)

    return _transform_samples_to_trinary(samples, num_wires)


def _transform_samples_to_trinary(samples, num_wires):
    """
    Returns a series of samples of a state for the JAX interface based on the PRNG.

    Args:
        samples (array[int]): The sampled values of the circuit
        num_wires (int): The number of wires being measured

    Returns:
        ndarray[int]: Sample values in ternary representation
    """
    res = np.zeros(samples.shape + (num_wires,), dtype=np.int64)
    for i in range(num_wires):
        res[..., -i] = (samples // (QUDIT_DIM**i)) % QUDIT_DIM
    return res
