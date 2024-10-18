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
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.measurements import ClassicalShadowMP, SampleMeasurement, ShadowExpvalMP, Shots
from pennylane.ops import Prod, SProd, Sum
from pennylane.typing import TensorLike

from .apply_operation import apply_operation
from .measure import flatten_state


def jax_random_split(prng_key, num: int = 2):
    """Get a new key with ``jax.random.split``."""
    if prng_key is None:
        return (None,) * num
    # pylint: disable=import-outside-toplevel
    from jax.random import split

    return split(prng_key, num=num)


def _group_measurements(mps: list[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]]):
    """
    Group the measurements such that:
      - measurements with pauli observables pairwise-commute in each group
      - measurements with observables that are not pauli words are all in different groups
      - measurements without observables are all in the same group
      - classical shadow measurements are all in different groups

    """
    # Note: this function is used by lightning qubit and so cannot yet be deleted.
    # TODO: delete this function when possible.

    if len(mps) == 1:
        return [mps], [[0]]

    # measurements with pauli-word observables
    mp_pauli_obs = []

    # measurements with non pauli-word observables
    mp_other_obs = []
    mp_other_obs_indices = []

    # measurements with no observables
    mp_no_obs = []
    mp_no_obs_indices = []
    for i, mp in enumerate(mps):
        if isinstance(mp.obs, (Sum, SProd, Prod)):
            mps[i].obs = qml.simplify(mp.obs)
        if isinstance(mp, (ClassicalShadowMP, ShadowExpvalMP)):
            mp_other_obs.append([mp])
            mp_other_obs_indices.append([i])
        elif mp.obs is None:
            mp_no_obs.append(mp)
            mp_no_obs_indices.append(i)
        elif qml.pauli.is_pauli_word(mp.obs):
            mp_pauli_obs.append((i, mp))
        else:
            mp_other_obs.append([mp])
            mp_other_obs_indices.append([i])
    if mp_pauli_obs:
        i_to_pauli_mp = dict(mp_pauli_obs)
        _, group_indices = qml.pauli.group_observables(
            [mp.obs for mp in i_to_pauli_mp.values()], list(i_to_pauli_mp.keys())
        )
        mp_pauli_groups = []
        for indices in group_indices:
            mp_group = [i_to_pauli_mp[i] for i in indices]
            mp_pauli_groups.append(mp_group)
    else:
        mp_pauli_groups, group_indices = [], []

    mp_no_obs_indices = [mp_no_obs_indices] if mp_no_obs else []
    mp_no_obs = [mp_no_obs] if mp_no_obs else []
    all_mp_groups = mp_pauli_groups + mp_no_obs + mp_other_obs
    all_indices = group_indices + mp_no_obs_indices + mp_other_obs_indices

    return all_mp_groups, all_indices


# pylint: disable=no-member
def get_num_shots_and_executions(tape: qml.tape.QuantumScript) -> tuple[int, int]:
    """Get the total number of qpu executions and shots.

    Args:
        tape (qml.tape.QuantumTape): the tape we want to get the number of executions and shots for

    Returns:
        int, int: the total number of QPU executions and the total number of shots

    """
    batch, _ = qml.transforms.split_non_commuting(tape)

    num_executions = 0
    num_shots = tape.shots.total_shots * len(batch) if tape.shots else tape.shots
    for t in batch:
        if isinstance(t.measurements[0], (ClassicalShadowMP, ShadowExpvalMP)):
            num_executions += t.shots.total_shots
        else:
            num_executions += 1

    if tape.batch_size:
        num_executions *= tape.batch_size
        if tape.shots:
            num_shots *= tape.batch_size
    return num_executions, num_shots


def _measure_classical_shadows(tape, state, is_state_batched, rng=None):
    num_wires = len(qml.math.shape(state)) - is_state_batched
    wire_order = qml.wires.Wires(list(range(num_wires)))
    results = []
    for s in tape.shots:
        r = tuple(
            mp.process_state_with_shots(state, wire_order, s, rng=rng) for mp in tape.measurements
        )
        results.append(r[0] if len(tape.measurements) == 1 else r)
    return tuple(results) if tape.shots.has_partitioned_shots else results[0]


def _sample_qwc_tape(tape, state, is_state_batched, rng=None, prng_key=None):
    if isinstance(tape.measurements[0], (ClassicalShadowMP, ShadowExpvalMP)):
        return _measure_classical_shadows(tape, state, is_state_batched, rng=rng)

    (tape,), _ = qml.transforms.diagonalize_measurements(tape)

    for op in tape.operations:
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    num_wires = len(qml.math.shape(state)) - is_state_batched
    wire_order = qml.wires.Wires(list(range(num_wires)))

    try:
        samples = sample_state(
            state,
            shots=tape.shots.total_shots,
            is_state_batched=is_state_batched,
            rng=rng,
            prng_key=prng_key,
        )
    except ValueError as e:
        if str(e) != "probabilities contain NaN":
            raise e
        samples = qml.math.full((tape.shots.total_shots, num_wires), 0)

    results = []
    for lower, upper in tape.shots.bins():
        sub_samples = samples[:, lower:upper] if is_state_batched else samples[lower:upper]

        def next_res(s):
            for mp in tape.measurements:
                r = mp.process_samples(s, wire_order)
                yield r if isinstance(r, dict) else qml.math.squeeze(r)

        results.append(tuple(next_res(sub_samples)))
    if len(tape.measurements) == 1:
        results = tuple(res[0] for res in results)
    if tape.shots.has_partitioned_shots:
        return tuple(results)
    return results[0]


# pylint:disable = too-many-arguments
def measure_with_samples(
    measurements: list[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    mid_measurements: dict = None,
) -> tuple[TensorLike]:
    """
    Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        measurements (List[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]]):
            The sample measurements to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
        mid_measurements (None, dict): Dictionary of mid-circuit measurements

    Returns:
        List[TensorLike[Any]]: Sample measurement results
    """
    # last N measurements are sampling MCMs in ``dynamic_one_shot`` execution mode
    mps = measurements[: -len(mid_measurements)] if mid_measurements else measurements
    if not mps:
        return tuple(mid_measurements.values()) if mid_measurements else tuple()

    tape = qml.tape.QuantumScript([], mps, shots=shots)
    batch, postprocessing = qml.transforms.split_non_commuting(tape)

    kwargs = {"state": state, "is_state_batched": is_state_batched, "rng": rng}
    if prng_key is not None:
        keys = jax_random_split(prng_key, len(batch))
        results = tuple(_sample_qwc_tape(t, **kwargs, prng_key=key) for t, key in zip(batch, keys))
    else:
        results = tuple(_sample_qwc_tape(t, **kwargs) for t in batch)
    results = postprocessing(results)

    if tape.shots.has_partitioned_shots and len(mps) == 1:
        results = tuple((val,) for val in results)
    else:
        results = (results,) if len(mps) == 1 else results

    # append MCM samples
    if mid_measurements:
        if shots.has_partitioned_shots:
            mcm_results = tuple(mid_measurements.values())
            results = tuple(r + mcm_r for r, mcm_r in zip(results, mcm_results))
        else:
            results += tuple(mid_measurements.values())

    return results


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

    total_indices = len(state.shape) - is_state_batched
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)

    flat_state = flatten_state(state, total_indices)
    with qml.queuing.QueuingManager.stop_recording():
        probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)
        # Keep same interface (e.g. jax) as in the device

    return sample_probs(probs, shots, num_wires, is_state_batched, rng, prng_key)


def sample_probs(probs, shots, num_wires, is_state_batched, rng, prng_key=None):
    """
    Sample from given probabilities, dispatching between JAX and NumPy implementations.

    Args:
        probs (array): The probabilities to sample from
        shots (int): The number of samples to take
        num_wires (int): The number of wires to sample
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
    """
    if qml.math.get_interface(probs) == "jax" or prng_key is not None:
        return _sample_probs_jax(probs, shots, num_wires, is_state_batched, prng_key, seed=rng)

    return _sample_probs_numpy(probs, shots, num_wires, is_state_batched, rng)


def _sample_probs_numpy(probs, shots, num_wires, is_state_batched, rng):
    """
    Sample from given probabilities using NumPy's random number generator.

    Args:
        probs (array): The probabilities to sample from
        shots (int): The number of samples to take
        num_wires (int): The number of wires to sample
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used
    """
    rng = np.random.default_rng(rng)
    norm = qml.math.sum(probs, axis=-1)
    norm_err = qml.math.abs(norm - 1.0)
    cutoff = 1e-07

    norm_err = norm_err[..., np.newaxis] if not is_state_batched else norm_err
    if qml.math.any(norm_err > cutoff):
        raise ValueError("probabilities do not sum to 1")

    basis_states = np.arange(2**num_wires)
    if is_state_batched:
        probs = probs / norm[:, np.newaxis] if norm.shape else probs / norm
        samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    else:
        probs = probs / norm
        samples = rng.choice(basis_states, shots, p=probs)

    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(np.int64)


def _sample_probs_jax(probs, shots, num_wires, is_state_batched, prng_key=None, seed=None):
    """
    Returns a series of samples of a state for the JAX interface based on the PRNG.

    Args:
        probs (array): The probabilities to sample from
        shots (int): The number of samples to take
        num_wires (int): The number of wires to sample
        is_state_batched (bool): whether the state is batched or not
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
        seed (Optional[int]): A seed for the random number generator. This is only used if ``prng_key``
            is not provided.

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """
    # pylint: disable=import-outside-toplevel
    import jax
    import jax.numpy as jnp

    if prng_key is None:
        prng_key = jax.random.PRNGKey(np.random.default_rng(seed).integers(100000))

    basis_states = jnp.arange(2**num_wires)

    if is_state_batched:
        keys = jax_random_split(prng_key, num=probs.shape[0])
        samples = jnp.array(
            [
                jax.random.choice(_key, basis_states, shape=(shots,), p=prob)
                for _key, prob in zip(keys, probs)
            ]
        )
    else:
        _, key = jax_random_split(prng_key)
        samples = jax.random.choice(key, basis_states, shape=(shots,), p=probs)

    powers_of_two = 1 << jnp.arange(num_wires, dtype=jnp.int64)[::-1]
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(jnp.int64)
