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
        results.append(
            tuple(mp.process_samples(sub_samples, wire_order) for mp in tape.measurements)
        )
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
) -> list[TensorLike]:
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

    tape = qml.tape.QuantumScript([], mps, shots=shots)
    batch, postprocessing = qml.transforms.split_non_commuting(tape)

    if prng_key is not None:
        keys = jax_random_split(prng_key, len(batch))
        kwargs = {"state": state, "is_state_batched": is_state_batched, "rng": rng}
        results = tuple(_sample_qwc_tape(t, **kwargs, prng_key=key) for t, key in zip(batch, keys))
    else:
        kwargs = {"is_state_batched": is_state_batched, "rng": rng}
        results = tuple(_sample_qwc_tape(t, state, **kwargs) for t in batch)
    print(results)
    results = postprocessing(results)
    print(results)

    # append MCM samples
    if mid_measurements:
        results += tuple(mid_measurements.values())

    if tape.shots.has_partitioned_shots and len(mps) == 1:
        return tuple((val,) for val in results)
    return (results,) if len(mps) == 1 else results


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
    if prng_key is not None or qml.math.get_interface(state) == "jax":
        return _sample_state_jax(
            state, shots, prng_key, is_state_batched=is_state_batched, wires=wires, seed=rng
        )

    rng = np.random.default_rng(rng)

    total_indices = len(state.shape) - is_state_batched
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(2**num_wires)

    flat_state = flatten_state(state, total_indices)
    with qml.queuing.QueuingManager.stop_recording():
        probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)

    # when using the torch interface with float32 as default dtype,
    # probabilities must be renormalized as they may not sum to one
    # see https://github.com/PennyLaneAI/pennylane/issues/5444
    norm = qml.math.sum(probs, axis=-1)
    abs_diff = qml.math.abs(norm - 1.0)
    cutoff = 1e-07

    if is_state_batched:
        normalize_condition = False

        for s in abs_diff:
            if s != 0:
                normalize_condition = True
            if s > cutoff:
                normalize_condition = False
                break

        if normalize_condition:
            probs = probs / norm[:, np.newaxis] if norm.shape else probs / norm

        # rng.choice doesn't support broadcasting
        samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    else:
        if not 0 < abs_diff < cutoff:
            norm = 1.0
        probs = probs / norm

        samples = rng.choice(basis_states, shots, p=probs)

    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(np.int64)


# pylint:disable = unused-argument
def _sample_state_jax(
    state,
    shots: int,
    prng_key,
    is_state_batched: bool = False,
    wires=None,
    seed=None,
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
        seed (numpy.random.Generator): seed to use to generate a key if a ``prng_key`` is not present. ``None`` by default.

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """
    # pylint: disable=import-outside-toplevel
    import jax
    import jax.numpy as jnp

    if prng_key is None:
        prng_key = jax.random.PRNGKey(np.random.default_rng(seed).integers(100000))

    total_indices = len(state.shape) - is_state_batched
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(2**num_wires)

    flat_state = flatten_state(state, total_indices)
    with qml.queuing.QueuingManager.stop_recording():
        probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)

    if is_state_batched:
        keys = jax_random_split(prng_key, num=len(state))
        samples = jnp.array(
            [
                jax.random.choice(_key, basis_states, shape=(shots,), p=prob)
                for _key, prob in zip(keys, probs)
            ]
        )
    else:
        _, key = jax_random_split(prng_key)
        samples = jax.random.choice(key, basis_states, shape=(shots,), p=probs)

    powers_of_two = 1 << np.arange(num_wires, dtype=int)[::-1]
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(int)
