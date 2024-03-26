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
import functools
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.ops import Sum
from pennylane.measurements import (
    Shots,
    SampleMeasurement,
    SampleMP,
    CountsMP,
    ExpectationMP,
    VarianceMP,
)
from pennylane.typing import TensorLike
from .utils import QUDIT_DIM, get_num_wires
from .measure import measure
from .apply_operation import apply_operation


def _apply_diagonalizing_gates(
    mp: SampleMeasurement, state: np.ndarray, is_state_batched: bool = False
):
    """Applies diagonalizing gates when necessary"""
    if mp.obs:
        for op in mp.diagonalizing_gates():
            state = apply_operation(op, state, is_state_batched=is_state_batched)

    return state


def _process_samples(
    mp,
    samples,
    wire_order,
):
    """Processes samples like SampleMP.process_samples, but fixed for qutrits"""
    wire_map = dict(zip(wire_order, range(len(wire_order))))
    mapped_wires = [wire_map[w] for w in mp.wires]

    if mapped_wires:
        # if wires are provided, then we only return samples from those wires
        samples = samples[..., mapped_wires]

    num_wires = samples.shape[-1]  # wires is the last dimension

    if mp.obs is None:
        # if no observable was provided then return the raw samples
        return samples

    # Replace the basis state in the computational basis with the correct eigenvalue.
    # Extract only the columns of the basis samples required based on ``wires``.
    powers_of_three = QUDIT_DIM ** qml.math.arange(num_wires)[::-1]
    indices = qml.math.array(samples @ powers_of_three)
    return mp.eigvals()[indices]


def _process_counts_samples(processed_sample, mp_has_obs):
    """Processes a set of samples and counts the results."""
    observables, counts = math.unique(processed_sample, return_counts=True, axis=0)
    if not mp_has_obs:
        observables = ["".join(observable.astype("str")) for observable in observables]
    return dict(zip(observables, counts))


def _process_expval_samples(processed_sample):
    """Processes a set of samples and returns the expectation value of an observable."""
    eigvals, counts = math.unique(processed_sample, return_counts=True)
    probs = counts / math.sum(counts)
    return math.dot(probs, eigvals)


def _process_variance_samples(processed_sample):
    """Processes a set of samples and returns the variance of an observable."""
    eigvals, counts = math.unique(processed_sample, return_counts=True)
    probs = counts / math.sum(counts)
    return math.dot(probs, (eigvals**2)) - math.dot(probs, eigvals) ** 2


# pylint:disable = too-many-arguments
def _measure_with_samples_diagonalizing_gates(
    mp: SampleMeasurement,
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
) -> TensorLike:
    """Returns the samples of the measurement process performed on the given state,
    by rotating the state into the measurement basis using the diagonalizing gates
    given by the measurement process.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # apply diagonalizing gates
    state = _apply_diagonalizing_gates(mp, state, is_state_batched)

    total_indices = get_num_wires(state, is_state_batched)
    wires = qml.wires.Wires(range(total_indices))

    def _process_single_shot(samples):
        samples_processed = _process_samples(mp, samples, wires)
        if isinstance(mp, SampleMP):
            return math.squeeze(samples_processed)
        if isinstance(mp, CountsMP):
            process_func = functools.partial(_process_counts_samples, mp_has_obs=mp.obs is not None)
        elif isinstance(mp, ExpectationMP):
            process_func = _process_expval_samples
        elif isinstance(mp, VarianceMP):
            process_func = _process_variance_samples
        else:
            raise NotImplementedError

        if is_state_batched:
            ret = []
            for processed_sample in samples_processed:
                ret.append(process_func(processed_sample))
            return math.squeeze(ret)
        return process_func(samples_processed)

    # if there is a shot vector, build a list containing results for each shot entry
    if shots.has_partitioned_shots:
        processed_samples = []
        for s in shots:
            # Like default.qubit currently calling sample_state for each shot entry,
            # but it may be better to call sample_state just once with total_shots,
            # then use the shot_range keyword argument
            samples = sample_state(
                state,
                shots=s,
                is_state_batched=is_state_batched,
                wires=wires,
                rng=rng,
                prng_key=prng_key,
            )
            processed_samples.append(_process_single_shot(samples))

        return tuple(processed_samples)

    samples = sample_state(
        state,
        shots=shots.total_shots,
        is_state_batched=is_state_batched,
        wires=wires,
        rng=rng,
        prng_key=prng_key,
    )

    return _process_single_shot(samples)


def _measure_sum_with_samples(
    mp: SampleMeasurement,
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
):
    """Compute expectation values of Sum or Hamiltonian Observables"""
    # mp.obs returns is the list of observables for Sum,
    # mp.obs.terms()[1] returns is the list of observables for Hamiltonian
    obs_terms = mp.obs if isinstance(mp.obs, Sum) else mp.obs.terms()[1]

    def _sum_for_single_shot(s):
        results = []
        for term in obs_terms:
            results.append(
                measure_with_samples(
                    ExpectationMP(term),
                    state,
                    s,
                    is_state_batched=is_state_batched,
                    rng=rng,
                    prng_key=prng_key,
                )
            )

        if isinstance(mp.obs, qml.ops.Hamiltonian):
            # If Hamiltonian apply coefficients
            return sum((c * res for c, res in zip(mp.obs.terms()[0], results)))

        return sum(results)

    if shots.has_partitioned_shots:
        return tuple(_sum_for_single_shot(type(shots)(s)) for s in shots)

    return _sum_for_single_shot(shots)


def _sample_state_jax(
    state,
    shots: int,
    prng_key,
    is_state_batched: bool = False,
    wires=None,
) -> np.ndarray:
    """Returns a series of samples of a state for the JAX interface based on the PRNG.

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

    with qml.queuing.QueuingManager.stop_recording():
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

    res = np.zeros(samples.shape + (num_wires,), dtype=np.int64)
    for i in range(num_wires):
        res[..., -(i + 1)] = (samples // (QUDIT_DIM**i)) % QUDIT_DIM
    return res


def sample_state(
    state,
    shots: int,
    is_state_batched: bool = False,
    wires=None,
    rng=None,
    prng_key=None,
) -> np.ndarray:
    """Returns a series of computational basis samples of a state.

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

    with qml.queuing.QueuingManager.stop_recording():
        probs = measure(qml.probs(wires=wires_to_sample), state, is_state_batched)

    if is_state_batched:
        # rng.choice doesn't support broadcasting
        samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    else:
        samples = rng.choice(basis_states, shots, p=probs)

    res = np.zeros(samples.shape + (num_wires,), dtype=np.int64)
    for i in range(num_wires):
        res[..., -(i + 1)] = (samples // (QUDIT_DIM**i)) % QUDIT_DIM
    return res


def measure_with_samples(
    mp: SampleMeasurement,
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
) -> TensorLike:
    """Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        mp (SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.

    Returns:
        TensorLike[Any]: Sample measurement results
    """

    if isinstance(mp, ExpectationMP) and isinstance(mp.obs, (qml.ops.Hamiltonian, Sum)):
        measure_fn = _measure_sum_with_samples
    else:
        # measure with the usual method (rotate into the measurement basis)
        measure_fn = _measure_with_samples_diagonalizing_gates

    return measure_fn(
        mp,
        state,
        shots,
        is_state_batched=is_state_batched,
        rng=rng,
        prng_key=prng_key,
    )
