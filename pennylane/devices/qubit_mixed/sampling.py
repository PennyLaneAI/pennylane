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
Submodule for sampling a qubit mixed state.
"""
# pylint: disable=too-many-positional-arguments, too-many-arguments
import functools
from typing import Callable

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    SampleMeasurement,
    SampleMP,
    Shots,
    VarianceMP,
)
from pennylane.ops import Sum
from pennylane.typing import TensorLike

from .apply_operation import _get_num_wires, apply_operation
from .measure import measure


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
    """Processes samples like SampleMP.process_samples, but different in need of some special cases e.g. CountsMP"""
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
    # This step converts e.g. 110 -> 6
    powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]  # e.g. [1, 2, 4, ...]
    indices = qml.math.array(samples @ powers_of_two)
    return mp.eigvals()[indices]


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
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Returns the samples of the measurement process performed on the given state,
    by rotating the state into the measurement basis using the diagonalizing gates
    given by the measurement process.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (TensorLike): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # apply diagonalizing gates
    state = _apply_diagonalizing_gates(mp, state, is_state_batched)

    total_indices = _get_num_wires(state, is_state_batched)
    wires = qml.wires.Wires(range(total_indices))

    def _process_single_shot_copy(samples):
        samples_processed = _process_samples(mp, samples, wires)
        if isinstance(mp, SampleMP):
            return math.squeeze(samples_processed)
        if isinstance(mp, CountsMP):
            process_func = functools.partial(mp.process_samples, wire_order=wires)
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
                readout_errors=readout_errors,
            )
            processed_samples.append(_process_single_shot_copy(samples))

        return tuple(processed_samples)

    samples = sample_state(
        state,
        shots=shots.total_shots,
        is_state_batched=is_state_batched,
        wires=wires,
        rng=rng,
        prng_key=prng_key,
        readout_errors=readout_errors,
    )

    return _process_single_shot_copy(samples)


def _measure_sum_with_samples(
    mp: SampleMeasurement,
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
):
    """Compute expectation values of Sum Observables"""

    def _sum_for_single_shot(s):
        results = []
        for term in mp.obs:
            results.append(
                measure_with_samples(
                    ExpectationMP(term),
                    state,
                    s,
                    is_state_batched=is_state_batched,
                    rng=rng,
                    prng_key=prng_key,
                    readout_errors=readout_errors,
                )
            )

        return sum(results)

    if shots.has_partitioned_shots:
        return tuple(_sum_for_single_shot(type(shots)(s)) for s in shots)

    return _sum_for_single_shot(shots)


def sample_state(
    state,
    shots: int,
    is_state_batched: bool = False,
    wires=None,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
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
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """

    total_indices = _get_num_wires(state, is_state_batched)
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)

    with qml.queuing.QueuingManager.stop_recording():
        probs = measure(qml.probs(wires=wires_to_sample), state, is_state_batched, readout_errors)

    # After getting the correct probs, there's no difference between mixed states and pure states.
    # Therefore, we directly re-use the sample_probs from the module qubit.
    return qml.devices.qubit.sampling.sample_probs(
        probs, shots, num_wires, is_state_batched, rng, prng_key=prng_key
    )


def measure_with_samples(
    mp: SampleMeasurement,
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
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
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike[Any]: Sample measurement results
    """

    if isinstance(mp, ExpectationMP) and isinstance(mp.obs, Sum):
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
        readout_errors=readout_errors,
    )
