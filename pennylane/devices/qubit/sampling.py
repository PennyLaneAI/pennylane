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

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
    SampleMeasurement,
    Shots,
    ExpectationMP,
    ClassicalShadowMP,
    ShadowExpvalMP,
)
from pennylane.typing import TensorLike
from .apply_operation import apply_operation


def measure_with_samples(
    mp: SampleMeasurement, state: np.ndarray, shots: Shots, rng=None
) -> TensorLike:
    """
    Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # if the measurement process involves a Sum or a Hamiltonian, measure each
    # of the terms separately and sum
    if isinstance(mp, ExpectationMP) and isinstance(mp.obs, Hamiltonian):
        return sum(
            c * measure_with_samples(ExpectationMP(t), state, shots, rng=rng)
            for c, t in zip(*mp.obs.terms())
        )

    if isinstance(mp, ExpectationMP) and isinstance(mp.obs, Sum):
        return sum(measure_with_samples(ExpectationMP(t), state, shots, rng=rng) for t in mp.obs)

    if isinstance(mp, (ClassicalShadowMP, ShadowExpvalMP)):
        return _measure_classical_shadow(mp, state, shots, rng=rng)

    # measure with the usual method (rotate into the measurement basis)
    return _measure_with_samples_diagonalizing_gates(mp, state, shots, rng=rng)


def _measure_with_samples_diagonalizing_gates(
    mp: SampleMeasurement, state: np.ndarray, shots: Shots, rng=None
) -> TensorLike:
    """
    Returns the samples of the measurement process performed on the given state,
    by rotating the state into the measurement basis using the diagonalizing gates
    given by the measurement process.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # apply diagonalizing gates
    pre_rotated_state = state
    for op in mp.diagonalizing_gates():
        pre_rotated_state = apply_operation(op, pre_rotated_state)

    wires = qml.wires.Wires(range(len(state.shape)))

    # if there is a shot vector, build a list containing results for each shot entry
    if shots.has_partitioned_shots:
        processed_samples = []
        for s in shots:
            # currently we call sample_state for each shot entry, but it may be
            # better to call sample_state just once with total_shots, then use
            # the shot_range keyword argument
            samples = sample_state(pre_rotated_state, shots=s, wires=wires, rng=rng)
            processed_samples.append(qml.math.squeeze(mp.process_samples(samples, wires)))

        return tuple(processed_samples)

    samples = sample_state(pre_rotated_state, shots=shots.total_shots, wires=wires, rng=rng)
    return qml.math.squeeze(mp.process_samples(samples, wires))


def _measure_classical_shadow(
    mp: Union[ClassicalShadowMP, ShadowExpvalMP], state: np.ndarray, shots: Shots, rng=None
):
    """
    Returns the result of a classical shadow measurement on the given state.

    A classical shadow measurement doesn't fit neatly into the current measurement API
    since different diagonalizing gates are used for each shot. Here it's treated as a
    state measurement with shots instead of a sample measurement.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    wires = qml.wires.Wires(range(len(state.shape)))

    if shots.has_partitioned_shots:
        return tuple(mp.process_state_with_shots(state, wires, s, rng=rng) for s in shots)

    return mp.process_state_with_shots(state, wires, shots.total_shots, rng=rng)


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
