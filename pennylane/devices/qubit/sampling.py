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
    mp: Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
) -> TensorLike:
    """
    Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        mp (Union[~.measurements.SampleMeasurement, ~.measurements.ClassicalShadowMP, ~.measurements.ShadowExpvalMP]):
            The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # if the measurement process involves a Sum or a Hamiltonian, measure each
    # of the terms separately and sum
    if isinstance(mp, ExpectationMP):
        if isinstance(mp.obs, Hamiltonian):

            def _sum_for_single_shot(s):
                return sum(
                    c
                    * measure_with_samples(
                        ExpectationMP(t), state, s, is_state_batched=is_state_batched, rng=rng
                    )
                    for c, t in zip(*mp.obs.terms())
                )

            unsqueezed_results = tuple(_sum_for_single_shot(Shots(s)) for s in shots)
            return unsqueezed_results if shots.has_partitioned_shots else unsqueezed_results[0]

        if isinstance(mp.obs, Sum):

            def _sum_for_single_shot(s):
                return sum(
                    measure_with_samples(
                        ExpectationMP(t), state, s, is_state_batched=is_state_batched, rng=rng
                    )
                    for t in mp.obs
                )

            unsqueezed_results = tuple(_sum_for_single_shot(Shots(s)) for s in shots)
            return unsqueezed_results if shots.has_partitioned_shots else unsqueezed_results[0]

    if isinstance(mp, (ClassicalShadowMP, ShadowExpvalMP)):
        return _measure_classical_shadow(mp, state, shots, rng=rng)

    # measure with the usual method (rotate into the measurement basis)
    return _measure_with_samples_diagonalizing_gates(
        mp, state, shots, is_state_batched=is_state_batched, rng=rng
    )


def _measure_with_samples_diagonalizing_gates(
    mp: SampleMeasurement, state: np.ndarray, shots: Shots, is_state_batched: bool = False, rng=None
) -> TensorLike:
    """
    Returns the samples of the measurement process performed on the given state,
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

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # apply diagonalizing gates
    pre_rotated_state = state
    for op in mp.diagonalizing_gates():
        pre_rotated_state = apply_operation(
            op, pre_rotated_state, is_state_batched=is_state_batched
        )

    total_indices = len(state.shape) - is_state_batched
    wires = qml.wires.Wires(range(total_indices))

    # if there is a shot vector, build a list containing results for each shot entry
    if shots.has_partitioned_shots:
        processed_samples = []
        for s in shots:
            # currently we call sample_state for each shot entry, but it may be
            # better to call sample_state just once with total_shots, then use
            # the shot_range keyword argument
            samples = sample_state(
                pre_rotated_state, shots=s, is_state_batched=is_state_batched, wires=wires, rng=rng
            )

            if not isinstance(processed := mp.process_samples(samples, wires), dict):
                processed = qml.math.squeeze(processed)

            processed_samples.append(processed)

        return tuple(processed_samples)

    samples = sample_state(
        pre_rotated_state,
        shots=shots.total_shots,
        is_state_batched=is_state_batched,
        wires=wires,
        rng=rng,
    )

    if not isinstance(processed := mp.process_samples(samples, wires), dict):
        processed = qml.math.squeeze(processed)

    return processed


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


def sample_state(
    state, shots: int, is_state_batched: bool = False, wires=None, rng=None
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

    Returns:
        ndarray[bool]: Sample values of the shape (shots, num_wires)
    """
    rng = np.random.default_rng(rng)

    total_indices = len(state.shape) - is_state_batched
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(2**num_wires)

    probs = qml.probs(wires=wires_to_sample).process_state(state, state_wires)

    if is_state_batched:
        # rng.choice doesn't support broadcasting
        samples = np.stack([rng.choice(basis_states, shots, p=p) for p in probs])
    else:
        samples = rng.choice(basis_states, shots, p=probs)

    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    return (samples[..., None] & powers_of_two).astype(np.bool8)
