# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A helper method for processing raw samples."""

import numpy as np

from pennylane import math
from pennylane.operation import EigvalsUndefinedError
from pennylane.typing import TensorLike

from .measurement_value import MeasurementValue
from .measurements import MeasurementProcess


def process_raw_samples(
    mp: MeasurementProcess, samples: TensorLike, wire_order, shot_range, bin_size
):
    """Slice the samples for a measurement process.

    Args:
        mp (MeasurementProcess): the measurement process containing the wires, observable, and mcms for the processing
        samples (TensorLike): the raw samples
        wire_order: the wire order for the raw samples
        shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
            to use. If not specified, all samples are used.
        bin_size (int): Divides the shot range into bins of size ``bin_size``, and
            returns the measurement statistic separately over each bin. If not
            provided, the entire shot range is treated as a single bin.

    This function matches `SampleMP.process_samples`, but does not have a dependence on the measurement process.

    """
    if shot_range is not None:
        # Indexing corresponds to: (potential broadcasting, shots, wires). Note that the last
        # colon (:) is required because shots is the second-to-last axis and the
        # Ellipsis (...) otherwise would take up broadcasting and shots axes.
        samples = samples[..., slice(*shot_range), :]

    print(mp)

    wire_map = dict(zip(wire_order, range(len(wire_order))))
    # Select the samples from samples that correspond to ``shot_range`` if provided

    # If we're sampling observables
    if mp.mv is not None and not isinstance(mp.mv, MeasurementValue):
        samples_for_mv = []
        for mv in mp.mv:
            mapped_wires = [wire_map[w] for w in mv.wires]
            subset_samples = samples[..., mapped_wires]

            _, processed_values = tuple(zip(*mv.items()))
            interface = math.get_deep_interface(processed_values)
            eigvals = math.asarray(processed_values, like=interface)
            samples_for_mv.append(_extract_from_eigvals(eigvals, subset_samples))

        return math.stack(samples_for_mv, axis=-1)

    mapped_wires = [wire_map[w] for w in mp.wires]
    if mapped_wires:
        # if wires are provided, then we only return samples from those wires
        samples = samples[..., mapped_wires]

    # If we're sampling wires or a list of mid-circuit measurements
    # pylint: disable=protected-access
    if mp.obs is None and mp._eigvals is None and mp.mv is None:
        # if no observable was provided then return the raw samples
        num_wires = samples.shape[-1]  # wires is the last dimension
        return samples if bin_size is None else samples.T.reshape(num_wires, bin_size, -1)

    try:
        eigvals = mp.eigvals()
    except EigvalsUndefinedError as e:
        # if observable has no info on eigenvalues, we cannot return this measurement
        raise EigvalsUndefinedError(f"Cannot compute samples of {mp.obs.name}.") from e

    samples = _extract_from_eigvals(eigvals, samples)

    if isinstance(mp.mv, MeasurementValue):
        samples = math.expand_dims(samples, -1)

    return samples if bin_size is None else samples.reshape((bin_size, -1))


def _extract_from_eigvals(eigvals, samples):
    if np.array_equal(eigvals, [1.0, -1.0]):
        # special handling for observables with eigvals +1/-1
        # (this is JIT-compatible, the next block is not)
        # type should be float
        return 1.0 - 2 * math.squeeze(samples, axis=-1)

    num_wires = samples.shape[-1]  # wires is the last dimension

    # Replace the basis state in the computational basis with the correct eigenvalue.
    # Extract only the columns of the basis samples required based on ``wires``.
    powers_of_two = 2 ** math.arange(num_wires)[::-1]
    indices = samples @ powers_of_two
    indices = math.array(indices)  # Add np.array here for Jax support.
    # This also covers statistics for mid-circuit measurements manipulated using
    # arithmetic operators
    if math.is_abstract(indices):
        return math.take(eigvals, indices, like=indices)
    return eigvals[indices]
