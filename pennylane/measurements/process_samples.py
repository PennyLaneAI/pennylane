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
from pennylane.typing import Sequence, TensorLike
from pennylane.wires import WiresLike

from .measurement_value import MeasurementValue
from .measurements import MeasurementProcess


# pylint: disable=too-many-arguments
def process_raw_samples(
    mp: MeasurementProcess,
    samples: TensorLike,
    wire_order: WiresLike,
    shot_range: Sequence[int],
    bin_size: int,
    dtype=None,
) -> TensorLike:
    """Slice the samples for a measurement process.

    Args:
        mp (MeasurementProcess): the measurement process containing the wires, observable, and mcms for the processing
        samples (TensorLike): the raw samples
        wire_order (WiresLike): the wire order for the raw samples
        shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
            to use. If not specified, all samples are used.
        bin_size (int): Divides the shot range into bins of size ``bin_size``, and
            returns the measurement statistic separately over each bin. If not
            provided, the entire shot range is treated as a single bin.
        dtype: The dtype of the samples returned by this measurement process.

    This function matches `SampleMP.process_samples`, but does not have a dependence on the measurement process.

    """

    wire_map = dict(zip(wire_order, range(len(wire_order))))
    mapped_wires = [wire_map[w] for w in mp.wires]
    # Select the samples from samples that correspond to ``shot_range`` if provided
    if shot_range is not None:
        # Indexing corresponds to: (potential broadcasting, shots, wires). Note that the last
        # colon (:) is required because shots is the second-to-last axis and the
        # Ellipsis (...) otherwise would take up broadcasting and shots axes.
        samples = samples[..., slice(*shot_range), :]

    if mapped_wires:
        # if wires are provided, then we only return samples from those wires
        samples = samples[..., mapped_wires]

    num_wires = samples.shape[-1]  # wires is the last dimension

    # If we're sampling wires or a list of mid-circuit measurements
    # pylint: disable=protected-access
    if mp.obs is None and not isinstance(mp.mv, MeasurementValue) and mp._eigvals is None:
        # if no observable was provided then return the raw samples
        samples = samples.astype(dtype) if dtype is not None else samples
        return samples if bin_size is None else samples.T.reshape(num_wires, bin_size, -1)

    # If we're sampling observables
    try:
        eigvals = mp.eigvals()
    except EigvalsUndefinedError as e:
        # if observable has no info on eigenvalues, we cannot return this measurement
        raise EigvalsUndefinedError(f"Cannot compute samples of {mp.obs.name}.") from e

    if np.array_equal(eigvals, [1.0, -1.0]):
        # special handling for observables with eigvals +1/-1
        # (this is JIT-compatible, the next block is not)
        # type should be float
        samples = 1.0 - 2 * math.squeeze(samples, axis=-1)
    else:
        # Replace the basis state in the computational basis with the correct eigenvalue.
        # Extract only the columns of the basis samples required based on ``wires``.
        powers_of_two = 2 ** math.arange(num_wires)[::-1]
        indices = samples @ powers_of_two
        indices = math.array(indices)  # Add np.array here for Jax support.
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        if math.is_abstract(indices):
            samples = math.take(eigvals, indices, like=indices)
        else:
            samples = eigvals[indices]

    samples = samples.astype(dtype) if dtype is not None else samples
    return samples if bin_size is None else samples.reshape((bin_size, -1))
