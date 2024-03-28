# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the qml.probs measurement.
"""
from typing import Sequence, Tuple

import numpy as np

import pennylane as qml
from pennylane.wires import Wires
from .measurements import Probability, SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue


def probs(wires=None, op=None) -> "ProbabilityMP":
    r"""Probability of each computational basis state.

    This measurement function accepts either a wire specification or
    an observable. Passing wires to the function
    instructs the QNode to return a flat array containing the
    probabilities :math:`|\langle i | \psi \rangle |^2` of measuring
    the computational basis state :math:`| i \rangle` given the current
    state :math:`| \psi \rangle`.

    Marginal probabilities may also be requested by restricting
    the wires to a subset of the full system; the size of the
    returned array will be ``[2**len(wires)]``.

    .. Note::
        If no wires or observable are given, the probability of all wires is returned.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        op (Observable or MeasurementValue): Observable (with a ``diagonalizing_gates``
            attribute) that rotates the computational basis, or a  ``MeasurementValue``
            corresponding to mid-circuit measurements.

    Returns:
        ProbabilityMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.probs(wires=[0, 1])

    Executing this QNode:

    >>> circuit()
    array([0.5, 0.5, 0. , 0. ])

    The returned array is in lexicographic order, so corresponds
    to a :math:`50\%` chance of measuring either :math:`|00\rangle`
    or :math:`|01\rangle`.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

        @qml.qnode(dev)
        def circuit():
            qml.Z(0)
            qml.X(1)
            return qml.probs(op=qml.Hermitian(H, wires=0))

    >>> circuit()
    array([0.14644661 0.85355339])

    The returned array is in lexicographic order, so corresponds
    to a :math:`14.6\%` chance of measuring the rotated :math:`|0\rangle` state
    and :math:`85.4\%` of measuring the rotated :math:`|1\rangle` state.

    Note that the output shape of this measurement process depends on whether
    the device simulates qubit or continuous variable quantum systems.
    """
    if isinstance(op, MeasurementValue):
        if len(op.measurements) > 1:
            raise ValueError(
                "Cannot use qml.probs() when measuring multiple mid-circuit measurements collected "
                "using arithmetic operators. To collect probabilities for multiple mid-circuit "
                "measurements, use a list of mid-circuit measurements with qml.probs()."
            )
        return ProbabilityMP(obs=op)

    if isinstance(op, Sequence):
        if not all(isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in op):
            raise qml.QuantumFunctionError(
                "Only sequences of single MeasurementValues can be passed with the op argument. "
                "MeasurementValues manipulated using arithmetic operators cannot be used when "
                "collecting statistics for a sequence of mid-circuit measurements."
            )

        return ProbabilityMP(obs=op)

    if isinstance(op, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
        raise qml.QuantumFunctionError("Hamiltonians are not supported for rotating probabilities.")

    if op is not None and not op.has_diagonalizing_gates:
        raise qml.QuantumFunctionError(
            f"{op} does not define diagonalizing gates : cannot be used to rotate the probability"
        )

    if wires is not None:
        if op is not None:
            raise qml.QuantumFunctionError(
                "Cannot specify the wires to probs if an observable is "
                "provided. The wires for probs will be determined directly from the observable."
            )
        wires = Wires(wires)
    return ProbabilityMP(obs=op, wires=wires)


class ProbabilityMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the probability of each computational basis state.

    Please refer to :func:`probs` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    @property
    def return_type(self):
        return Probability

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        num_shot_elements = (
            sum(s.copies for s in shots.shot_vector) if shots.has_partitioned_shots else 1
        )
        len_wires = len(self.wires)
        if len_wires == 0:
            len_wires = len(device.wires) if device.wires else 0
        dim = self._get_num_basis_states(len_wires, device)

        return (dim,) if num_shot_elements == 1 else tuple((dim,) for _ in range(num_shot_elements))

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        if shot_range is not None:
            # Indexing corresponds to: (potential broadcasting, shots, wires). Note that the last
            # colon (:) is required because shots is the second-to-last axis and the
            # Ellipsis (...) otherwise would take up broadcasting and shots axes.
            samples = samples[..., slice(*shot_range), :]

        if mapped_wires:
            # if wires are provided, then we only return samples from those wires
            samples = samples[..., mapped_wires]

        num_wires = qml.math.shape(samples)[-1]
        # convert samples from a list of 0, 1 integers, to base 10 representation
        powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]
        indices = samples @ powers_of_two

        # `samples` typically has two axes ((shots, wires)) but can also have three with
        # broadcasting ((batch_size, shots, wires)) so that we simply read out the batch_size.
        batch_size = samples.shape[0] if qml.math.ndim(samples) == 3 else None
        dim = 2**num_wires
        # count the basis state occurrences, and construct the probability vector
        new_bin_size = bin_size or samples.shape[-2]
        new_shape = (-1, new_bin_size) if batch_size is None else (batch_size, -1, new_bin_size)
        indices = indices.reshape(new_shape)
        prob = self._count_samples(indices, batch_size, dim)
        return qml.math.squeeze(prob) if bin_size is None else prob

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        prob = qml.math.real(state) ** 2 + qml.math.imag(state) ** 2
        if self.wires == Wires([]):
            # no need to marginalize
            return prob

        # determine which subsystems are to be summed over
        inactive_wires = Wires.unique_wires([wire_order, self.wires])

        # translate to wire labels used by device
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        inactive_wires = [wire_map[w] for w in inactive_wires]

        # reshape the probability so that each axis corresponds to a wire
        num_device_wires = len(wire_order)
        shape = [2] * num_device_wires
        desired_axes = np.argsort(np.argsort(mapped_wires))
        flat_shape = (-1,)
        expected_size = 2**num_device_wires
        batch_size = qml.math.get_batch_size(prob, (expected_size,), expected_size)
        if batch_size is not None:
            # prob now is reshaped to have self.num_wires+1 axes in the case of broadcasting
            shape.insert(0, batch_size)
            inactive_wires = [idx + 1 for idx in inactive_wires]
            desired_axes = np.insert(desired_axes + 1, 0, 0)
            flat_shape = (batch_size, -1)

        prob = qml.math.reshape(prob, shape)
        # sum over all inactive wires
        prob = qml.math.sum(prob, axis=tuple(inactive_wires))
        # rearrange wires if necessary
        prob = qml.math.transpose(prob, desired_axes)
        # flatten and return probabilities
        return qml.math.reshape(prob, flat_shape)

    def process_counts(self, counts: dict, wire_order: Wires) -> np.ndarray:
        with qml.QueuingManager.stop_recording():
            helper_counts = qml.counts(wires=self.wires, all_outcomes=False)
        mapped_counts = helper_counts.process_counts(counts, wire_order)

        num_shots = sum(mapped_counts.values())
        num_wires = len(next(iter(mapped_counts)))
        dim = 2**num_wires

        # constructs the probability vector
        # converts outcomes from binary strings to integers (base 10 representation)
        prob_vector = qml.math.zeros((dim), dtype="float64")
        for outcome, occurrence in mapped_counts.items():
            prob_vector[int(outcome, base=2)] = occurrence / num_shots

        return prob_vector

    @staticmethod
    def _count_samples(indices, batch_size, dim):
        """Count the occurrences of sampled indices and convert them to relative
        counts in order to estimate their occurrence probability."""
        num_bins, bin_size = indices.shape[-2:]
        if batch_size is None:
            prob = qml.math.zeros((dim, num_bins), dtype="float64")
            # count the basis state occurrences, and construct the probability vector for each bin
            for b, idx in enumerate(indices):
                basis_states, counts = qml.math.unique(idx, return_counts=True)
                prob[basis_states, b] = counts / bin_size

            return prob

        prob = qml.math.zeros((batch_size, dim, num_bins), dtype="float64")
        indices = indices.reshape((batch_size, num_bins, bin_size))

        # count the basis state occurrences, and construct the probability vector
        # for each bin and broadcasting index
        for i, _indices in enumerate(indices):  # First iterate over broadcasting dimension
            for b, idx in enumerate(_indices):  # Then iterate over bins dimension
                basis_states, counts = qml.math.unique(idx, return_counts=True)
                prob[i, basis_states, b] = counts / bin_size

        return prob
