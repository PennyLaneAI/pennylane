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
import warnings
from collections.abc import Sequence

import numpy as np

from pennylane import math
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops import LinearCombination
from pennylane.ops.qubit.observables import Hermitian
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .counts import CountsMP
from .measurements import SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue


class ProbabilityMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the probability of each computational basis state.

    Please refer to :func:`pennylane.probs` for detailed documentation.

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

    _shortname = "probs"

    @classmethod
    def _abstract_eval(cls, n_wires=None, has_eigvals=False, shots=None, num_device_wires=0):
        n_wires = num_device_wires if n_wires == 0 else n_wires
        shape = (2**n_wires,)
        return shape, float

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple[int]:
        len_wires = len(self.wires) if self.wires else num_device_wires
        return (2**len_wires,)

    def process_samples(
        self,
        samples: TensorLike,
        wire_order: Wires,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
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

        num_wires = math.shape(samples)[-1]
        # convert samples from a list of 0, 1 integers, to base 10 representation
        powers_of_two = 2 ** math.arange(num_wires)[::-1]
        indices = samples @ powers_of_two

        # `samples` typically has two axes ((shots, wires)) but can also have three with
        # broadcasting ((batch_size, shots, wires)) so that we simply read out the batch_size.
        batch_size = samples.shape[0] if math.ndim(samples) == 3 else None
        dim = 2**num_wires
        # count the basis state occurrences, and construct the probability vector
        new_bin_size = bin_size or samples.shape[-2]
        new_shape = (-1, new_bin_size) if batch_size is None else (batch_size, -1, new_bin_size)
        indices = indices.reshape(new_shape)
        prob = self._count_samples(indices, batch_size, dim)
        return math.squeeze(prob) if bin_size is None else prob

    def process_state(self, state: TensorLike, wire_order: Wires):
        prob = math.real(state) ** 2 + math.imag(state) ** 2
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
        batch_size = math.get_batch_size(prob, (expected_size,), expected_size)
        if batch_size is not None:
            # prob now is reshaped to have self.num_wires+1 axes in the case of broadcasting
            shape.insert(0, batch_size)
            inactive_wires = [idx + 1 for idx in inactive_wires]
            desired_axes = np.insert(desired_axes + 1, 0, 0)
            flat_shape = (batch_size, -1)

        prob = math.reshape(prob, shape)
        # sum over all inactive wires
        prob = math.sum(prob, axis=tuple(inactive_wires))
        # rearrange wires if necessary
        prob = math.transpose(prob, desired_axes)
        # flatten and return probabilities
        return math.reshape(prob, flat_shape)

    def process_counts(self, counts: dict, wire_order: Wires) -> np.ndarray:
        with QueuingManager.stop_recording():
            helper_counts = CountsMP(wires=self.wires, all_outcomes=False)
        mapped_counts = helper_counts.process_counts(counts, wire_order)

        num_shots = sum(mapped_counts.values())
        num_wires = len(next(iter(mapped_counts)))
        dim = 2**num_wires

        # constructs the probability vector
        # converts outcomes from binary strings to integers (base 10 representation)
        prob_vector = math.zeros((dim), dtype="float64")
        for outcome, occurrence in mapped_counts.items():
            prob_vector[int(outcome, base=2)] = occurrence / num_shots

        return prob_vector

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires):
        if len(math.shape(density_matrix)) == 2:
            prob = math.diagonal(density_matrix)
        else:
            prob = math.stack(
                [math.diagonal(density_matrix[i]) for i in range(math.shape(density_matrix)[0])]
            )

        # Since we only care about the probabilities, we can simplify the task here by creating a 'pseudo-state' to carry the diagonal elements and reuse the process_state method
        prob = math.convert_like(prob, density_matrix)
        p_state = math.sqrt(prob)
        return self.process_state(p_state, wire_order)

    @staticmethod
    def _count_samples(indices, batch_size, dim):
        """Count the occurrences of sampled indices and convert them to relative
        counts in order to estimate their occurrence probability."""
        num_bins, bin_size = indices.shape[-2:]
        interface = math.get_deep_interface(indices)

        if math.is_abstract(indices):

            def _count_samples_core(indices, dim, interface):
                return math.array(
                    [[math.sum(idx == p) for idx in indices] for p in range(dim)],
                    like=interface,
                )

        else:

            def _count_samples_core(indices, dim, *_):
                probabilities = math.zeros((dim, num_bins), dtype="float64")
                for b, idx in enumerate(indices):
                    basis_states, counts = math.unique(idx, return_counts=True)
                    probabilities[basis_states, b] = counts
                return probabilities

        if batch_size is None:
            return _count_samples_core(indices, dim, interface) / bin_size

        # count the basis state occurrences, and construct the probability vector
        # for each bin and broadcasting index
        indices = indices.reshape((batch_size, num_bins, bin_size))
        probabilities = math.array(
            [_count_samples_core(_indices, dim, interface) for _indices in indices],
            like=interface,
        )
        return probabilities / bin_size


def probs(wires=None, op=None) -> ProbabilityMP:
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
        op (Operator or MeasurementValue or Sequence[MeasurementValue]): Observable (with a ``diagonalizing_gates``
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

    .. warning::

       ``qml.probs`` is not compatible with :class:`~.Hermitian`. When using
       ``qml.probs`` with a Hermitian observable, the output might be different than
       expected as the lexicographical ordering of eigenvalues is not guaranteed and
       the diagonalizing gates may exist in a degenerate subspace.

    **Example:**

    The order of the output might be different when using ``qml.Hermitian``, as in the
    following example:

    .. code-block:: python3

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

        @qml.qnode(dev)
        def circuit():
            qml.H(wires=0)
            return qml.probs(op=qml.Hermitian(H, wires=0)), qml.probs(op=qml.Hadamard(wires=0))

    >>> circuit()
    (array([0.14644661, 0.85355339]), array([0.85355339, 0.14644661]))

    **Example:**

    The output might also be different than expected when using ``qml.Hermitian``,
    because the probability vector can be expressed in the eigenbasis obtained from
    diagonalizing the matrix of the observable, as in the following example:

    .. code-block:: python3

        ob = qml.X(0) @ qml.Y(1)
        h = qml.Hermitian(ob.matrix(), wires=[0, 1])

        @qml.qnode(dev)
        def circuit():
            return qml.probs(op=h), qml.probs(op=ob)

    >>> circuit()
    (array([0.5, 0. , 0. , 0.5]), array([0.25, 0.25, 0.25, 0.25]))

    Both outputs are in the eigenbasis of the observable, but at different locations in a degenerate subspace.  Both
    correspond to half in the ``-1`` eigenvalue and half in the ``+1`` eigenvalue.

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
        if not math.is_abstract(op[0]) and not all(
            isinstance(o, MeasurementValue) and not o.has_processing for o in op
        ):
            raise QuantumFunctionError(
                "Only sequences of unprocessed MeasurementValues can be passed with the op argument. "
                "MeasurementValues manipulated using arithmetic operators cannot be used when "
                "collecting statistics for a sequence of mid-circuit measurements."
            )

        return ProbabilityMP(obs=op)

    if isinstance(op, LinearCombination):
        raise QuantumFunctionError("Hamiltonians are not supported for rotating probabilities.")

    if op is not None and not math.is_abstract(op) and not op.has_diagonalizing_gates:
        raise QuantumFunctionError(
            f"{op} does not define diagonalizing gates : cannot be used to rotate the probability"
        )

    if wires is not None:
        if op is not None:
            raise QuantumFunctionError(
                "Cannot specify the wires to probs if an observable is "
                "provided. The wires for probs will be determined directly from the observable."
            )
        wires = Wires(wires)

    if isinstance(op, Hermitian):
        warnings.warn(
            "Using qml.probs with a Hermitian observable might return different results than expected as the "
            "lexicographical ordering of eigenvalues is not guaranteed.",
            UserWarning,
        )

    return ProbabilityMP(obs=op, wires=wires)
