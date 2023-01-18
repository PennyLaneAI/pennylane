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

import pennylane as qml
from pennylane.wires import Wires

from .measurements import MeasurementShapeError, Probability, SampleMeasurement, StateMeasurement


def probs(wires=None, op=None):
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
        op (Observable): Observable (with a ``diagonalizing_gates`` attribute) that rotates
            the computational basis

    Returns:
        ProbabilityMP: measurement process instance

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
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            return qml.probs(op=qml.Hermitian(H, wires=0))

    >>> circuit()

    array([0.14644661 0.85355339])

    The returned array is in lexicographic order, so corresponds
    to a :math:`14.6\%` chance of measuring the rotated :math:`|0\rangle` state
    and :math:`85.4\%` of measuring the rotated :math:`|1\rangle` state.

    Note that the output shape of this measurement process depends on whether
    the device simulates qubit or continuous variable quantum systems.
    """

    if isinstance(op, qml.Hamiltonian):
        raise qml.QuantumFunctionError("Hamiltonians are not supported for rotating probabilities.")

    if isinstance(op, (qml.ops.Sum, qml.ops.SProd, qml.ops.Prod)):  # pylint: disable=no-member
        raise qml.QuantumFunctionError(
            "Symbolic Operations are not supported for rotating probabilities yet."
        )

    if op is not None and not qml.operation.defines_diagonalizing_gates(op):
        raise qml.QuantumFunctionError(
            f"{op} does not define diagonalizing gates : cannot be used to rotate the probability"
        )

    if wires is not None:
        if op is not None:
            raise qml.QuantumFunctionError(
                "Cannot specify the wires to probs if an observable is "
                "provided. The wires for probs will be determined directly from the observable."
            )
        wires = qml.wires.Wires(wires)
    return ProbabilityMP(obs=op, wires=wires)


class ProbabilityMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the probability of each computational basis state.

    Please refer to :func:`probs` for detailed documentation.

    Args:
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
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

    def shape(self, device=None):
        if qml.active_return():
            return self._shape_new(device)
        if device is None:
            raise MeasurementShapeError(
                "The device argument is required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        num_shot_elements = (
            1 if device.shot_vector is None else sum(s.copies for s in device.shot_vector)
        )
        len_wires = len(self.wires)
        dim = self._get_num_basis_states(len_wires, device)

        return (num_shot_elements, dim)

    def _shape_new(self, device=None):
        if device is None:
            raise MeasurementShapeError(
                "The device argument is required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        num_shot_elements = (
            1 if device.shot_vector is None else sum(s.copies for s in device.shot_vector)
        )
        len_wires = len(self.wires)
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
        num_wires = len(wire_order)
        dim = 2**num_wires
        # Compute batch_size
        expected_shape = [2] * num_wires
        expected_size = dim
        size = qml.math.size(state)
        batch_size = (
            size // expected_size
            if qml.math.ndim(state) > len(expected_shape) or size > expected_size
            else None
        )
        flat_state = qml.math.reshape(
            state, (batch_size, dim) if batch_size is not None else (dim,)
        )
        real_state = qml.math.real(flat_state)
        imag_state = qml.math.imag(flat_state)
        return self.marginal_prob(real_state**2 + imag_state**2, wire_order, batch_size)

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

    def marginal_prob(self, prob, wire_order, batch_size):
        r"""Return the marginal probability of the computational basis
        states by summing the probabilities on the non-specified wires.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            If the provided wires are not in the order as they appear on the device,
            the returned marginal probabilities take this permutation into account.

            For example, if the addressable wires on this device are ``Wires([0, 1, 2])`` and
            this function gets passed ``wires=[2, 0]``, then the returned marginal
            probability vector will take this 'reversal' of the two wires
            into account:

            .. math::

                \mathbb{P}^{(2, 0)}
                            = \left[
                               |00\rangle, |10\rangle, |01\rangle, |11\rangle
                              \right]

        Args:
            prob: The probabilities to return the marginal probabilities
                for
            wire_order (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array[float]: array of the resulting marginal probabilities.
        """
        # TODO: Add when ``qml.probs()`` is supported
        # if self.wires == Wires([]):
        #     # no need to marginalize
        #     return prob

        # determine which subsystems are to be summed over
        inactive_wires = Wires.unique_wires([wire_order, self.wires])

        # translate to wire labels used by device
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        inactive_wires = [wire_map[w] for w in inactive_wires]

        # reshape the probability so that each axis corresponds to a wire
        num_device_wires = len(wire_order)
        shape = [2] * num_device_wires
        if batch_size is not None:
            shape.insert(0, batch_size)
            inactive_wires = [idx + 1 for idx in inactive_wires]
        # prob now is reshaped to have self.num_wires+1 axes in the case of broadcasting
        prob = qml.math.reshape(prob, shape)

        # sum over all inactive wires
        flat_shape = (-1,) if batch_size is None else (batch_size, -1)
        prob = qml.math.reshape(qml.math.sum(prob, axis=tuple(inactive_wires)), flat_shape)

        # The wires provided might not be in consecutive order (i.e., wires might be [2, 0]).
        # If this is the case, we must permute the marginalized probability so that
        # it corresponds to the orders of the wires passed.
        num_wires = len(mapped_wires)
        basis_states = qml.QubitDevice.generate_basis_states(num_wires)
        basis_states = basis_states[:, qml.math.argsort(qml.math.argsort(mapped_wires))]

        powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]
        perm = basis_states @ powers_of_two
        return prob[:, perm] if batch_size is not None else prob[perm]
