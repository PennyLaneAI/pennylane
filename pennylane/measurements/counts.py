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
This module contains the qml.counts measurement.
"""
from typing import Sequence, Tuple, Optional
import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import AllCounts, Counts, SampleMeasurement
from .mid_measure import MeasurementValue


def counts(
    op=None,
    wires=None,
    all_outcomes=False,
) -> "CountsMP":
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning the number of counts for each sample. If no observable is provided then basis state
    samples are returned directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or MeasurementValue or None): a quantum observable object. To get counts
            for mid-circuit measurements, ``op`` should be a ``MeasurementValue``.
        wires (Sequence[int] or int or None): the wires we wish to sample from, ONLY set wires if
            op is None
        all_outcomes(bool): determines whether the returned dict will contain only the observed
            outcomes (default), or whether it will display all possible outcomes for the system

    Returns:
        CountsMP: Measurement process instance

    Raises:
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    .. note::

        Differentiation of QNodes that return ``counts`` is currently not supported. Please refer to
        :func:`~.pennylane.sample` if differentiability is required.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    {-1: 2, 1: 2}

    If no observable is provided, then the raw basis state samples obtained
    from device are returned (e.g., for a qubit device, samples from the
    computational device are returned). In this case, ``wires`` can be specified
    so that sample results only include measurement results of the qubits of interest.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

    Executing this QNode:

    >>> circuit(0.5)
    {'00': 3, '01': 1}

    By default, outcomes that were not observed will not be included in the dictionary.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts()

    Executing this QNode shows only the observed outcomes:

    >>> circuit()
    {'10': 4}

    Passing all_outcomes=True will create a dictionary that displays all possible outcomes:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts(all_outcomes=True)

    Executing this QNode shows counts for all states:

    >>> circuit()
    {'00': 0, '01': 0, '10': 4, '11': 0}

    """
    if isinstance(op, MeasurementValue):
        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if isinstance(op, Sequence):
        if not all(isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in op):
            raise qml.QuantumFunctionError(
                "Only sequences of single MeasurementValues can be passed with the op argument. "
                "MeasurementValues manipulated using arithmetic operators cannot be used when "
                "collecting statistics for a sequence of mid-circuit measurements."
            )

        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is provided. The wires "
                "to sample will be determined directly from the observable."
            )
        wires = Wires(wires)

    return CountsMP(obs=op, wires=wires, all_outcomes=all_outcomes)


class CountsMP(SampleMeasurement):
    """Measurement process that samples from the supplied observable and returns the number of
    counts for each sample.

    Please refer to :func:`counts` for detailed documentation.

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
        all_outcomes(bool): determines whether the returned dict will contain only the observed
            outcomes (default), or whether it will display all possible outcomes for the system
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        obs: Optional[Operator] = None,
        wires=None,
        eigvals=None,
        id: Optional[str] = None,
        all_outcomes: bool = False,
    ):
        self.all_outcomes = all_outcomes
        if wires is not None:
            wires = Wires(wires)
        super().__init__(obs, wires, eigvals, id)

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("all_outcomes", self.all_outcomes))
        return (self.obs or self.mv, self._eigvals), metadata

    def __repr__(self):
        if self.mv:
            return f"CountsMP({repr(self.mv)}, all_outcomes={self.all_outcomes})"
        if self.obs:
            return f"CountsMP({self.obs}, all_outcomes={self.all_outcomes})"
        if self._eigvals is not None:
            return f"CountsMP(eigvals={self._eigvals}, wires={self.wires.tolist()}, all_outcomes={self.all_outcomes})"

        return f"CountsMP(wires={self.wires.tolist()}, all_outcomes={self.all_outcomes})"

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            getattr(self.obs, "hash", "None"),
            str(self._eigvals),  # eigvals() could be expensive to compute for large observables
            tuple(self.wires.tolist()),
            self.all_outcomes,
        )

        return hash(fingerprint)

    @property
    def return_type(self):
        return AllCounts if self.all_outcomes else Counts

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        with qml.queuing.QueuingManager.stop_recording():
            samples = qml.sample(op=self.obs or self.mv, wires=self._wires).process_samples(
                samples, wire_order, shot_range, bin_size
            )

        if bin_size is None:
            return self._samples_to_counts(samples)

        num_wires = len(self.wires) if self.wires else len(wire_order)
        samples = (
            samples.reshape((num_wires, -1)).T.reshape(-1, bin_size, num_wires)
            if self.obs is None and not isinstance(self.mv, MeasurementValue)
            else samples.reshape((-1, bin_size))
        )

        return [self._samples_to_counts(bin_sample) for bin_sample in samples]

    def _samples_to_counts(self, samples):
        """Groups the samples into a dictionary showing number of occurrences for
        each possible outcome.

        The format of the dictionary depends on the all_outcomes attribute. By default,
        the dictionary will only contain the observed outcomes. Optionally (all_outcomes=True)
        the dictionary will instead contain all possible outcomes, with a count of 0
        for those not observed. See example.

        Args:
            samples: An array of samples, with the shape being ``(shots,len(wires))`` if an observable
                is provided, with sample values being an array of 0s or 1s for each wire. Otherwise, it
                has shape ``(shots,)``, with sample values being scalar eigenvalues of the observable

        Returns:
            dict: dictionary with format ``{'outcome': num_occurrences}``, including all
                outcomes for the sampled observable

        **Example**

            >>> samples
            tensor([[0, 0],
                    [0, 0],
                    [1, 0]], requires_grad=True)

            By default, this will return:
            >>> self._samples_to_counts(samples)
            {'00': 2, '10': 1}

            However, if ``all_outcomes=True``, this will return:
            >>> self._samples_to_counts(samples)
            {'00': 2, '01': 0, '10': 1, '11': 0}

            The variable all_outcomes can be set when running measurements.counts, i.e.:

             .. code-block:: python3

                dev = qml.device("default.qubit", wires=2, shots=4)

                @qml.qnode(dev)
                def circuit(x):
                    qml.RX(x, wires=0)
                    return qml.counts(all_outcomes=True)

        """

        outcomes = []

        # if an observable was provided, batched samples will have shape (batch_size, shots)
        batched_ndims = 2
        shape = qml.math.shape(samples)

        if self.obs is None and not isinstance(self.mv, MeasurementValue):
            # convert samples and outcomes (if using) from arrays to str for dict keys
            batched_ndims = 3  # no observable was provided, batched samples will have shape (batch_size, shots, len(wires))

            # remove nans
            mask = qml.math.isnan(samples)
            num_wires = shape[-1]
            if np.any(mask):
                mask = np.logical_not(np.any(mask, axis=tuple(range(1, samples.ndim))))
                samples = samples[mask, ...]

            def convert(sample):
                # convert array of ints to string
                return "".join(str(s) for s in sample)

            new_shape = samples.shape[:-1]
            # Flatten broadcasting axis
            flattened_samples = np.reshape(samples, (-1, shape[-1])).astype(np.int8)
            samples = list(map(convert, flattened_samples))
            samples = np.reshape(np.array(samples), new_shape)

            if self.all_outcomes:

                def convert_from_int(x):
                    # convert int to binary string
                    return f"{x:0{num_wires}b}"

                num_wires = len(self.wires) if len(self.wires) > 0 else shape[-1]
                outcomes = list(map(convert_from_int, range(2**num_wires)))

        elif self.all_outcomes:
            # This also covers statistics for mid-circuit measurements manipulated using
            # arithmetic operators
            outcomes = self.eigvals()

        batched = len(shape) == batched_ndims
        if not batched:
            samples = samples[None]

        # generate empty outcome dict, populate values with state counts
        base_dict = {k: qml.math.int64(0) for k in outcomes}
        outcome_dicts = [base_dict.copy() for _ in range(shape[0])]
        results = [qml.math.unique(batch, return_counts=True) for batch in samples]

        for result, outcome_dict in zip(results, outcome_dicts):
            states, _counts = result
            for state, count in zip(qml.math.unwrap(states), _counts):
                outcome_dict[state] = count

        def outcome_to_eigval(outcome: str):
            return self.eigvals()[int(outcome, 2)]

        if self._eigvals is not None:
            outcome_dicts = [
                {outcome_to_eigval(outcome): count for outcome, count in outcome_dict.items()}
                for outcome_dict in outcome_dicts
            ]

        return outcome_dicts if batched else outcome_dicts[0]

    # pylint: disable=redefined-outer-name
    def process_counts(self, counts: dict, wire_order: Wires) -> dict:
        mapped_counts = self._map_counts(counts, wire_order)
        if self.all_outcomes:
            self._include_all_outcomes(mapped_counts)
        else:
            _remove_unobserved_outcomes(mapped_counts)
        return mapped_counts

    def _map_counts(self, counts_to_map: dict, wire_order: Wires) -> dict:
        """
        Args:
            counts_to_map: Dictionary where key is binary representation of the outcome and value is its count
            wire_order: Order of wires to which counts_to_map should be ordered in

        Returns:
            Dictionary where counts_to_map has been reordered according to wire_order
        """
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]

        mapped_counts = {}
        for outcome, occurrence in counts_to_map.items():
            mapped_outcome = "".join(outcome[i] for i in mapped_wires)
            mapped_counts[mapped_outcome] = mapped_counts.get(mapped_outcome, 0) + occurrence

        return mapped_counts

    def _include_all_outcomes(self, outcome_counts: dict) -> None:
        """
        Includes missing outcomes in outcome_counts.
        If an outcome is not present in outcome_counts, it's count is considered 0

        Args:
            outcome_counts(dict): Dictionary where key is binary representation of the outcome and value is its count
        """
        num_wires = len(self.wires)
        num_outcomes = 2**num_wires
        if num_outcomes == len(outcome_counts.keys()):
            return

        binary_pattern = "{0:0" + str(num_wires) + "b}"
        for outcome in range(num_outcomes):
            outcome_binary = binary_pattern.format(outcome)
            if outcome_binary not in outcome_counts:
                outcome_counts[outcome_binary] = 0


def _remove_unobserved_outcomes(outcome_counts: dict):
    """
    Removes unobserved outcomes, i.e. whose count is 0 from the outcome_count dictionary.

    Args:
        outcome_counts(dict): Dictionary where key is binary representation of the outcome and value is its count
    """
    for outcome in list(outcome_counts.keys()):
        if outcome_counts[outcome] == 0:
            del outcome_counts[outcome]
