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
import warnings
from collections.abc import Sequence

import numpy as np

from pennylane import math
from pennylane.exceptions import QuantumFunctionError
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .capture_measurements import _get_abstract_measurement
from .measurement_value import MeasurementValue
from .measurements import SampleMeasurement
from .process_samples import process_raw_samples


class CountsMP(SampleMeasurement):
    """Measurement process that samples from the supplied observable and returns the number of
    counts for each sample.

    Please refer to :func:`pennylane.counts` for detailed documentation.

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

    _shortname = "counts"

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        obs: Operator | None = None,
        wires=None,
        eigvals=None,
        id: str | None = None,
        all_outcomes: bool = False,
    ):
        self.all_outcomes = all_outcomes
        self._shortname = "allcounts" if all_outcomes else "counts"
        if wires is not None:
            wires = Wires(wires)
        super().__init__(obs, wires, eigvals, id)

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("all_outcomes", self.all_outcomes))
        return (self.obs or self.mv, self._eigvals), metadata

    def __repr__(self):
        if self.mv is not None:
            return f"CountsMP({repr(self.mv)}, all_outcomes={self.all_outcomes})"
        if self.obs:
            return f"CountsMP({self.obs}, all_outcomes={self.all_outcomes})"
        if self._eigvals is not None:
            return f"CountsMP(eigvals={self._eigvals}, wires={self.wires.tolist()}, all_outcomes={self.all_outcomes})"

        return f"CountsMP(wires={self.wires.tolist()}, all_outcomes={self.all_outcomes})"

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ) -> tuple:
        raise NotImplementedError(
            "CountsMP returns a dictionary, which is not compatible with capture."
        )

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

    def process_samples(
        self,
        samples: TensorLike,
        wire_order: Wires,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
    ):
        dummy_mp = CountsMP(obs=self.obs or self.mv, wires=self._wires)
        # cant use `self` due to eigvals differences
        samples = process_raw_samples(
            dummy_mp, samples, wire_order, shot_range=shot_range, bin_size=bin_size
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

                from functools import partial
                dev = qml.device("default.qubit", wires=2)

                @partial(qml.set_shots, shots=4)
                @qml.qnode(dev)
                def circuit(x):
                    qml.RX(x, wires=0)
                    return qml.counts(all_outcomes=True)

        """

        outcomes = []

        # if an observable was provided, batched samples will have shape (batch_size, shots)
        batched_ndims = 2
        shape = math.shape(samples)

        if self.obs is None and not isinstance(self.mv, MeasurementValue):
            # convert samples and outcomes (if using) from arrays to str for dict keys
            batched_ndims = 3  # no observable was provided, batched samples will have shape (batch_size, shots, len(wires))

            # remove nans
            mask = math.isnan(samples)
            num_wires = shape[-1]
            if math.any(mask):
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
        base_dict = {k: math.int64(0) for k in outcomes}
        outcome_dicts = [base_dict.copy() for _ in range(shape[0])]
        results = [math.unique(batch, return_counts=True) for batch in samples]

        for result, outcome_dict in zip(results, outcome_dicts):
            states, _counts = result
            for state, count in zip(math.unwrap(states), _counts):
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

        if self.eigvals() is not None:
            eigvals = self.eigvals()
            eigvals_dict = {k: math.int64(0) for k in eigvals}
            for outcome, count in mapped_counts.items():
                val = eigvals[int(outcome, 2)]
                eigvals_dict[val] += count
            if not self.all_outcomes:
                _remove_unobserved_outcomes(eigvals_dict)
            return eigvals_dict

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


# pylint: disable=protected-access, unused-argument
if CountsMP._wires_primitive is not None:

    CountsMP._wires_primitive.multiple_results = True

    @CountsMP._wires_primitive.def_impl
    def _(*args, **kwargs):
        raise NotImplementedError("Counts has no execution implementation with program capture.")

    def _keys_eval(n_wires=None, has_eigvals=False, shots=None, num_device_wires=0):
        if shots is None:
            raise ValueError("finite shots are required to use CountsMP")
        n_wires = n_wires or num_device_wires
        return (2**n_wires,), int

    def _values_eval(n_wires=None, has_eigvals=False, shots=None, num_device_wires=0):
        if shots is None:
            raise ValueError("finite shots are required to use CountsMP")
        n_wires = n_wires or num_device_wires
        return (2**n_wires,), int

    abstract_mp = _get_abstract_measurement()

    @CountsMP._wires_primitive.def_abstract_eval
    def _(*args, has_eigvals=False, all_outcomes=False):
        if not all_outcomes:
            warnings.warn(
                "all_outcomes=False is unsupported with program capture and qjit. Using all_outcomes=True",
                UserWarning,
            )
        n_wires = len(args) - 1 if has_eigvals else len(args)
        keys = abstract_mp(_keys_eval, n_wires=n_wires, has_eigvals=has_eigvals)
        values = abstract_mp(_values_eval, n_wires=n_wires, has_eigvals=has_eigvals)
        return keys, values


def counts(
    op=None,
    wires=None,
    all_outcomes=False,
) -> CountsMP:
    r"""Sample from the supplied observable, with the number of shots
    determined from QNode,
    returning the number of counts for each sample. If no observable is provided then basis state
    samples are returned directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Operator or MeasurementValue or None): a quantum observable object. To get counts
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

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=4)
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

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=4)
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

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=4)
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
        if not all(
            math.is_abstract(o) or (isinstance(o, MeasurementValue) and not o.has_processing)
            for o in op
        ):
            raise QuantumFunctionError(
                "Only sequences of unprocessed MeasurementValues can be passed with the op argument. "
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


def _remove_unobserved_outcomes(outcome_counts: dict):
    """
    Removes unobserved outcomes, i.e. whose count is 0 from the outcome_count dictionary.

    Args:
        outcome_counts(dict): Dictionary where key is binary representation of the outcome and value is its count
    """
    for outcome in list(outcome_counts.keys()):
        if outcome_counts[outcome] == 0:
            del outcome_counts[outcome]
