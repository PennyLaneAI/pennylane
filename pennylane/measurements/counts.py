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
from typing import Sequence, Tuple, Optional

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import AllCounts, Counts, SampleMeasurement
from .mid_measure import MeasurementValue


def _sample_to_str(sample):
    """Converts a bit-array to a string. For example, ``[0, 1]`` would become '01'."""
    return "".join(map(str, sample))


def counts(op=None, wires=None, all_outcomes=False) -> "CountsMP":
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
            return qml.counts(qml.PauliY(0))

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
            qml.PauliX(wires=0)
            return qml.counts()

    Executing this QNode shows only the observed outcomes:

    >>> circuit()
    {'10': 4}

    Passing all_outcomes=True will create a dictionary that displays all possible outcomes:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.counts(all_outcomes=True)

    Executing this QNode shows counts for all states:

    >>> circuit()
    {'00': 0, '01': 0, '10': 4, '11': 0}

    """
    if isinstance(op, MeasurementValue):
        return CountsMP(obs=op, all_outcomes=all_outcomes)

    if op is not None and not op.is_hermitian:  # None type is also allowed for op
        warnings.warn(f"{op.name} might not be hermitian.")

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is "
                "provided. The wires to sample will be determined directly from the observable."
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
        super().__init__(obs, wires, eigvals, id)

    def __repr__(self):
        if self.obs is None:
            if self._eigvals is None:
                return f"CountsMP(wires={self.wires.tolist()}, all_outcomes={self.all_outcomes})"
            return f"CountsMP(eigvals={self._eigvals}, wires={self.wires.tolist()}, all_outcomes={self.all_outcomes})"

        return f"CountsMP({self.obs}, all_outcomes={self.all_outcomes})"

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
        samples = qml.sample(op=self.obs, wires=self._wires).process_samples(
            samples, wire_order, shot_range, bin_size
        )

        if bin_size is None:
            return self._samples_to_counts(samples)

        num_wires = len(self.wires) if self.wires else len(wire_order)
        samples = (
            samples.reshape((num_wires, -1)).T.reshape(-1, bin_size, num_wires)
            if self.obs is None or isinstance(self.obs, MeasurementValue)
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

        if self.obs is None or isinstance(self.obs, MeasurementValue):
            # convert samples and outcomes (if using) from arrays to str for dict keys
            samples = qml.math.cast_like(samples, qml.math.int8(0))
            samples = qml.math.apply_along_axis(_sample_to_str, -1, samples)
            batched_ndims = 3  # no observable was provided, batched samples will have shape (batch_size, shots, len(wires))
            if self.all_outcomes:
                num_wires = len(self.wires) if len(self.wires) > 0 else shape[-1]
                outcomes = list(
                    map(_sample_to_str, qml.QubitDevice.generate_basis_states(num_wires))
                )
        elif self.all_outcomes:
            outcomes = qml.eigvals(self.obs)

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

        return outcome_dicts if batched else outcome_dicts[0]
