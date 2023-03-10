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
import copy
import warnings
from typing import Sequence, Tuple, Union

import pennylane as qml
from pennylane.operation import Observable
from pennylane.wires import Wires

from .measurements import AllCounts, Counts, SampleMeasurement


def counts(op=None, wires=None, all_outcomes=False):
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning the number of counts for each sample. If no observable is provided then basis state
    samples are returned directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or None): a quantum observable object
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
        def circuit(x):
            qml.PauliX(wires=0)
            return qml.counts(all_outcomes=True)

    Executing this QNode shows counts for all states:

    >>> circuit()
    {'00': 0, '01': 0, '10': 4, '11': 0}

    .. note::

        QNodes that return samples cannot, in general, be differentiated, since the derivative
        with respect to a sample --- a stochastic process --- is ill-defined. The one exception
        is if the QNode uses the parameter-shift method (``diff_method="parameter-shift"``), in
        which case ``qml.sample(obs)`` is interpreted as a single-shot expectation value of the
        observable ``obs``.
    """
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
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
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
        obs: Union[Observable, None] = None,
        wires=None,
        eigvals=None,
        id=None,
        all_outcomes=False,
    ):
        self.all_outcomes = all_outcomes
        super().__init__(obs, wires, eigvals, id)

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
            if self.obs is None
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
            samples: samples in an array of dimension ``(shots,len(wires))``

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

        if self.obs is None:
            # convert samples and outcomes (if using) from arrays to str for dict keys
            num_wires = len(self.wires) if len(self.wires) > 0 else qml.math.shape(samples)[-1]
            samples = ["".join([str(s.item()) for s in sample]) for sample in samples]
            if self.all_outcomes:
                outcomes = qml.QubitDevice.generate_basis_states(num_wires)
                outcomes = ["".join([str(o.item()) for o in outcome]) for outcome in outcomes]
        elif self.all_outcomes:
            outcomes = qml.eigvals(self.obs)

        # generate empty outcome dict, populate values with state counts
        outcome_dict = {k: qml.math.int64(0) for k in outcomes}
        states, _counts = qml.math.unique(samples, return_counts=True)
        for s, c in zip(states, _counts):
            outcome_dict[s] = c

        return outcome_dict

    def __copy__(self):
        return self.__class__(
            obs=copy.copy(self.obs),
            eigvals=self._eigvals,
            wires=self._wires,
            all_outcomes=self.all_outcomes,
        )
