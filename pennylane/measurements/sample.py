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
This module contains the qml.sample measurement.
"""
import functools
from typing import Sequence, Tuple, Optional, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import MeasurementShapeError, Sample, SampleMeasurement
from .mid_measure import MeasurementValue


def sample(
    op: Optional[Union[Operator, MeasurementValue]] = None,
    wires=None,
) -> "SampleMP":
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning raw samples. If no observable is provided then basis state samples are returned
    directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or MeasurementValue): a quantum observable object. To get samples
            for mid-circuit measurements, ``op`` should be a``MeasurementValue``.
        wires (Sequence[int] or int or None): the wires we wish to sample from; ONLY set wires if
            op is ``None``

    Returns:
        SampleMP: Measurement process instance

    Raises:
        ValueError: Cannot set wires if an observable is provided

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    .. note::

        QNodes that return samples cannot, in general, be differentiated, since the derivative
        with respect to a sample --- a stochastic process --- is ill-defined. An alternative
        approach would be to use single-shot expectation values. For example, instead of this:

        .. code-block:: python

            dev = qml.device("default.qubit", shots=10)

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(angle):
                qml.RX(angle, wires=0)
                return qml.sample(qml.PauliX(0))

            angle = qml.numpy.array(0.1)
            res = qml.jacobian(circuit)(angle)

        Consider using :func:`~pennylane.expval` and a sequence of single shots, like this:

        .. code-block:: python

            dev = qml.device("default.qubit", shots=[(1, 10)])

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(angle):
                qml.RX(angle, wires=0)
                return qml.expval(qml.PauliX(0))

            def cost(angle):
                return qml.math.hstack(circuit(angle))

            angle = qml.numpy.array(0.1)
            res = qml.jacobian(cost)(angle)

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    array([ 1.,  1.,  1., -1.])

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
            return qml.sample()

    Executing this QNode:

    >>> circuit(0.5)
    array([[0, 1],
           [0, 0],
           [1, 1],
           [0, 0]])

    """
    return SampleMP(obs=op, wires=wires)


class SampleMP(SampleMeasurement):
    """Measurement process that returns the samples of a given observable. If no observable is
    provided then basis state samples are returned directly from the device.

    Please refer to :func:`sample` for detailed documentation.

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

    def __init__(self, obs=None, wires=None, eigvals=None, id=None):

        if isinstance(obs, MeasurementValue):
            super().__init__(obs=obs)
            return

        if isinstance(obs, Sequence):
            if not all(isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in obs):
                raise qml.QuantumFunctionError(
                    "Only sequences of single MeasurementValues can be passed with the op "
                    "argument. MeasurementValues manipulated using arithmetic operators cannot be "
                    "used when collecting statistics for a sequence of mid-circuit measurements."
                )

            super().__init__(obs=obs)
            return

        if wires is not None:
            if obs is not None:
                raise ValueError(
                    "Cannot specify the wires to sample if an observable is provided. The wires "
                    "to sample will be determined directly from the observable."
                )
            wires = Wires(wires)

        super().__init__(obs=obs, wires=wires, eigvals=eigvals, id=id)

    @property
    def return_type(self):
        return Sample

    @property
    @functools.lru_cache()
    def numeric_type(self):
        # Note: we only assume an integer numeric type if the observable is a
        # built-in observable with integer eigenvalues or a tensor product thereof
        if self.obs is None:
            # Computational basis samples
            return int
        int_eigval_obs = {qml.X, qml.Y, qml.Z, qml.Hadamard, qml.Identity}
        tensor_terms = self.obs.obs if isinstance(self.obs, qml.operation.Tensor) else [self.obs]
        every_term_standard = all(o.__class__ in int_eigval_obs for o in tensor_terms)
        return int if every_term_standard else float

    def shape(self, device, shots):
        if not shots:
            raise MeasurementShapeError(
                "Shots are required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        if self.obs:
            num_values_per_shot = 1  # one single eigenvalue
        else:
            # one value per wire
            num_values_per_shot = len(self.wires) if len(self.wires) > 0 else len(device.wires)

        def _single_int_shape(shot_val, num_values):
            # singleton dimensions, whether in shot val or num_wires are squeezed away
            inner_shape = []
            if shot_val != 1:
                inner_shape.append(shot_val)
            if num_values != 1:
                inner_shape.append(num_values)
            return tuple(inner_shape)

        if not shots.has_partitioned_shots:
            return _single_int_shape(shots.total_shots, num_values_per_shot)

        shape = []
        for s in shots.shot_vector:
            for _ in range(s.copies):
                shape.append(_single_int_shape(s.shots, num_values_per_shot))

        return tuple(shape)

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        name = self.obs.name if self.obs is not None else None
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
        if self.obs is None and not isinstance(self.mv, MeasurementValue) and self._eigvals is None:
            # if no observable was provided then return the raw samples
            return samples if bin_size is None else samples.T.reshape(num_wires, bin_size, -1)

        # If we're sampling observables
        if str(name) in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            # Process samples for observables with eigenvalues {1, -1}
            samples = 1 - 2 * qml.math.squeeze(samples, axis=-1)
        else:
            # Replace the basis state in the computational basis with the correct eigenvalue.
            # Extract only the columns of the basis samples required based on ``wires``.
            powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]
            indices = samples @ powers_of_two
            indices = qml.math.array(indices)  # Add np.array here for Jax support.
            try:
                # This also covers statistics for mid-circuit measurements manipulated using
                # arithmetic operators
                samples = self.eigvals()[indices]
            except qml.operation.EigvalsUndefinedError as e:
                # if observable has no info on eigenvalues, we cannot return this measurement
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute samples of {self.obs.name}."
                ) from e

        return samples if bin_size is None else samples.reshape((bin_size, -1))

    def process_counts(self, counts: dict, wire_order: Wires):
        samples = []
        mapped_counts = self._map_counts(counts, wire_order)
        for outcome, count in mapped_counts.items():
            outcome_sample = self._compute_outcome_sample(outcome)
            if len(self.wires) == 1:
                # If only one wire is sampled, flatten the list
                outcome_sample = outcome_sample[0]
            samples.extend([outcome_sample] * count)

        return np.array(samples)

    def _map_counts(self, counts_to_map, wire_order) -> dict:
        """
        Args:
            counts_to_map: Dictionary where key is binary representation of the outcome and value is its count
            wire_order: Order of wires to which counts_to_map should be ordered in

        Returns:
            Dictionary where counts_to_map has been reordered according to wire_order
        """
        with qml.QueuingManager.stop_recording():
            helper_counts = qml.counts(wires=self.wires, all_outcomes=False)
        return helper_counts.process_counts(counts_to_map, wire_order)

    def _compute_outcome_sample(self, outcome) -> list:
        """
        Args:
            outcome (str): The binary string representation of the measurement outcome.

        Returns:
            list: A list of outcome samples for given binary string.
                If eigenvalues exist, the binary outcomes are mapped to their corresponding eigenvalues.
        """
        outcome_samples = [int(bit) for bit in outcome]

        if self.eigvals() is not None:
            eigvals = self.eigvals()
            outcome_samples = [eigvals[outcome] for outcome in outcome_samples]

        return outcome_samples
