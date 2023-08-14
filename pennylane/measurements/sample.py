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
import warnings
from typing import Sequence, Tuple, Optional, Union

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import MeasurementShapeError, Sample, SampleMeasurement
from .mid_measure import MeasurementValue


def sample(
    op: Optional[Union[Operator, Sequence[MeasurementValue]]] = None, wires=None
) -> "SampleMP":
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device,
    returning raw samples. If no observable is provided then basis state samples are returned
    directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Observable or Sequence[MeasurementValue]): a quantum observable object. To get samples
            for mid-circuit measurements, ``op`` should be specified as a sequence of
            their respective ``MeasurementValue``'s.
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
        with respect to a sample --- a stochastic process --- is ill-defined. The one exception
        is if the QNode uses the parameter-shift method (``diff_method="parameter-shift"``), in
        which case ``qml.sample(obs)`` is interpreted as a single-shot expectation value of the
        observable ``obs``.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliY(0))

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
    if isinstance(op, MeasurementValue):
        op = (op,)
    if isinstance(op, Sequence):
        return SampleMP(obs=tuple(op))

    if op is not None and not op.is_hermitian:  # None type is also allowed for op
        warnings.warn(f"{op.name} might not be hermitian.")

    if wires is not None:
        if op is not None:
            raise ValueError(
                "Cannot specify the wires to sample if an observable is "
                "provided. The wires to sample will be determined directly from the observable."
            )
        wires = Wires(wires)

    return SampleMP(obs=op, wires=wires)


class SampleMP(SampleMeasurement):
    """Measurement process that returns the samples of a given observable. If no observable is
    provided then basis state samples are returned directly from the device.

    Please refer to :func:`sample` for detailed documentation.

    Args:
        obs (Union[.Operator, Tuple[.MeasurementValue]]): The observable that is to be measured
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
        return Sample

    @property
    @functools.lru_cache()
    def numeric_type(self):
        # Note: we only assume an integer numeric type if the observable is a
        # built-in observable with integer eigenvalues or a tensor product thereof
        if self.obs is None:
            # Computational basis samples
            return int
        int_eigval_obs = {qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity}
        tensor_terms = self.obs.obs if hasattr(self.obs, "obs") else [self.obs]
        every_term_standard = all(o.__class__ in int_eigval_obs for o in tensor_terms)
        return int if every_term_standard else float

    def _shape_legacy(self, device, shots):
        if not shots:
            raise MeasurementShapeError(
                "Shots are required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        if shots.has_partitioned_shots:
            if self.obs is None:
                # TODO: revisit when qml.sample without an observable fully
                # supports shot vectors
                raise MeasurementShapeError(
                    "Getting the output shape of a measurement returning samples along with "
                    "a device with a shot vector is not supported."
                )
            return tuple(
                (s.shots,) * s.copies if s.shots != 1 else tuple() * s.copies
                for s in shots.shot_vector
            )
        len_wires = len(self.wires) if len(self.wires) > 0 else len(device.wires)
        return (1, shots.total_shots) if self.obs is not None else (1, shots.total_shots, len_wires)

    def shape(self, device, shots):
        if not qml.active_return():
            return self._shape_legacy(device, shots)
        if not shots:
            raise MeasurementShapeError(
                "Shots are required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        len_wires = len(self.wires) if len(self.wires) > 0 else len(device.wires)

        def _single_int_shape(shot_val, num_wires):
            # singleton dimensions, whether in shot val or num_wires are squeezed away
            inner_shape = []
            if shot_val != 1:
                inner_shape.append(shot_val)
            if num_wires != 1:
                inner_shape.append(num_wires)
            return tuple(inner_shape)

        if not shots.has_partitioned_shots:
            return _single_int_shape(shots.total_shots, len_wires)

        shape = []
        for s in shots.shot_vector:
            for _ in range(s.copies):
                shape.append(_single_int_shape(s.shots, len_wires))

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
        name = self.obs.name if self.obs is not None and not isinstance(self.obs, tuple) else None
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

        if self.obs is None or isinstance(self.obs, tuple):
            # if no observable was provided then return the raw samples
            return samples if bin_size is None else samples.T.reshape(num_wires, bin_size, -1)

        if str(name) in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            # Process samples for observables with eigenvalues {1, -1}
            samples = 1 - 2 * qml.math.squeeze(samples)
            if samples.shape == ():
                samples = samples.reshape((1,))
        else:
            # Replace the basis state in the computational basis with the correct eigenvalue.
            # Extract only the columns of the basis samples required based on ``wires``.
            powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]
            indices = samples @ powers_of_two
            indices = qml.math.array(indices)  # Add np.array here for Jax support.
            try:
                samples = self.obs.eigvals()[indices]
            except qml.operation.EigvalsUndefinedError as e:
                # if observable has no info on eigenvalues, we cannot return this measurement
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute samples of {self.obs.name}."
                ) from e

        return samples if bin_size is None else samples.reshape((bin_size, -1))
