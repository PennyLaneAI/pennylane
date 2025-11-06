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
from collections.abc import Sequence

import numpy as np

from pennylane import math
from pennylane.exceptions import MeasurementShapeError, QuantumFunctionError
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from .counts import CountsMP
from .measurements import SampleMeasurement
from .mid_measure import MeasurementValue
from .process_samples import process_raw_samples


class SampleMP(SampleMeasurement):
    """Measurement process that returns the samples of a given observable. If no observable is
    provided then basis state samples are returned directly from the device.

    Please refer to :func:`pennylane.sample` for detailed documentation.

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
        dtype (str or None): The dtype of the samples returned by this measurement process.
    """

    _shortname = "sample"

    # pylint: disable=too-many-arguments
    def __init__(self, obs=None, wires=None, eigvals=None, id=None, dtype=None):

        self._dtype = dtype

        if isinstance(obs, MeasurementValue):
            super().__init__(obs=obs)
            return

        if isinstance(obs, Sequence):
            if not all(
                isinstance(o, MeasurementValue) and len(o.measurements) == 1 for o in obs
            ) and not all(math.is_abstract(o) for o in obs):
                raise QuantumFunctionError(
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

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ) -> tuple[tuple[int, ...], type]:
        if shots is None:
            raise ValueError("finite shots are required to use SampleMP")
        sample_eigvals = n_wires is None or has_eigvals
        dtype = float if sample_eigvals else int

        if sample_eigvals:
            return (shots,), dtype
        dim = num_device_wires if n_wires == 0 else n_wires
        return (shots, dim), dtype

    @property
    def numeric_type(self):
        if self._dtype is not None:
            return self._dtype
        if self.obs is None:
            # Computational basis samples
            return int
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        if not shots:
            raise MeasurementShapeError(
                "Shots are required to obtain the shape of the measurement "
                f"{self.__class__.__name__}."
            )
        if self.obs:
            return (shots,)

        if self.mv is not None:
            num_values_per_shot = 1 if isinstance(self.mv, MeasurementValue) else len(self.mv)
        else:
            num_values_per_shot = len(self.wires) if len(self.wires) > 0 else num_device_wires

        return (shots, num_values_per_shot)

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: WiresLike,
        shot_range: None | tuple[int, ...] = None,
        bin_size: None | int = None,
    ) -> TensorLike:

        return process_raw_samples(
            self, samples, wire_order, shot_range=shot_range, bin_size=bin_size, dtype=self._dtype
        )

    def process_counts(self, counts: dict, wire_order: WiresLike) -> np.ndarray:
        samples = []
        mapped_counts = self._map_counts(counts, wire_order)
        for outcome, count in mapped_counts.items():
            outcome_sample = self._compute_outcome_sample(outcome)
            if len(self.wires) == 1 and self.eigvals() is None:
                # For sampling wires, if only one wire is sampled, flatten the list
                outcome_sample = outcome_sample[0]
            samples.extend([outcome_sample] * count)

        return np.array(samples)

    def _map_counts(self, counts_to_map: dict, wire_order: WiresLike) -> dict:
        """
        Args:
            counts_to_map (dict): Dictionary where key is binary representation of the outcome and value is its count
            wire_order (WiresLike): Order of wires to which counts_to_map should be ordered in

        Returns:
            Dictionary where counts_to_map has been reordered according to wire_order
        """
        with QueuingManager.stop_recording():
            helper_counts = CountsMP(wires=self.wires, all_outcomes=False)
        return helper_counts.process_counts(counts_to_map, wire_order)

    def _compute_outcome_sample(self, outcome: str) -> list:
        """
        Args:
            outcome (str): The binary string representation of the measurement outcome.

        Returns:
            list: A list of outcome samples for given binary string.
                If eigenvalues exist, the binary outcomes are mapped to their corresponding eigenvalues.
        """
        if self.eigvals() is not None:
            eigvals = self.eigvals()
            return eigvals[int(outcome, 2)]

        return [int(bit) for bit in outcome]


def sample(
    op: Operator | MeasurementValue | Sequence[MeasurementValue] | None = None,
    wires: WiresLike = None,
    dtype=None,
) -> SampleMP:
    r"""Sample from the supplied observable, with the number of shots
    determined from QNode,
    returning raw samples. If no observable is provided, then basis state samples are returned
    directly from the device.

    Note that the output shape of this measurement process depends on the shots
    specified on the device.

    Args:
        op (Operator or MeasurementValue): a quantum observable object. To get samples
            for mid-circuit measurements, ``op`` should be a ``MeasurementValue``.
        wires (Sequence[int] or int or None): the wires we wish to sample from; ONLY set wires if
            op is ``None``.
        dtype: The dtype of the samples returned by this measurement process.

    Returns:
        SampleMP: Measurement process instance

    Raises:
        ValueError: Cannot set wires if an observable is provided

    .. warning::

        In v0.42, a breaking change removed the squeezing of singleton dimensions, eliminating the need for
        specialized, error-prone handling for finite-shot results.
        For the QNode:

        >>> @qml.qnode(qml.device('default.qubit'))
        ... def circuit(wires):
        ...     return qml.sample(wires=wires)

        We previously squeezed out singleton dimensions like:

        >>> qml.set_shots(circuit, 1)(wires=1)
        array(0)
        >>> qml.set_shots(circuit, 2)(0)
        array([0, 0])
        >>> qml.set_shots(circuit, 1)((0,1))
        array([0, 0])

        With v0.42 and newer, the above circuit will **always** return an array of shape ``(shots, num_wires)``.

        >>> qml.set_shots(circuit, 1)(wires=1)
        array([[0]])
        >>> qml.set_shots(circuit, 2)(0)
        array([[0],
        [0]])
        >>> qml.set_shots(circuit, 1)((0,1))
        array([[0, 0]])

        Previous behavior can be recovered by applying ``qml.math.squeeze(result)`` to the array.

    The samples are drawn from the eigenvalues :math:`\{\lambda_i\}` of the observable.
    The probability of drawing eigenvalue :math:`\lambda_i` is given by
    :math:`p(\lambda_i) = |\langle \xi_i | \psi \rangle|^2`, where :math:`| \xi_i \rangle`
    is the corresponding basis state from the observable's eigenbasis.

    .. note::

        QNodes that return samples cannot, in general, be differentiated, since the derivative
        with respect to a sample --- a stochastic process --- is ill-defined. An alternative
        approach would be to use single-shot expectation values. For example, instead of this:

        .. code-block:: python

            from functools import partial
            dev = qml.device("default.qubit")

            @partial(qml.set_shots, shots=10)
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(angle):
                qml.RX(angle, wires=0)
                return qml.sample(qml.PauliX(0))

            angle = qml.numpy.array(0.1)
            res = qml.jacobian(circuit)(angle)

        Consider using :func:`~pennylane.expval` and a sequence of single shots, like this:

        .. code-block:: python

            from functools import partial
            dev = qml.device("default.qubit")

            @partial(qml.set_shots, shots=[(1, 10)])
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

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=4)
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
    from the device are returned (e.g., for a qubit device, samples from the
    computational basis are returned). In this case, ``wires`` can be specified
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
            return qml.sample()

    Executing this QNode:

    >>> circuit(0.5)
    array([[0, 1],
           [0, 0],
           [1, 1],
           [0, 0]])

    .. details::
            :title: Setting the precision of the samples

            The ``dtype`` argument can be used to set the type and precision of the samples returned by this measurement process
            when the ``op`` argument does not contain mid-circuit measurements. Otherwise, the ``dtype`` argument is ignored.

            By default, the samples will be returned as floating point numbers if an observable is provided,
            and as integers if no observable is provided. The ``dtype`` argument can be used to specify further details,
            and set the precision to any valid interface-like dtype, e.g. ``'float32'``, ``'int8'``, ``'uint16'``, etc.

            We show two examples below using the JAX and PyTorch interfaces.
            This argument is compatible with all interfaces currently supported by PennyLane.

            **Example:**

            .. code-block:: python3

                @qml.set_shots(1000000)
                @qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
                def circuit():
                    qml.Hadamard(0)
                    return qml.sample(dtype="int8")

            Executing this QNode, we get:

            >>> samples = circuit()
            >>> samples.dtype
            dtype('int8')
            >>> type(samples)
            jaxlib._jax.ArrayImpl

            If an observable is provided, the samples will be floating point numbers:

            .. code-block:: python3

                @qml.set_shots(1000000)
                @qml.qnode(qml.device("default.qubit", wires=1), interface="torch")
                def circuit():
                    qml.Hadamard(0)
                    return qml.sample(qml.Z(0), dtype="float32")

            Executing this QNode, we get:

            >>> samples = circuit()
            >>> samples.dtype
            torch.float32
            >>> type(samples)
            torch.Tensor

    """
    return SampleMP(obs=op, wires=None if wires is None else Wires(wires), dtype=dtype)
