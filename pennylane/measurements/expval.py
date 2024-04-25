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
This module contains the qml.expval measurement.
"""
from typing import Sequence, Tuple, Union

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import Expectation, SampleMeasurement, StateMeasurement
from .sample import SampleMP
from .mid_measure import MeasurementValue


def expval(
    op: Union[Operator, MeasurementValue],
):
    r"""Expectation value of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Y(0))

    Executing this QNode:

    >>> circuit(0.5)
    -0.4794255386042029

    Args:
        op (Union[Observable, MeasurementValue]): a quantum observable object. To
            get expectation values for mid-circuit measurements, ``op`` should be
            a ``MeasurementValue``.

    Returns:
        ExpectationMP: measurement process instance
    """
    if isinstance(op, MeasurementValue):
        return ExpectationMP(obs=op)

    if isinstance(op, Sequence):
        raise ValueError(
            "qml.expval does not support measuring sequences of measurements or observables"
        )

    if isinstance(op, qml.Identity) and len(op.wires) == 0:
        # temporary solution to merge https://github.com/PennyLaneAI/pennylane/pull/5106
        # allow once we have testing and confidence in qml.expval(I())
        raise NotImplementedError(
            "Expectation values of qml.Identity() without wires are currently not allowed."
        )

    return ExpectationMP(obs=op)


class ExpectationMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the expectation value of the supplied observable.

    Please refer to :func:`expval` for detailed documentation.

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
        return Expectation

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        if not self.wires:
            return qml.math.squeeze(self.eigvals())
        # estimate the ev
        op = self.mv if self.mv is not None else self.obs
        with qml.queuing.QueuingManager.stop_recording():
            samples = SampleMP(
                obs=op,
                eigvals=self._eigvals,
                wires=self.wires if self._eigvals is not None else None,
            ).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )

        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.mean(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        # we use ``self.wires`` instead of ``self.obs`` because the observable was
        # already applied to the state
        if not self.wires:
            return qml.math.squeeze(self.eigvals())
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
        return self._calculate_expectation(prob)

    def process_counts(self, counts: dict, wire_order: Wires):
        with qml.QueuingManager.stop_recording():
            probs = qml.probs(wires=self.wires).process_counts(counts=counts, wire_order=wire_order)
        return self._calculate_expectation(probs)

    def _calculate_expectation(self, probabilities):
        """
        Calculate the of expectation set of probabilities.

        Args:
            probabilities (array): the probabilities of collapsing to eigen states
        """
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")
        return qml.math.dot(probabilities, eigvals)
