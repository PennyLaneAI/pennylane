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
from collections.abc import Sequence

from pennylane import math
from pennylane.operation import Operator
from pennylane.ops import I
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .measurement_value import MeasurementValue
from .measurements import SampleMeasurement, StateMeasurement
from .probs import probs
from .sample import SampleMP


class ExpectationMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the expectation value of the supplied observable.

    Please refer to :func:`pennylane.expval` for detailed documentation.

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

    _shortname = "expval"

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
    ):
        if not self.wires:
            return math.squeeze(self.eigvals())
        # estimate the ev
        op = self.mv if self.mv is not None else self.obs
        with QueuingManager.stop_recording():
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
        return math.squeeze(math.mean(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        # we use ``self.wires`` instead of ``self.obs`` because the observable was
        # already applied to the state
        if not self.wires:
            return math.squeeze(self.eigvals())
        with QueuingManager.stop_recording():
            probabilities = probs(wires=self.wires).process_state(
                state=state, wire_order=wire_order
            )
        # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
        return self._calculate_expectation(probabilities)

    def process_counts(self, counts: dict, wire_order: Wires):
        with QueuingManager.stop_recording():
            probabilties = probs(wires=self.wires).process_counts(
                counts=counts, wire_order=wire_order
            )
        return self._calculate_expectation(probabilties)

    def process_density_matrix(self, density_matrix: Sequence[complex], wire_order: Wires):
        if not self.wires:
            return math.squeeze(self.eigvals())
        with QueuingManager.stop_recording():
            probabilities = probs(wires=self.wires).process_density_matrix(
                density_matrix=density_matrix, wire_order=wire_order
            )
        return self._calculate_expectation(probabilities)

    def _calculate_expectation(self, probabilities):
        """
        Calculate the of expectation set of probabilities.

        Args:
            probabilities (array): the probabilities of collapsing to eigen states
        """
        return math.dot(probabilities, self.eigvals())


def expval(
    op: Operator | MeasurementValue,
) -> ExpectationMP:
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
        op (Union[Operator, MeasurementValue]): a quantum observable object. To
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

    if isinstance(op, I) and len(op.wires) == 0:
        # temporary solution to merge https://github.com/PennyLaneAI/pennylane/pull/5106
        # allow once we have testing and confidence in qml.expval(I())
        raise NotImplementedError(
            "Expectation values of qml.Identity() without wires are currently not allowed."
        )

    return ExpectationMP(obs=op)
