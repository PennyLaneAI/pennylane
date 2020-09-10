# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""
This module contains the functions for computing different types of measurement
outcomes from quantum observables - expectation values, variances of expectations,
and measurement samples using AnnotatedQueues.
"""
import pennylane as qml
from pennylane.operation import Expectation, Observable, Probability, Sample, Variance
from pennylane.qnodes import QuantumFunctionError


from .queuing import QueuingContext


class MeasurementProcess:
    """Represents a measurement process occurring at the end of a
    quantum variational circuit.

    Args:
        return_type (.ObservableReturnTypes): The type of measurement process.
            This includes ``Expectation``, ``Variance``, ``Sample``, or ``Probability``.
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to. If the measurement
            process includes observables, this argument should be the union of all
            observable wires.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, return_type, obs=None, wires=None):
        self.return_type = return_type
        self.wires = wires
        self.obs = obs


        # TODO: remove the following lines once devices
        # have been refactored to accept and understand recieving
        # measurement processes rather than specific observables.

        # The following lines are only applicable for measurement processes
        # that do no have corresponding observables (e.g., Probability). We use
        # them to 'trick' the device into thinking it has recieved an observable.

        # Below, we imitate an identity observable, so that the
        # device undertakes no action upon recieving this observable.
        self.name = "Identity"
        self.diagonalizing_gates = lambda: []
        self.data = []


def expval(op):
    r"""Expectation value of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    -0.4794255386042029

    Args:
        op (Observable): a quantum observable object

    Returns:
        MeasurementProcess: measurement process representing the measurement

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not isinstance(op, Observable):
        raise QuantumFunctionError(
            "{} is not an observable: cannot be used with expval".format(op.name)
        )

    meas_op = MeasurementProcess(Expectation, obs=op)
    QueuingContext.update_info(op, owner=meas_op)
    QueuingContext.append(meas_op, owns=op)
    return meas_op


def var(op):
    r"""Variance of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    0.7701511529340698

    Args:
        op (Observable): a quantum observable object

    Returns:
        MeasurementProcess: measurement process representing the measurement

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not isinstance(op, Observable):
        raise QuantumFunctionError(
            "{} is not an observable: cannot be used with var".format(op.name)
        )

    meas_op = MeasurementProcess(Variance, obs=op)
    QueuingContext.update_info(op, owner=meas_op)
    QueuingContext.append(meas_op, owns=op)
    return meas_op


def sample(op):
    r"""Sample from the supplied observable, with the number of shots
    determined from the ``dev.shots`` attribute of the corresponding device.

    **Example:**

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

    Args:
        op (Observable): a quantum observable object

    Returns:
        MeasurementProcess: measurement process representing the measurement

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not isinstance(op, Observable):
        raise QuantumFunctionError(
            "{} is not an observable: cannot be used with sample".format(op.name)
        )

    meas_op = MeasurementProcess(Sample, obs=op)
    QueuingContext.update_info(op, owner=meas_op)
    QueuingContext.append(meas_op, owns=op)
    return meas_op


def probs(wires):
    r"""Probability of each computational basis state.

    This measurement function accepts no observables, and instead
    instructs the QNode to return a flat array containing the
    probabilities of each quantum state.

    Marginal probabilities may also be requested by restricting
    the wires to a subset of the full system; the size of the
    returned array will be ``[2**len(wires)]``.

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

    Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Returns:
        MeasurementProcess: measurement process representing the measurement
    """
    # pylint: disable=protected-access
    meas_op = MeasurementProcess(Probability, wires=qml.wires.Wires(wires))
    QueuingContext.append(meas_op)
    return meas_op
