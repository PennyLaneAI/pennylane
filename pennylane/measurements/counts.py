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
# pylint: disable=protected-access
"""
This module contains the qml.counts measurement.
"""
import warnings

from pennylane.wires import Wires

from .measurements import AllCounts, Counts, MeasurementProcess


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

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
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
    {'01': 4}

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

    if all_outcomes:
        return MeasurementProcess(AllCounts, obs=op, wires=wires)

    return MeasurementProcess(Counts, obs=op, wires=wires)
