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
This module contains the qml.probs measurement.
"""
import pennylane as qml

from .measurements import MeasurementProcess, Probability


def probs(wires=None, op=None):
    r"""Probability of each computational basis state.

    This measurement function accepts either a wire specification or
    an observable. Passing wires to the function
    instructs the QNode to return a flat array containing the
    probabilities :math:`|\langle i | \psi \rangle |^2` of measuring
    the computational basis state :math:`| i \rangle` given the current
    state :math:`| \psi \rangle`.

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

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            return qml.probs(op=qml.Hermitian(H, wires=0))

    >>> circuit()

    array([0.14644661 0.85355339])

    The returned array is in lexicographic order, so corresponds
    to a :math:`14.6\%` chance of measuring the rotated :math:`|0\rangle` state
    and :math:`85.4\%` of measuring the rotated :math:`|1\rangle` state.

    Note that the output shape of this measurement process depends on whether
    the device simulates qubit or continuous variable quantum systems.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        op (Observable): Observable (with a diagonalzing_gates attribute) that rotates
         the computational basis
    """
    # pylint: disable=protected-access

    if wires is None and op is None:
        raise qml.QuantumFunctionError(
            "qml.probs requires either the wires or the observable to be passed."
        )

    if isinstance(op, qml.Hamiltonian):
        raise qml.QuantumFunctionError("Hamiltonians are not supported for rotating probabilities.")

    if isinstance(op, (qml.ops.Sum, qml.ops.SProd, qml.ops.Prod)):  # pylint: disable=no-member
        raise qml.QuantumFunctionError(
            "Symbolic Operations are not supported for rotating probabilities yet."
        )

    if op is not None and not qml.operation.defines_diagonalizing_gates(op):
        raise qml.QuantumFunctionError(
            f"{op} does not define diagonalizing gates : cannot be used to rotate the probability"
        )

    if wires is not None:
        if op is not None:
            raise qml.QuantumFunctionError(
                "Cannot specify the wires to probs if an observable is "
                "provided. The wires for probs will be determined directly from the observable."
            )
        return MeasurementProcess(Probability, wires=qml.wires.Wires(wires))
    return MeasurementProcess(Probability, obs=op)
