# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains functions for tape expansion
"""
# pylint: disable=protected-access
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator, StatePrepBase
from pennylane.queuing import QueuingManager

from .qscript import QuantumScript
from .tape import _validate_computational_basis_sampling, rotations_and_diagonal_measurements


def expand_tape(tape, depth=1, stop_at=None, expand_measurements=False):
    """Expand all objects in a tape to a specific depth.

    Args:
        tape (QuantumTape): The tape to expand
        depth (int): the depth the tape should be expanded
        stop_at (Callable): A function which accepts a queue object,
            and returns ``True`` if this object should *not* be expanded.
            If not provided, all objects that support expansion will be expanded.
        expand_measurements (bool): If ``True``, measurements will be expanded
            to basis rotations and computational basis measurements.

    Returns:
        QuantumTape: The expanded version of ``tape``.

    .. seealso:: :func:`~.pennylane.devices.preprocess.decompose` for a transform that
        performs the same job and fits into the current transform architecture.

    .. warning::

        This method cannot be used with a tape with non-commuting measurements, even if
        ``expand_measurements=False``.

        >>> from pennylane.tape import expand_tape
        >>> mps = [qml.expval(qml.X(0)), qml.expval(qml.Y(0))]
        >>> tape = qml.tape.QuantumScript([], mps)
        >>> expand_tape(tape)
        Traceback (most recent call last):
            ...
        pennylane.exceptions.QuantumFunctionError: Only observables that are qubit-wise commuting Pauli words can be returned on the same wire, some of the following measurements do not commute:
        [expval(X(0)), expval(Y(0))]

        Since commutation is determined by pauli word arithmetic, non-pauli words cannot share
        wires with other measurements, even if they commute:

        >>> measurements = [qml.expval(qml.Projector([0], 0)), qml.probs(wires=0)]
        >>> tape = qml.tape.QuantumScript([], measurements)
        >>> expand_tape(tape)
        Traceback (most recent call last):
            ...
        pennylane.exceptions.QuantumFunctionError: Only observables that are qubit-wise commuting Pauli words can be returned on the same wire, some of the following measurements do not commute:
        [expval(Projector(array([0]), wires=[0])), probs(wires=[0])]

        For this reason, we recommend the use of :func:`~.pennylane.devices.preprocess.decompose` instead.

    .. details::
        :title: Usage Details

        >>> from pennylane.tape import expand_tape
        >>> ops = [qml.Permute((2,1,0), wires=(0,1,2)), qml.X(0)]
        >>> measurements = [qml.expval(qml.X(0))]
        >>> tape = qml.tape.QuantumScript(ops, measurements)
        >>> expanded_tape = expand_tape(tape)
        >>> print(expanded_tape.draw())
        0: ─╭SWAP──RX─╭GlobalPhase─┤  <X>
        2: ─╰SWAP─────╰GlobalPhase─┤

        Specifying a depth greater than one decomposes operations multiple times.

        >>> expanded_tape2 = expand_tape(tape, depth=2)
        >>> print(expanded_tape2.draw())
        0: ─╭●─╭X─╭●──RX─┤  <X>
        2: ─╰X─╰●─╰X─────┤

        The ``stop_at`` callable allows the specification of terminal
        operations that should no longer be decomposed. In this example, the ``X``
        operator is not decomposed because ``stop_at(qml.X(0)) == True``.

        >>> def stop_at(obj):
        ...     return isinstance(obj, qml.X)
        >>> expanded_tape = expand_tape(tape, stop_at=stop_at)
        >>> print(expanded_tape.draw())
        0: ─╭SWAP──X─┤  <X>
        2: ─╰SWAP────┤

        .. warning::

            If an operator does not have a decomposition, it will not be decomposed, even if
            ``stop_at(obj) == False``.  If you want to decompose to reach a certain gateset,
            you will need an extra validation pass to ensure you have reached the gateset.

            >>> def stop_at(obj):
            ...     return getattr(obj, "name", "") in {"RX", "RY"}
            >>> tape = qml.tape.QuantumScript([qml.RZ(0.1, 0)])
            >>> expand_tape(tape, stop_at=stop_at).circuit
            [RZ(0.1, wires=[0])]

        If more than one observable exists on a wire, the diagonalizing gates will be applied
        and the observable will be substituted for an analogous combination of ``qml.Z`` operators.
        This will happen even if ``expand_measurements=False``.

        >>> mps = [qml.expval(qml.X(0)), qml.expval(qml.X(0) @ qml.X(1))]
        >>> tape = qml.tape.QuantumScript([], mps)
        >>> expanded_tape = expand_tape(tape)
        >>> print(expanded_tape.draw())
        0: ──RY─┤  <Z> ╭<Z@Z>
        1: ──RY─┤      ╰<Z@Z>

        Setting ``expand_measurements=True`` applies any diagonalizing gates and converts
        the measurement into a wires+eigvals representation.

        .. warning::
            Many components of PennyLane do not support the wires + eigvals representation.
            Setting ``expand_measurements=True`` should be used with extreme caution.

        >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0))])
        >>> expand_tape(tape, expand_measurements=True).circuit
        [H(0), expval(eigvals=[ 1. -1.], wires=[0])]


    """
    if depth == 0:
        return tape

    if stop_at is None:
        # by default expand all objects
        def stop_at(obj):  # pylint: disable=unused-argument
            return False

    new_ops = []
    new_measurements = []

    # Check for observables acting on the same wire. If present, observables must be
    # qubit-wise commuting Pauli words. In this case, the tape is expanded with joint
    # rotations and the observables updated to the computational basis. Note that this
    # expansion acts on the original tape in place.
    if tape.samples_computational_basis and len(tape.measurements) > 1:
        _validate_computational_basis_sampling(tape)

    diagonalizing_gates, diagonal_measurements = rotations_and_diagonal_measurements(tape)
    for queue, new_queue in [
        (tape.operations + diagonalizing_gates, new_ops),
        (diagonal_measurements, new_measurements),
    ]:
        for obj in queue:
            stop_at_meas = not expand_measurements and isinstance(obj, MeasurementProcess)

            if stop_at_meas or stop_at(obj):
                # do not expand out the object; append it to the
                # new tape, and continue to the next object in the queue
                new_queue.append(obj)
                continue

            if isinstance(obj, Operator):
                if obj.has_decomposition:
                    with QueuingManager.stop_recording():
                        obj = QuantumScript(obj.decomposition())
                else:
                    new_queue.append(obj)
                    continue
            elif isinstance(obj, qml.measurements.MeasurementProcess):
                if obj.obs is not None and obj.obs.has_diagonalizing_gates:
                    new_mp = type(obj)(eigvals=obj.obs.eigvals(), wires=obj.obs.wires)
                    obj = QuantumScript(obj.obs.diagonalizing_gates(), [new_mp])
                else:
                    new_queue.append(obj)
                    continue
            # recursively expand out the newly created tape
            expanded_tape = expand_tape(obj, stop_at=stop_at, depth=depth - 1)

            new_ops.extend(expanded_tape.operations)
            new_measurements.extend(expanded_tape.measurements)

    # preserves inheritance structure
    # if tape is a QuantumTape, returned object will be a quantum tape
    new_tape = tape.__class__(new_ops, new_measurements, shots=tape.shots)

    # Update circuit info
    new_tape._batch_size = tape._batch_size
    return new_tape


def expand_tape_state_prep(tape, skip_first=True):
    """Expand all instances of StatePrepBase operations in the tape.

    Args:
        tape (QuantumScript): The tape to expand.
        skip_first (bool): If ``True``, will not expand a ``StatePrepBase`` operation if
            it is the first operation in the tape.

    Returns:
        QuantumTape: The expanded version of ``tape``.

    **Example**

    If a ``StatePrepBase`` occurs as the first operation of a tape, the operation will not be expanded:

    >>> ops = [qml.StatePrep([0, 1], wires=0), qml.Z(1), qml.StatePrep([1, 0], wires=0)]
    >>> tape = qml.tape.QuantumScript(ops, [])
    >>> new_tape = qml.tape.expand_tape_state_prep(tape)
    >>> new_tape.operations
    [StatePrep(array([0, 1]), wires=[0]), Z(1), MottonenStatePreparation(array([1, 0]), wires=[0])]

    To force expansion, the keyword argument ``skip_first`` can be set to ``False``:

    >>> new_tape = qml.tape.expand_tape_state_prep(tape, skip_first=False)
    >>> new_tape.operations
    [MottonenStatePreparation(array([0, 1]), wires=[0]), Z(1), MottonenStatePreparation(array([1, 0]), wires=[0])]
    """
    first_op = tape.operations[0]
    new_ops = (
        [first_op]
        if not isinstance(first_op, StatePrepBase) or skip_first
        else first_op.decomposition()
    )

    for op in tape.operations[1:]:
        if isinstance(op, StatePrepBase):
            new_ops.extend(op.decomposition())
        else:
            new_ops.append(op)

    # preserves inheritance structure
    # if tape is a QuantumTape, returned object will be a quantum tape
    new_tape = tape.__class__(new_ops, tape.measurements, shots=tape.shots)

    # Update circuit info
    new_tape._batch_size = tape._batch_size
    return new_tape
