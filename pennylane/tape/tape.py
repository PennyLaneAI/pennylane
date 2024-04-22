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
This module contains the base quantum tape.
"""
# pylint: disable=too-many-instance-attributes,protected-access,too-many-branches,too-many-public-methods, too-many-arguments
import copy
from threading import RLock

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    MeasurementProcess,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
)
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue

from .qscript import QuantumScript


class TapeError(ValueError):
    """An error raised with a quantum tape."""


def _err_msg_for_some_meas_not_qwc(measurements):
    """Error message for the case when some operators measured on the same wire are not qubit-wise commuting."""
    return (
        "Only observables that are qubit-wise commuting "
        "Pauli words can be returned on the same wire, "
        f"some of the following measurements do not commute:\n{measurements}"
    )


def _validate_computational_basis_sampling(tape):
    """Auxiliary function for validating computational basis state sampling with other measurements considering the
    qubit-wise commutativity relation."""
    measurements = tape.measurements
    n_meas = len(measurements)
    n_mcms = sum(isinstance(op, MidMeasureMP) for op in tape.operations)
    non_comp_basis_sampling_obs = []
    comp_basis_sampling_obs = []
    comp_basis_indices = []
    for i, o in enumerate(measurements):
        if o.samples_computational_basis:
            comp_basis_sampling_obs.append(o)
            comp_basis_indices.append(i)
        else:
            non_comp_basis_sampling_obs.append(o)

    if non_comp_basis_sampling_obs:
        all_wires = []
        empty_wires = qml.wires.Wires([])
        for idx, (cb_obs, global_idx) in enumerate(
            zip(comp_basis_sampling_obs, comp_basis_indices)
        ):
            if cb_obs.wires == empty_wires:
                all_wires = qml.wires.Wires.all_wires([m.wires for m in measurements])
                break
            if global_idx < n_meas - n_mcms:
                all_wires.append(cb_obs.wires)
            if idx == len(comp_basis_sampling_obs) - 1:
                all_wires = qml.wires.Wires.all_wires(all_wires)

        # This happens when a MeasurementRegisterMP is the only computational basis state measurement
        if all_wires == empty_wires:
            return

        with QueuingManager.stop_recording():  # stop recording operations - the constructed operator is just aux
            pauliz_for_cb_obs = (
                qml.Z(all_wires)
                if len(all_wires) == 1
                else qml.operation.Tensor(*[qml.Z(w) for w in all_wires])
            )

        for obs in non_comp_basis_sampling_obs:
            # Cover e.g., qml.probs(wires=wires) case by checking obs attr
            if obs.obs is not None and not qml.pauli.utils.are_pauli_words_qwc(
                [obs.obs, pauliz_for_cb_obs]
            ):
                raise qml.QuantumFunctionError(_err_msg_for_some_meas_not_qwc(measurements))


def rotations_and_diagonal_measurements(tape):
    """Compute the rotations for overlapping observables, and return them along with the diagonalized observables."""
    if not tape._obs_sharing_wires:
        return [], tape.measurements

    with QueuingManager.stop_recording():  # stop recording operations to active context when computing qwc groupings
        try:
            rotations, diag_obs = qml.pauli.diagonalize_qwc_pauli_words(tape._obs_sharing_wires)
        except (TypeError, ValueError) as e:
            if any(isinstance(m, (ProbabilityMP, SampleMP, CountsMP)) for m in tape.measurements):
                raise qml.QuantumFunctionError(
                    "Only observables that are qubit-wise commuting "
                    "Pauli words can be returned on the same wire.\n"
                    "Try removing all probability, sample and counts measurements "
                    "this will allow for splitting of execution and separate measurements "
                    "for each non-commuting observable."
                ) from e

            raise qml.QuantumFunctionError(_err_msg_for_some_meas_not_qwc(tape.measurements)) from e

        measurements = copy.copy(tape.measurements)

        for o, i in zip(diag_obs, tape._obs_sharing_wires_id):
            new_m = tape.measurements[i].__class__(obs=o)
            measurements[i] = new_m

    return rotations, measurements


# TODO: move this function to its own file and rename
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

    Raises:
        QuantumFunctionError: if some observables in the tape are not qubit-wise commuting

    **Example**

    Consider the following nested tape:

    .. code-block:: python

        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 1]), wires=[0, 'a'])

            with QuantumTape() as tape2:
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, 'a'])
            qml.RY(0.2, wires='a')
            qml.probs(wires=0), qml.probs(wires='a')

    The nested structure is preserved:

    >>> tape.operations
    [BasisState(array([1, 1]), wires=[0, 'a']),
     <QuantumTape: wires=[0], params=3>,
     CNOT(wires=[0, 'a']),
     RY(0.2, wires=['a'])]

    Calling ``expand_tape`` will return a tape with all nested tapes
    expanded, resulting in a single tape of quantum operations:

    >>> new_tape = qml.tape.tape.expand_tape(tape)
    >>> new_tape.operations
    [BasisStatePreparation([1, 1], wires=[0, 'a']),
    Rot(0.543, 0.1, 0.4, wires=[0]),
    CNOT(wires=[0, 'a']),
    RY(0.2, wires=['a'])]
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
                        obj = QuantumScript(obj.decomposition(), _update=False)
                else:
                    new_queue.append(obj)
                    continue
            elif isinstance(obj, qml.measurements.MeasurementProcess):
                # Object is an operation; query it for its expansion
                try:
                    obj = obj.expand()
                except DecompositionUndefinedError:
                    # Object does not define an expansion; treat this as
                    # a stopping condition.
                    new_queue.append(obj)
                    continue

            # recursively expand out the newly created tape
            expanded_tape = expand_tape(obj, stop_at=stop_at, depth=depth - 1)

            new_ops.extend(expanded_tape.operations)
            new_measurements.extend(expanded_tape.measurements)

    # preserves inheritance structure
    # if tape is a QuantumTape, returned object will be a quantum tape
    new_tape = tape.__class__(new_ops, new_measurements, shots=tape.shots, _update=False)

    # Update circuit info
    new_tape.wires = copy.copy(tape.wires)
    new_tape.num_wires = tape.num_wires
    new_tape._batch_size = tape._batch_size
    new_tape._output_dim = tape._output_dim
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
    >>> new_tape = qml.tape.tape.expand_tape_state_prep(tape)
    >>> new_tape.operations
    [StatePrep(array([0, 1]), wires=[0]), Z(1), MottonenStatePreparation(array([1, 0]), wires=[0])]

    To force expansion, the keyword argument ``skip_first`` can be set to ``False``:

    >>> new_tape = qml.tape.tape.expand_tape_state_prep(tape, skip_first=False)
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
    new_tape = tape.__class__(new_ops, tape.measurements, shots=tape.shots, _update=False)

    # Update circuit info
    new_tape.wires = copy.copy(tape.wires)
    new_tape.num_wires = tape.num_wires
    new_tape._batch_size = tape._batch_size
    new_tape._output_dim = tape._output_dim
    return new_tape


# pylint: disable=too-many-public-methods
class QuantumTape(QuantumScript, AnnotatedQueue):
    """A quantum tape recorder, that records and stores variational quantum programs.

    Args:
        ops (Iterable[Operator]): An iterable of the operations to be performed
        measurements (Iterable[MeasurementProcess]): All the measurements to be performed
        prep (Iterable[Operator]): Arguments to specify state preparations to
            perform at the start of the circuit. These should go at the beginning of ``ops``
            instead.

    Keyword Args:
        shots (None, int, Sequence[int], ~.Shots): Number and/or batches of shots for execution.
            Note that this property is still experimental and under development.
        trainable_params (None, Sequence[int]): the indices for which parameters are trainable
        _update=True (bool): Whether or not to set various properties on initialization. Setting
            ``_update=False`` reduces computations if the tape is only an intermediary step.


    **Example**

    Tapes can be constructed by directly providing operations and measurements:

    >>> ops = [qml.BasisState([1,0], wires=0), qml.S(0), qml.T(1)]
    >>> measurements = [qml.state()]
    >>> tape = qml.tape.QuantumTape(ops, measurements)
    >>> tape.circuit
    [BasisState([1, 0], wires=[0]), S(wires=[0]), T(wires=[1]), state(wires=[])]

    They can also be populated into a recording tape via queuing.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            qml.expval(qml.Z(0))

    A ``QuantumTape`` can also be constructed directly from an :class:`~.AnnotatedQueue`:

    .. code-block:: python

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            qml.expval(qml.Z(0))

        tape = qml.tape.QuantumTape.from_queue(q)

    Once constructed, the tape may act as a quantum circuit and information
    about the quantum circuit can be queried:

    >>> list(tape)
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a']), expval(Z(0))]
    >>> tape.operations
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
    >>> tape.observables
    [expval(Z(0))]
    >>> tape.get_parameters()
    [0.432, 0.543, 0.133]
    >>> tape.wires
    <Wires = [0, 'a']>
    >>> tape.num_params
    3

    The existing circuit is overriden upon exiting a recording context.

    Iterating over the quantum circuit can be done by iterating over the tape
    object:

    >>> for op in tape:
    ...     print(op)
    RX(0.432, wires=[0])
    RY(0.543, wires=[0])
    CNOT(wires=[0, 'a'])
    RX(0.133, wires=['a'])
    expval(Z(0))

    Tapes can also as sequences and support indexing and the ``len`` function:

    >>> tape[0]
    RX(0.432, wires=[0])
    >>> len(tape)
    5

    The :class:`~.CircuitGraph` can also be accessed:

    >>> tape.graph
    <pennylane.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

    Once constructed, the quantum tape can be executed directly on a supported
    device via the :func:`~.pennylane.execute` function:

    >>> dev = qml.device("default.qubit", wires=[0, 'a'])
    >>> qml.execute([tape], dev, gradient_fn=None)
    [array([0.77750694])]

    A new tape can be created by passing new parameters along with the indices
    to be updated to :meth:`~pennylane.tape.QuantumScript.bind_new_parameters`:

    >>> new_tape = tape.bind_new_parameters(params=[0.56], indices=[0])
    >>> tape.get_parameters()
    [0.432, 0.543, 0.133]
    >>> new_tape.get_parameters()
    [0.56, 0.543, 0.133]


    To prevent the tape from being queued use :meth:`~.queuing.QueuingManager.stop_recording`.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape1:
            with qml.QueuingManager.stop_recording():
                with qml.tape.QuantumTape() as tape2:
                    qml.RX(0.123, wires=0)

    Here, tape2 records the RX gate, but tape1 doesn't record tape2.

    >>> tape1.operations
    []
    >>> tape2.operations
    [RX(0.123, wires=[0])]

    This is useful for when you want to transform a tape first before applying it.
    """

    _lock = RLock()
    """threading.RLock: Used to synchronize appending to/popping from global QueueingContext."""

    def __init__(
        self,
        ops=None,
        measurements=None,
        shots=None,
        trainable_params=None,
        _update=True,
    ):  # pylint: disable=too-many-arguments
        AnnotatedQueue.__init__(self)
        QuantumScript.__init__(
            self, ops, measurements, shots, trainable_params=trainable_params, _update=_update
        )

    def __enter__(self):
        QuantumTape._lock.acquire()
        QueuingManager.append(self)
        QueuingManager.add_active_queue(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        QueuingManager.remove_active_queue()
        QuantumTape._lock.release()
        self._process_queue()
        self._trainable_params = None

    def adjoint(self):
        adjoint_tape = super().adjoint()
        QueuingManager.append(adjoint_tape)
        return adjoint_tape

    # ========================================================
    # construction methods
    # ========================================================

    # This is a temporary attribute to fix the operator queuing behaviour.
    # Tapes may be nested and therefore processed into the `_ops` list.
    _queue_category = "_ops"

    def _process_queue(self):
        """Process the annotated queue, creating a list of quantum
        operations and measurement processes.

        Sets:
            _ops (list[~.Operation]): Main tape operations
            _measurements (list[~.MeasurementProcess]): Tape measurements

        Also calls `_update()` which sets many attributes.
        """
        self._ops, self._measurements = process_queue(self)
        self._update()

    def __getitem__(self, key):
        """
        Overrides the default because QuantumTape is both a QuantumScript and an AnnotatedQueue.
        If key is an int, the caller is likely indexing the backing QuantumScript. Otherwise, the
        caller is likely indexing the backing AnnotatedQueue.
        """
        if isinstance(key, int):
            return QuantumScript.__getitem__(self, key)
        return AnnotatedQueue.__getitem__(self, key)

    def __setitem__(self, key, val):
        AnnotatedQueue.__setitem__(self, key, val)

    def __hash__(self):
        return QuantumScript.__hash__(self)


register_pytree(QuantumTape, QuantumTape._flatten, QuantumTape._unflatten)
