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
import contextlib
import copy
from warnings import warn

# pylint: disable=too-many-instance-attributes,protected-access,too-many-branches,too-many-public-methods, too-many-arguments
from threading import RLock

import pennylane as qml
from pennylane.measurements import Counts, Sample, AllCounts, Probability
from pennylane.operation import DecompositionUndefinedError, Operator
from pennylane.queuing import AnnotatedQueue, QueuingManager
from .qscript import QuantumScript


class TapeError(ValueError):
    """An error raised with a quantum tape."""


def get_active_tape():
    """Returns the currently recording tape.
    If no tape is currently recording, ``None`` is returned.

    **Example**

    >>> with qml.tape.QuantumTape():
    ...     qml.RX(0.2, wires="a")
    ...     tape = qml.tape.get_active_tape()
    ...     qml.RY(0.1, wires="b")
    >>> print(tape)
    <QuantumTape: wires=['a', 'b'], params=2>
    >>> print(qml.tape.get_active_tape())
    None
    """
    message = (
        "qml.tape.get_active_tape is now deprecated."
        " Please use qml.QueuingManager.active_context"
    )
    warn(message, UserWarning)
    return QueuingManager.active_context()


def _err_msg_for_some_meas_not_qwc(measurements):
    """Error message for the case when some operators measured on the same wire are not qubit-wise commuting."""
    msg = (
        "Only observables that are qubit-wise commuting "
        "Pauli words can be returned on the same wire, "
        f"some of the following measurements do not commute:\n{measurements}"
    )
    return msg


def _validate_computational_basis_sampling(measurements):
    """Auxiliary function for validating computational basis state sampling with other measurements considering the
    qubit-wise commutativity relation."""
    wires = qml.wires.Wires.all_wires([m.wires for m in measurements])
    with QueuingManager.stop_recording():  # stop recording operations - the constructed operator is just aux
        all_wire_pauliz = (
            qml.PauliZ(wires) if len(wires) == 1 else qml.prod(*[qml.PauliZ(w) for w in wires])
        )

    all_obs_minus_comp_basis_sampling = [
        o for o in measurements if not o.samples_computational_basis
    ]

    should_raise = False
    for obs in all_obs_minus_comp_basis_sampling:
        if obs.obs is not None:
            should_raise = qml.grouping.utils.are_pauli_words_qwc([obs.obs, all_wire_pauliz])

        if should_raise:
            _err_msg_for_some_meas_not_qwc(measurements)
            return


# TODO: move this function to its own file and rename
def expand_tape(qscript, depth=1, stop_at=None, expand_measurements=False):
    """Expand all objects in a tape to a specific depth.

    Args:
        qscript (QuantumScript): The Quantum Script to expand
        depth (int): the depth the tape should be expanded
        stop_at (Callable): A function which accepts a queue object,
            and returns ``True`` if this object should *not* be expanded.
            If not provided, all objects that support expansion will be expanded.
        expand_measurements (bool): If ``True``, measurements will be expanded
            to basis rotations and computational basis measurements.

    Returns:
        QuantumScript: The expanded version of ``qscript``.

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
        return qscript

    if stop_at is None:
        # by default expand all objects
        def stop_at(obj):  # pylint: disable=unused-argument
            return False

    new_prep = []
    new_ops = []
    new_measurements = []

    # Check for observables acting on the same wire. If present, observables must be
    # qubit-wise commuting Pauli words. In this case, the tape is expanded with joint
    # rotations and the observables updated to the computational basis. Note that this
    # expansion acts on the original qscript in place.
    need_to_validate_comp_basis_sampling = (
        qscript.samples_computational_basis and len(qscript.measurements) > 1
    )
    if need_to_validate_comp_basis_sampling:
        # TODO: edge case: multiple obs=None measurement
        _validate_computational_basis_sampling(qscript.measurements)

    if qscript._obs_sharing_wires:
        with QueuingManager.stop_recording():  # stop recording operations to active context when computing qwc groupings
            try:
                rotations, diag_obs = qml.grouping.diagonalize_qwc_pauli_words(
                    qscript._obs_sharing_wires
                )
            except (TypeError, ValueError) as e:
                if any(
                    m.return_type in (Probability, Sample, Counts, AllCounts)
                    for m in qscript.measurements
                ):
                    raise qml.QuantumFunctionError(
                        "Only observables that are qubit-wise commuting "
                        "Pauli words can be returned on the same wire.\n"
                        "Try removing all probability, sample and counts measurements "
                        "this will allow for splitting of execution and separate measurements "
                        "for each non-commuting observable."
                    ) from e

                raise qml.QuantumFunctionError(
                    _err_msg_for_some_meas_not_qwc(qscript.measurements)
                ) from e

            qscript._ops.extend(rotations)

            for o, i in zip(diag_obs, qscript._obs_sharing_wires_id):
                new_m = qml.measurements.MeasurementProcess(
                    qscript.measurements[i].return_type, obs=o
                )
                qscript._measurements[i] = new_m

    for queue, new_queue in [
        (qscript._prep, new_prep),
        (qscript._ops, new_ops),
        (qscript._measurements, new_measurements),
    ]:
        for obj in queue:
            stop = stop_at(obj)

            if not expand_measurements:
                # Measurements should not be expanded; treat measurements
                # as a stopping condition
                stop = stop or isinstance(obj, qml.measurements.MeasurementProcess)

            if stop:
                # do not expand out the object; append it to the
                # new qscript, and continue to the next object in the queue
                new_queue.append(obj)
                continue

            if isinstance(obj, (Operator, qml.measurements.MeasurementProcess)):
                # Object is an operation; query it for its expansion
                try:
                    obj = obj.expand()
                except DecompositionUndefinedError:
                    # Object does not define an expansion; treat this as
                    # a stopping condition.
                    new_queue.append(obj)
                    continue

            # recursively expand out the newly created qscript
            expanded_qscript = expand_tape(obj, stop_at=stop_at, depth=depth - 1)

            new_prep.extend(expanded_qscript._prep)
            new_ops.extend(expanded_qscript._ops)
            new_measurements.extend(expanded_qscript._measurements)

    # preserves inheritance structure
    # if qscript is a QuantumTape, returned object will be a quantum tape
    new_qscript = qscript.__class__(new_ops, new_measurements, new_prep, _update=False)

    # Update circuit info
    new_qscript.wires = copy.copy(qscript.wires)
    new_qscript.num_wires = qscript.num_wires
    new_qscript.is_sampled = qscript.is_sampled
    new_qscript.all_sampled = qscript.all_sampled
    new_qscript.samples_computational_basis = qscript.samples_computational_basis
    new_qscript._batch_size = qscript.batch_size
    new_qscript._output_dim = qscript.output_dim
    new_qscript._qfunc_output = qscript._qfunc_output
    return new_qscript


# pylint: disable=too-many-public-methods
class QuantumTape(QuantumScript, AnnotatedQueue):
    """A quantum tape recorder, that records and stores variational quantum programs.

    Args:
        ops (Iterable[Operator]): An iterable of the operations to be performed
        measurements (Iterable[MeasurementProcess]): All the measurements to be performed
        prep (Iterable[Operator]): Any state preparations to perform at the start of the circuit

    Keyword Args:
        name (str): a name given to the quantum script
        do_queue=True (bool): Whether or not to queue. Defaults to ``True`` for ``QuantumTape``.
        _update=True (bool): Whether or not to set various properties on initialization. Setting
            ``_update=False`` reduces computations if the script is only an intermediary step.


    **Example**

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            qml.expval(qml.PauliZ(wires=[0]))

    Once constructed, the tape may act as a quantum circuit and information
    about the quantum circuit can be queried:

    >>> list(tape)
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a']), expval(PauliZ(wires=[0]))]
    >>> tape.operations
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
    >>> tape.observables
    [expval(PauliZ(wires=[0]))]
    >>> tape.get_parameters()
    [0.432, 0.543, 0.133]
    >>> tape.wires
    <Wires = [0, 'a']>
    >>> tape.num_params
    3

    Tapes can also be constructed by directly providing operations, measurements, and state preparations:

    >>> ops = [qml.S(0), qml.T(1)]
    >>> measurements = [qml.state()]
    >>> prep = [qml.BasisState([1,0], wires=0)]
    >>> tape = qml.tape.QuantumTape(ops, measurements, prep=prep)
    >>> tape.circuit
    [BasisState([1, 0], wires=[0]), S(wires=[0]), T(wires=[1]), state(wires=[])]

    The existing circuit is overriden upon exiting a recording context.

    Iterating over the quantum circuit can be done by iterating over the tape
    object:

    >>> for op in tape:
    ...     print(op)
    RX(0.432, wires=[0])
    RY(0.543, wires=[0])
    CNOT(wires=[0, 'a'])
    RX(0.133, wires=['a'])
    expval(PauliZ(wires=[0]))

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

    The trainable parameters of the tape can be explicitly set, and the values of
    the parameters modified in-place:

    >>> tape.trainable_params = [0] # set only the first parameter as trainable
    >>> tape.set_parameters([0.56])
    >>> tape.get_parameters()
    [0.56]
    >>> tape.get_parameters(trainable_only=False)
    [0.56, 0.543, 0.133]


    When using a tape with ``do_queue=False``, that tape will not be queued in a parent tape context.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape(do_queue=False) as tape2:
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
        self, ops=None, measurements=None, prep=None, name=None, do_queue=True, _update=True
    ):
        self.do_queue = do_queue
        AnnotatedQueue.__init__(self)
        QuantumScript.__init__(self, ops, measurements, prep, name=name, _update=_update)

    def __enter__(self):
        QuantumTape._lock.acquire()
        try:
            if self.do_queue:
                QueuingManager.append(self)
            return super().__enter__()
        except Exception as _:
            QuantumTape._lock.release()
            raise

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            AnnotatedQueue.__exit__(self, exception_type, exception_value, traceback)
            # After other optimizations in #2963, #2986 and follow-up work, we should check whether
            # calling `_process_queue` only if there is no `exception_type` saves time. This would
            # be done via the following:
            # if exception_type is None:
            #    self._process_queue()
            self._process_queue()
        finally:
            QuantumTape._lock.release()

    # pylint: disable=no-self-use
    @contextlib.contextmanager
    def stop_recording(self):
        """Context manager to temporarily stop recording operations
        onto the tape. This is useful is scratch space is needed.

        **Deprecated Method:** Please use ``qml.QueuingManager.stop_recording`` instead.

        **Example**

        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(0, wires=0)
        ...     with tape.stop_recording():
        ...         qml.RY(1.0, wires=1)
        ...     qml.RZ(2, wires=1)
        >>> tape.operations
        [RX(0, wires=[0]), RZ(2, wires=[1])]
        """
        warn(
            "QuantumTape.stop_recording has moved to qml.QueuingManager.stop_recording.",
            UserWarning,
        )
        with QueuingManager.stop_recording():
            yield

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
            _prep (list[~.Operation]): Preparation operations
            _ops (list[~.Operation]): Main tape operations
            _measurements (list[~.MeasurementProcess]): Tape measurements

        Also calls `_update()` which sets many attributes.
        """
        self._prep = []
        self._ops = []
        self._measurements = []
        list_order = {"_prep": 0, "_ops": 1, "_measurements": 2}
        current_list = "_prep"

        for obj, info in self._queue.items():

            if "owner" not in info and getattr(obj, "_queue_category", None) is not None:
                if list_order[obj._queue_category] > list_order[current_list]:
                    current_list = obj._queue_category
                elif list_order[obj._queue_category] < list_order[current_list]:
                    raise ValueError(
                        f"{obj._queue_category[1:]} operation {obj} must occur prior "
                        f"to {current_list[1:]}. Please place earlier in the queue."
                    )
                getattr(self, obj._queue_category).append(obj)

        self._update()

    def inv(self):
        """Inverts the processed operations.

        Inversion is performed in-place.

        .. note::

            This method only inverts the quantum operations/unitary recorded
            by the quantum tape; state preparations and measurements are left unchanged.

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.BasisState(np.array([1, 1]), wires=[0, 'a'])
                qml.RX(0.432, wires=0)
                qml.Rot(0.543, 0.1, 0.4, wires=0).inv()
                qml.CNOT(wires=[0, 'a'])
                qml.probs(wires=0), qml.probs(wires='a')

        This tape has the following properties:

        >>> tape.operations
        [BasisState(array([1, 1]), wires=[0, 'a']),
         RX(0.432, wires=[0]),
         Rot.inv(0.543, 0.1, 0.4, wires=[0]),
         CNOT(wires=[0, 'a'])]
        >>> tape.get_parameters()
        [array([1, 1]), 0.432, 0.543, 0.1, 0.4]

        Here, let's set some trainable parameters:

        >>> tape.trainable_params = [1, 2]
        >>> tape.get_parameters()
        [0.432, 0.543]

        Inverting the tape:

        >>> tape.inv()
        >>> tape.operations
        [BasisState(array([1, 1]), wires=[0, 'a']),
         CNOT.inv(wires=[0, 'a']),
         Rot(0.543, 0.1, 0.4, wires=[0]),
         RX.inv(0.432, wires=[0])]

        Tape inversion also modifies the order of tape parameters:

        >>> tape.get_parameters(trainable_only=False)
        [array([1, 1]), 0.543, 0.1, 0.4, 0.432]
        >>> tape.get_parameters(trainable_only=True)
        [0.543, 0.432]
        >>> tape.trainable_params
        [1, 4]
        """
        # we must remap the old parameter
        # indices to the new ones after the operation order is reversed.
        parameter_indices = []
        param_count = 0

        for queue in [self._prep, self._ops, self.observables]:
            # iterate through all queues

            obj_params = []

            for obj in queue:
                # index the number of parameters on each operation
                num_obj_params = len(obj.data)
                obj_params.append(list(range(param_count, param_count + num_obj_params)))

                # keep track of the total number of parameters encountered so far
                param_count += num_obj_params

            if queue == self._ops:
                # reverse the list representing operator parameters
                obj_params = obj_params[::-1]

            parameter_indices.extend(obj_params)

        # flatten the list of parameter indices after the reversal
        parameter_indices = [item for sublist in parameter_indices for item in sublist]
        parameter_mapping = dict(zip(parameter_indices, range(len(parameter_indices))))

        # map the params
        self.trainable_params = [parameter_mapping[i] for i in self.trainable_params]
        self._par_info = {parameter_mapping[k]: v for k, v in self._par_info.items()}

        for idx, op in enumerate(self._ops):
            self._ops[idx] = qml.adjoint(op, lazy=False)

        self._ops = list(reversed(self._ops))
