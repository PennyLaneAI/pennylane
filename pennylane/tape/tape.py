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

# pylint: disable=too-many-instance-attributes,protected-access,too-many-branches,too-many-public-methods
from collections import Counter, defaultdict, deque
from threading import RLock
from typing import List

import pennylane as qml
from pennylane.measurements import Counts, Sample, Shadow
from pennylane.operation import DecompositionUndefinedError, Operation
from pennylane.queuing import AnnotatedQueue, QueuingContext, QueuingError

from .unwrap import UnwrapTape

OPENQASM_GATES = {
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "S.inv": "sdg",
    "T": "t",
    "T.inv": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
}
"""
dict[str, str]: Maps PennyLane gate names to equivalent QASM gate names.

Note that QASM has two native gates:

- ``U`` (equivalent to :class:`~.U3`)
- ``CX`` (equivalent to :class:`~.CNOT`)

All other gates are defined in the file stdgates.inc:
https://github.com/Qiskit/openqasm/blob/master/examples/stdgates.inc
"""


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
    return QueuingContext.active_context()


def expand_tape(tape, depth=1, stop_at=None, expand_measurements=False):
    """Expand all objects in a tape to a specific depth.

    Args:
        depth (int): the depth the tape should be expanded
        stop_at (Callable): A function which accepts a queue object,
            and returns ``True`` if this object should *not* be expanded.
            If not provided, all objects that support expansion will be expanded.
        expand_measurements (bool): If ``True``, measurements will be expanded
            to basis rotations and computational basis measurements.

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

    new_tape = QuantumTape()

    # Check for observables acting on the same wire. If present, observables must be
    # qubit-wise commuting Pauli words. In this case, the tape is expanded with joint
    # rotations and the observables updated to the computational basis. Note that this
    # expansion acts on the original tape in place.
    if tape._obs_sharing_wires:
        with qml.tape.stop_recording():  # stop recording operations to active context when computing qwc groupings
            try:
                rotations, diag_obs = qml.grouping.diagonalize_qwc_pauli_words(
                    tape._obs_sharing_wires
                )
            except (TypeError, ValueError) as e:
                raise qml.QuantumFunctionError(
                    "Only observables that are qubit-wise commuting "
                    "Pauli words can be returned on the same wire"
                ) from e

            tape._ops.extend(rotations)

            for o, i in zip(diag_obs, tape._obs_sharing_wires_id):
                new_m = qml.measurements.MeasurementProcess(tape.measurements[i].return_type, obs=o)
                tape._measurements[i] = new_m

    for queue in ("_prep", "_ops", "_measurements"):
        for obj in getattr(tape, queue):
            stop = stop_at(obj)

            if not expand_measurements:
                # Measurements should not be expanded; treat measurements
                # as a stopping condition
                stop = stop or isinstance(obj, qml.measurements.MeasurementProcess)

            if stop:
                # do not expand out the object; append it to the
                # new tape, and continue to the next object in the queue
                getattr(new_tape, queue).append(obj)
                continue

            if isinstance(obj, (qml.operation.Operator, qml.measurements.MeasurementProcess)):
                # Object is an operation; query it for its expansion
                try:
                    obj = obj.expand()
                except DecompositionUndefinedError:
                    # Object does not define an expansion; treat this as
                    # a stopping condition.
                    getattr(new_tape, queue).append(obj)
                    continue

            # recursively expand out the newly created tape
            expanded_tape = expand_tape(obj, stop_at=stop_at, depth=depth - 1)

            new_tape._prep += expanded_tape._prep
            new_tape._ops += expanded_tape._ops
            new_tape._measurements += expanded_tape._measurements

    # Update circuit info
    new_tape.wires = copy.copy(tape.wires)
    new_tape.num_wires = tape.num_wires
    new_tape.is_sampled = tape.is_sampled
    new_tape.all_sampled = tape.all_sampled
    new_tape._batch_size = tape.batch_size
    new_tape._output_dim = tape.output_dim
    new_tape._qfunc_output = tape._qfunc_output
    return new_tape


# pylint: disable=too-many-public-methods
class QuantumTape(AnnotatedQueue):
    """A quantum tape recorder, that records, validates and executes variational quantum programs.

    Args:
        name (str): a name given to the quantum tape
        do_queue (bool): Whether to queue this tape in a parent tape context.

    **Example**

    .. code-block:: python

        import pennylane.tape

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

    def __init__(self, name=None, do_queue=True):
        super().__init__()
        self.name = name
        self.do_queue = do_queue
        self._prep = []
        """list[.Operation]: Tape state preparations."""

        self._ops = []
        """list[.Operation]: quantum operations recorded by the tape."""

        self._measurements = []
        """list[.MeasurementProcess]: measurement processes recorded by the tape."""

        self._par_info = {}
        """dict[int, dict[str, Operation or int]]: Parameter information. Keys are
        parameter indices (in the order they appear on the tape), and values are a
        dictionary containing the corresponding operation and operation parameter index."""

        self._trainable_params = []
        self._graph = None
        self._specs = None
        self._depth = None
        self._output_dim = 0
        self._batch_size = None
        self._qfunc_output = None

        self.wires = qml.wires.Wires([])
        self.num_wires = 0

        self.is_sampled = False
        self.all_sampled = False
        self.inverse = False

        self._obs_sharing_wires = []
        """list[.Observable]: subset of the observables that share wires with another observable,
        i.e., that do not have their own unique set of wires."""
        self._obs_sharing_wires_id = []

    def __repr__(self):
        return f"<{self.__class__.__name__}: wires={self.wires.tolist()}, params={self.num_params}>"

    def __enter__(self):
        QuantumTape._lock.acquire()
        try:
            if self.do_queue:
                QueuingContext.append(self)
            return super().__enter__()
        except Exception as _:
            QuantumTape._lock.release()
            raise

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            super().__exit__(exception_type, exception_value, traceback)
            self._process_queue()
        finally:
            QuantumTape._lock.release()

    @property
    def circuit(self):
        """Returns the quantum circuit recorded by the tape.

        The circuit is created with the assumptions that:

        * The ``operations`` attribute contains quantum operations and
          mid-circuit measurements and
        * The ``measurements`` attribute contains terminal measurements.

        Note that the resulting list could contain MeasurementProcess objects
        that some devices may not support.

        Returns:

            list[.Operator, .MeasurementProcess]: the quantum circuit
            containing quantum operations and measurements as recorded by the
            tape.
        """
        return self.operations + self.measurements

    def __iter__(self):
        """list[.Operator, .MeasurementProcess]: Return an iterator to the
        underlying quantum circuit object."""
        return iter(self.circuit)

    def __getitem__(self, idx):
        """list[.Operator]: Return the indexed operator from underlying quantum
        circuit object."""
        return self.circuit[idx]

    def __len__(self):
        """int: Return the number of operations and measurements in the
        underlying quantum circuit object."""
        return len(self.circuit)

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the quantum tape (if any)"""
        return None

    @contextlib.contextmanager
    def stop_recording(self):
        """Context manager to temporarily stop recording operations
        onto the tape. This is useful is scratch space is needed.

        **Example**

        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(0, wires=0)
        ...     with tape.stop_recording():
        ...         qml.RY(1.0, wires=1)
        ...     qml.RZ(2, wires=1)
        >>> tape.operations
        [RX(0, wires=[0]), RZ(2, wires=[1])]
        """
        if QueuingContext.active_context() is not self:
            raise QueuingError(
                "Cannot stop recording requested tape as it is not currently recording."
            )

        active_contexts = QueuingContext._active_contexts
        QueuingContext._active_contexts = deque()
        yield
        QueuingContext._active_contexts = active_contexts

    # ========================================================
    # construction methods
    # ========================================================

    # This is a temporary attribute to fix the operator queuing behaviour.
    # Tapes may be nested and therefore processed into the `_ops` list.
    _queue_category = "_ops"

    def _process_queue(self):
        """Process the annotated queue, creating a list of quantum
        operations and measurement processes.
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

    def _update_circuit_info(self):
        """Update circuit metadata

        Sets:
            wires (~.Wires): Wires
            num_wires (int): Number of wires
            is_sampled (bool): Whether any measurement is of type ``Sample`` or ``Counts``
            all_sampled (bool): Whether all measurements are of type ``Sample`` or ``Counts``
        """
        self.wires = qml.wires.Wires.all_wires(
            dict.fromkeys(op.wires for op in self.operations + self.observables)
        )
        self.num_wires = len(self.wires)

        is_sample_type = [m.return_type in (Sample, Counts, Shadow) for m in self.measurements]
        self.is_sampled = any(is_sample_type)
        self.all_sampled = all(is_sample_type)

    def _update_batch_size(self):
        """Infer the batch_size of the tape from the batch sizes of its operations
        and check the latter for consistency.

        Sets:
            _batch_size (int): The common batch size of the tape operations, if any has one
        """
        candidate = None
        for op in self.operations:
            op_batch_size = getattr(op, "batch_size", None)
            if op_batch_size is None:
                continue
            if candidate:
                if op_batch_size != candidate:
                    raise ValueError(
                        "The batch sizes of the tape operations do not match, they include "
                        f"{candidate} and {op_batch_size}."
                    )
            else:
                candidate = op_batch_size

        self._batch_size = candidate

    def _update_output_dim(self):
        """Update the dimension of the output of the tape.

        Sets:
            self._output_dim (int): Size of the tape output (when flattened)
        """
        self._output_dim = 0
        for m in self.measurements:
            # attempt to infer the output dimension
            if m.return_type is qml.measurements.Probability:
                # TODO: what if we had a CV device here? Having the base as
                # 2 would have to be swapped to the cutoff value
                self._output_dim += 2 ** len(m.wires)
            elif m.return_type is not qml.measurements.State:
                self._output_dim += 1
        if self.batch_size:
            self._output_dim *= self.batch_size

    def _update_observables(self):
        """Update information about observables, including the wires that are acted upon and
        identifying any observables that share wires.

        Sets:
            _obs_sharing_wires (list[~.Observable]): Observables that share wires with
                any other observable
            _obs_sharing_wires_id (list[int]): Indices of the measurements that contain
                the observables in _obs_sharing_wires
        """
        obs_wires = [wire for m in self.measurements for wire in m.wires if m.obs is not None]
        self._obs_sharing_wires = []
        self._obs_sharing_wires_id = []

        if len(obs_wires) != len(set(obs_wires)):
            c = Counter(obs_wires)
            repeated_wires = {w for w in obs_wires if c[w] > 1}

            for i, m in enumerate(self.measurements):
                if m.obs is not None and len(set(m.wires) & repeated_wires) > 0:
                    self._obs_sharing_wires.append(m.obs)
                    self._obs_sharing_wires_id.append(i)

    def _update_par_info(self):
        """Update the parameter information dictionary.

        Sets:
            _par_info (dict): Parameter information dictionary
        """
        param_count = 0

        for obj in self.operations + self.observables:

            for p in range(len(obj.data)):
                info = self._par_info.get(param_count, {})
                info.update({"op": obj, "p_idx": p})

                self._par_info[param_count] = info
                param_count += 1

    def _update_trainable_params(self):
        """Set the trainable parameters

        Sets:
            _trainable_params (list[int]): Tape parameter indices of trainable parameters

        self._par_info.keys() is assumed to be sorted
        As its order is maintained, this assumes that self._par_info
        is created in a sorted manner, as in _update_par_info
        """
        self._trainable_params = list(self._par_info)

    def _update(self):
        """Update all internal tape metadata regarding processed operations and observables"""
        self._graph = None
        self._specs = None
        self._depth = None
        self._update_circuit_info()  # Updates wires, num_wires, is_sampled, all_sampled; O(ops+obs)
        self._update_par_info()  # Updates the _par_info dictionary; O(ops+obs)
        self._update_trainable_params()  # Updates the _trainable_params; O(1)
        self._update_observables()  # Updates _obs_sharing_wires and _obs_sharing_wires_id
        self._update_batch_size()  # Updates _batch_size; O(ops)
        self._update_output_dim()  # Updates _output_dim; O(obs)

    def expand(self, depth=1, stop_at=None, expand_measurements=False):
        """Expand all operations in the processed queue to a specific depth.

        Args:
            depth (int): the depth the tape should be expanded
            stop_at (Callable): A function which accepts a queue object,
                and returns ``True`` if this object should *not* be expanded.
                If not provided, all objects that support expansion will be expanded.
            expand_measurements (bool): If ``True``, measurements will be expanded
                to basis rotations and computational basis measurements.

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

        Calling ``.expand`` will return a tape with all nested tapes
        expanded, resulting in a single tape of quantum operations:

        >>> new_tape = tape.expand(depth=2)
        >>> new_tape.operations
        [PauliX(wires=[0]),
        PauliX(wires=['a']),
        RZ(0.543, wires=[0]),
        RY(0.1, wires=[0]),
        RZ(0.4, wires=[0]),
        CNOT(wires=[0, 'a']),
        RY(0.2, wires=['a'])]
        """
        new_tape = expand_tape(
            self, depth=depth, stop_at=stop_at, expand_measurements=expand_measurements
        )
        new_tape._update()
        return new_tape

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
            try:
                self._ops[idx] = op.adjoint()
            except qml.operation.AdjointUndefinedError:
                op.inverse = not op.inverse

        self._ops = list(reversed(self._ops))

    def adjoint(self):
        """Create a tape that is the adjoint of this one.

        Adjointed tapes are the conjugated and transposed version of the
        original tapes. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        Returns:
            ~.QuantumTape: the adjointed tape
        """
        with qml.tape.stop_recording():
            new_tape = self.copy(copy_operations=True)
            new_tape.inv()

        # the current implementation of the adjoint
        # transform requires that the returned inverted object
        # is automatically queued.
        with QuantumTape._lock:
            QueuingContext.append(new_tape)

        return new_tape

    # ========================================================
    # Parameter handling
    # ========================================================

    @property
    def trainable_params(self):
        """Store or return a list containing the indices of parameters that support
        differentiability. The indices provided match the order of appearence in the
        quantum circuit.

        Setting this property can help reduce the number of quantum evaluations needed
        to compute the Jacobian; parameters not marked as trainable will be
        automatically excluded from the Jacobian computation.

        The number of trainable parameters determines the number of parameters passed to
        :meth:`~.set_parameters`, and changes the default output size of method :meth:`~.get_parameters()`.

        .. note::

            For devices that support native backpropagation (such as
            ``default.qubit.tf`` and ``default.qubit.autograd``), this
            property contains no relevant information when using
            backpropagation to compute gradients.

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        >>> tape.trainable_params
        [0, 1, 2]
        >>> tape.trainable_params = [0] # set only the first parameter as trainable
        >>> tape.get_parameters()
        [0.432]
        """
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, param_indices):
        """Store the indices of parameters that support differentiability.

        Args:
            param_indices (list[int]): parameter indices
        """
        if any(not isinstance(i, int) or i < 0 for i in param_indices):
            raise ValueError("Argument indices must be non-negative integers.")

        num_params = len(self._par_info)
        if any(i > num_params for i in param_indices):
            raise ValueError(f"Tape only has {num_params} parameters.")

        self._trainable_params = sorted(set(param_indices))

    def get_operation(self, idx):
        """Returns the trainable operation, and the corresponding operation argument
        index, for a specified trainable parameter index.

        Args:
            idx (int): the trainable parameter index

        Returns:
            tuple[.Operation, int]: tuple containing the corresponding
            operation, and an integer representing the argument index,
            for the provided trainable parameter.
        """
        # get the index of the parameter in the tape
        t_idx = self.trainable_params[idx]

        # get the info for the parameter
        info = self._par_info[t_idx]

        # get the corresponding operation
        op = info["op"]

        # get the corresponding operation parameter index
        # (that is, index of the parameter within the operation)
        p_idx = info["p_idx"]
        return op, p_idx

    def get_parameters(
        self, trainable_only=True, operations_only=False, **kwargs
    ):  # pylint:disable=unused-argument
        """Return the parameters incident on the tape operations.

        The returned parameters are provided in order of appearance
        on the tape.

        Args:
            trainable_only (bool): if True, returns only trainable parameters
            operations_only (bool): if True, returns only the parameters of the
                operations excluding parameters to observables of measurements

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        By default, all parameters are trainable and will be returned:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> tape.trainable_params = [1] # set the second parameter as trainable
        >>> tape.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> tape.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        params = []
        iterator = self.trainable_params if trainable_only else self._par_info

        for p_idx in iterator:
            op = self._par_info[p_idx]["op"]
            if operations_only and hasattr(op, "return_type"):
                continue

            op_idx = self._par_info[p_idx]["p_idx"]
            params.append(op.data[op_idx])
        return params

    def set_parameters(self, params, trainable_only=True):
        """Set the parameters incident on the tape operations.

        Args:
            params (list[float]): A list of real numbers representing the
                parameters of the quantum operations. The parameters should be
                provided in order of appearance in the quantum tape.
            trainable_only (bool): if True, set only trainable parameters

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        By default, all parameters are trainable and can be modified:

        >>> tape.set_parameters([0.1, 0.2, 0.3])
        >>> tape.get_parameters()
        [0.1, 0.2, 0.3]

        Setting the trainable parameter indices will result in only the specified
        parameters being modifiable. Note that this only modifies the number of
        parameters that must be passed.

        >>> tape.trainable_params = [0, 2] # set the first and third parameter as trainable
        >>> tape.set_parameters([-0.1, 0.5])
        >>> tape.get_parameters(trainable_only=False)
        [-0.1, 0.2, 0.5]

        The ``trainable_only`` argument can be set to ``False`` to instead set
        all parameters:

        >>> tape.set_parameters([4, 1, 6], trainable_only=False)
        >>> tape.get_parameters(trainable_only=False)
        [4, 1, 6]
        """
        if trainable_only:
            iterator = zip(self.trainable_params, params)
            required_length = self.num_params
        else:
            iterator = enumerate(params)
            required_length = len(self._par_info)

        if len(params) != required_length:
            raise ValueError("Number of provided parameters does not match.")

        for idx, p in iterator:
            op = self._par_info[idx]["op"]
            op.data[self._par_info[idx]["p_idx"]] = p
            op._check_batching(op.data)
        self._update_batch_size()
        self._update_output_dim()

    @staticmethod
    def _single_measurement_shape(measurement_process, device):
        """Auxiliary function of shape that determines the output
        shape of a tape with a single measurement.

        Args:
            measurement_process (MeasurementProcess): the measurement process
                associated with the single measurement
            device (~.Device): a PennyLane device

        Returns:
            tuple: output shape
        """
        return measurement_process.shape(device)

    @staticmethod
    def _multi_homogenous_measurement_shape(mps, device):
        """Auxiliary function of shape that determines the output
        shape of a tape with multiple homogenous measurements.

        .. note::

            Assuming multiple probability measurements where not all
            probability measurements have the same number of wires specified,
            the output shape of the tape is a sum of the output shapes produced
            by each probability measurement.

            Consider the `qml.probs(wires=[0]), qml.probs(wires=[1,2])`
            multiple probability measurement with an analytic device as an
            example.

            The output shape will be a one element tuple `(6,)`, where the
            element `6` is equal to `2 ** 1 + 2 ** 2 = 6`. The base of each
            term is determined by the number of basis states and the exponent
            of each term comes from the length of the wires specified for the
            probability measurements: `1 == len([0]) and 2 == len([1, 2])`.
        """
        shape = tuple()

        # We know that there's one type of return_type, gather it from the
        # first one
        ret_type = mps[0].return_type
        if ret_type == qml.measurements.State:
            raise TapeError(
                "Getting the output shape of a tape with multiple state measurements is not supported."
            )

        shot_vector = device._shot_vector
        if shot_vector is None:
            if ret_type in (qml.measurements.Expectation, qml.measurements.Variance):

                shape = (len(mps),)

            elif ret_type == qml.measurements.Probability:

                wires_num_set = {len(meas.wires) for meas in mps}
                same_num_wires = len(wires_num_set) == 1
                if same_num_wires:
                    # All probability measurements have the same number of
                    # wires, gather the length from the first one

                    len_wires = len(mps[0].wires)
                    dim = mps[0]._get_num_basis_states(len_wires, device)
                    shape = (len(mps), dim)

                else:
                    # There are a varying number of wires that the probability
                    # measurement processes act on
                    shape = (sum(2 ** len(m.wires) for m in mps),)

            elif ret_type == qml.measurements.Sample:

                shape = (len(mps), device.shots)

            # No other measurement type to check

        else:
            shape = QuantumTape._shape_shot_vector_multi_homogenous(mps, device)

        return shape

    @staticmethod
    def _shape_shot_vector_multi_homogenous(mps, device):
        """Auxiliary function for determining the output shape of the tape for
        multiple homogenous measurements for a device with a shot vector.

        Note: it is assumed that getting the output shape of a tape with
        multiple state measurements is not supported.
        """
        shape = tuple()

        ret_type = mps[0].return_type
        shot_vector = device._shot_vector

        # Shot vector was defined
        if ret_type in (qml.measurements.Expectation, qml.measurements.Variance):
            num = sum(shottup.copies for shottup in shot_vector)
            shape = (num, len(mps))

        elif ret_type == qml.measurements.Probability:

            wires_num_set = {len(meas.wires) for meas in mps}
            same_num_wires = len(wires_num_set) == 1
            if same_num_wires:
                # All probability measurements have the same number of
                # wires, gather the length from the first one

                len_wires = len(mps[0].wires)
                dim = mps[0]._get_num_basis_states(len_wires, device)
                shot_copies_sum = sum(s.copies for s in shot_vector)
                shape = (shot_copies_sum, len(mps), dim)

            else:
                # There is a varying number of wires that the probability
                # measurement processes act on
                # TODO: revisit when issues with this case are resolved
                raise TapeError(
                    "Getting the output shape of a tape with multiple probability measurements "
                    "along with a device that defines a shot vector is not supported."
                )

        elif ret_type == qml.measurements.Sample:
            shape = []
            for shot_val in device.shot_vector:
                shots = shot_val.shots
                if shots != 1:
                    shape.extend((shots, len(mps)) for _ in range(shot_val.copies))
                else:
                    shape.extend((len(mps),) for _ in range(shot_val.copies))
        return shape

    def shape(self, device):
        """Produces the output shape of the tape by inspecting its measurements
        and the device used for execution.

        .. note::

            The computed shape is not stored because the output shape may be
            dependent on the device used for execution.

        Args:
            device (.Device): the device that will be used for the tape execution

        Raises:
            TapeError: raised for unsupported cases for
                example when the tape contains heterogeneous measurements

        Returns:
            Union[tuple[int], list[tuple[int]]]: the output shape(s) of the
            tape result

        **Example:**

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            a = np.array([0.1, 0.2, 0.3])

            def func(a):
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)

            with qml.tape.QuantumTape() as tape:
                func(a)
                qml.state()

        .. code-block:: pycon

            >>> tape.shape(dev)
            (1, 4)
        """
        output_shape = tuple()

        if len(self._measurements) == 1:
            output_shape = self._single_measurement_shape(self._measurements[0], device)
        else:
            num_measurements = len({meas.return_type for meas in self._measurements})
            if num_measurements == 1:
                output_shape = self._multi_homogenous_measurement_shape(self._measurements, device)
            else:
                raise TapeError(
                    "Getting the output shape of a tape that contains multiple types of measurements is unsupported."
                )
        return output_shape

    @property
    def numeric_type(self):
        """Returns the expected numeric type of the tape result by inspecting
        its measurements.

        Raises:
            TapeError: raised for unsupported cases for
                example when the tape contains heterogeneous measurements

        Returns:
            type: the numeric type corresponding to the result type of the
            tape

        **Example:**

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            a = np.array([0.1, 0.2, 0.3])

            def func(a):
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)

            with qml.tape.QuantumTape() as tape:
                func(a)
                qml.state()

        .. code-block:: pycon

            >>> tape.numeric_type
            complex
        """
        measurement_types = {meas.return_type for meas in self._measurements}
        if len(measurement_types) > 1:
            raise TapeError(
                "Getting the numeric type of a tape that contains multiple types of measurements is unsupported."
            )

        if list(measurement_types)[0] == qml.measurements.Sample:
            for observable in self._measurements:
                # Note: if one of the sample measurements contains outputs that
                # are real, then the entire result will be real
                if observable.numeric_type is float:
                    return observable.numeric_type

            return int

        return self._measurements[0].numeric_type

    def unwrap(self):
        """A context manager that unwraps a tape with tensor-like parameters
        to NumPy arrays.

        Args:
            tape (.QuantumTape): the quantum tape to unwrap

        Returns:

            .QuantumTape: the unwrapped quantum tape

        **Example**

        >>> with tf.GradientTape():
        ...     with qml.tape.QuantumTape() as tape:
        ...         qml.RX(tf.Variable(0.1), wires=0)
        ...         qml.RY(tf.constant(0.2), wires=0)
        ...         qml.RZ(tf.Variable(0.3), wires=0)
        ...     with tape.unwrap():
        ...         print("Trainable params:", tape.trainable_params)
        ...         print("Unwrapped params:", tape.get_parameters())
        Trainable params: [0, 2]
        Unwrapped params: [0.1, 0.3]
        >>> print("Original parameters:", tape.get_parameters())
        Original parameters: [<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.1>,
          <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.3>]
        """
        return UnwrapTape(self)

    # ========================================================
    # Tape properties
    # ========================================================

    @property
    def operations(self) -> List[Operation]:
        """Returns the operations on the quantum tape.

        Returns:
            list[.Operation]: recorded quantum operations

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        >>> tape.operations
        [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
        """
        return self._prep + self._ops

    @property
    def observables(self):
        """Returns the observables on the quantum tape.

        Returns:
            list[.Observable]: list of recorded quantum operations

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        >>> tape.observables
        [expval(PauliZ(wires=[0]))]
        """
        # TODO: modify this property once devices
        # have been refactored to accept and understand recieving
        # measurement processes rather than specific observables.
        obs = []

        for m in self._measurements:
            if m.obs is not None:
                m.obs.return_type = m.return_type
                obs.append(m.obs)
            else:
                obs.append(m)

        return obs

    @property
    def measurements(self):
        """Returns the measurements on the quantum tape.

        Returns:
            list[.MeasurementProcess]: list of recorded measurement processess

        **Example**

        .. code-block:: python

            with QuantumTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        >>> tape.measurements
        [expval(PauliZ(wires=[0]))]
        """
        return self._measurements

    @property
    def num_params(self):
        """Returns the number of trainable parameters on the quantum tape."""
        return len(self.trainable_params)

    @property
    def batch_size(self):
        r"""The batch size of the quantum tape inferred from the batch sizes
        of the used operations for parameter broadcasting.

        .. seealso:: :attr:`~.Operator.batch_size` for details.

        Returns:
            int or None: The batch size of the quantum tape if present, else ``None``.
        """
        return self._batch_size

    @property
    def output_dim(self):
        """The (inferred) output dimension of the quantum tape."""
        return self._output_dim

    @property
    def diagonalizing_gates(self):
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Returns:
            List[~.Operation]: the operations that diagonalize the observables
        """
        rotation_gates = []

        for observable in self.observables:
            # some observables do not have diagonalizing gates,
            # in which case we just don't append any
            with contextlib.suppress(qml.operation.DiagGatesUndefinedError):
                rotation_gates.extend(observable.diagonalizing_gates())
        return rotation_gates

    @property
    def graph(self):
        """Returns a directed acyclic graph representation of the recorded
        quantum circuit:

        >>> tape.graph
        <pennylane.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

        Note that the circuit graph is only constructed once, on first call to this property,
        and cached for future use.

        Returns:
            .CircuitGraph: the circuit graph object
        """
        if self._graph is None:
            self._graph = qml.CircuitGraph(
                self.operations, self.observables, self.wires, self._par_info, self.trainable_params
            )

        return self._graph

    @property
    def specs(self):
        """Resource information about a quantum circuit.

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain tape specifications

        **Example**

        .. code-block:: python3

            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=0)
                qml.RZ(0.26, wires=1)
                qml.CNOT(wires=[1, 0])
                qml.Rot(1.8, -2.7, 0.2, wires=0)
                qml.Hadamard(wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        Asking for the specs produces a dictionary as shown below:

        >>> tape.specs['gate_sizes']
        defaultdict(int, {1: 4, 2: 2})
        >>> tape.specs['gate_types']
        defaultdict(int, {'Hadamard': 2, 'RZ': 1, 'CNOT': 2, 'Rot': 1})

        As ``defaultdict`` objects, any key not present in the dictionary returns 0.

        >>> tape.specs['gate_types']['RX']
        0

        """
        if self._specs is None:
            self._specs = {"gate_sizes": defaultdict(int), "gate_types": defaultdict(int)}

            for op in self.operations:
                # don't use op.num_wires to allow for flexible gate classes like QubitUnitary
                self._specs["gate_sizes"][len(op.wires)] += 1
                self._specs["gate_types"][op.name] += 1

            self._specs["num_operations"] = len(self.operations)
            self._specs["num_observables"] = len(self.observables)
            self._specs["num_diagonalizing_gates"] = len(self.diagonalizing_gates)
            self._specs["num_used_wires"] = self.num_wires
            self._specs["depth"] = self.graph.get_depth()
            self._specs["num_trainable_params"] = self.num_params

        return self._specs

    # pylint: disable=too-many-arguments
    def draw(
        self,
        wire_order=None,
        show_all_wires=False,
        decimals=None,
        max_length=100,
        show_matrices=False,
    ):
        """Draw the quantum tape as a circuit diagram. See :func:`~.drawer.tape_text` for more information.

        Args:
            wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
            show_all_wires (bool): If True, all wires, including empty wires, are printed.
            decimals (int): How many decimal points to include when formatting operation parameters.
                Default ``None`` will omit parameters from operation labels.
            max_length (Int) : Maximum length of a individual line.  After this length, the diagram will
                begin anew beneath the previous lines.
            show_matrices=False (bool): show matrix valued parameters below all circuit diagrams

        Returns:
            str: the circuit representation of the tape
        """
        return qml.drawer.tape_text(
            self,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            max_length=max_length,
            show_matrices=show_matrices,
        )

    def to_openqasm(self, wires=None, rotations=True, measure_all=True, precision=None):
        """Serialize the circuit as an OpenQASM 2.0 program.

        Measurements are assumed to be performed on all qubits in the computational basis. An
        optional ``rotations`` argument can be provided so that output of the OpenQASM circuit is
        diagonal in the eigenbasis of the tape's observables. The measurement outputs can be
        restricted to only those specified in the tape by setting ``measure_all=False``.

        .. note::

            The serialized OpenQASM program assumes that gate definitions
            in ``qelib1.inc`` are available.

        Args:
            wires (Wires or None): the wires to use when serializing the circuit
            rotations (bool): in addition to serializing user-specified
                operations, also include the gates that diagonalize the
                measured wires such that they are in the eigenbasis of the circuit observables.
            measure_all (bool): whether to perform a computational basis measurement on all qubits
                or just those specified in the tape
            precision (int): decimal digits to display for parameters

        Returns:
            str: OpenQASM serialization of the circuit
        """
        wires = wires or self.wires

        # add the QASM headers
        qasm_str = "OPENQASM 2.0;\n"
        qasm_str += 'include "qelib1.inc";\n'

        if self.num_wires == 0:
            # empty circuit
            return qasm_str

        # create the quantum and classical registers
        qasm_str += f"qreg q[{len(wires)}];\n"
        qasm_str += f"creg c[{len(wires)}];\n"

        # get the user applied circuit operations
        operations = self.operations

        if rotations:
            # if requested, append diagonalizing gates corresponding
            # to circuit observables
            operations += self.diagonalizing_gates

        with QuantumTape() as tape:
            for op in operations:
                op.queue()

        # decompose the queue
        # pylint: disable=no-member
        operations = tape.expand(depth=2, stop_at=lambda obj: obj.name in OPENQASM_GATES).operations

        # create the QASM code representing the operations
        for op in operations:
            try:
                gate = OPENQASM_GATES[op.name]
            except KeyError as e:
                raise ValueError(f"Operation {op.name} not supported by the QASM serializer") from e

            wire_labels = ",".join([f"q[{wires.index(w)}]" for w in op.wires.tolist()])
            params = ""

            if op.num_params > 0:
                # If the operation takes parameters, construct a string
                # with parameter values.
                if precision is not None:
                    params = "(" + ",".join([f"{p:.{precision}}" for p in op.parameters]) + ")"
                else:
                    # use default precision
                    params = "(" + ",".join([str(p) for p in op.parameters]) + ")"

            qasm_str += f"{gate}{params} {wire_labels};\n"

        # apply computational basis measurements to each quantum register
        # NOTE: This is not strictly necessary, we could inspect self.observables,
        # and then only measure wires which are requested by the user. However,
        # some devices which consume QASM require all registers be measured, so
        # measure all wires by default to be safe.
        if measure_all:
            for wire in range(len(wires)):
                qasm_str += f"measure q[{wire}] -> c[{wire}];\n"
        else:
            measured_wires = qml.wires.Wires.all_wires([m.wires for m in self.measurements])

            for w in measured_wires:
                wire_indx = self.wires.index(w)
                qasm_str += f"measure q[{wire_indx}] -> c[{wire_indx}];\n"

        return qasm_str

    @property
    def data(self):
        """Alias to :meth:`~.get_parameters` and :meth:`~.set_parameters`
        for backwards compatibilities with operations."""
        return self.get_parameters(trainable_only=False)

    @data.setter
    def data(self, params):
        self.set_parameters(params, trainable_only=False)

    def copy(self, copy_operations=False):
        """Returns a shallow copy of the quantum tape.

        Args:
            copy_operations (bool): If True, the tape operations are also shallow copied.
                Otherwise, if False, the copied tape operations will simply be references
                to the original tape operations; changing the parameters of one tape will likewise
                change the parameters of all copies.

        Returns:
            .QuantumTape: a shallow copy of the tape
        """
        tape = QuantumTape()

        if copy_operations:
            # Perform a shallow copy of all operations in the state prep, operation, and measurement
            # queues. The operations will continue to share data with the original tape operations
            # unless modified.
            tape._prep = [copy.copy(op) for op in self._prep]
            tape._ops = [copy.copy(op) for op in self._ops]
            tape._measurements = [copy.copy(op) for op in self._measurements]
        else:
            # Perform a shallow copy of the state prep, operation, and measurement queues. The
            # operations within the queues will be references to the original tape operations;
            # changing the original operations will always alter the operations on the copied tape.
            tape._prep = self._prep.copy()
            tape._ops = self._ops.copy()
            tape._measurements = self._measurements.copy()

        tape._graph = None
        tape._specs = None
        tape._depth = None
        tape.wires = copy.copy(self.wires)
        tape.num_wires = self.num_wires
        tape.is_sampled = self.is_sampled
        tape.all_sampled = self.all_sampled
        tape._update_par_info()
        tape.trainable_params = self.trainable_params.copy()
        tape._obs_sharing_wires = self._obs_sharing_wires
        tape._obs_sharing_wires_id = self._obs_sharing_wires_id
        tape._batch_size = self.batch_size
        tape._output_dim = self.output_dim

        return tape

    def __copy__(self):
        return self.copy(copy_operations=True)

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the quantum tape"""
        fingerprint = []
        fingerprint.extend(op.hash for op in self.operations)
        fingerprint.extend(m.hash for m in self.measurements)
        fingerprint.extend(self.trainable_params)
        return hash(tuple(fingerprint))
