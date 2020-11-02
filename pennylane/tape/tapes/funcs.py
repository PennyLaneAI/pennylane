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
"""
This module contains the base quantum tape.
"""
import pennylane as qml
from copy import copy
from itertools import product


def _all_paths_through_branch_tape(batch_tape, path={}):
    """Generator that adds all paths through a branch tape to an existing path.
        Paths are represented by dictionaries of the form

        ..code-block:: python

            {batch1.name (str): element (int),  batch2.name (str): element (int), ...},

        where ``element`` is an index in ``range(batch.n_batches)`` defining which object in the
        tape queue is considered by this path.

    Arg:
        batch_tape (BatchTape): the batch tape to consider
        path (dict): a path through previous batches that this function may extend

    **Example**

    .. code-block:: python

        with BranchTape(name="b3") as b3:
            qml.RX(0.2, wires='a')
            qml.RX(0.5, wires='a')

        for p in _paths_through_batch_tape(b3, path={"b1":2, "b2":0}):
            print(p)

        # {"b1":2, "b2":0, "b3": 0}
        # {"b1":2, "b2":0, "b3": 1}

    Batch tapes can be nested structures of other batch tapes. The recursion takes care of
    the nesting.

    .. code-block:: python

        with BranchTape(name="b3") as b3:
            qml.RX(0.2, wires='a')
            with BatchTape(name="b4"):
                qml.RY(0.5, wires='a')
                qml.RZ(0.5, wires='a')

        for p in _paths_through_batch_tape(b3, path={"b1":2, "b2":0}):
            print(p)

        # {"b1":2, "b2":0, "b3": 0}
        # {"b1":2, "b2":0, "b3": 1, "b4": 0}
        # {"b1":2, "b2":0, "b3": 1, "b4": 1}

    Branch tapes can also nest quantum tapes (which in turn can consist of branch tapes).

    .. code-block:: python

        with BranchTape(name="b3") as b3:
            qml.RX(0.2, wires='a')

            with QuantumTape(name="q4"):

                with BatchTape(name="b5"):
                    qml.RY(0.5, wires='a')
                    qml.RZ(0.5, wires='a')

                with BatchTape(name="b6"):
                    qml.Hadamard(wires='a')
                    qml.Hadamard(wires='b')

        for p in _paths_through_batch_tape(b3, path={"b1":2, "b2":0}):
            print(p)

        # {"b1":2, "b2":0, "b3": 0}
        # {"b1":2, "b2":0, "b3": 1, "b5": 0, "b6":0}
        # {"b1":2, "b2":0, "b3": 1, "b5": 0, "b6":1}
        # {"b1":2, "b2":0, "b3": 1, "b5": 1, "b6":0}
        # {"b1":2, "b2":0, "b3": 1, "b5": 1, "b6":1}
    """

    counter = 0
    add_batch = True

    if batch_tape.name in path:
        # we use the information from the
        # batch already in the path
        add_batch = False

    for obj_ in batch_tape.iterator():

        new_path = copy(path)

        if add_batch:
            new_path.update({batch_tape.name: counter})

        counter += 1

        if isinstance(obj_, qml.tape.BranchTape):
            yield from _all_paths_through_branch_tape(obj_, path=new_path)

        else:
            if isinstance(obj_, qml.tape.QuantumTape):
                yield from _all_paths(obj_, path=new_path)

            else:
                # is an operation
                yield new_path


def _all_paths(tape, path={}):
    """
    Generator that adds all possible paths through batches in
    a tape to an existing path.

    Paths are represented by dictionaries of the form

    .. code-block::
        {batch1.name (str): element (int),  batch2.name (str): element (int), ...},

    where ``element`` is an index in ``range(batch.n_branches)`` defining which object
    in a branched queue is considered by this path.

    Arg:
        tape (QuantumTape): tape that may have a batched structure
        path (dict): a path through previous batches that this function may extend

    **Example**

    If the quantum tape does not contain batch tapes at any nesting level, the
    path will be returned unmodified.

    .. code-block:: python

        with QuantumTape(name="q3") as q3:
            qml.RX(0.2, wires='a')
            qml.RX(0.5, wires='a')

        for p in _paths_through_quantum_tape(q3, path={"b1":2, "b2":0}):
            print(p)
        # {"b1":2, "b2":0}
        # {"b1":2, "b2":0}

    If the recursion encounters batch tapes, their paths will be combined using a
    tensor product and added to the original path.

    .. code-block:: python

        with QuantumTape(name="q3") as q3:

            qml.RX(0.2, wires='a')

            with BatchTape(name="b4"):
                qml.RX(0.2, wires='a')
                qml.RX(0.5, wires='a')

            with BatchTape(name="b5"):
                qml.RX(0.2, wires='a')
                qml.RX(0.2, wires='b')

        for p in _paths_through_quantum_tape(q3, path={"b1":2, "b2":0}):
            print(p)

        # {"b1":2, "b2":0, "b4":0, "b5":0}
        # {"b1":2, "b2":0, "b4":0, "b5":1}
        # {"b1":2, "b2":0, "b4":1, "b5":0}
        # {"b1":2, "b2":0, "b4":1, "b5":1}
    """
    # If the tape is itself a batch tape, call batch method
    if type(tape) == qml.tape.BranchTape:
        yield from _all_paths_through_branch_tape(tape, path=path)

    else:
        new_paths = []
        for obj in tape.iterator():

            # todo: do not collect the items from the generator here to save memory
            if isinstance(obj, qml.tape.QuantumTape):
                new_paths.append(list(_all_paths(obj, path=path)))

        # todo: avoid duplicates from batches with the same name
        kron_dicts = product(*new_paths)

        for extension in kron_dicts:
            new_path = copy(path)
            for batch in extension:
                new_path.update(batch)
            yield new_path


def _unraveled_tape(tape, path, new_tape=None):
    """Record a new quantum tape that uses the batching alternatives defined in path.

    Args:
        tape (QuantumTape): tape to unravel

        path (dict): dictionary representing a path by defining batch
            names (keys) and the objects selected (values)

        new_tape (QuantumTape): new tape to which objects are queued, empty tape at first call

    Returns:
        QuantumTape: new tape recorded by following path in tape

    **Example**

    ..code-block:: python

        with QuantumTape() as tape:

            qml.RX(0.2, wires='a')

            with QuantumTape(name="q2"):

                with BranchTape(name="b1"):
                    qml.RY(0.5, wires='b')
                    qml.RY(0.5, wires='c')

                with BatchTape(name="b2"):
                    qml.RZ(0.1, wires='d')
                    qml.RZ(0.1, wires='e')

        path = {"b1": 0, "b2": 1}

        new_tape = _tape_from_path(q1, path)

        print(new_tape.operations)
        # [RX(0.2, wires=['a']), RY(0.5, wires=['b']), RZ(0.1, wires=['e'])]
    """

    if new_tape is None:
        new_tape = tape.__class__(name=tape.name)

    if isinstance(tape, qml.tape.BranchTape):

        # select object in the batch according to path
        idx = path[tape.name]
        # todo: faster if we iterate "select" times and grab the last
        batch = list(tape.iterator())
        obj = batch[idx]
        # recursion in case the object contains more batches
        new_tape = _unraveled_tape(obj, path, new_tape)

    elif isinstance(tape, qml.tape.QuantumTape):
        # tape is a QuantumTape, iterate through to queue its objects
        for obj in tape.iterator():
            new_tape = _unraveled_tape(obj, path, new_tape)
    else:
        # tape is actually the object to queue
        obj = tape

        # queue the object in the new tape
        with new_tape:
            # todo: update when more elegant API for queueing available
            if isinstance(obj, qml.beta.queuing.MeasurementProcess):
                obj.obs.queue()
            obj.queue()

    return new_tape


def unravel(tape, paths=None):
    """
    Returns a generator that yields tapes constructed from combining the
    alternative queues represented by ``BranchTape`` objects in this tape.

    Args:
        tape (QuantumTape): tape to unravel

    Returns:
        GeneratorObject: generator of quantum tapes

    **Examples**

    ..code-block:: python

        with QuantumTape() as tape:
            qml.RX(0.2, wires='a')

            with BranchTape(name="b1"):
                qml.RY(0.5, wires='b')
                qml.RY(0.5, wires='c')

            with BranchTape(name="measure"):
                qml.tape.measure.expval(qml.PauliZ(wires='a'))
                qml.tape.measure.expval(qml.PauliZ(wires='b'))

        for t in _unravel(tape):
            print(t.operations)
            print(t.measurements)

        # [RX(0.2, wires=['a']), RY(0.5, wires=['b'])]
        # [<pennylane.beta.queuing.measure.MeasurementProcess object at ...>]

        # [RX(0.2, wires=['a']), RY(0.5, wires=['b'])]
        # [<pennylane.beta.queuing.measure.MeasurementProcess object at ...>]

        # [RX(0.2, wires=['a']), RY(0.5, wires=['c'])]
        # [<pennylane.beta.queuing.measure.MeasurementProcess object at ...>]

        # [RX(0.2, wires=['a']), RY(0.5, wires=['c'])]
        # [<pennylane.beta.queuing.measure.MeasurementProcess object at ...>]
    """

    if paths is None:
        paths = _all_paths(tape)

    for path in paths:

        yield _unraveled_tape(tape, path)
