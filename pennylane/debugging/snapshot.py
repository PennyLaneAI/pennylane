# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This file contains the snapshots function which extracts measurements from the qnode.
"""
import warnings
from functools import partial

from pennylane.devices import LegacyDeviceFacade
from pennylane.ops import Snapshot
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn


def _is_snapshot_compatible(dev):
    # The `_debugger` attribute is a good enough proxy for snapshot compatibility
    if isinstance(dev, LegacyDeviceFacade):
        return _is_snapshot_compatible(dev.target_device)
    return hasattr(dev, "_debugger")


class _SnapshotDebugger:
    """A debugging context manager.

    Without an active debugging context, devices will not save their internal state when
    encoutering Snapshot operations. The debugger also serves as storage for the device states.

    Args:
        dev (Device): device to attach the debugger to
    """

    def __init__(self, dev):
        self.snapshots = {}
        self.active = False
        self.device = dev
        dev._debugger = self

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.active = False
        self.device._debugger = None


@transform
def snapshots(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""This transform processes :class:`~pennylane.Snapshot` instances contained in a circuit,
    depending on the compatibility of the execution device.
    For supported devices, the snapshots' measurements are computed as the execution progresses.
    Otherwise, the :func:`QuantumTape <pennylane.tape.QuantumTape>` gets split into several, one for each snapshot, with each aggregating
    all the operations up to that specific snapshot.

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    If tape splitting is carried out, the transform will be conservative about the wires that it includes in each tape.
    So, if all operations preceding a snapshot in a 3-qubit circuit has been applied to only one wire,
    the tape would only be looking at this wire. This can be overridden by the configuration of the execution device
    and its nature.

    Regardless of the transform's behaviour, the output is a dictionary where each key is either
    the tag supplied to the snapshot or its index in order of appearance, in addition to an
    ``"execution_results"`` entry that returns the final output of the quantum circuit. The post-processing
    function is responsible for aggregating the results into this dictionary.

    When the transform is applied to a QNode, the ``shots`` configuration is inherited from the device.
    Therefore, the snapshot measurements must be supported by the device's nature. An exception of this are
    the :func:`qml.state <pennylane.state>` measurements on finite-shot simulators.

    .. warning::

        For devices that do not support snapshots (e.g QPUs, external plug-in simulators), be mindful of
        additional costs that you might incur due to the 1 separate execution/snapshot behaviour.

    .. warning::

        ``Snapshot`` captures the internal execution state at a point in the circuit, but compilation transforms
        (e.g., ``combine_global_phases``, ``merge_rotations``) may reorder or modify operations across the snapshot.
        As a result, the captured state may differ from the original intent.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            qml.Snapshot("sample", measurement=qml.sample(), shots=5)
            return qml.expval(qml.X(0))

    >>> qml.snapshots(circuit)()
    {0: 1.0,
    1: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]),
    'sample': array([[0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1]]),
    'execution_results': 0.0}

    .. code-block:: python3

        from functools import partial
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.set_shots, shots=200)
        @qml.snapshots
        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Snapshot()
            qml.CNOT(wires=[0, 1])
            return qml.counts()

    >>> circuit()
    {0: array([0.70710678+0.j, 0.        +0.j, 0.70710678+0.j, 0.        +0.j]),
    'execution_results': {'00': 101, '11': 99}}

    Here one can see how a device that does not natively support snapshots executes two different circuits. Additionally, a warning
    is raised along with the results:

    .. code-block:: python3

        dev = qml.device("lightning.qubit", wires=2)

        @partial(qml.set_shots, shots=200)
        @qml.snapshots
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0),
            qml.Snapshot(qml.counts())
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with circuit.device.tracker:
            out = circuit()

    >>> circuit.device.tracker.totals
    UserWarning: Snapshots are not supported for the given device. Therefore, a tape will be created for each snapshot, resulting in a total of n_snapshots + 1 executions.
      warnings.warn(
    {'batches': 1,
     'simulations': 2,
     'executions': 2,
     'shots': 200,
     'results': -0.16}

    >>> out
    {0: {'00': tensor(52, requires_grad=True),
      '10': tensor(48, requires_grad=True)},
     'execution_results': tensor(-0.1, requires_grad=True)}

    **Integration with Mid-Circuit Measurements:**

    Mid-circuit measurements can be handled either by executing the circuit ``shots`` times (``mcm_method="one-shot"``),
    or constructing and executing a tree of all possible branches (``mcm_method="tree-traversal"``).

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit'))
        def c(y):
            qml.H(0)
            qml.Snapshot(measurement=qml.probs(0))
            m0 = qml.measure(0)
            qml.cond(m0, qml.RX)(0.5, 0)
            qml.Snapshot(measurement=qml.probs(0))
            qml.RY(y, 0)
            qml.measure(0)
            qml.Snapshot(measurement=qml.probs(0))
            return qml.probs(0)

    If we use ``"one-shot"`` together with three shots, you can see that each snapshot has three
    probabilities in a list.  Each entry corresponds to a different execution of the circuit. Note that
    postselection will not be applied on snapshots with ``"one-shot"``.

    >>> one_shot = qml.set_shots(c, 3).update(mcm_method="one-shot")
    >>> qml.snapshots(one_shot)(1.0)
    {0: [array([0., 1.]), array([0., 1.]), array([1., 0.])],
    1: [array([1., 0.]), array([1., 0.]), array([1., 0.])],
    2: [array([1., 0.]), array([1., 0.]), array([1., 0.])],
    'execution_results': array([1., 0.])}

    Note that for a single shot ``"one-shot"`` execution, the results will not be in a list.

    >>> one_shot = qml.set_shots(c, 1).update(mcm_method="one-shot")
    >>> qml.snapshots(one_shot)(1.0)
    {0: array([0., 1.]),
    1: array([0., 1.]),
    2: array([0., 1.]),
    'execution_results': array([0., 1.])}

    With ``"tree-traversal"``, we have different numbers of probabilities based on the number of mcm's earlier
    in the circuit. Before any mcm's, we have one state. After the first mcm, we have two states.  And after
    two mcm's, we have four states.

    >>> qml.snapshots(c.update(mcm_method="tree-traversal"))(1.0)
    {0: [array([0.5, 0.5])],
    1: [array([1., 0.]), array([0.06120872, 0.93879128])],
    2: [array([1., 0.]), array([0., 1.]), array([1., 0.]), array([0., 1.])],
    'execution_results': array([0.51653561, 0.48346439])}

    Note that without the ``RY`` rotation, the third snapshot only has **three** probabilities, not **four**.
    This is due to the fact that one of the four branches can never occur.

    >>> qml.snapshots(c.update(mcm_method="tree-traversal"))(0)
    {0: array([0.5, 0.5]),
    1: [array([1., 0.]), array([0.06120872, 0.93879128])],
    2: [array([1., 0.]), array([1., 0.]), array([0., 1.])],
    'execution_results': array([0.53060436, 0.46939564])}


    **Application to tapes**

    Here you can see the default behaviour of the transform for unsupported devices and you can see how the amount of wires included
    in each resulting tape is minimal:

    .. code-block:: python3

        ops = [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ]

        measurements = [qml.expval(qml.PauliX(0))]

        tape = qml.tape.QuantumTape(ops, measurements)

        tapes, collect_results_into_dict = qml.snapshots(tape)

    >>> print(tapes)
    [<QuantumTape: wires=[], params=0>, <QuantumTape: wires=[0], params=0>, <QuantumTape: wires=[0, 1], params=0>, <QuantumTape: wires=[0, 1], params=0>]

    """

    new_tapes = []
    accumulated_ops = []
    snapshot_tags = []

    for op in tape.operations:
        if isinstance(op, Snapshot):
            if isinstance(op.tag, int):
                raise ValueError("User provided Snapshot tags can only be of type 'str'. Got int.")
            snapshot_tags.append(op.tag or len(new_tapes))
            meas_op = op.hyperparameters["measurement"]
            shots = (
                tape.shots
                if op.hyperparameters["shots"] == "workflow"
                else op.hyperparameters["shots"]
            )
            new_tapes.append(
                tape.copy(operations=accumulated_ops, measurements=[meas_op], shots=shots)
            )
        else:
            accumulated_ops.append(op)

    # Create an additional final tape if a return measurement exists
    if tape.measurements:
        snapshot_tags.append("execution_results")
        new_tapes.append(tape.copy(operations=accumulated_ops))

    def postprocessing_fn(results, snapshot_tags):
        return dict(zip(snapshot_tags, results))

    return new_tapes, partial(postprocessing_fn, snapshot_tags=snapshot_tags)


def _null_postprocessing(results):
    return results[0]


@transform
def _add_snapshot_tags(tape):
    """Makes sure every snapshot has a tag.  Snapshots that do not have tags gain a tag corresponding to their
    index into all snapshots."""
    new_ops = []
    num_snapshots = 0
    for op in tape.operations:
        if isinstance(op, Snapshot):
            if op.tag is None:
                new_ops.append(op.update_tag(num_snapshots))
            elif isinstance(op.tag, int):
                raise ValueError("User provided Snapshot tags can only be of type 'str'. Got int.")
            else:
                new_ops.append(op)
            num_snapshots += 1
        else:
            new_ops.append(op)
    return (tape.copy(ops=new_ops),), _null_postprocessing


@snapshots.custom_qnode_transform
def snapshots_qnode(self, qnode, targs, tkwargs):
    """A custom QNode wrapper for the snapshot transform :func:`~.snapshots`.
    Depending on whether the QNode's device supports snapshots, an efficient execution
    would be used. Otherwise, the QNode's tape would be split into several around the
    present snapshots and execute each individually.
    """

    def get_snapshots(*args, **kwargs):

        with _SnapshotDebugger(qnode.device) as dbg:
            results = _add_snapshot_tags(qnode)(*args, **kwargs)  # pylint: disable=not-callable

        dbg.snapshots["execution_results"] = results
        return dbg.snapshots

    if _is_snapshot_compatible(qnode.device):
        return get_snapshots

    warnings.warn(
        "Snapshots are not supported for the given device. Therefore, a tape will be "
        "created for each snapshot, resulting in a total of n_snapshots + 1 executions.",
        UserWarning,
    )

    return self.default_qnode_transform(qnode, targs, tkwargs)
