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
"""Code for the tape transform implementing the deferred measurement principle."""
import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.ops.op_math import ctrl
from pennylane.queuing import apply
from pennylane.tape import QuantumTape
from pennylane.transforms import qfunc_transform
from pennylane.wires import Wires

# pylint: disable=too-many-branches


@qfunc_transform
def defer_measurements(tape: QuantumTape):
    """Quantum function transform that substitutes operations conditioned on
    measurement outcomes to controlled operations.

    This transform uses the `deferred measurement principle
    <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`_ and
    applies to qubit-based quantum functions.

    Support for mid-circuit measurements is device-dependent. If a device
    doesn't support mid-circuit measurements natively, then the QNode will
    apply this transform.

    .. note::

        The transform uses the :func:`~.ctrl` transform to implement operations
        controlled on mid-circuit measurement outcomes. The set of operations
        that can be controlled as such depends on the set of operations
        supported by the chosen device.

    .. note::

        Devices that inherit `QubitDevice` **must** be initialized with an additional
        wire for each mid-circuit measurement after which the measured wire is reused
        or reset for `defer_measurements` to transform the quantum tape correctly.
        Such devices should also be initialized without custom wire labels for correct
        behaviour.

    .. note::

        This transform does not change the list of terminal measurements returned by
        the quantum function.

    .. note::

        When applying the transform on a quantum function that returns
        :func:`~pennylane.state` as the terminal measurement or contains the
        :class:`~.Snapshot` instruction, state information corresponding to
        simulating the transformed circuit will be obtained. No
        post-measurement states are considered.

    Args:
        tape (.QuantumTape): a quantum tape

    **Example**

    Suppose we have a quantum function with mid-circuit measurements and
    conditional operations:

    .. code-block:: python3

        def qfunc(par):
            qml.RY(0.123, wires=0)
            qml.Hadamard(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(par, wires=0)
            return qml.expval(qml.PauliZ(0))

    The ``defer_measurements`` transform allows executing such quantum
    functions without having to perform mid-circuit measurements:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = qml.defer_measurements(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> par = np.array(np.pi/2, requires_grad=True)
    >>> qnode(par)
    tensor(0.43487747, requires_grad=True)

    We can also differentiate parameters passed to conditional operations:

    >>> qml.grad(qnode)(par)
    tensor(-0.49622252, requires_grad=True)

    Reusing and reseting measured wires will work as expected with the
    ``defer_measurements`` transform:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1, reset=True)

            qml.cond(m_0, qml.RY)(y, wires=0)
            qml.RX(np.pi/4, wires=1)
            return qml.probs(wires=[0, 1])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.76960924, 0.13204407, 0.08394415, 0.01440254], requires_grad=True)
    """

    cv_types = (qml.operation.CVOperation, qml.operation.CVObservable)
    ops_cv = any(isinstance(op, cv_types) for op in tape.operations)
    obs_cv = any(isinstance(getattr(op, "obs", None), cv_types) for op in tape.measurements)
    if ops_cv or obs_cv:
        raise ValueError("Continuous variable operations and observables are not supported.")

    # Find wires that are reused after measurement
    measured_wires = set()
    reused_measurement_wires = set()

    for op in tape.operations:
        if isinstance(op, MidMeasureMP):
            measured_wires.add(op.wires[0])
            if op.reset:
                reused_measurement_wires.add(op.wires[0])

        elif op.__class__.__name__ == "Conditional":
            reused_measurement_wires = reused_measurement_wires.union(
                measured_wires.intersection(op.then_op.wires.toset())
            )

        else:
            reused_measurement_wires = reused_measurement_wires.union(
                measured_wires.intersection(op.wires.toset())
            )

    # Apply controlled operations to store measurement outcomes and replace
    # classically controlled operations
    new_wires = {}
    cur_wire = max(tape.wires) + 1

    for op in tape:
        if isinstance(op, MidMeasureMP):
            # Only store measurement outcome in new wire if wire gets reused
            if op.wires[0] in reused_measurement_wires:
                new_wires[op.id] = cur_wire

                qml.CNOT([op.wires[0], cur_wire])
                if op.reset:
                    qml.CNOT([cur_wire, op.wires[0]])

                cur_wire += 1
            else:
                new_wires[op.id] = op.wires[0]

        elif op.__class__.__name__ == "Conditional":
            _add_control_gate(op, new_wires)
        else:
            apply(op)

    return tape._qfunc_output  # pylint: disable=protected-access


def _add_control_gate(op, control_wires):
    """Helper function to add control gates"""
    control = [control_wires[m.id] for m in op.meas_val.measurements]
    for branch, value in op.meas_val._items():  # pylint: disable=protected-access
        if value:
            ctrl(
                lambda: apply(op.then_op),  # pylint: disable=cell-var-from-loop
                control=Wires(control),
                control_values=branch,
            )()
