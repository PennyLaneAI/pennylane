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
from pennylane.transforms import qfunc_transform
from pennylane.wires import Wires


@qfunc_transform
def defer_measurements(tape):
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

        This transform does not change the list of terminal measurements returned by
        the quantum function.

    .. note::

        When applying the transform on a quantum function that returns
        :func:`~pennylane.state` as the terminal measurement or contains the
        :class:`~.Snapshot` instruction, state information corresponding to
        simulating the transformed circuit will be obtained. No
        post-measurement states are considered.

    Args:
        qfunc (function): a quantum function

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
    """
    new_wires = {}

    cv_types = (qml.operation.CVOperation, qml.operation.CVObservable)
    ops_cv = any(isinstance(op, cv_types) for op in tape.operations)
    obs_cv = any(isinstance(getattr(op, "obs", None), cv_types) for op in tape.measurements)
    if ops_cv or obs_cv:
        raise ValueError("Continuous variable operations and observables are not supported.")

    # Current wire in which pre-measurement state will be saved
    new_wire_latest = tape.num_wires
    for op in tape.operations:
        if isinstance(op, MidMeasureMP):
            new_wires[op.id] = new_wire_latest
            qml.CNOT([op.wires[0], new_wire_latest])

            if op.reset:
                qml.CNOT([new_wire_latest, op.wires[0]])

            new_wire_latest += 1

        elif op.__class__.__name__ == "Conditional":
            _add_control_gate(op, new_wires)
        else:
            apply(op)


def _add_control_gate(op, control_wires):
    """Helper function to add control gates"""
    control = [control_wires[m_id] for m_id in op.measurement_ids]
    for branch, value in op._items():  # pylint: disable=protected-access
        if value:
            ctrl(
                lambda: apply(op.then_op),  # pylint: disable=cell-var-from-loop
                control=Wires(control),
                control_values=branch,
            )()
