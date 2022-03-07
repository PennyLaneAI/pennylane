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
from pennylane.wires import Wires

import pennylane as qml
from pennylane.transforms import qfunc_transform, ctrl
from pennylane.queuing import apply


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
            qml.cond(m_0, qml.RY(par, wires=0))
            return qml.expval(qml.PauliZ(0))

    The ``defer_measurements`` transform allows executing such quantum
    functions without having to perform mid-circuit measurements:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = qml.defer_measurements(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> par = np.array(np.pi/2, requires_grad=True)
    >>> qnode(par)
    tensor(-0.12269009, requires_grad=True)

    We can also differentiate parameters passed to conditional operations:

    >>> qml.grad(qnode)(par)
    -0.9924450321351936

    .. note::

        When applying the transform on a quantum function that returns
        :func:`~measurements.state` as the terminal measurement, the state vector corresponding
        to the pre-measurement state of the transformed circuit will be
        obtained. No post-measurement states are considered.
    """
    measured_wires = {}

    if any(
        isinstance(op, (qml.operation.CVOperation, qml.operation.CVObservable)) for op in tape.queue
    ):
        raise ValueError("Continuous variable operations and observables are not supported.")

    for op in tape.queue:
        op_wires_measured = set(wire for wire in op.wires if wire in measured_wires.values())
        if len(op_wires_measured) > 0:
            raise ValueError(
                f"Cannot apply operations on {op.wires} as the following wires have been measured already: {op_wires_measured}."
            )

        if (
            isinstance(op, qml.measurements.MeasurementProcess)
            and op.return_type == qml.operation.MidMeasure
        ):
            measured_wires[op.id] = op.wires[0]

        elif op.__class__.__name__ == "Conditional":
            control_wire = [measured_wires[m_id] for m_id in op.meas_val.measurements]
            for value in op.meas_val.branches.values():
                if value == op.meas_val.control_value:
                    if op.meas_val.control_value == 0:
                        qml.PauliX(Wires(control_wire))

                    ctrl(
                        lambda: apply(op.then_op),  # pylint: disable=cell-var-from-loop
                        control=Wires(control_wire),
                    )()

                    if op.meas_val.control_value == 0:
                        qml.PauliX(Wires(control_wire))
        else:
            apply(op)
