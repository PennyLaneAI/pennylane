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
from pennylane.tape import QuantumTape, get_active_tape


@qfunc_transform
def defer_measurements(tape):
    """Quantum function transform that substitutes operations conditioned on
    measurement outcomes to controlled operations.

    This transform uses the `deferred measurement principle
    <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`_ and
    applies to qubit based quantum functions.

    .. note::

        The transform uses the `:func:~.ctrl` transform to implement operations
        controlled on mid-circuit measurement outcomes. The set of operations
        that can be controlled as such depends on the set of operations
        supported by the chosen device.

    .. note::

        This transform does not extend the terminal measurements returned by
        the quantum function with the mid-circuit measurements.

    Args:
        qfunc (function): a quantum function

    **Example**

    Suppose we have a quantum function with mid-circuit measurements and
    conditional operations:

    .. code-block:: python3

        def qfunc(par):
            qml.RY(0.123, wires=0)
            qml.Hadamard(wires=1)
            m_0 = qml.mid_measure(1)
            qml.if_then(m_0, qml.RY)(par, wires=0)
            return qml.expval(qml.PauliZ(0))

    The ``defer_measurements`` transform allows executing such quantum
    functions without having to perform mid-circuit measurements.

    The original circuit is:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = qml.defer_measurements(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> par = np.array(np.pi/2, requires_grad=True)
    >>> qnode(par)
    tensor(0.43487747, requires_grad=True)

    We can also optimize parameters passed to conditional operations:

    >>> steps = 100
    >>> for _ in range(steps):
    ...     par, cost = opt.step_and_cost(qnode, par)
    >>> print(par, cost)
    3.018529732412975 -0.0037774828357067247
    """
    # TODO: do we need a map or can we just have a list?
    measured_wires = {}

    if any(
        [
            isinstance(op, qml.operation.CVOperation) or isinstance(op, qml.operation.CVObservable)
            for op in tape.queue
        ]
    ):
        raise ValueError("Continuous variable operations and observables are not supported.")

    for op in tape.queue:
        if any([wire in measured_wires.values() for wire in op.wires]):
            raise ValueError(
                "Cannot apply operations on {op.wires} as some has been measured already."
            )

        if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
            measured_wires[op.measurement_id] = op.wires[0]

        elif op.__class__.__name__ == "_IfOp":
            # TODO: Why does op.dependant_measurements store the wire ids instead of labels?
            control = [measured_wires[m_id] for m_id in op.dependant_measurements]
            for branch, value in op.branches.items():
                if value:
                    ctrl(lambda: apply(op.then_op), control=Wires(control))()
        else:
            apply(op)
