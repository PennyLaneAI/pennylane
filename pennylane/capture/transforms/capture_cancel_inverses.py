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
"""Transform for cancelling adjacent inverse gates in quantum circuits."""
# pylint: disable=protected-access
import pennylane as qml
from pennylane.ops.qubit.attributes import symmetric_over_all_wires, symmetric_over_control_wires
from pennylane.transforms.optimization.cancel_inverses import _are_inverses
from pennylane.wires import Wires


class CancelInversesInterpreter(qml.capture.PlxprInterpreter):
    """Plxpr Interpreter for applying the ``cancel_inverses`` transform to callables or jaxpr
    when program capture is enabled.
    """

    def __init__(self):
        super().__init__()
        self.previous_ops = {}

    def cleanup(self) -> None:
        """Perform any final steps after iterating through all equations."""
        self.previous_ops = {}

    def interpret_operation(self, op):
        """Interpret a PennyLane operation instance.

        Args:
            op (Operator): a pennylane operator instance

        Returns:
            Any

        This method is only called when the operator's output is a dropped variable,
        so the output will not affect later equations in the circuit.

        See also: :meth:`~.interpret_operation_eqn`.

        """
        if len(op.wires) == 0:
            return super().interpret_operation(op)

        prev_op = self.previous_ops.get(op.wires[0], None)
        if prev_op is None:
            for w in op.wires:
                self.previous_ops[w] = op
            return []

        cancel = False
        if _are_inverses(op, prev_op):
            # Same wires, cancel
            if op.wires == prev_op.wires:
                cancel = True
            # Full overlap over wires
            elif len(Wires.shared_wires([op.wires, prev_op.wires])) == len(op.wires):
                # symmetric op + full wire overlap; cancel
                if op in symmetric_over_all_wires:
                    cancel = True
                # symmetric over control wires, full overlap over control wires; cancel
                elif op in symmetric_over_control_wires and (
                    len(Wires.shared_wires([op.wires[:-1], prev_op.wires[:-1]]))
                    == len(op.wires) - 1
                ):
                    cancel = True
            # No or partial overlap over wires; can't cancel

        if cancel:
            for w in op.wires:
                self.previous_ops.pop(w)
            return []

        previous_ops_on_wires = set(self.previous_ops.get(w) for w in op.wires)
        for o in previous_ops_on_wires:
            if o is not None:
                for w in o.wires:
                    self.previous_ops.pop(w)
        for w in op.wires:
            self.previous_ops[w] = op

        res = []
        for o in previous_ops_on_wires:
            res.append(super().interpret_operation(o))
        return res
