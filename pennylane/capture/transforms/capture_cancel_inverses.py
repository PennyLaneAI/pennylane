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

    .. note::

        In the process of transforming plxpr, this interpreter may reorder operations that do
        not share any wires. This will not impact the correctness of the circuit.
    """

    def __init__(self):
        super().__init__()
        self.previous_ops = {}

    def setup(self) -> None:
        """Initialize the instance before interpreting equations."""
        self.previous_ops = {}

    def interpret_operation(self, op: qml.operation.Operator):
        """Interpret a PennyLane operation instance.

        This method cancels operations that are the adjoint of the previous
        operation on the same wires, and otherwise, applies it.

        Args:
            op (Operator): a pennylane operator instance

        Returns:
            Any

        This method is only called when the operator's output is a dropped variable,
        so the output will not affect later equations in the circuit.

        See also: :meth:`~.interpret_operation_eqn`.

        """
        # pylint: disable=too-many-branches
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

        # Putting the operations in a set to avoid applying the same op multiple times
        # Using a set causes order to no longer be guaranteed, so the new order of the
        # operations might differ from the original order. However, this only impacts
        # operators without any shared wires, so correctness will not be impacted.
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

    def interpret_all_previous_ops(self) -> None:
        """Interpret all ops in ``previous_ops``. This is done whenever any
        operators that haven't been interpreted that are saved to be cancelled
        no longer need to be saved."""
        ops_remaining = set(self.previous_ops.values())
        for op in ops_remaining:
            super().interpret_operation(op)

        all_wires = tuple(self.previous_ops.keys())
        for w in all_wires:
            self.previous_ops.pop(w)

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: list, *args) -> list:
        """Evaluate a jaxpr.

        Args:
            jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args (tuple[TensorLike]): The arguments for the jaxpr.

        Returns:
            list[TensorLike]: the results of the execution.

        """
        self._env = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            self._env[constvar] = const

        for eqn in jaxpr.eqns:

            custom_handler = self._primitive_registrations.get(eqn.primitive, None)
            if custom_handler:
                # Interpret any stored ops so that they are applied before the custom
                # primitive is handled
                self.interpret_all_previous_ops()
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = custom_handler(self, *invals, **eqn.params)
            elif isinstance(eqn.outvars[0].aval, qml.capture.AbstractOperator):
                outvals = self.interpret_operation_eqn(eqn)
            elif isinstance(eqn.outvars[0].aval, qml.capture.AbstractMeasurement):
                self.interpret_all_previous_ops()
                outvals = self.interpret_measurement_eqn(eqn)
            else:
                # Transform primitives don't have custom handlers, so we check for them here
                # to purge the stored ops in self.previous_ops
                if eqn.primitive.name.endswith("_transform"):
                    self.interpret_all_previous_ops
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = eqn.primitive.bind(*invals, **eqn.params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                self._env[outvar] = outval

        # The following is needed because any operations inside self.previous_ops have not yet
        # been applied. At this point, we **know** that any operations that should be cancelled
        # have been cancelled, and operations left inside self.previous_ops should be applied
        self.interpret_all_previous_ops()

        # Read the final result of the Jaxpr from the environment
        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if isinstance(outval, qml.operation.Operator):
                outvals.append(super().interpret_operation(outval))
            else:
                outvals.append(outval)
        self.cleanup()
        self._env = {}
        return outvals
