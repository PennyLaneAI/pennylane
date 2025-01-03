# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
# pylint: disable=too-many-branches
from functools import lru_cache, partial

from pennylane.ops.op_math import Adjoint
from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

from .optimization_utils import find_next_gate


def _can_cancel_ops(op1, op2):
    """Checks if two operators can be cancelled

    Args:
        op1 (~.Operator)
        op2 (~.Operator)

    Returns:
        Bool
    """
    # Make sure that if one of the ops is Adjoint, it is always op2 by swapping
    # the ops if op1 is Adjoint
    if isinstance(op1, Adjoint):
        op1, op2 = op2, op1

    are_self_inverses_without_wires = op1 in self_inverses and op1.name == op2.name
    are_inverses_without_wires = (
        isinstance(op2, Adjoint)
        and op1.__class__ == op2.base.__class__
        and op1.data == op2.base.data
        and op1.hyperparameters == op2.base.hyperparameters
    )
    if are_self_inverses_without_wires or are_inverses_without_wires:

        # If the wires are the same, then we can safely cancel both
        if op1.wires == op2.wires:
            return True
        # If wires are not equal, there are two things that can happen.
        # 1. There is not full overlap in the wires; we cannot cancel
        if len(Wires.shared_wires([op1.wires, op2.wires])) != len(op1.wires):
            return False

        # 2. There is full overlap, but the wires are in a different order.
        # If the wires are in a different order, gates that are "symmetric"
        # over all wires (e.g., CZ), can be cancelled.
        if op1 in symmetric_over_all_wires:
            return True
        # For other gates, as long as the control wires are the same, we can still
        # cancel (e.g., the Toffoli gate).
        if op1 in symmetric_over_control_wires:
            # TODO[David Wierichs]: This assumes single-qubit targets of controlled gates
            if len(Wires.shared_wires([op1.wires[:-1], op2.wires[:-1]])) == len(op1.wires) - 1:
                return True

    return False


@lru_cache
def _get_plxpr_cancel_inverses():  # pylint: disable=missing-function-docstring,too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import AbstractMeasurement, AbstractOperator, PlxprInterpreter
        from pennylane.operation import Operator
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class CancelInversesInterpreter(PlxprInterpreter):
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

        def interpret_operation(self, op: Operator):
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

            cancel = _can_cancel_ops(op, prev_op)
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
            """Interpret all operators in ``previous_ops``. This is done when any previously
            uninterpreted operators, saved for cancellation, no longer need to be stored."""
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
            # pylint: disable=too-many-branches,attribute-defined-outside-init
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
                elif len(eqn.outvars) > 0 and isinstance(eqn.outvars[0].aval, AbstractOperator):
                    outvals = self.interpret_operation_eqn(eqn)
                elif len(eqn.outvars) > 0 and isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                    self.interpret_all_previous_ops()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    # Transform primitives don't have custom handlers, so we check for them here
                    # to purge the stored ops in self.previous_ops
                    if eqn.primitive.name.endswith("_transform"):
                        self.interpret_all_previous_ops()
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
                if isinstance(outval, Operator):
                    outvals.append(super().interpret_operation(outval))
                else:
                    outvals.append(outval)
            self.cleanup()
            self._env = {}
            return outvals

    def cancel_inverses_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        interpreter = CancelInversesInterpreter()

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return CancelInversesInterpreter, cancel_inverses_plxpr_to_plxpr


CancelInversesInterpreter, cancel_inverses_plxpr_to_plxpr = _get_plxpr_cancel_inverses()


@partial(transform, plxpr_transform=cancel_inverses_plxpr_to_plxpr)
def cancel_inverses(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverses or adjoint.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    You can apply the cancel inverses transform directly on :class:`~.QNode`.

    >>> dev = qml.device('default.qubit', wires=3)

    .. code-block:: python

        @cancel_inverses
        @qml.qnode(device=dev)
        def circuit(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=2)
            qml.RY(y, wires=1)
            qml.X(1)
            qml.RZ(z, wires=0)
            qml.RX(y, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.X(1)
            return qml.expval(qml.Z(0))

    >>> circuit(0.1, 0.2, 0.3)
    0.999999999999999

    .. details::
        :title: Usage Details

        You can also apply it on quantum functions:

        .. code-block:: python

            def qfunc(x, y, z):
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=1)
                qml.Hadamard(wires=0)
                qml.RX(x, wires=2)
                qml.RY(y, wires=1)
                qml.X(1)
                qml.RZ(z, wires=0)
                qml.RX(y, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.X(1)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(1, 2, 3))
        0: ──H─────────H─────────RZ(3.00)─╭●────┤  <Z>
        1: ──H─────────RY(2.00)──X────────│───X─┤
        2: ──RX(1.00)──RX(2.00)───────────╰X────┤

        We can see that there are two adjacent Hadamards on the first qubit that
        should cancel each other out. Similarly, there are two Pauli-X gates on the
        second qubit that should cancel. We can obtain a simplified circuit by running
        the ``cancel_inverses`` transform:

        >>> optimized_qfunc = cancel_inverses(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RZ(3.00)───────────╭●─┤  <Z>
        1: ──H─────────RY(2.00)─│──┤
        2: ──RX(1.00)──RX(2.00)─╰X─┤

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    operations = []

    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)

        # Find the next gate that acts on at least one of the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy)

        # If no such gate is found queue the operation and move on
        if next_gate_idx is None:
            operations.append(current_gate)
            continue

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx]

        # If operators are inverses, cancel
        if _can_cancel_ops(current_gate, next_gate):
            list_copy.pop(next_gate_idx)
            continue
        # Apply gate any cases where
        # - there is no wire symmetry
        # - the control wire symmetry does not apply because the control wires are not the same
        # - neither of the flags are_self_inverses and are_inverses are true
        operations.append(current_gate)

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
