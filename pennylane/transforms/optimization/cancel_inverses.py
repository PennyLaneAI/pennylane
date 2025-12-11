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

from functools import lru_cache, partial

from pennylane.math import is_abstract
from pennylane.operation import Operator
from pennylane.ops.op_math import Adjoint
from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn, TensorLike
from pennylane.wires import Wires

from .optimization_utils import find_next_gate


def _check_equality(items1: TensorLike | Wires, items2: TensorLike | Wires) -> bool:
    """Checks if two data objects are equal, considering abstractness."""

    for d1, d2 in zip(items1, items2, strict=True):
        if is_abstract(d1) or is_abstract(d2):
            if d1 is not d2:
                return False
        elif d1 != d2:
            return False

    return True


def _ops_equal(op1: Operator, op2: Operator) -> bool:
    """Checks if two operators are equal up to class, data, hyperparameters, and wires"""
    return (
        op1.__class__ is op2.__class__
        and _check_equality(op1.data, op2.data)
        and (op1.hyperparameters == op2.hyperparameters)
    )


def _are_inverses(op1: Operator, op2: Operator) -> bool:
    """Checks if two operators are inverses of each other

    Args:
        op1 (~.Operator)
        op2 (~.Operator)

    Returns:
        Bool
    """
    # op1 is self-inverse and the next gate is of the same type as op1
    if op1 in self_inverses and op1.name == op2.name:
        return True

    # op2 is an `Adjoint` class and its base is equal to op1
    if isinstance(op2, Adjoint) and _ops_equal(op2.base, op1):
        return True

    return False


@lru_cache
def _get_plxpr_cancel_inverses():  # pylint: disable=too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.capture.primitives import measure_prim

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
            if len(op.wires) == 0:
                return super().interpret_operation(op)

            # Throughout we will use that any pair of operators we want to cancel must act on the
            # same set of wires. We use the marker (1) to indicate that we used this fact.

            # previous operator that last acted on first wire of ``op``.
            # Only need to look at one wire for this (1).
            prev_op = self.previous_ops.get(op.wires[0], None)
            dyn_wires = {w for w in op.wires if is_abstract(w)}
            other_saved_wires = set(self.previous_ops.keys()) - dyn_wires

            if prev_op is None or (dyn_wires and other_saved_wires):
                # If there are dynamic wires, we need to make sure that there are no
                # other wires in `self.previous_ops`, otherwise we can't cancel. If
                # there are other wires but no other op on the same dynamic wire(s),
                # there isn't anything to cancel, so we just add the current op to
                # `self.previous_ops` and return.
                if dyn_wires and (prev_op is None or other_saved_wires):
                    self.interpret_all_previous_ops()
                for w in op.wires:
                    self.previous_ops[w] = op
                return []

            if _can_cancel(prev_op, op):
                # If we can cancel the previous op with the current op, we simply don't interpret
                # either and remove the previous op from `self.previous_ops`. We do not need to
                # remove ops on any other wires (1).
                for w in op.wires:
                    self.previous_ops.pop(w)
                return []

            # If we can't cancel, get all previous ops with wire overlap with `op`, interpret them,
            # and pop them. They won't cancel with any other operator as `op` blocks them (1).
            previous_ops_on_wires = list(
                dict.fromkeys(o for w in op.wires if (o := self.previous_ops.get(w)) is not None)
            )
            # pylint: disable=super-with-arguments
            res = [
                super(CancelInversesInterpreter, self).interpret_operation(o)
                for o in previous_ops_on_wires
            ]
            for o in previous_ops_on_wires:
                for w in o.wires:
                    self.previous_ops.pop(w)

            # Record `op` as last op that acted on its wires.
            for w in op.wires:
                self.previous_ops[w] = op

            return res

        def interpret_all_previous_ops(self) -> None:
            """Interpret all operators in ``previous_ops``. This is done when any previously
            uninterpreted operators, saved for cancellation, no longer need to be stored."""

            ops_remaining = list(dict.fromkeys(self.previous_ops.values()))

            for op in ops_remaining:
                super().interpret_operation(op)

            self.previous_ops.clear()

        def eval(self, jaxpr: "jax.extend.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.

            """
            # pylint: disable=attribute-defined-outside-init
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
                elif getattr(eqn.primitive, "prim_type", "") == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif getattr(eqn.primitive, "prim_type", "") == "measurement":
                    self.interpret_all_previous_ops()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    outvals = eqn.primitive.bind(*subfuns, *invals, **params)

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

    @CancelInversesInterpreter.register_primitive(measure_prim)
    def _(_, *invals, **params):
        subfuns, params = measure_prim.get_bind_params(params)
        return measure_prim.bind(*subfuns, *invals, **params)

    def cancel_inverses_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args):
        """Function for applying the ``cancel_inverses`` transform on plxpr."""
        tkwargs = dict(tkwargs)

        interpreter = CancelInversesInterpreter(*targs, **tkwargs)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return CancelInversesInterpreter, cancel_inverses_plxpr_to_plxpr


CancelInversesInterpreter, cancel_inverses_plxpr_to_plxpr = _get_plxpr_cancel_inverses()


def _num_shared_wires(wires1, wires2):
    if any(is_abstract(w) for w in [*wires1, *wires2]):
        # Rely on `id`s to check object equality instead of value equality for abstract wires
        wire_ids1 = {id(w) for w in wires1}
        wire_ids2 = {id(w) for w in wires2}
        return len(wire_ids1 & wire_ids2)
    return len(Wires.shared_wires([wires1, wires2]))


def _can_cancel(op1, op2):
    # Make sure that if one of the operators is an adjoint it is the latter
    if isinstance(op1, Adjoint):
        op1, op2 = op2, op1

    if _are_inverses(op1, op2):
        # If the wires are exactly the same, then we can safely remove both
        if _check_equality(op1.wires, op2.wires):
            return True
        # If wires are not exactly equal, they don't have full overlap, or differ by a permutation
        # 1. There is not full overlap in the wires; we cannot cancel
        if _num_shared_wires(op1.wires, op2.wires) != len(op1.wires):
            return False
        # 2. There is full overlap, but the wires are in a different order.
        # If the wires are in a different order, gates that are "symmetric"
        # over all wires (e.g., CZ), can be cancelled.
        if op1 in symmetric_over_all_wires:
            return True
        # For gates that are symmetric over controls and have a single target (e.g., Toffoli),
        # we can still cancel as long as the target wire is the same
        if op1 in symmetric_over_control_wires and _check_equality(op1.wires[-1:], op2.wires[-1:]):
            return True
    return False


def _try_to_cancel_with_next(current_gate, list_copy):
    cancelled = False
    next_gate_idx = find_next_gate(current_gate.wires, list_copy)
    # If no next gate is found: can not cancel
    if next_gate_idx is None:
        return list_copy, cancelled
    # Otherwise, get the next gate
    next_gate = list_copy[next_gate_idx]
    if _can_cancel(current_gate, next_gate):
        list_copy.pop(next_gate_idx)
        cancelled = True
    return list_copy, cancelled


@partial(
    transform,
    plxpr_transform=cancel_inverses_plxpr_to_plxpr,
    pass_name="cancel-inverses",
)
def cancel_inverses(
    tape: QuantumScript, recursive: bool = True
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverses or adjoint.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        recursive (bool): Whether or not to recursively cancel inverses after a first pair
            of mutual inverses has been cancelled. Enabled by default.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:
            The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    You can apply the cancel inverses transform directly on :class:`~.QNode`.

    >>> dev = qml.device('default.qubit', wires=3)

    .. code-block:: python

        @qml.transforms.cancel_inverses
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

    >>> print(circuit(0.1, 0.2, 0.3))
    1.0

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

        >>> optimized_qfunc = qml.transforms.cancel_inverses(qfunc)
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
        current_gate = list_copy.pop(0)

        list_copy, cancelled = _try_to_cancel_with_next(current_gate, list_copy)
        if cancelled:
            if not recursive:
                continue
            while cancelled and operations:
                list_copy, cancelled = _try_to_cancel_with_next(operations[-1], list_copy)
                if cancelled:
                    operations.pop(-1)
        else:
            operations.append(current_gate)

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
