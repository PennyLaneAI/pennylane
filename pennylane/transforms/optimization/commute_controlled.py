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
"""Transforms for pushing commuting gates through targets/control qubits."""

from collections import deque
from collections.abc import Sequence
from functools import lru_cache, partial
from itertools import islice

from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

from .optimization_utils import find_next_gate


@lru_cache
def _get_plxpr_commute_controlled():  # pylint: disable=too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.capture.primitives import measure_prim
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name
    class CommuteControlledInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for applying the ``commute_controlled`` transform to callables or jaxpr
        when program capture is enabled.

        .. note::
            If the direction is set to ``"right"``, this class interprets the operations by scanning them
            backward after the jaxpr has been traversed (pushing the gates to the right of controlled gates).
            This is because we can only traverse the jaxpr in a forward direction with the current implementation when
            program capture is enabled.

            This is less efficient than setting the direction to ``"left"``, which allows the interpreter
            to push the gates to the left of controlled gates as it interprets the jaxpr.

            Despite this, the default direction is ``"right"`` to maintain compatibility with the
            current default value of the transform.
        """

        def __init__(self, direction="right"):
            """Initialize the interpreter."""

            if direction not in ("left", "right"):
                raise ValueError(
                    f"Direction for commute_controlled must be 'left' or 'right'. Got {direction}"
                )

            self.direction = direction
            self.op_deque = deque()
            self._env = {}
            self.current_index = 0

        def cleanup(self) -> None:
            """Clean up the instance after interpreting equations."""
            self.op_deque.clear()

        def _interpret_operation_left(self, op: Operator) -> list:
            """Interpret a PennyLane operation and push it through controlled operations as far left as possible."""

            # This function follows the same logic used in the `_commute_controlled_left` function.

            if not _can_be_pushed_through(op):
                self.current_index += 1
                self.op_deque.append(op)
                return []

            prev_gate_idx = _find_previous_gate_on_wires(op.wires, self.op_deque)
            new_index = self.current_index

            while prev_gate_idx is not None:
                prev_gate = self.op_deque[new_index - (prev_gate_idx + 1)]

                if not _can_push_through(prev_gate) or not _can_commute(op, prev_gate):
                    break

                new_index -= prev_gate_idx + 1

                prev_gate_idx = _find_previous_gate_on_wires(
                    op.wires, tuple(islice(self.op_deque, new_index))
                )

            self.op_deque.insert(new_index, op)
            self.current_index += 1
            return []

        def _interpret_all_operations_right(self) -> None:
            """Push all single-qubit gates as far right as possible through controlled operations."""

            # This function follows the same logic used in the `_commute_controlled_right` function.

            self.current_index = len(self.op_deque) - 1

            while self.current_index >= 0:
                current_gate = self.op_deque[self.current_index]

                if not _can_be_pushed_through(current_gate):
                    self.current_index -= 1
                    continue

                next_gate_idx = find_next_gate(
                    current_gate.wires,
                    tuple(islice(self.op_deque, self.current_index + 1, len(self.op_deque))),
                )

                new_index = self.current_index

                while next_gate_idx is not None:
                    next_gate = self.op_deque[new_index + next_gate_idx + 1]

                    if not _can_push_through(next_gate) or not _can_commute(
                        current_gate, next_gate
                    ):
                        break

                    new_index += next_gate_idx + 1

                    next_gate_idx = find_next_gate(
                        current_gate.wires,
                        tuple(islice(self.op_deque, new_index + 1, len(self.op_deque))),
                    )

                self.op_deque.insert(new_index + 1, current_gate)
                del self.op_deque[self.current_index]
                self.current_index -= 1

        def interpret_operation(self, op: Operator):
            """Interpret a PennyLane operation instance."""

            if self.direction == "left":
                return self._interpret_operation_left(op)

            # If the direction is right, we append the operator
            # to the list while we scan through the operators forwards.
            self.op_deque.append(op)
            return []

        def interpret_all_previous_ops(self) -> None:
            """Interpret all previous operations stored in the instance."""

            if self.direction == "left":
                for op in self.op_deque:
                    super().interpret_operation(op)
                self.op_deque.clear()
                self.current_index = 0
                return

            # If the direction is right, push the gates in each sub-list
            # created at this stage by traversing it backwards.
            self._interpret_all_operations_right()

            for op in self.op_deque:
                super().interpret_operation(op)
            self.op_deque.clear()

        def eval(self, jaxpr: "jax.extend.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.
            """

            self.setup()
            self._env = {}

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:

                prim_type = getattr(eqn.primitive, "prim_type", "")

                custom_handler = self._primitive_registrations.get(eqn.primitive, None)
                if custom_handler:
                    self.interpret_all_previous_ops()
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif prim_type == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif prim_type == "measurement":
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

            self.interpret_all_previous_ops()

            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, Operator):
                    outvals.append(super().interpret_operation(outval))
                else:
                    outvals.append(outval)

            self._env = {}
            self.cleanup()
            return outvals

    @CommuteControlledInterpreter.register_primitive(measure_prim)
    def _(_, *invals, **params):
        _, params = measure_prim.get_bind_params(params)
        return measure_prim.bind(*invals, **params)

    def commute_controlled_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        interpreter = CommuteControlledInterpreter(direction=tkwargs.get("direction", "right"))

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return CommuteControlledInterpreter, commute_controlled_plxpr_to_plxpr


CommuteControlledInterpreter, commute_controlled_plxpr_to_plxpr = _get_plxpr_commute_controlled()


def _find_previous_gate_on_wires(wires: Wires, prevs_ops: Sequence) -> int | None:
    """Finds the previous gate index that shares wires."""

    return find_next_gate(wires, reversed(prevs_ops))


def _shares_control_wires(op: Operator, ctrl_gate: Operator) -> bool:
    """Check if the operation shares wires with the control wires of the provided controlled gate."""

    return len(Wires.shared_wires([Wires(op.wires), ctrl_gate.control_wires])) > 0


def _can_commute(op1: Operator, op2: Operator) -> bool:
    """Helper that determines if op1 can commute with a single-qubit gate op2 based on their basis and control wires."""

    # Case 1: overlap is on the control wires. Only Z-type gates go through
    if _shares_control_wires(op1, op2):
        return op1.basis == "Z"

    # Case 2: since we know the gates overlap somewhere, and it's a
    # single-qubit gate, if it wasn't on a control it's the target.
    return op1.basis == op2.basis


def _can_push_through(op: Operator) -> bool:
    """Check if the provided gate can be pushed through."""

    # Only go ahead if information is available.
    # If the gate does not have control_wires defined, it is not
    # controlled so we won't push through.
    return hasattr(op, "basis") and len(op.control_wires) > 0


def _can_be_pushed_through(op: Operator) -> bool:
    """Check if the provided gate is a single-qubit gate that can be pushed through."""

    # We are looking only at the gates that can be pushed through
    # controls/targets; these are single-qubit gates with the basis
    # property specified.
    return hasattr(op, "basis") and len(op.wires) == 1


def _commute_controlled_right(op_list):
    """Push commuting single qubit gates to the right of controlled gates.

    Args:
        op_list (list[Operation]): The initial list of operations.

    Returns:
        list[Operation]: The modified list of operations with all single-qubit
        gates as far right as possible.
    """
    # We will go through the list backwards; whenever we find a single-qubit
    # gate, we will extract it and push it through 2-qubit gates as far as
    # possible to the right.
    current_location = len(op_list) - 1

    while current_location >= 0:
        current_gate = op_list[current_location]

        if not _can_be_pushed_through(current_gate):
            current_location -= 1
            continue

        # Find the next gate that contains an overlapping wire
        next_gate_idx = find_next_gate(current_gate.wires, op_list[current_location + 1 :])

        new_location = current_location

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = op_list[new_location + next_gate_idx + 1]

            if not _can_push_through(next_gate) or not _can_commute(current_gate, next_gate):
                break

            new_location += next_gate_idx + 1

            next_gate_idx = find_next_gate(current_gate.wires, op_list[new_location + 1 :])

        # After we have gone as far as possible, move the gate to new location
        op_list.insert(new_location + 1, current_gate)
        op_list.pop(current_location)
        current_location -= 1

    return op_list


def _commute_controlled_left(op_list):
    """Push commuting single qubit gates to the left of controlled gates.

    Args:
        op_list (list[Operation]): The initial list of operations.

    Returns:
        list[Operation]: The modified list of operations with all single-qubit
        gates as far left as possible.
    """
    # We will go through the list forwards; whenever we find a single-qubit
    # gate, we will extract it and push it through 2-qubit gates as far as
    # possible back to the left.
    current_location = 0

    while current_location < len(op_list):
        current_gate = op_list[current_location]

        if not _can_be_pushed_through(current_gate):
            current_location += 1
            continue

        # Pass a backwards copy of the list
        prev_gate_idx = find_next_gate(current_gate.wires, op_list[:current_location][::-1])

        new_location = current_location

        while prev_gate_idx is not None:
            prev_gate = op_list[new_location - prev_gate_idx - 1]

            if not _can_push_through(prev_gate) or not _can_commute(current_gate, prev_gate):
                break

            new_location -= prev_gate_idx + 1

            prev_gate_idx = find_next_gate(current_gate.wires, op_list[:new_location][::-1])

        op_list.pop(current_location)
        op_list.insert(new_location, current_gate)
        current_location += 1

    return op_list


@partial(transform, plxpr_transform=commute_controlled_plxpr_to_plxpr)
def commute_controlled(
    tape: QuantumScript, direction="right"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to move commuting gates past control and target qubits of controlled operations.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        direction (str): The direction in which to move single-qubit gates.
            Options are "right" (default), or "left". Single-qubit gates will
            be pushed through controlled operations as far as possible in the
            specified direction.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        @partial(commute_controlled, direction="right")
        @qml.qnode(device=dev)
        def circuit(theta):
            qml.CZ(wires=[0, 2])
            qml.X(2)
            qml.S(wires=0)

            qml.CNOT(wires=[0, 1])

            qml.Y(1)
            qml.CRY(theta, wires=[0, 1])
            qml.PhaseShift(theta/2, wires=0)

            qml.Toffoli(wires=[0, 1, 2])
            qml.T(wires=0)
            qml.RZ(theta/2, wires=1)

            return qml.expval(qml.Z(0))

    >>> circuit(0.5)
    0.9999999999999999

    .. details::
        :title: Usage Details

        You can also apply it on quantum function.

        .. code-block:: python

            def qfunc(theta):
                qml.CZ(wires=[0, 2])
                qml.X(2)
                qml.S(wires=0)

                qml.CNOT(wires=[0, 1])

                qml.Y(1)
                qml.CRY(theta, wires=[0, 1])
                qml.PhaseShift(theta/2, wires=0)

                qml.Toffoli(wires=[0, 1, 2])
                qml.T(wires=0)
                qml.RZ(theta/2, wires=1)

                return qml.expval(qml.Z(0))

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(0.5))
        0: ─╭●──S─╭●────╭●─────────Rϕ(0.25)─╭●──T────────┤  <Z>
        1: ─│─────╰X──Y─╰RY(0.50)───────────├●──RZ(0.25)─┤
        2: ─╰Z──X───────────────────────────╰X───────────┤

        Diagonal gates on either side of control qubits do not affect the outcome
        of controlled gates; thus we can push all the single-qubit gates on the
        first qubit together on the right (and fuse them if desired). Similarly, X
        gates commute with the target of ``CNOT`` and ``Toffoli`` (and ``PauliY``
        with ``CRY``). We can use the transform to push single-qubit gates as
        far as possible through the controlled operations:

        >>> optimized_qfunc = commute_controlled(qfunc, direction="right")
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(0.5))
        0: ─╭●─╭●─╭●───────────╭●──S─────────Rϕ(0.25)──T─┤  <Z>
        1: ─│──╰X─╰RY(0.50)──Y─├●──RZ(0.25)──────────────┤
        2: ─╰Z─────────────────╰X──X─────────────────────┤

    """
    if direction not in ("left", "right"):
        raise ValueError("Direction for commute_controlled must be 'left' or 'right'")

    if direction == "right":
        op_list = _commute_controlled_right(tape.operations)
    else:
        op_list = _commute_controlled_left(tape.operations)

    new_tape = tape.copy(operations=op_list)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
