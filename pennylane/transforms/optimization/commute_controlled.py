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

from functools import lru_cache
from typing import Optional

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

from .optimization_utils import find_next_gate


@lru_cache
def _get_plxpr_commute_controlled():  # pylint: disable=missing-function-docstring,too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.operation import Operator
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

        def __init__(self, direction: Optional[str] = "right"):
            """Initialize the interpreter."""

            if direction not in ("left", "right"):
                raise ValueError("Direction for commute_controlled must be 'left' or 'right'")

            self.direction = direction
            self.op_list = []
            self._env = {}
            self.current_location = 0

        def setup(self) -> None:
            """Initialize the instance before interpreting equations."""
            self.op_list.clear()

        def cleanup(self) -> None:
            """Clean up the instance after interpreting equations."""
            self.op_list.clear()

        def _interpret_operation_left(self, op: Operator) -> list:
            """Interpret a PennyLane operation instance and push it through controlled operations as far left as possible."""

            # This function follows the same logic used in the `_commute_controlled_left` function.

            if op.basis is None or len(op.wires) != 1:
                self.current_location += 1
                self.op_list.append(op)
                return []

            prev_gate_idx = find_next_gate(op.wires, self.op_list[:][::-1])
            new_location = self.current_location

            while prev_gate_idx is not None:
                prev_gate = self.op_list[new_location - prev_gate_idx - 1]

                if prev_gate.basis is None:
                    break

                if len(prev_gate.control_wires) == 0:
                    break

                shared_controls = Wires.shared_wires([Wires(op.wires), prev_gate.control_wires])

                if len(shared_controls) > 0:
                    if op.basis == "Z":
                        new_location = new_location - prev_gate_idx - 1
                    else:
                        break

                else:
                    if op.basis == prev_gate.basis:
                        new_location = new_location - prev_gate_idx - 1
                    else:
                        break

                prev_gate_idx = find_next_gate(op.wires, self.op_list[:new_location][::-1])

            self.op_list.insert(new_location, op)
            self.current_location += 1
            return []

        def _interpret_all_operations_right(self) -> None:
            """Push all single-qubit gates as far right as possible through controlled operations."""

            # This function follows the same logic used in the `_commute_controlled_right` function.

            current_location = len(self.op_list) - 1

            while current_location >= 0:
                current_gate = self.op_list[current_location]

                if getattr(current_gate, "basis", None) is None or len(current_gate.wires) != 1:
                    current_location -= 1
                    continue

                next_gate_idx = find_next_gate(
                    current_gate.wires, self.op_list[current_location + 1 :]
                )

                new_location = current_location

                while next_gate_idx is not None:
                    next_gate = self.op_list[new_location + next_gate_idx + 1]

                    if getattr(next_gate, "basis", None) is None:
                        break

                    if len(next_gate.control_wires) == 0:
                        break
                    shared_controls = Wires.shared_wires(
                        [Wires(current_gate.wires), next_gate.control_wires]
                    )

                    if len(shared_controls) > 0:
                        if current_gate.basis == "Z":
                            new_location += next_gate_idx + 1
                        else:
                            break

                    else:
                        if current_gate.basis == next_gate.basis:
                            new_location += next_gate_idx + 1
                        else:
                            break

                    next_gate_idx = find_next_gate(
                        current_gate.wires, self.op_list[new_location + 1 :]
                    )

                self.op_list.insert(new_location + 1, current_gate)
                self.op_list.pop(current_location)
                current_location -= 1

        def interpret_operation(self, op: Operator):
            """Interpret a PennyLane operation instance."""

            if self.direction == "left":
                return self._interpret_operation_left(op)

            # If the direction is right, we simply append the operation
            # to the list while we scan through the operations forwards.
            self.op_list.append(op)
            return []

        def interpret_all_previous_ops(self) -> None:
            """Interpret all previous operations stored in the instance."""

            if self.direction == "left":
                for op in self.op_list:
                    super().interpret_operation(op)
                self.op_list.clear()

            # If the direction is right, push the gates in each sub-list
            # created at this stage by traversing the list backwards.
            self._interpret_all_operations_right()

            for op in self.op_list:
                super().interpret_operation(op)

            self.op_list.clear()

        def eval(self, jaxpr: "jax.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.
            Args:
                jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
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
                    if prim_type == "transform":
                        self.interpret_all_previous_ops()
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

    def commute_controlled_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        interpreter = CommuteControlledInterpreter()

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return CommuteControlledInterpreter, commute_controlled_plxpr_to_plxpr


CommuteControlledInterpreter, commute_controlled_plxpr_to_plxpr = _get_plxpr_commute_controlled()


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

        # We are looking only at the gates that can be pushed through
        # controls/targets; these are single-qubit gates with the basis
        # property specified.
        if getattr(current_gate, "basis", None) is None or len(current_gate.wires) != 1:
            current_location -= 1
            continue

        # Find the next gate that contains an overlapping wire
        next_gate_idx = find_next_gate(current_gate.wires, op_list[current_location + 1 :])

        new_location = current_location

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = op_list[new_location + next_gate_idx + 1]

            # Only go ahead if information is available
            if getattr(next_gate, "basis", None) is None:
                break

            # If the next gate does not have control_wires defined, it is not
            # controlled so we can't push through.
            if len(next_gate.control_wires) == 0:
                break
            shared_controls = Wires.shared_wires(
                [Wires(current_gate.wires), next_gate.control_wires]
            )

            # Case 1: overlap is on the control wires. Only Z-type gates go through
            if len(shared_controls) > 0:
                if current_gate.basis == "Z":
                    new_location += next_gate_idx + 1
                else:
                    break

            # Case 2: since we know the gates overlap somewhere, and it's a
            # single-qubit gate, if it wasn't on a control it's the target.
            else:
                if current_gate.basis == next_gate.basis:
                    new_location += next_gate_idx + 1
                else:
                    break

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

        if current_gate.basis is None or len(current_gate.wires) != 1:
            current_location += 1
            continue

        # Pass a backwards copy of the list
        prev_gate_idx = find_next_gate(current_gate.wires, op_list[:current_location][::-1])

        new_location = current_location

        while prev_gate_idx is not None:
            prev_gate = op_list[new_location - prev_gate_idx - 1]

            if prev_gate.basis is None:
                break

            if len(prev_gate.control_wires) == 0:
                break
            shared_controls = Wires.shared_wires(
                [Wires(current_gate.wires), prev_gate.control_wires]
            )

            if len(shared_controls) > 0:
                if current_gate.basis == "Z":
                    new_location = new_location - prev_gate_idx - 1
                else:
                    break

            else:
                if current_gate.basis == prev_gate.basis:
                    new_location = new_location - prev_gate_idx - 1
                else:
                    break

            prev_gate_idx = find_next_gate(current_gate.wires, op_list[:new_location][::-1])

        op_list.pop(current_location)
        op_list.insert(new_location, current_gate)
        current_location += 1

    return op_list


@transform
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
