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

from functools import partial

from pennylane.core.operator import Operator
from pennylane.core.qscript import QuantumScript, QuantumScriptBatch
from pennylane.math import is_abstract
from pennylane.ops.op_math import Adjoint
from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)
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


@partial(transform, pass_name="cancel-inverses")
def cancel_inverses(
    tape: QuantumScript, recursive: bool = True
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverses or adjoint.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit (QNode or quantum function).
        recursive (bool): Whether or not to recursively cancel inverses after a first pair of mutual inverses has been cancelled. Enabled by default.

            .. note::
                This argument is not supported within a :func:`~.qjit` workflow.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:
            The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.


    **Example**


    You can apply it on quantum functions:

    .. code-block:: python

        def qfunc(x, y, z):
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)
            qp.Hadamard(wires=0)
            qp.RX(x, wires=2)
            qp.RY(y, wires=1)
            qp.X(1)
            qp.RZ(z, wires=0)
            qp.RX(y, wires=2)
            qp.CNOT(wires=[0, 2])
            qp.X(1)
            return qp.expval(qp.Z(0))

    The circuit before optimization:

    >>> dev = qp.device("default.qubit")
    >>> qnode = qp.QNode(qfunc, dev)
    >>> print(qp.draw(qnode)(1, 2, 3))
    0: ──H─────────H─────────RZ(3.00)─╭●────┤  <Z>
    1: ──H─────────RY(2.00)──X────────│───X─┤
    2: ──RX(1.00)──RX(2.00)───────────╰X────┤

    We can see that there are two adjacent Hadamards on the first qubit that
    should cancel each other out. Similarly, there are two ``X`` gates on the
    second qubit that should cancel. We can obtain a simplified circuit by running
    the ``cancel_inverses`` transform:

    >>> optimized_qnode = qp.transforms.cancel_inverses(qnode)
    >>> print(qp.draw(optimized_qnode)(1, 2, 3))
    0: ──RZ(3.00)───────────╭●─┤  <Z>
    1: ──H─────────RY(2.00)─│──┤
    2: ──RX(1.00)──RX(2.00)─╰X─┤

    .. details::
        :title: Usage with qjit

        There are three key differences to note when using ``cancel_inverses`` with ``qjit``:

        * ``cancel_inverses`` must be applied to a QNode. Quantum functions are not supported as input.

        * The ``recursive`` argument is not supported, and an error will be raised if a value for ``recursive`` is specified.

        * Only the following gates can be optimized by ``cancel_inverses`` with ``qjit``:

          - :class:`qp.Hadamard <pennylane.Hadamard>`,
          - :class:`qp.PauliX <pennylane.PauliX>`,
          - :class:`qp.PauliY <pennylane.PauliY>`,
          - :class:`qp.PauliZ <pennylane.PauliZ>`
          - :class:`qp.CNOT <pennylane.CNOT>`,
          - :class:`qp.CY <pennylane.CY>`,
          - :class:`qp.CZ <pennylane.CZ>`,
          - :class:`qp.SWAP <pennylane.SWAP>`
          - :class:`qp.Toffoli <pennylane.Toffoli>`

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=1)

            @qp.qjit(capture=True)
            @qp.transforms.cancel_inverses
            @qp.qnode(dev)
            def circuit():
                qp.RX(0.1, wires=0)
                qp.Hadamard(wires=0)
                qp.Hadamard(wires=0)
                return qp.expval(qp.PauliZ(0))

        >>> print(qp.specs(circuit, level=1)())
        Device: lightning.qubit
        Device wires: 1
        Shots: Shots(total=None)
        Level: cancel-inverses
        <BLANKLINE>
        Quantum operations:
        - Total: 1
          - RX: 1
        Measurement processes:
        - expval(PauliZ): 1
        Wire allocations: 1
        Circuit Depth: Not computed

        Additionally, the ``cancel_inverses`` transform with ``qjit`` supports
        `loop-boundary optimization <https://pennylane.ai/compilation/loop-boundary-optimization>`_.

        For more technical information on how this transform behaves, consult the Catalyst
        documentation for :func:`catalyst.passes.cancel_inverses`.

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
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
