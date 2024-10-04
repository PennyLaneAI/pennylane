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
"""
A transform for decomposing quantum circuits into user defined gate sets. Offers an alternative to the more device-focused decompose transform.
"""
# pylint: disable=protected-access, too-many-arguments

import warnings
from collections.abc import Callable, Generator, Sequence
from typing import Optional

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms.core import transform


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def _operator_decomposition_gen(
    op: qml.operation.Operator,
    acceptance_function: Callable[[qml.operation.Operator], bool],
    decomposer: Callable[[qml.operation.Operator], Sequence[qml.operation.Operator]],
    max_expansion: Optional[int] = None,
    current_depth=0,
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted."""
    max_depth_reached = False
    if max_expansion is not None and max_expansion <= current_depth:
        max_depth_reached = True
    if acceptance_function(op) or max_depth_reached:
        yield op
    else:
        try:
            decomp = decomposer(op)
            current_depth += 1
        except qml.operation.DecompositionUndefinedError as e:
            raise UserWarning(
                f"Operator {op.name} has no supported decomposition and is not in the gate set."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(
                sub_op,
                acceptance_function,
                decomposer=decomposer,
                max_expansion=max_expansion,
                current_depth=current_depth,
            )


@transform
def apply_decomposition(tape, gate_set=None, gate_rules=None, max_expansion=None):
    """Decomposes quantum circuit into a desired gate set.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_set (set, optional): A set of decomposition gates. Defaults to None. If ``None``, gate set defaults to all operators not currently present.
        gate_rules (Callable[qml.operation.Operator, bool], optional): A rule set for the gate set to follow. Defaults to None.
        max_expansion (int, optional): The maximum depth of the expansion. Defaults to None.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    .. seealso:: :func:`~.pennylane.devices.preprocess.decompose` for a transform that is designed particularly for restricted gate sets on specific device architectures.

    **Examples:**

    >>> tape = qml.tape.QuantumScript([qml.IsingXX(1.2, wires=(0,1))], [qml.expval(qml.Z(0))])
    >>> batch, fn = apply_decomposition(tape, gate_set = {"CNOT", "RX", "RZ"})
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]),
    RX(1.2, wires=[0]),
    CNOT(wires=[0, 1]),
    expval(Z(0))]

    >>> @partial(apply_decomposition, gate_rules = lambda obj: len(obj.wires) <= 2)
    >>> @qml.qnode(device)
    >>> def circuit():
    >>>     qml.Toffoli(wires = range(3))
    >>>
    >>> print(qml.draw(circuit)())
    0: ───────────╭●───────────╭●────╭●──T──╭●─┤
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

    """
    universal_ops = set(qml.ops.__all__)

    if gate_set is None:
        gate_set = universal_ops - set(op.name for op in tape.operations)

    if gate_rules is None:

        def gate_rules(op):
            return not op.has_matrix

    if isinstance(gate_set, str):
        gate_set = set([gate_set])

    if isinstance(gate_set, (list, tuple)):
        gate_set = set(gate_set)

    def decomposer(op):
        return op.decomposition()

    def stopping_condition(op):
        if not op.has_decomposition:
            if op.name not in gate_set:
                warnings.warn(
                    f"Operator {op.name} has no supported decomposition and was not found in the gate set.",
                    UserWarning,
                )
            return True
        return (op.name in gate_set) or gate_rules(op)

    if all(stopping_condition(op) for op in tape.operations):
        return (tape,), null_postprocessing

    try:
        new_ops = [
            final_op
            for op in tape.operations
            for final_op in _operator_decomposition_gen(
                op,
                stopping_condition,
                decomposer,
                max_expansion=max_expansion,
            )
        ]
    except RecursionError as e:
        raise RecursionError(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
            "Setting ``max_expansion`` will terminate the decomposition after a set number."
        ) from e

    tape = QuantumScript(new_ops, tape.measurements, shots=tape.shots)

    return (tape,), null_postprocessing
