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

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms.core import transform


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def decompose(tape, gate_set=None, max_expansion=None):
    """Decomposes a quantum circuit into the provided gate set.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_rules (set[Union[str, Operator]] or Callable[Operator, bool], optional): Decomposition gates defined either by a set of operators or a rule that they must follow.
        Defaults to None. If ``None``, gate set defaults to all available operators given by ``~.pennylane.ops.__all__``.
        max_expansion (int, optional): The maximum depth of the expansion. Defaults to None. If ``None``, circuit will be decomposed until no further decompositions are possible.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    .. seealso:: :func:`~.pennylane.devices.preprocess.decompose` for a transform that is designed particularly for gate sets designed for specific device architectures.

    **Examples:**

    >>> @partial(decompose, gate_set={qml.Toffoli, "RX", "RZ"})
    >>> @qml.qnode(dev)
    >>> def circuit():
    >>>     qml.Hadamard(wires=[0])
    >>>     qml.Toffoli(wires=[0,1,2])
    >>>     return qml.expval(qml.Z(0))
    >>>
    >>> print(qml.draw(circuit)())
    0: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭●─┤  <Z>
    1: ───────────────────────────────├●─┤
    2: ───────────────────────────────╰X─┤

    >>> @partial(decompose, gate_set=lambda op: len(op.wires) <= 2)
    >>> @qml.qnode(dev)
    >>> def circuit():
    >>>     qml.Hadamard(wires=[0])
    >>>     qml.Toffoli(wires=[0,1,2])
    >>>     return qml.expval(qml.Z(0))
    >>>
    >>> print(qml.draw(circuit)())
    0: ──H────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

    >>> tape = qml.tape.QuantumScript([qml.IsingXX(1.2, wires=(0,1))], [qml.expval(qml.Z(0))])
    >>> batch, fn = qml.transforms.decompose(tape, gate_set={"CNOT", "RX", "RZ"})
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]), RX(1.2, wires=[0]), CNOT(wires=[0, 1]), expval(Z(0))]

    """

    if gate_set is None:
        gate_set = set(qml.ops.__all__)

    if isinstance(gate_set, (str, type)):
        gate_set = set([gate_set])

    if isinstance(gate_set, (list, tuple)):
        gate_set = set(gate_set)

    if isinstance(gate_set, set):
        gate_types = tuple(gate for gate in gate_set if isinstance(gate, type))
        gate_names = set(gate for gate in gate_set if isinstance(gate, str))
        gate_set = lambda op: (op.name in gate_names) or isinstance(op, gate_types)

    def decomposer(op):
        return op.decomposition()

    def stopping_condition(op):
        if not op.has_decomposition:
            if not gate_set(op):
                warnings.warn(
                    f"Operator {op.name} has no supported decomposition and was not found in the set of allowed decomposition gates."
                    f"To remove this warning, add the operator name ({op.name}) or type ({type(op)}) to the allowed set of gates.",
                    UserWarning,
                )
            return True
        return gate_set(op)

    if all(stopping_condition(op) for op in tape.operations):
        return (tape,), null_postprocessing

    try:
        new_ops = [
            final_op
            for op in tape.operations
            for final_op in qml.devices.preprocess._operator_decomposition_gen(
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
