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
# pylint: disable=protected-access
# pylint: disable=unnecessary-lambda-assignment

import warnings
from collections.abc import Callable, Generator
from typing import Iterable, Optional

import pennylane as qml
from pennylane.transforms.core import transform


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def _operator_decomposition_gen(
    op: qml.operation.Operator,
    acceptance_function: Callable[[qml.operation.Operator], bool],
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
        decomp = op.decomposition()
        current_depth += 1

        for sub_op in decomp:
            yield from _operator_decomposition_gen(
                sub_op,
                acceptance_function,
                max_expansion=max_expansion,
                current_depth=current_depth,
            )


@transform
def decompose(tape, gate_set=None, max_expansion=None):
    """Decomposes a quantum circuit into a user-specified gate set.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_set (Iterable[Union[str, type]] or Callable[Operator, bool], optional): Decomposition gates defined by either (1) a gate set of operators or (2) a rule that they must follow.
            Defaults to None. If ``None``, gate set defaults to all available :doc:`quantum operators </introduction/operations>`.
        max_expansion (int, optional): The maximum depth of the decomposition. Defaults to None. If ``None``, the circuit will be decomposed until the target gate set is reached.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    .. note::

        This function does not guarantee a decomposition to the target gate set. During the decomposition, if an unsupported operation is encountered
        the function will leave it in the circuit and raise a ``UserWarning`` indicating no defined decomposition. To waive this warning, simply add the operator
        to the defined gate set.

    .. seealso:: :func:`qml.devices.preprocess.decompose <.pennylane.devices.preprocess.decompose>` for a transform that is intended for device developers. This function will decompose a quantum circuit into a set of basis gates available on a specific device architecture.

    **Example**

    Consider the following tape:

    >>> ops = [qml.IsingXX(1.2, wires=(0,1))]
    >>> tape = qml.tape.QuantumScript(ops, measurements=[qml.expval(qml.Z(0))])

    You can then decompose the circuit into a set of gates:

    >>> batch, fn = qml.transforms.decompose(tape, gate_set={qml.CNOT, qml.RX})
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]), RX(1.2, wires=[0]), CNOT(wires=[0, 1]), expval(Z(0))]

    You can also apply the transform directly on a :class:`~.pennylane.QNode`:

    .. code-block:: python

        from functools import partial

        @partial(qml.transforms.decompose, gate_set={qml.Toffoli, "RX", "RZ"})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(wires=[0])
            qml.Toffoli(wires=[0,1,2])
            return qml.expval(qml.Z(0))

    Since the Hadamard gate is not defined in our gate set, it will be decomposed into rotations:

    >>> print(qml.draw(circuit)())
    0: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭●─┤  <Z>
    1: ───────────────────────────────├●─┤
    2: ───────────────────────────────╰X─┤

    You can also use callable functions to build a decomposition gate set:

    .. code-block:: python

        @partial(qml.transforms.decompose, gate_set=lambda op: len(op.wires)<=2)
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(wires=[0])
            qml.Toffoli(wires=[0,1,2])
            return qml.expval(qml.Z(0))

    The circuit will be decomposed into single or two-qubit operators,

    >>> print(qml.draw(circuit)())
    0: ──H────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

    You can use the ``max_expansion`` kwarg to have control over the number
    of decomposition stages applied to the circuit. By default, the function will decompose
    the circuit until the desired gate set is reached.

    The example below demonstrates how the user can visualize the decomposition in stages:

    .. code-block:: python

        phase = 1
        target_wires = [0]
        unitary = qml.RX(phase, wires=0).matrix()
        n_estimation_wires = 3
        estimation_wires = range(1, n_estimation_wires + 1)

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            # Start in the |+> eigenstate of the unitary
            qml.Hadamard(wires=target_wires)
            qml.QuantumPhaseEstimation(
                unitary,
                target_wires=target_wires,
                estimation_wires=estimation_wires,
            )

    >>> print(qml.draw(qml.transforms.decompose(circuit, max_expansion=0))())
    0: ──H─╭QuantumPhaseEstimation─┤
    1: ────├QuantumPhaseEstimation─┤
    2: ────├QuantumPhaseEstimation─┤
    3: ────╰QuantumPhaseEstimation─┤

    >>> print(qml.draw(qml.transforms.decompose(circuit, max_expansion=1))())
    0: ──H─╭U(M0)⁴─╭U(M0)²─╭U(M0)¹───────┤
    1: ──H─╰●──────│───────│───────╭QFT†─┤
    2: ──H─────────╰●──────│───────├QFT†─┤
    3: ──H─────────────────╰●──────╰QFT†─┤

    >>> print(qml.draw(qml.transforms.decompose(circuit, max_expansion=2))())
    0: ──H──RZ(11.00)──RY(1.14)─╭X──RY(-1.14)──RZ(-9.42)─╭X──RZ(-1.57)──RZ(1.57)──RY(1.00)─╭X──RY(-1.00)
    1: ──H──────────────────────╰●───────────────────────╰●────────────────────────────────│────────────
    2: ──H─────────────────────────────────────────────────────────────────────────────────╰●───────────
    3: ──H──────────────────────────────────────────────────────────────────────────────────────────────
    ───RZ(-6.28)─╭X──RZ(4.71)──RZ(1.57)──RY(0.50)─╭X──RY(-0.50)──RZ(-6.28)─╭X──RZ(4.71)─────────────────
    ─────────────│────────────────────────────────│────────────────────────│──╭SWAP†────────────────────
    ─────────────╰●───────────────────────────────│────────────────────────│──│─────────────╭(Rϕ(1.57))†
    ──────────────────────────────────────────────╰●───────────────────────╰●─╰SWAP†─────H†─╰●──────────
    ────────────────────────────────────┤
    ──────╭(Rϕ(0.79))†─╭(Rϕ(1.57))†──H†─┤
    ───H†─│────────────╰●───────────────┤
    ──────╰●────────────────────────────┤
    """

    if gate_set is None:
        gate_set = set(qml.ops.__all__)

    if isinstance(gate_set, (str, type)):
        gate_set = set([gate_set])

    if isinstance(gate_set, Iterable):
        gate_types = tuple(gate for gate in gate_set if isinstance(gate, type))
        gate_names = set(gate for gate in gate_set if isinstance(gate, str))
        gate_set = lambda op: (op.name in gate_names) or isinstance(op, gate_types)

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
            for final_op in _operator_decomposition_gen(
                op,
                stopping_condition,
                max_expansion=max_expansion,
            )
        ]
    except RecursionError as e:
        raise RecursionError(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
            "Setting ``max_expansion`` will terminate the decomposition after a set number."
        ) from e

    tape = tape.copy(operations=new_ops)

    return (tape,), null_postprocessing
