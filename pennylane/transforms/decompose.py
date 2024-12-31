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
from collections.abc import Callable, Generator, Iterable
from functools import lru_cache, partial
from typing import Optional

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


@lru_cache
def _get_plxpr_decompose():  # pylint: disable=missing-docstring
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture.primitives import ctrl_transform_prim
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class DecomposeInterpreter(qml.capture.PlxprInterpreter):
        """Plxpr Interpreter for applying the ``decompose`` transform to callables or jaxpr
        when program capture is enabled.
        """

        def __init__(self, gate_set=None, max_expansion=None):
            self.max_expansion = max_expansion

            if gate_set is None:
                gate_set = set(qml.ops.__all__)

            if isinstance(gate_set, (str, type)):
                gate_set = set([gate_set])

            if isinstance(gate_set, Iterable):
                gate_types = tuple(gate for gate in gate_set if isinstance(gate, type))
                gate_names = set(gate for gate in gate_set if isinstance(gate, str))
                self.gate_set = lambda op: (op.name in gate_names) or isinstance(op, gate_types)
            else:
                self.gate_set = gate_set

            super().__init__()

        def stopping_condition(self, op: qml.operation.Operator) -> bool:
            """Function to determine whether or not an operator needs to be decomposed or not.

            Args:
                op (qml.operation.Operator): Operator to check.

            Returns:
                bool: Whether or not ``op`` is valid or needs to be decomposed. ``True`` means
                that the operator does not need to be decomposed.
            """
            if not op.has_decomposition:
                if not self.gate_set(op):
                    warnings.warn(
                        f"Operator {op.name} does not define a decomposition and was not "
                        f"found in the target gate set. To remove this warning, add the operator name "
                        f"({op.name}) or type ({type(op)}) to the gate set.",
                        UserWarning,
                    )
                return True
            return self.gate_set(op)

        def decompose_operation(self, op: qml.operation.Operator):
            """Decompose a PennyLane operation instance if it does not satisfy the
            provided gate set.

            Args:
                op (Operator): a pennylane operator instance

            Returns:
                Any

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`, :meth:`~.interpret_operation`.
            """
            if self.gate_set(op):
                return self.interpret_operation(op)

            qml.capture.disable()
            try:
                decomposition = list(
                    _operator_decomposition_gen(
                        op, self.stopping_condition, max_expansion=self.max_expansion
                    )
                )
            finally:
                qml.capture.enable()

            return [self.interpret_operation(decomp_op) for decomp_op in decomposition]

        def interpret_operation_eqn(self, eqn):
            """Interpret an equation corresponding to an operator.

            Args:
                eqn (jax.core.JaxprEqn): a jax equation for an operator.

            See also: :meth:`~.interpret_operation`.

            """
            invals = (self.read(invar) for invar in eqn.invars)
            with qml.QueuingManager.stop_recording():
                op = eqn.primitive.impl(*invals, **eqn.params)
            if eqn.outvars[0].__class__.__name__ == "DropVar":
                return self.decompose_operation(op)
            return op

    # pylint: disable=unused-variable,missing-function-docstring
    @DecomposeInterpreter.register_primitive(ctrl_transform_prim)
    def handle_ctrl_transform(*_, **__):
        raise NotImplementedError

    def decompose_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        """Function from decomposing jaxpr."""
        decomposer = DecomposeInterpreter(
            gate_set=tkwargs.pop("gate_set", None), max_expansion=tkwargs.pop("max_expansion", None)
        )

        def wrapper(*inner_args):
            return decomposer.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return DecomposeInterpreter, decompose_plxpr_to_plxpr


DecomposeInterpreter, decompose_plxpr_to_plxpr = _get_plxpr_decompose()


@partial(transform, plxpr_transform=decompose_plxpr_to_plxpr)
def decompose(tape, gate_set=None, max_expansion=None):
    """Decomposes a quantum circuit into a user-specified gate set.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_set (Iterable[str or type] or Callable, optional): The target gate set specified as
            either (1) a sequence of operator types and/or names or (2) a function that returns
            ``True`` if the operator belongs to the target gate set. Defaults to ``None``, in which
            case the gate set is considered to be all available :doc:`quantum operators </introduction/operations>`.
        max_expansion (int, optional): The maximum depth of the decomposition. Defaults to None.
            If ``None``, the circuit will be decomposed until the target gate set is reached.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    .. note::

        This function does not guarantee a decomposition to the target gate set. If an operation
        with no defined decomposition is encountered during decomposition, it will be left in the
        circuit even if it does not belong in the target gate set. In this case, a ``UserWarning``
        will be raised. To suppress this warning, simply add the operator to the gate set.

    .. seealso::

        :func:`qml.devices.preprocess.decompose <.pennylane.devices.preprocess.decompose>` for a
        transform that is intended for device developers. This function will decompose a quantum
        circuit into a set of basis gates available on a specific device architecture.

    **Example**

    Consider the following tape:

    >>> ops = [qml.IsingXX(1.2, wires=(0,1))]
    >>> tape = qml.tape.QuantumScript(ops, measurements=[qml.expval(qml.Z(0))])

    You can decompose the circuit into a set of gates:

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

    You can also use a function to build a decomposition gate set:

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

    You can use the ``max_expansion`` argument to control the number of decomposition stages
    applied to the circuit. By default, the function will decompose the circuit until the desired
    gate set is reached.

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
                    f"Operator {op.name} does not define a decomposition and was not "
                    f"found in the target gate set. To remove this warning, add the operator name "
                    f"({op.name}) or type ({type(op)}) to the gate set.",
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
            "Reached recursion limit trying to decompose operations. Operator decomposition may "
            "have entered an infinite loop. Setting max_expansion will terminate the decomposition "
            "at a fixed recursion depth."
        ) from e

    tape = tape.copy(operations=new_ops)

    return (tape,), null_postprocessing
