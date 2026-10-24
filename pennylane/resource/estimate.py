# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code for projecting the resources of a circuit onto a target gate set"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable
from functools import partial

from pennylane import capture
from pennylane.core.transforms import CompilePipeline
from pennylane.transforms import decompose

from ._utils import (
    apply_partial_args,
    build_circuit_specs,
    preprocess_level_input,
    unwrap_partial,
    unwrap_qjit_qnode,
)
from .mlir_specs import resources_from_analysis_pass
from .resource import CircuitSpecs


def _decomps_from_pipeline(pipeline: CompilePipeline) -> dict:
    """Collect the ``fixed_decomps`` and ``alt_decomps`` of the ``decompose`` transforms in
    ``pipeline``. If more than one fixes a rule for the same operator, the last one is used."""
    # pylint: disable=protected-access
    fixed_decomps, alt_decomps = {}, {}
    for bound_transform in pipeline:
        if bound_transform._transform is decompose:
            fixed_decomps |= bound_transform.kwargs.get("fixed_decomps", {})
            for op, rules in bound_transform.kwargs.get("alt_decomps", {}).items():
                alt_decomps.setdefault(op, []).extend(rules)
    return {"fixed_decomps": fixed_decomps or None, "alt_decomps": alt_decomps or None}


def _estimate_qjit(func, level, target, *args, **kwargs) -> CircuitSpecs:
    """Decompose a qjit'd QNode at ``level`` into the ``target`` gate set and return its specs."""
    # pylint: disable=import-outside-toplevel,protected-access,no-value-for-parameter
    original_qnode = unwrap_qjit_qnode(func, fn_name="qp.estimate")
    from catalyst.utils.exceptions import CompileError

    if level == "device":
        raise NotImplementedError("qp.estimate does not support level='device' yet.")
    levels = preprocess_level_input(level, original_qnode.compile_pipeline)
    if levels != [0] or not isinstance(level, (int, str)) or level == "all":
        raise NotImplementedError(
            f"qp.estimate only supports level='top' or level=0, instead got: {level!r}."
        )

    if isinstance(target, str):
        raise NotImplementedError(
            f"qp.estimate does not support target={target!r} yet. Provide a gate set instead."
        )

    uses_capture = func.compile_options.capture
    if uses_capture == "global":
        uses_capture = capture.enabled()
    if not uses_capture:
        raise ValueError(
            "qp.estimate requires program capture. Compile the QNode with qp.qjit(capture=True)."
        )

    # At level="top", the user transforms are skipped and only the decomposition is applied
    estimate_qnode = copy.copy(original_qnode)
    estimate_qnode._compile_pipeline = CompilePipeline(
        decompose(gate_set=target, **_decomps_from_pipeline(original_qnode.compile_pipeline))
    )

    try:
        resources = resources_from_analysis_pass(func, estimate_qnode, 1, {}, {}, *args, **kwargs)
    except CompileError as exc:
        missing_rule = re.search(r"Decomposition rule not found for operator '([^'{]+)", str(exc))
        if missing_rule is None:
            raise
        op = missing_rule.group(1)
        raise ValueError(
            f"Decomposition rule not found for operator {op}: it cannot be decomposed into the "
            f"target gate set. Add {op} to the target, or use a gate set it can be decomposed into."
        ) from None

    return build_circuit_specs(original_qnode, next(iter(resources.values())), "top")


def estimate(
    qnode,
    level: str | int = "top",
    *,
    target: str | Iterable[type | str] | dict[type | str, float],
) -> Callable[..., CircuitSpecs]:
    r"""Provides a resource estimate of a quantum circuit with respect to the provided compilation target,
    via rapid analysis of decomposition pathways.

    .. note::

        Only QNodes which have been decorated with :func:`@qp.qjit(capture=True) <~pennylane.qjit>` are supported.

    The ``estimate`` function never executes the circuit. It only provides resource estimates.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the qjit'd QNode for which to estimate resources.
        level (str | int): The level to compile to before projecting the resources onto the
            ``target``. Only ``"top"``, ``0``, or the name of a marker placed before all passes,
            are currently supported, indicating the original circuit before any compilation
            passes have been applied. Defaults to ``"top"``.
        target (str | Iterable[type | str] | dict[type | str, float]): The gate set to project
            the resources onto, in any form accepted by the ``gate_set`` argument of
            :func:`~.decompose`.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the projected resources, including
        gate and measurement data, wire allocations, device information, shots, and more.

    .. seealso:: :func:`~.analyze`, :func:`~.track`

    **Example**

    Consider the following circuit.

    .. code-block:: python

        dev = qp.device("null.qubit", wires=2)

        @qp.qjit(capture=True)
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit(x):
            qp.X(0)
            qp.X(0)
            qp.CZ([0, 1])
            qp.RX(x, wires=1)
            return qp.probs()

    By calling ``estimate`` on this circuit with ``level="top"``, the :func:`~.cancel_inverses`
    pass is ignored. Subsequently, the operations in the original circuit will have their
    resources projected into the ``target`` gate set by analyzing decomposition pathways.

    >>> print(qp.estimate(circuit, target={"PauliX", "Hadamard", "CNOT", "RX"})(1.23))
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: top
    <BLANKLINE>
    Quantum operations:
    - Total: 6
      - CNOT: 1
      - Hadamard: 2
      - PauliX: 2
      - RX: 1
    Measurement processes:
    - probs(all wires): 1
    Total wires: 2
    Circuit Depth: Not computed

    Since :func:`~.cancel_inverses` is ignored, both ``PauliX`` gates are counted. The ``CZ`` gate
    is decomposed into ``Hadamard`` and ``CNOT`` gates.

    .. details::
        :title: Custom decomposition rules

        To reach the ``target`` gate set, ``estimate`` uses the ``fixed_decomps`` and
        ``alt_decomps`` of the :func:`~.decompose` transforms in the compile pipeline.

        For example, the following circuit fixes a decomposition rule for ``CZ`` that uses
        ``PauliZ`` and ``RY`` gates instead of ``Hadamard`` gates:

        .. code-block:: python

            @qp.register_resources({qp.Z: 2, qp.RY: 2, qp.CNOT: 1})
            def cz_with_ry(wires, **_):
                qp.Z(wires[1])
                qp.RY(np.pi / 2, wires[1])
                qp.CNOT(wires)
                qp.Z(wires[1])
                qp.RY(np.pi / 2, wires[1])

            @qp.qjit(capture=True)
            @qp.decompose(gate_set={"CNOT", "PauliZ", "RY"}, fixed_decomps={qp.CZ: cz_with_ry})
            @qp.qnode(qp.device("null.qubit", wires=2))
            def circuit():
                qp.CZ([0, 1])
                return qp.probs()

        The ``CZ`` gate is decomposed with this rule:

        >>> qp.estimate(circuit, target={"CNOT", "PauliZ", "RY"})().resources.quantum_operations
        {'CNOT': 1, 'PauliZ': 2, 'RY': 2}

    .. details::
        :title: Symbolic resource counts

        Since the circuit is not executed, the projected resources are subject to the same
        compile-time constraints as :func:`~.analyze`. In particular, operations inside a loop
        whose number of iterations depends on runtime values are counted symbolically.

        For example, the number of ``CZ`` gates in the following circuit, and therefore the number
        of gates they decompose into, depends on the value of ``n``:

        .. code-block:: python

            @qp.qjit(capture=True)
            @qp.qnode(qp.device("null.qubit", wires=2))
            def circuit(n):
                qp.Hadamard(0)

                @qp.for_loop(n)
                def loop(i):
                    qp.CZ([0, 1])

                loop()
                return qp.expval(qp.Z(0))

        >>> resources = qp.estimate(circuit, target={"Hadamard", "CNOT"})(5).resources
        >>> print(resources)
        Symbolic Variables: a
        Quantum operations:
        - Total: 3*a + 1
          - CNOT: a
          - Hadamard: 2*a + 1
        Measurement processes:
        - expval(PauliZ): 1
        Total wires: 2
        Circuit Depth: Not computed

        Concrete values can be estimated by substituting each symbolic variable with an integer,
        using the ``.subs`` method:

        >>> resources.subs(a=5).quantum_operations
        {'CNOT': 5, 'Hadamard': 11}

        .. note::

            Loops inside the decomposition rules used to reach the ``target`` gate set are counted
            symbolically as well, so the result can be symbolic even if the circuit has no dynamic
            control flow.
    """
    # TODO: [sc-133408] Add a qp.hint example to the docstring once hints survive decomposition
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    return apply_partial_args(
        partial(_estimate_qjit, qnode, level, target), partial_args, partial_kwargs
    )
