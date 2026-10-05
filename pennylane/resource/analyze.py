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
"""Code for analyzing the resources of a compiled circuit"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial

from ._utils import (
    apply_partial_args,
    build_circuit_specs,
    get_marker_level_map,
    preprocess_level_input,
    unwrap_partial,
    unwrap_qjit_qnode,
)
from .mlir_specs import resources_from_analysis_pass
from .resource import CircuitSpecs, SpecsResources


def _run_resource_analysis(qjit, original_qnode, level, *args, **kwargs) -> tuple[
    SpecsResources | list[SpecsResources] | dict[str, SpecsResources | list[SpecsResources]],
    str | dict[int, str],
]:
    """Compile a qjit'd QNode with the ``resource-analysis`` pass inserted at the given levels.

    Returns the resources found at each level along with the name of each level.
    """
    # Note that this only gets transforms manually applied by the user
    compile_pipeline = original_qnode.compile_pipeline

    return_single_level: bool = isinstance(level, (int, str)) and level != "all"

    # Easier to assume level is always a sorted list of int levels
    level = preprocess_level_input(level, compile_pipeline)

    # Map to convert back and forth between marker name and int level
    marker_to_level = get_marker_level_map(compile_pipeline)
    level_to_markers = defaultdict(list)  # Multiple markers can correspond to the same level
    for marker, lvl in marker_to_level.items():
        level_to_markers[lvl].append(marker)

    level_to_name: dict[int, str] = {}

    # Handle MLIR passes
    resources = resources_from_analysis_pass(
        qjit,
        original_qnode,
        level,
        level_to_markers,
        level_to_name,
        *args,
        **kwargs,
    )

    # Unpack dictionary to single item if only 1 level was given as input
    if return_single_level:
        resources = next(iter(resources.values()))
        level_to_name = next(iter(level_to_name.values()))

    return resources, level_to_name


def _analyze_qjit(qjit, level, *args, **kwargs) -> CircuitSpecs:
    """Compile a qjit'd QNode up to the given level and return the specs found by analyzing it."""
    original_qnode = unwrap_qjit_qnode(qjit, fn_name="qp.analyze")

    if level == "device":
        raise NotImplementedError("qp.analyze does not support level='device' yet.")

    resources, level = _run_resource_analysis(qjit, original_qnode, level, *args, **kwargs)

    return build_circuit_specs(original_qnode, resources, level)


def analyze(
    qnode,
    level: str | int | Iterable[int | str] = "user",
) -> Callable[..., CircuitSpecs]:
    r"""Provides a compile-time resource estimate of a quantum circuit, obtained by analyzing its
    intermediate representation at the specified compilation level.

    This transform converts a QNode into a callable that compiles the circuit up to ``level``
    and inspects the resulting representation, without executing the circuit or unrolling its
    control flow. The resource information is therefore subject to compile-time constraints.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the (qjit'd) QNode for which to estimate resources.
            ``functools.partial`` wrappers around supported callables are also accepted.
        level (str | int | Iterable[int | str]): The level of compilation at which to estimate
            resources. Defaults to ``"user"``. See the note below for the accepted values.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications,
        including gate and measurement data, wire allocations, device information, shots, and
        more.

    .. seealso:: :func:`~.specs`, which provides the same analysis, and :func:`~.track`, which
        counts the resources used after device preprocessing by mock-executing the circuit.

    .. note::

        The available options for ``level`` are:

        * ``"top"`` or ``0``: The original circuit before any transforms have been applied.
        * An ``int``: The circuit after the specified number of user-specified transforms have
          been applied.
        * ``"user"``: The circuit after all user-specified transforms have been applied.
        * A marker name (str): The circuit after the transforms that precede the given
          :func:`qp.marker <pennylane.marker>` have been applied.
        * An iterable: A ``list``, ``tuple``, or similar containing ints and/or marker names.
          Should be sorted in ascending transform order with no duplicates. The output will
          provide resource information for each level.
        * The string ``"all"``: To provide information at each stage of compilation with respect
          to user-specified transforms.

        Levels that include the device preprocessing transforms, such as ``"device"``, are not
        currently supported.

    **Example**

    .. code-block:: python

        dev = qp.device("null.qubit", wires=2)

        @qp.qjit
        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.X(0)
            qp.X(0)
            qp.CNOT([0, 1])
            return qp.probs()

    >>> print(qp.analyze(circuit, level="user")(1.23))
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: merge-rotations
    <BLANKLINE>
    Quantum operations:
    - Total: 2
      - CNOT: 1
      - RX: 1
    Measurement processes:
    - probs(all wires): 1
    Total wires: 2
    Circuit Depth: Not computed

    Using ``level="all"`` shows how the resources change after each transform:

    >>> all_specs = qp.analyze(circuit, level="all")(1.23)
    >>> print(all_specs)
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Levels:
    - 0: Before MLIR Passes
    - 1: cancel-inverses
    - 2: merge-rotations
    <BLANKLINE>
    ↓Metric         Level→ |  0 |  1 |  2
    -------------------------------------
    Quantum operations:    |
    - Total                |  5 |  3 |  2
      - CNOT               |  1 |  1 |  1
      - PauliX             |  2 |  0 |  0
      - RX                 |  2 |  2 |  1
    Measurement processes: |
    - probs(all wires)     |  1 |  1 |  1
    Total wires            |  2 |  2 |  2

    The resources at a given level can be accessed with the name of the corresponding transform
    or marker:

    >>> all_specs.resources["cancel-inverses"].quantum_operations
    {'CNOT': 1, 'RX': 2}

    .. details::
        :title: Compile-time constraints of resource analysis

        Since the circuit is not executed, some resources cannot be counted exactly:

        * Resources inside a ``for`` loop whose number of iterations depends on runtime values
          are counted symbolically, using :class:`~.resource.Expression` instances instead of
          integers.
        * Resources inside a ``while`` loop are counted as if only one iteration occurred.
        * For conditional branches from ``if`` or ``switch`` statements, each gate is counted with
          its maximum count across all branches, providing an upper bound.

        For example, the number of ``PauliX`` gates in the following circuit depends on the
        value of ``n``:

        .. code-block:: python

            @qp.qjit(autograph=True)
            @qp.qnode(qp.device("null.qubit", wires=1))
            def circuit(n):
                qp.Hadamard(0)
                for _ in range(n):
                    qp.PauliX(0)
                return qp.expval(qp.Z(0))

        >>> resources = qp.analyze(circuit, level=0)(5).resources
        >>> print(resources)
        Symbolic Variables: a
        Quantum operations:
        - Total: a + 1
          - Hadamard: 1
          - PauliX: a
        Measurement processes:
        - expval(PauliZ): 1
        Total wires: 1
        Circuit Depth: Not computed

        Concrete values can be estimated by substituting each symbolic variable with an integer,
        using the ``.subs`` method:

        >>> resources.subs(a=5).quantum_operations
        {'Hadamard': 1, 'PauliX': 5}

        To get concrete counts for given arguments instead, use :func:`~.track`, which executes
        the circuit and therefore unrolls its control flow.
    """
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    return apply_partial_args(partial(_analyze_qjit, qnode, level), partial_args, partial_kwargs)
