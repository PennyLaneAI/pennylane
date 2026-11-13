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

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

from pennylane import capture
from pennylane.core.transforms import CompilePipeline
from pennylane.workflow._levels import (
    device_level,
    is_single_level,
    levels_to_numbers,
    pipeline_at_slice,
    split_device_preprocessing,
)
from pennylane.workflow.get_compile_pipeline import unwrap_qjit_qnode

from ._utils import (
    apply_partial_args,
    build_circuit_specs,
    check_no_tape_transforms,
    unwrap_partial,
)
from .mlir_specs import resources_from_analysis_pass
from .resource import CircuitSpecs, SpecsResources

if TYPE_CHECKING:
    from pennylane.workflow import QNode


def _qnode_at_slice(qnode: QNode, level: slice) -> tuple[QNode, int]:
    """Return a copy of ``qnode`` that only applies the compile pipeline selected by ``level``,
    as given by :func:`~.workflow.get_compile_pipeline`, along with the level at which to analyze
    the resources of the copy."""
    # The device applies its preprocessing itself, after the other transforms
    pipeline, has_device_preprocessing = split_device_preprocessing(
        pipeline_at_slice(qnode.compile_pipeline, level)
    )
    if level.indices(device_level(qnode.compile_pipeline))[0] > 0:
        # The markers describe the circuit after all the transforms before them, so they do not
        # apply to a circuit where the transforms before the slice are skipped
        pipeline = CompilePipeline(list(pipeline))
    sliced_qnode = qnode.update_pipelines(
        pipeline, post_device_pipeline=CompilePipeline(qnode.post_device_pipeline)
    )
    return sliced_qnode, device_level(pipeline) if has_device_preprocessing else len(pipeline)


def _run_resource_analysis(func, original_qnode, level, *args, **kwargs) -> tuple[
    SpecsResources | list[SpecsResources] | dict[str, SpecsResources | list[SpecsResources]],
    str | dict[int, str],
]:
    """Compile a qjit'd QNode with the ``resource-analysis`` pass inserted at the given levels.

    Returns the resources found at each level along with the name of each level.
    """
    if isinstance(level, slice):
        # Only apply the portion of the compile pipeline selected by the slice
        original_qnode, level = _qnode_at_slice(original_qnode, level)

    # Note that this only gets transforms manually applied by the user
    compile_pipeline = original_qnode.compile_pipeline
    level_numbers = levels_to_numbers(level, compile_pipeline)
    # Only the transforms up to the highest level are applied
    check_no_tape_transforms(compile_pipeline[: max(level_numbers)])

    has_device_level: bool = device_level(compile_pipeline) in level_numbers
    if has_device_level:
        if not func.compile_options.capture or (
            func.compile_options.capture == "global" and not capture.enabled()
        ):
            raise ValueError("Device level is only supported when capture is enabled.")

    # Levels with markers are named after them
    markers_by_level = compile_pipeline._level_to_markers()  # pylint: disable=protected-access

    level_to_name: dict[int, str] = {}

    # Handle MLIR passes
    resources = resources_from_analysis_pass(
        func,
        original_qnode,
        level_numbers,
        markers_by_level,
        level_to_name,
        *args,
        **kwargs,
    )

    # Unpack dictionary to single item if only 1 level was given as input
    if is_single_level(level):
        resources = next(iter(resources.values()))
        level_to_name = next(iter(level_to_name.values()))

    return resources, level_to_name


def _analyze_qjit(func, level, *args, **kwargs) -> CircuitSpecs:
    """Compile a qjit'd QNode up to the given level and return the specs found by analyzing it."""
    original_qnode = unwrap_qjit_qnode(func, fn_name="qp.analyze")

    resources, level = _run_resource_analysis(func, original_qnode, level, *args, **kwargs)

    return build_circuit_specs(original_qnode, resources, level)


def analyze(
    qnode,
    level: str | int | slice | Sequence[str | int] = "user",
) -> Callable[..., CircuitSpecs]:
    r"""Provides a compile-time resource estimate of a quantum circuit, obtained by analyzing its
    intermediate representation at the specified compilation level.

    This transform converts a QNode into a callable that compiles the circuit with the
    portion(s) of its :class:`~.CompilePipeline` selected by ``level``, and inspects the
    resulting representation, without executing the circuit or unrolling its
    control flow. The resource information is therefore subject to compile-time constraints.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the (qjit'd) QNode for which to estimate resources.
            ``functools.partial`` wrappers around supported callables are also accepted.
        level (str | int | slice | Sequence[str | int]): The portion of the
            :class:`~.CompilePipeline` to apply before analyzing resources. See
            :func:`~.workflow.get_compile_pipeline` for accepted values. If ``level`` is a
            ``Sequence``, ``"all-user"``, or ``"all"``, resources are analyzed independently for
            each provided case. Defaults to ``"user"``.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications,
        including gate and measurement data, wire allocations, device information, shots, and
        more.

    .. seealso:: :func:`~pennylane.estimate`, :func:`~.track`

    **Example**

    .. code-block:: python

        dev = qp.device("null.qubit", wires=2)

        @qp.qjit(capture=True)
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
    - 3: Device Preprocessing
    <BLANKLINE>
    ↓Metric         Level→ |  0 |  1 |  2 |  3
    ------------------------------------------
    Quantum operations:    |
    - Total                |  5 |  3 |  2 |  2
      - CNOT               |  1 |  1 |  1 |  1
      - PauliX             |  2 |  0 |  0 |  0
      - RX                 |  2 |  2 |  1 |  1
    Measurement processes: |
    - probs(all wires)     |  1 |  1 |  1 |  1
    Total wires            |  2 |  2 |  2 |  2

    The resources at a given level can be accessed with the name of the corresponding transform
    or marker:

    >>> all_specs.resources["cancel-inverses"].quantum_operations
    {'CNOT': 1, 'RX': 2}

    A ``slice`` only applies that portion of the compile pipeline. For example, the resources
    after applying ``merge_rotations``, but not ``cancel_inverses``, are given by:

    >>> qp.analyze(circuit, level=slice(1, 2))(1.23).resources.quantum_operations
    {'CNOT': 1, 'PauliX': 2, 'RX': 1}

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
