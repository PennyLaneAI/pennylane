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
"""Code for resource estimation"""

import copy
import dataclasses
import json
import os
import tempfile
import time
from functools import partial
from typing import TYPE_CHECKING

import pennylane as qp

from ._utils import (
    get_last_tape_transform_level,
    get_marker_level_map,
    make_level_name_unique,
    preprocess_level_input,
)
from .specs import specs

_RESOURCE_ESTIMATION_PREFIX = "pennylane_resource_estimation"

if TYPE_CHECKING:
    from pennylane.transforms.core import CompilePipeline


def _decomposition_to_levels(
    level: list[int], compile_pipeline: "CompilePipeline"
) -> dict[int, list[int]]:
    """Convert a list of levels into a dictionary mapping a number of ``decompose`` passes to
    a list of all levels which have that number of decompositions before them.

    This effectively converts each level number into a list of how many decompositions come before it.

    For example, in the following compile pipeline:
    [1] l1
    [2] decompose
    [3] l2
    [4] l3
    [5] decompose
    The list ``[1,2,3,4,5]`` would be converted to ``{0: [1], 1: [2, 3, 4], 2: [5]}``.

    .. note::

        This function assumes that `level` is a sorted list of integers.

    .. warning::

        This function is intended for internal use only and may change or be removed in future releases.

    Args:
        level (list[int]): The levels to convert.
        compile_pipeline (CompilePipeline): The compile pipeline to use for decomposition.

    Returns:
        dict[int, list[int]]: A dictionary mapping a number of decompositions to a list of all
            levels which have that number of decompositions before them.
    """
    decompose_count = 0
    decompose_levels = {}
    if 0 in level:
        decompose_levels[0] = [0]
    for i, mlir_pass in enumerate(compile_pipeline, start=1):
        if mlir_pass.pass_name == "graph-decomposition":
            decompose_count += 1
        if i in level:
            if decompose_count not in decompose_levels:
                decompose_levels[decompose_count] = []
            decompose_levels[decompose_count].append(i)
    return decompose_levels


def _get_decomposition_rules(
    qnode, compile_pipeline: "CompilePipeline", level: list[int] | int, *args, **kwargs
) -> list[dict[str, dict[str, int]]]:
    """
    Get all of the decomposition rules by querying the graph solver.

    Args:
        qnode: The (qjit'd) QNode to estimate resources for.
        compile_pipeline (CompilePipeline): The compile pipeline to use for decomposition.
        level (list[int] | int): The level(s) to estimate resources for.
        *args: Additional positional arguments to pass to the QNode.
        **kwargs: Additional keyword arguments to pass to the QNode.

    Returns:
        list[dict[str, dict[str, int]]]: A list containing the decomposition rules
    """
    from catalyst import QJIT

    all_decomps = []
    resource_fnames = []

    max_level = max(level) if isinstance(level, (list, tuple)) else level

    # This new compile pipeline will ONLY contain decomposition passes (used to generate the
    #   decomposition rules for each level)
    new_compile_pipeline = qp.CompilePipeline()

    with tempfile.TemporaryDirectory(
        prefix=f"{_RESOURCE_ESTIMATION_PREFIX}_{os.getpid()}_"
    ) as tmpdirname:
        fname_prefix = f"{tmpdirname}/{_RESOURCE_ESTIMATION_PREFIX}_{time.time_ns()}_level_"

        for i, mlir_pass in enumerate(compile_pipeline):
            if i > max_level:
                break
            # TODO: Should also support some other passes such as lowerings to PPR/PPM
            if mlir_pass.pass_name == "graph-decomposition":
                estimation_fpath = f"{fname_prefix}{i}.json"
                resource_fnames.append(estimation_fpath)
                new_pass = copy.deepcopy(mlir_pass)
                new_pass.kwargs["est_json_path"] = estimation_fpath
                new_compile_pipeline += new_pass

        new_qnode = copy.copy(qnode.original_function)
        # Need to manually set the compile pipeline internal variable for the new qnode
        new_qnode._compile_pipeline = new_compile_pipeline
        compile_options = copy.deepcopy(qnode.compile_options)

        # TODO: Investigate issue that every AOT compile causes a crash
        new_qnode = QJIT(new_qnode, compile_options)
        new_qnode.jit_compile(args, **kwargs)

        # Load in the decomposition rules from the JSON files generated by the graph decomposition pass
        for resource_fname in resource_fnames:
            with open(resource_fname, "r") as f:
                decomps = json.load(f)
                all_decomps.append(decomps)

    return all_decomps


def _estimate_impl(qnode, level, *args, **kwargs):
    compile_pipeline = qnode.original_function.compile_pipeline

    if get_last_tape_transform_level(compile_pipeline) != 0:
        raise ValueError("Resource estimation is not supported with tape transforms.")

    return_single_level: bool = isinstance(level, (int, str)) and level != "all"

    level_to_markers = get_marker_level_map(compile_pipeline)
    level_to_name = {}

    if level == "all":
        level = [
            0,
            *(
                i + 1
                for i in range(len(compile_pipeline))
                if compile_pipeline[i].pass_name == "graph-decomposition"
            ),
        ]
    else:
        # Easier to assume level is always a sorted list of int levels
        level = preprocess_level_input(level, level_to_markers, len(compile_pipeline), 0)

    initial_specs = specs(qnode, level=0)(*args, **kwargs)
    all_decomps = _get_decomposition_rules(qnode, compile_pipeline, level, *args, **kwargs)

    # Convert level into the index notation used by the output from _get_decomposition_rules
    decompose_levels = _decomposition_to_levels(level, compile_pipeline)

    current_operations = initial_specs.resources.quantum_operations
    all_resources = {}

    if 0 in level:
        level_name = (
            ", ".join(level_to_markers[0]) if 0 in level_to_markers else "Before MLIR Passes"
        )
        level_to_name[0] = level_name
        all_resources[level_name] = initial_specs.resources

    for decomp_idx, decomps in enumerate(all_decomps, start=1):
        new_resources = {}
        for gate, count in current_operations.items():
            if gate not in decomps:
                raise RuntimeError(f"Gate {gate} not found in decomposition rules")
            for decomp_gate, decomp_count in decomps[gate].items():
                if decomp_gate not in new_resources:
                    new_resources[decomp_gate] = 0
                new_resources[decomp_gate] += count * decomp_count
        current_operations = new_resources
        if decomp_idx in decompose_levels:
            for pass_level in decompose_levels[decomp_idx]:
                level_name = (
                    ", ".join(level_to_markers[pass_level])
                    if pass_level in level_to_markers
                    else compile_pipeline[pass_level - 1].pass_name or f"Level {pass_level}"
                )
                level_name = make_level_name_unique(level_name, frozenset(level_to_name.values()))

                level_to_name[pass_level] = level_name
                all_resources[level_name] = dataclasses.replace(
                    initial_specs.resources, counts=current_operations
                )

    if return_single_level:
        all_resources = next(iter(all_resources.values()))
        level_to_name = next(iter(level_to_name.values()))

    # TODO: How best to handle level ordering?
    return dataclasses.replace(initial_specs, resources=all_resources, level=level_to_name)


def estimate(qnode, level="user"):
    r"""Provides resource estimates for a quantum circuit.

    This transform converts a QNode into a callable that provides estimated resource information
    about the circuit after applying the specified decompositions.

    .. note::

        This function only provides high-level resource estimates, and does not provide exact counts
        of gates or measurements. Namely, any compilation passes other than decomposition are
        ignored, which can lead to inaccurate gate counts. For more precise resource information,
        see :func:`~pennylane.resource.specs`.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the QNode to calculate the specifications for.
            ``functools.partial`` wrappers around supported callables are also accepted.

    Keyword Args:
        level (str | int | iter[int | str] | None): An indication of which transforms, expansions,
            and passes to apply before computing the resource information. Default: ``"user"``.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications, including
        gate and measurement data, wire allocations, device information, shots, and more.

    **Example**

    .. code-block:: python

        qp.decomposition.enable_graph()


        @decomposition_rule(op_type=qp.CNOT)
        def my_cnot(wires):
            qp.H(wires=wires[1])
            qp.CZ(wires=wires)
            qp.H(wires=wires[1])


        @qp.qjit(capture=True)#, verbose=True)
        @graph_decomposition(
            gate_set={"H", "CZ", "GlobalPhase", "PauliZ"},
            fixed_decomps={qp.CNOT: my_cnot},
        )
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(i):
            qp.H(0)
            qp.H(0)
            qp.CNOT(wires=[0, 1])
            qp.GlobalPhase(i)

            # register custom decomposition rules
            my_cnot(ShapedArray((2,), int))

            return qp.state()

    >>> print(qp.estimate(circuit)(1.23))
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: After decomposition 1

    Quantum operations:
    - Total: 6
    - CZ: 1
    - Hadamard: 4
    - GlobalPhase: 1
    Measurement processes:
    - state(all wires): 1
    Wire allocations: 2
    Circuit Depth: Not computed

    Note that the ``cancel-inverses`` pass does not get applied, so the number of Hadamards is not
    reduced.

    The :class:`~.resource.SpecsResources` can be accessed using the ``.resources`` attribute,
    which provides more direct access to the data fields. For example:

    >>> qp.estimate(circuit)(1.23).resources.quantum_operations
    {'CZ': 1, 'Hadamard': 4, 'GlobalPhase': 1}

    .. details::

        .. note::

            This functionality is specific to workflows with ``qjit``.

        **Respource estimation** functions by analyzing the intermediate representations of compiled
        circuits. This can be helpful for determining how circuit resources change as they reach
        various stages of decomposition.

        .. warning::
            It is not always possible to determine exact resource usage from intermediate
            representations. The output of this function is a high-level algorithmic estimate of the
            resources used by a circuit, and may not reflect the exact resources used by the final
            compiled circuit.

            This is partly due to the fact that :func:`estimate` does not apply any compilation
            passes other than decomposition, which can lead to inaccurate gate counts but can also
            be due to limitations about what is known at compile time.

            For example, resources contained in a ``for`` loop with a non-static range or a
            ``while`` loop will be counted symbolically. Additionally, resources contained in
            conditional branches from ``if`` or ``switch`` statements will take a union of resources
            over all branches, providing a tight upper-bound.

            Due to similar technical limitations, circuit depth is not available.

        The following ``level`` arguments are supported:

        * An ``int``: the desired pass level of a user-applied pass, see the note below
        * A marker name (str): The name of an applied :func:`qp.marker <pennylane.marker>` pass
        * An iterable: A ``list``, ``tuple``, or similar containing ints and/or marker names. Should
          be sorted in ascending pass order with no duplicates
        * The string ``"user"``: To provide information after all user-specified transforms have
          been applied
        * The string ``"top"``: To provide information about the original circuit before any
          user-specified transforms have been applied
        * The string ``"all"``: To provide information at each stage of compilation with respect to
          user-specified transforms

        Consider the following circuit:

        .. code-block:: python

            qp.decomposition.enable_graph()

            @decomposition_rule(op_type=qp.PauliZ)
            def quad_t(wires):
                qp.T(wires=wires)
                qp.T(wires=wires)
                qp.T(wires=wires)
                qp.T(wires=wires)

            @decomposition_rule(op_type=qp.CNOT)
            def my_cnot(wires):
                qp.H(wires=wires[1])
                qp.CZ(wires=wires)
                qp.H(wires=wires[1])

            @qp.qjit(capture=True)#, verbose=True)
            @graph_decomposition(
                gate_set={"H", "CZ", "GlobalPhase", "T"},
                fixed_decomps={qp.PauliZ: quad_t},
            )
            @qp.transforms.cancel_inverses
            @graph_decomposition(
                gate_set={"H", "CZ", "GlobalPhase", "PauliZ"},
                fixed_decomps={qp.CNOT: my_cnot},
            )
            @qp.qnode(qp.device("lightning.qubit", wires=2))
            def circuit(i):
                qp.H(0)
                qp.H(0)
                qp.PauliZ(1)
                qp.CNOT(wires=[0, 1])
                qp.GlobalPhase(i)  # Include this to prevent AOT compilation

                # register custom decomposition rules
                my_cnot(ShapedArray((2,), int))
                quad_t(ShapedArray((1,), int))

                return qp.state()

        We can get a pass-by-pass overview of the resources using ``level="all"``:

        >>> all_specs = qp.estimate(circuit, level="all")(1.23)
        >>> print(all_specs)
        Device: lightning.qubit
        Device wires: 2
        Shots: Shots(total=None)
        Levels:
        - 0: Before any decomposition
        - 1: After decomposition 1
        - 2: After decomposition 2
        <BLANKLINE>
        ↓Metric         Level→ |  0 |  1 |  2
        -------------------------------------
        Quantum operations:    |
        - Total                |  5 |  7 | 10
        - CNOT                 |  1 |  0 |  0
        - GlobalPhase          |  1 |  1 |  1
        - Hadamard             |  2 |  4 |  4
        - PauliZ               |  1 |  1 |  0
        - CZ                   |  0 |  1 |  1
        - T                    |  0 |  0 |  4
        Measurement processes: |
        - state(all wires)     |  1 |  1 |  1
        Wire allocations       |  2 |  2 |  2

        When invoked with an iterable of levels, or ``"all"`` as above, the resources at different
        levels can be accessed from the the returned :class:`~.resource.CircuitSpecs` object's
        ``.resources`` attribute, using the name of a pass or marker. For example:

        >>> print(all_specs.resources['After decomposition 2'])
        Quantum operations:
        - Total: 10
          - GlobalPhase: 1
          - Hadamard: 2
          - CZ: 1
          - T: 4
        Measurement processes:
        - state(all wires): 1
        Wire allocations: 2
        Circuit Depth: Not computed

        A shortcut to access the resources after all user-specified transforms and passes have been
        applied is to use the ``"user"`` level.

        >>> print(qp.estimate(circuit, level="user")(1.23).resources)
        Quantum operations:
        - Total: 10
          - GlobalPhase: 1
          - Hadamard: 2
          - CZ: 1
          - T: 4
        Measurement processes:
        - state(all wires): 1
        Wire allocations: 2
        Circuit Depth: Not computed
    """
    return partial(_estimate_impl, qnode, level)
