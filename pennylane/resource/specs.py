# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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

from __future__ import annotations

import copy
import json
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import pennylane as qml

from .resource import CircuitSpecs, SpecsResources, resources_from_tape

if TYPE_CHECKING:
    from pennylane.transforms.core import CompilePipeline

# Used for device-level qjit resource tracking
_RESOURCE_TRACKING_FILEPATH = "__qml_specs_qjit_resources.json"


def _make_level_name_unique(level_name: str, existing_names: set[str]) -> str:
    """Helper function to make a level name unique by appending a suffix if necessary.

    Args:
        level_name (str): The original level name
        existing_names (set[str]): The set of existing level names to check against

    Returns:
        str: A unique level name

    Example:
        >>> existing = {"cancel-inverses", "merge-rotations", "cancel-inverses-2"}
        >>> _make_level_name_unique("cancel-inverses", existing)
        'cancel-inverses-3'
    """
    unique_name = level_name
    counter = 1
    while unique_name in existing_names:
        counter += 1
        unique_name = f"{level_name}-{counter}"
    return unique_name


def _specs_qnode(qnode, level, compute_depth, *args, **kwargs) -> CircuitSpecs:
    """Returns information on the structure and makeup of provided QNode.

    Returns:
        CircuitSpecs: result object that contains QNode specifications
    """
    if level is None:
        level = "gradient"

    if compute_depth is None:
        compute_depth = True

    batch, _ = qml.workflow.construct_batch(qnode, level=level)(*args, **kwargs)

    resources = [resources_from_tape(tape, compute_depth) for tape in batch]

    if len(resources) == 1:
        resources = resources[0]

    return CircuitSpecs(
        resources=resources,
        num_device_wires=len(qnode.device.wires) if qnode.device.wires is not None else None,
        device_name=qnode.device.name,
        level=level,
        shots=qnode.shots,
    )


def _specs_qjit_device_level_tracking(
    qjit, original_qnode, compute_depth, *args, **kwargs
) -> SpecsResources:  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    from catalyst import QJIT

    from ..devices import NullQubit

    if compute_depth is None:
        compute_depth = True

    # When running at the device level, execute on null.qubit directly with resource tracking,
    # which will give resource usage information for after all compiler passes have completed
    # TODO: Find a way to inherit all devices args from input
    original_device = original_qnode.device
    spoofed_dev = NullQubit(
        target_device=original_device,
        wires=original_device.wires,
        track_resources=True,
        resources_filename=_RESOURCE_TRACKING_FILEPATH,
        compute_depth=compute_depth,
    )

    new_qnode = qjit.original_function.update(device=spoofed_dev)
    new_qjit = QJIT(new_qnode, copy.deepcopy(qjit.compile_options))

    if os.path.exists(_RESOURCE_TRACKING_FILEPATH):
        # TODO: Warn that something has gone wrong here
        os.remove(_RESOURCE_TRACKING_FILEPATH)

    try:
        # Execute on null.qubit with resource tracking
        new_qjit(*args, **kwargs)

        with open(_RESOURCE_TRACKING_FILEPATH, encoding="utf-8") as f:
            resource_data = json.load(f)

        return SpecsResources(
            gate_types=resource_data["gate_types"],
            gate_sizes={int(k): v for (k, v) in resource_data["gate_sizes"].items()},
            measurements=resource_data["measurements"],
            num_allocs=resource_data["num_wires"],
            depth=resource_data["depth"],
        )
    finally:
        # Ensure we clean up the resource tracking file
        if os.path.exists(_RESOURCE_TRACKING_FILEPATH):
            os.remove(_RESOURCE_TRACKING_FILEPATH)


def _get_last_tape_transform_level(compile_pipeline: CompilePipeline) -> int:
    """Helper function to get the last level which is a tape transform and not an MLIR pass.

    Note that this includes an implicit level 0 which corresponds to the original circuit.

    Args:
        compile_pipeline: The compile pipeline of the QNode, which contains both user-applied tape transforms and MLIR passes

    Returns:
        int: The last level which is a tape transform and not an MLIR pass, or 0 if there are no tape transforms
    """
    # Find the seam where transforms end and MLIR passes begin
    # If the pass name is None, it indicates a transform which is NOT also a Catalyst pass
    for i, trans in reversed(list(enumerate(compile_pipeline))):
        if trans.pass_name is None:
            #  Add 1 to account for the implicit "Before Tape Transforms" at level=0
            return i + 1
    return 0


def _preprocess_level_input(  # pylint: disable=too-many-branches
    level: str | int | slice | list[int | str],
    marker_to_level: dict[str, int],
    pipeline_len: int,
    num_tape_levels: int,
) -> list[int]:
    """Preprocesses the level input to always return a sorted list of integers.

    Args:
        level (str | int | slice | iter[int | str]): The level input to preprocess
        marker_to_level (dict[str, int]): Mapping from marker names to their associated level numbers.
            Note that this should already account for any inserted lowering pass.
        pipeline_len (int): The length of the compile pipeline (number of transforms and passes)
        num_tape_levels (int): The number of tape levels in the compile pipeline (including the implicit level 0)
    Returns:
        list[int]: The preprocessed level input
    """
    # Account for "Before MLIR passes" level
    max_level = pipeline_len + 1

    if num_tape_levels > 1:
        # Account for an additional "Before Tape Transforms" level
        max_level += 1

    if level == "all":
        return list(range(0, max_level))
    if level == "all-mlir":
        return list(range(num_tape_levels, max_level))

    if isinstance(level, (int, str)):
        level = [level]
    elif isinstance(level, slice):
        level = list(range(level.start or 0, level.stop, level.step or 1))
    else:
        level = list(level)

    # Convert marker names to the associated level number
    for i, lvl in enumerate(level):
        if isinstance(lvl, str):
            if lvl not in marker_to_level:
                raise ValueError(f"Marker name '{lvl}' not found in the compile pipeline.")
            level[i] = marker_to_level[lvl]
        elif isinstance(lvl, int):
            if lvl < 0:
                level[i] = max_level - abs(lvl)
        else:
            raise ValueError(f"Invalid level '{lvl}' in level list, expected int or str.")

    level_sorted = sorted(set(level))
    if level != level_sorted:
        warnings.warn(
            "The 'level' argument to qml.specs for QJIT'd QNodes has been sorted to be in ascending "
            "order with no duplicate levels.",
            UserWarning,
        )

    if len(bad_levels := [lvl for lvl in level_sorted if lvl >= max_level or lvl < 0]) > 0:
        raise ValueError(
            "The 'level' argument to qml.specs for QJIT'd QNodes is out of bounds, got "
            f"{', '.join(str(lvl) for lvl in bad_levels)}."
        )

    return level_sorted


def _mlir_resources_to_specs_resources(resources) -> SpecsResources:  # pragma: no cover
    # This is function is covered by integration tests within the Catalyst frontend
    """Helper function to convert the output of mlir_specs (which is in terms of ResourcesResult) to SpecsResources."""

    gate_types = {}
    gate_sizes = defaultdict(int)

    for res_name, sizes in resources.operations.items():
        for size, count in sizes.items():
            gate_sizes[size] += count

        if res_name in ("PPM", "PPR-pi/2", "PPR-pi/4", "PPR-pi/8", "PPR-Phi"):
            # Separate out PPMs and PPRs by weight
            for size, count in sizes.items():
                gate_types[f"{res_name}-w{size}"] = count
        else:
            gate_types[res_name] = sum(sizes.values())

    return SpecsResources(
        gate_types=gate_types,
        gate_sizes=dict(gate_sizes),
        measurements=dict(resources.measurements),
        num_allocs=resources.num_allocs,
        depth=None,  # Can't get depth from MLIR pass results
    )


def _specs_qjit_intermediate_passes(qjit, original_qnode, level, *args, **kwargs) -> tuple[
    SpecsResources | list[SpecsResources] | dict[int, SpecsResources | list[SpecsResources]],
    dict[int, str],
]:  # pragma: no cover
    # pylint: disable=import-outside-toplevel,too-many-branches,too-many-statements
    from catalyst.python_interface.inspection import mlir_specs

    # Note that this only gets transforms manually applied by the user
    compile_pipeline = original_qnode.compile_pipeline

    # This value is used to determine the last level which is a transform and not an MLIR pass
    num_tape_levels = _get_last_tape_transform_level(compile_pipeline)
    if num_tape_levels != 0:
        # Account for the "Before Tape Transforms" tape at level 0
        num_tape_levels += 1

    # Maps to convert back and forth between marker name and int level
    marker_to_level: dict[str, int] = {}
    for marker in compile_pipeline.markers:
        lvl = compile_pipeline.get_marker_level(marker)
        marker_to_level[marker] = lvl

        # Account for the MLIR lowering pass if necessary
        if 0 < num_tape_levels <= lvl:
            marker_to_level[marker] += 1

    # Multiple markers can correspond to the same level
    level_to_markers = defaultdict(list)
    for marker, lvl in marker_to_level.items():
        level_to_markers[lvl].append(marker)
    mlir_level_to_markers = {
        lvl - num_tape_levels: markers
        for lvl, markers in level_to_markers.items()
        if lvl >= num_tape_levels
    }

    # Easier to assume level is always a sorted list of int levels (if not "all" or "all-mlir")
    return_single_level = isinstance(level, (int, str)) and level not in ("all", "all-mlir")
    level = _preprocess_level_input(level, marker_to_level, len(compile_pipeline), num_tape_levels)
    output_level: dict[int, str] = {}  # This will be a map of level to its name

    tape_levels = [lvl for lvl in level if lvl < num_tape_levels]
    mlir_levels = [lvl - num_tape_levels for lvl in level if lvl >= num_tape_levels]

    resources = {}

    # Handle tape transforms
    if len(tape_levels) > 0:
        for tape_level in tape_levels:
            # User transforms always come first, so level and tape_level align correctly
            batch, _ = qml.workflow.construct_batch(original_qnode, level=tape_level)(
                *args, **kwargs
            )
            res = [resources_from_tape(tape, False) for tape in batch]

            if len(res) == 1:
                res = res[0]

            if tape_level in level_to_markers:
                trans_name: str = ", ".join(level_to_markers[tape_level])
            elif tape_level == 0:
                trans_name = "Before Tape Transforms"
            else:
                trans_name = compile_pipeline[tape_level - 1].tape_transform.__name__

            trans_name = _make_level_name_unique(trans_name, set(output_level.values()))
            resources[trans_name] = res
            output_level[tape_level] = trans_name

    # Handle MLIR passes
    if len(mlir_levels) > 0:
        try:
            results = mlir_specs(
                qjit,
                mlir_levels,
                *args,
                **kwargs,
                level_to_markers=mlir_level_to_markers,
                existing_level_names=set(output_level.values()),
            )
        except ValueError as ve:
            levels = re.match("Requested specs levels (.*) not found in MLIR pass list.", str(ve))
            bad_levels = [str(int(lvl) + num_tape_levels) for lvl in levels[1].split(", ")]
            raise ValueError(
                f"Requested specs levels {', '.join(bad_levels)} not found in MLIR pass list."
            ) from ve

        for lvl, (level_name, result) in zip(mlir_levels, results.items()):
            output_level[lvl + num_tape_levels] = level_name

            if isinstance(result, list):
                result = [_mlir_resources_to_specs_resources(res) for res in result]
            else:
                result = _mlir_resources_to_specs_resources(result)

            resources[level_name] = result

    # Unpack dictionary to single item if only 1 level was given as input
    if return_single_level:
        resources = next(iter(resources.values()))
        output_level = next(iter(output_level.values()))

    return resources, output_level


# NOTE: Some information is missing from specs_qjit compared to specs_qnode
def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> CircuitSpecs:  # pragma: no cover
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests

    if level is None:
        level = "device"

    # Unwrap the original QNode if any passes have been applied
    if isinstance(qjit.original_function, qml.QNode):
        original_qnode = qjit.original_function
    else:
        raise ValueError(
            "qml.specs can only be applied to a QNode or qjit'd QNode, instead got:",
            qjit.original_function,
        )

    device = original_qnode.device

    if level == "device":
        resources = _specs_qjit_device_level_tracking(
            qjit, original_qnode, compute_depth, *args, **kwargs
        )

    elif isinstance(level, (int, tuple, list, range, str)):
        if compute_depth:
            warnings.warn(
                "Cannot calculate circuit depth for intermediate transformations or compilation passes."
                " To compute the depth, please use level='device'.",
                UserWarning,
            )
        resources, level = _specs_qjit_intermediate_passes(
            qjit, original_qnode, level, *args, **kwargs
        )

    else:
        raise NotImplementedError(f"Unsupported level argument '{level}' for QJIT'd code.")

    return CircuitSpecs(
        resources=resources,
        shots=original_qnode.shots,
        device_name=device.name,
        num_device_wires=(
            len(original_qnode.device.wires) if original_qnode.device.wires is not None else None
        ),
        level=level,
    )


def specs(
    qnode,
    level: str | int | slice | None = None,
    compute_depth: bool | None = None,
) -> Callable[..., CircuitSpecs]:
    r"""Provides the specifications of a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit after applying the specified transforms, expansions, and/or compilation passes.

    Args:
        qnode (:class:`~pennylane.QNode` | :class:`~catalyst.jit.QJIT`): the QNode to calculate the specifications for.

    Keyword Args:
        level (str | int | slice | iter[int]): An indication of which transforms, expansions, and passes to apply before
            computing the resource information. See :func:`~pennylane.workflow.get_compile_pipeline` for more details
            on the available levels. For ``qjit``-compiled workflows, see the sections below for more information.
            Default is ``"device"`` for qjit-compiled workflows or ``"gradient"`` otherwise.
        compute_depth (bool): Whether to compute the depth of the circuit. If ``False``, circuit
            depth will not be included in the output. By default, ``specs`` will always attempt to calculate circuit
            depth (behaves as ``True``), except where not available, such as in pass-by-pass analysis with :func:`~pennylane.qjit` present.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications, including gate and
        measurement data, wire allocations, device information, shots, and more.

    .. warning::

        Computing circuit depth is computationally expensive and can lead to slower ``specs`` calculations.
        If circuit depth is not needed, set ``compute_depth=False``.

    **Example**

    .. code-block:: python

        from pennylane import numpy as pnp

        dev = qml.device("default.qubit", wires=2)
        x = pnp.array([0.1, 0.2])
        Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])
        gradient_kwargs = {"shifts": pnp.pi / 4}

        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
        def circuit(x, add_ry=True):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=(0,1))
            qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
            if add_ry:
                qml.RY(x[1], wires=1)
            qml.TrotterProduct(Hamiltonian, time=1.0, n=4, order=4)
            return qml.probs(wires=(0,1))

    >>> print(qml.specs(circuit)(x, add_ry=False))
    Device: default.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: gradient
    <BLANKLINE>
    Wire allocations: 2
    Total gates: 98
    Gate counts:
    - RX: 1
    - CNOT: 1
    - Evolution: 96
    Measurements:
    - probs(all wires): 1
    Depth: 98

    .. note::

        The available options for ``levels`` are different for circuits which have been compiled using Catalyst.
        There are 2 broad ways to use ``specs`` on ``qjit`` compiled QNodes:

        * Runtime resource tracking via mock circuit execution
        * Pass-by-pass resource collection for user applied compilation passes

    .. details::
        :title: Specs with Tape Transforms

        Here you can see how the number of gates and their types change as we apply different amounts of transforms
        through the ``level`` argument:

        .. code-block:: python

            dev = qml.device("default.qubit")
            gradient_kwargs = {"shifts": pnp.pi / 4}

            @qml.transforms.merge_rotations
            @qml.transforms.undo_swaps
            @qml.transforms.cancel_inverses
            @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
            def circuit(x):
                qml.RandomLayers(pnp.array([[1.0, 2.0]]), wires=(0, 1))
                qml.RX(x, wires=0)
                qml.RX(-x, wires=0)
                qml.SWAP((0, 1))
                qml.X(0)
                qml.X(0)
                return qml.expval(qml.X(0) + qml.Y(1))

        First, we can check the resource information of the QNode without any modifications by specifying ``level=0``. Note that ``level=top`` would
        return the same results:

        >>> print(qml.specs(circuit, level=0)(0.1).resources)
        Wire allocations: 2
        Total gates: 6
        Gate counts:
        - RandomLayers: 1
        - RX: 2
        - SWAP: 1
        - PauliX: 2
        Measurements:
        - expval(Sum(num_wires=2, num_terms=2)): 1
        Depth: 6

        We can analyze the effects of, for example, applying the first two transforms
        (:func:`~pennylane.transforms.cancel_inverses` and :func:`~pennylane.transforms.undo_swaps`) by setting
        ``level=2``. The result will show that ``SWAP`` and ``PauliX`` are not present in the circuit:

        >>> print(qml.specs(circuit, level=2)(0.1).resources)
        Wire allocations: 2
        Total gates: 3
        Gate counts:
        - RandomLayers: 1
        - RX: 2
        Measurements:
        - expval(Sum(num_wires=2, num_terms=2)): 1
        Depth: 3

        We can then check the resources after applying all transforms with ``level="device"`` (which, in this particular example, would be equivalent to ``level=3``):

        >>> print(qml.specs(circuit, level="device")(0.1).resources)
        Wire allocations: 2
        Total gates: 2
        Gate counts:
        - RY: 1
        - RX: 1
        Measurements:
        - expval(Sum(num_wires=2, num_terms=2)): 1
        Depth: 1

        If a QNode with a tape-splitting transform is supplied to the function, with the transform included in the
        desired transforms, the specs output's resources field is instead returned as a list with a
        :class:`~.resource.SpecsResources` for each resulting tape:

        .. code-block:: python

            dev = qml.device("default.qubit")
            H = qml.Hamiltonian([0.2, -0.543], [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Y(2)])
            gradient_kwargs = {"shifts": pnp.pi / 4}

            @qml.transforms.split_non_commuting
            @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
            def circuit():
                qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0, 1))
                return qml.expval(H)

        >>> from pprint import pprint
        >>> pprint(qml.specs(circuit, level="user")())
        CircuitSpecs(device_name='default.qubit',
                     num_device_wires=None,
                     shots=Shots(total_shots=None, shot_vector=()),
                     level='user',
                     resources=[SpecsResources(gate_types={'RandomLayers': 1},
                                               gate_sizes={2: 1},
                                               measurements={'expval(Prod(num_wires=2, num_terms=2))': 1},
                                               num_allocs=2,
                                               depth=1),
                                SpecsResources(gate_types={'RandomLayers': 1},
                                               gate_sizes={2: 1},
                                               measurements={'expval(Prod(num_wires=2, num_terms=2))': 1},
                                               num_allocs=3,
                                               depth=1)])

    .. details::
        :title: Runtime Specs with Catalyst

        **Runtime resource tracking** (specified by ``level="device"``) works by mock-executing the desired
        workflow and tracking the number of times a given gate has been applied. This mock-execution happens
        after all compilation steps, and should be highly accurate to the final gatecounts of running on
        a real device.

        .. code-block:: python

            dev = qml.device("lightning.qubit", wires=3)

            @qml.qjit
            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(x, wires=0)
                qml.RX(x, wires=0)
                qml.X(0)
                qml.X(0)
                qml.CNOT([0, 1])
                return qml.probs()

        >>> print(qml.specs(circuit, level="device")(1.23))
        Device: lightning.qubit
        Device wires: 3
        Shots: Shots(total=None)
        Level: device
        <BLANKLINE>
        Wire allocations: 3
        Total gates: 2
        Gate counts:
        - CNOT: 1
        - RX: 1
        Measurements:
        - probs(all wires): 1
        Depth: 2

    .. details::
        :title: Pass-by-pass Specs with Catalyst

        **Pass-by-pass specs** analyze the intermediate representations of compiled circuits.
        This can be helpful for determining how circuit resources change after a given transform or compilation pass.

        .. warning::
            Some resource information from pass-by-pass specs may be estimated, since it is not always
            possible to determine exact resource usage from intermediate representations.
            For example, resources contained in a ``for`` loop with a non-static range or a ``while`` loop will only be counted as if one iteration occurred.
            Additionally, resources contained in conditional branches from ``if`` or ``switch`` statements will take a union of resources over all branches, providing a tight upper-bound.

            Due to similar technical limitations, depth computation is not available for pass-by-pass specs.

        Pass-by-pass specs can be obtained by providing one of the following values for the ``level`` argument:

        * An ``int``: the desired pass level of a user-applied pass, see the note below
        * A marker name (str): The name of an applied :func:`qml.marker <pennylane.marker>` pass
        * An iterable: A ``list``, ``tuple``, or similar containing ints and/or marker names. Should be sorted in
          ascending pass order with no duplicates
        * The string ``"all"``: To output information about all user-applied transforms and compilation passes
        * The string ``"all-mlir"``: To output information about all compilation passes at the MLIR level only

        .. note::
            The level arguments only take into account user-applied transforms and compilation passes.
            Level ``0`` always corresponds to the original circuit before any user-specified
            tape transforms or compilation passes have been applied,
            and incremental levels correspond to the aggregate of user-specified transforms and passes
            in the order in which they are applied.

            This may include an MLIR "lowering" pass that indicates that the program had to be lowered into MLIR for
            further compilation with Catalyst. If such a pass is included, it will be placed after all tape transforms
            but before all other MLIR passes.

        Here is an example using ``level="all"`` on the circuit from the previous code example:

        >>> all_specs = qml.specs(circuit, level="all")(1.23)
        >>> print(all_specs)
        Device: lightning.qubit
        Device wires: 3
        Shots: Shots(total=None)
        Levels:
        - 0: Before MLIR Passes
        - 1: cancel-inverses
        - 2: merge-rotations
        <BLANKLINE>
        ↓Metric     Level→ |  0 |  1 |  2
        ---------------------------------
        Wire allocations   |  3 |  3 |  3
        Total gates        |  5 |  3 |  2
        Gate counts:       |
        - RX               |  2 |  2 |  1
        - PauliX           |  2 |  0 |  0
        - CNOT             |  1 |  1 |  1
        Measurements:      |
        - probs(all wires) |  1 |  1 |  1

        When invoked with an iterable of levels, or ``"all"`` as above, the returned :class:`~.resource.CircuitSpecs`
        object's ``resources`` field is a dictionary mapping transform names (or marker labels) to their associated
        :class:`~.resource.SpecsResources` object. The keys to this dictionary have human readable names. To use the int
        level name directly, use the ``level`` attribute of the returned :class:`~.resource.CircuitSpecs` object, which
        maps int levels to their associated transform or pass name. For example, the level names for the above example

        >>> print(all_specs.level)
        {0: 'Before MLIR Passes', 1: 'cancel-inverses', 2: 'merge-rotations'}

        The resources associated with a particular level can be accessed using the returned level name as follows:

        >>> print(all_specs.resources['merge-rotations'])
        Wire allocations: 3
        Total gates: 2
        Gate counts:
        - RX: 1
        - CNOT: 1
        Measurements:
        - probs(all wires): 1
        Depth: Not computed

        Or, equivalently, by using the int level directly:

        >>> print(all_specs.resources[all_specs.level[2]])
        Wire allocations: 3
        Total gates: 2
        Gate counts:
        - RX: 1
        - CNOT: 1
        Measurements:
        - probs(all wires): 1
        Depth: Not computed

        .. note::

            The level argument to ``specs`` may be negative. When negative integers are used for
            ``level``, these indices are interpreted as counting backwards with respect to the
            final transform or pass in the sequence. This is similar to how Python slices
            operate, where ``-1`` corresponds to the last applied transform or pass,
            ``-2`` to the second to last, and so on.

        For example, the following will also return the resources after the ``merge-rotations`` pass:

        >>> print(qml.specs(circuit, level=-1)(1.23).resources)
        Wire allocations: 3
        Total gates: 2
        Gate counts:
        - RX: 1
        - CNOT: 1
        Measurements:
        - probs(all wires): 1
        Depth: Not computed

        .. warning::
            Certain transforms, like the ``split-non-commuting`` transform, can result in multiple output tapes.
            In this case, the resources for that level will be returned as a list of :class:`~.resource.SpecsResources`
            objects. When printed, these split tapes will be shown as individual columns.

        .. code-block:: python

            dev = qml.device("lightning.qubit", wires=3)

            @qml.qjit
            @qml.transforms.cancel_inverses
            @qml.transforms.split_non_commuting
            @qml.qnode(dev)
            def circuit():
                qml.X(0)
                qml.X(0)
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

        >>> print(qml.specs(circuit, level="all")())
        Device: lightning.qubit
        Device wires: 3
        Shots: Shots(total=None)
        Levels:
        - 0: Before Tape Transforms
        - 1: split_non_commuting
        - 2: Before MLIR Passes
        - 3: cancel-inverses
        <BLANKLINE>
        ↓Metric   Level→ |    0 |  1-a |  1-b |    2 |    3
        ---------------------------------------------------
        Wire allocations |    1 |    1 |    1 |    6 |    6
        Total gates      |    2 |    2 |    2 |    4 |    0
        Gate counts:     |
        - PauliX         |    2 |    2 |    2 |    4 |    0
        Measurements:    |
        - expval(PauliZ) |    1 |    1 |    0 |    1 |    1
        - expval(PauliX) |    1 |    0 |    1 |    1 |    1

        Note that in the above example, the ``split_non_commuting`` transform results in two tapes, which are labeled as
        ``1-a`` and ``1-b`` in the output. The resources for these tapes are shown separately, and the level name for
        both tapes is the same since they come from the same transform. Multiple tapes may not display as separate
        columns for MLIR passes since MLIR passes do not operate on tapes directly.
    """
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed

    if isinstance(qnode, qml.QNode):
        return partial(_specs_qnode, qnode, level, compute_depth)

    try:
        from ..qnn.torch import TorchLayer

        if isinstance(qnode, TorchLayer) and isinstance(qnode.qnode, qml.QNode):
            return partial(_specs_qnode, qnode, level, compute_depth)
    except ImportError:  # pragma: no cover
        pass

    try:  # pragma: no cover
        # This is tested by integration tests within the Catalyst frontend
        import catalyst

        if isinstance(qnode, catalyst.jit.QJIT):
            return partial(_specs_qjit, qnode, level, compute_depth)
    except ImportError:  # pragma: no cover
        pass

    raise ValueError("qml.specs can only be applied to a QNode or qjit'd QNode")
