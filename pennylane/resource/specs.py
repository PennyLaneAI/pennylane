# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import pennylane as qp

from ._utils import (
    apply_partial_args,
    get_last_tape_transform_level,
    get_marker_level_map,
    make_level_name_unique,
    preprocess_level_input,
    unwrap_partial,
)
from .mlir_specs import resources_from_analysis_pass
from .resource import CircuitSpecs, SpecsResources, resources_from_tape

# Used for device-level qjit resource tracking
_RESOURCE_TRACKING_PREFIX = "pennylane_specs_qjit_resources"


def _specs_qnode(qnode, level, compute_depth, *args, **kwargs) -> CircuitSpecs:
    """Returns information on the structure and makeup of provided QNode.

    Returns:
        CircuitSpecs: result object that contains QNode specifications
    """
    if level is None:
        level = "gradient"

    if compute_depth is None:
        compute_depth = True

    batch, _ = qp.workflow.construct_batch(qnode, level=level)(*args, **kwargs)

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

    with tempfile.TemporaryDirectory(
        prefix=f"{_RESOURCE_TRACKING_PREFIX}_{os.getpid()}_"
    ) as tmpdirname:
        filepath = Path(f"{tmpdirname}/{_RESOURCE_TRACKING_PREFIX}_{time.time_ns()}.json")

        # When running at the device level, execute on null.qubit directly with resource tracking,
        # which will give resource usage information for after all compiler passes have completed
        # TODO: Find a way to inherit all devices args from input
        original_device = original_qnode.device
        spoofed_dev = NullQubit(
            target_device=original_device,
            wires=original_device.wires,
            track_resources=True,
            resources_filename=str(filepath),
            compute_depth=compute_depth,
        )

        new_qnode = qjit.original_function.update(device=spoofed_dev)
        new_qjit = QJIT(new_qnode, copy.deepcopy(qjit.compile_options))

        # Execute on null.qubit with resource tracking
        new_qjit(*args, **kwargs)

        with filepath.open("r", encoding="utf-8") as f:
            resource_data = json.load(f)

        return SpecsResources(
            gate_types=resource_data["gate_types"],
            gate_sizes={int(k): v for (k, v) in resource_data["gate_sizes"].items()},
            measurements=resource_data["measurements"],
            num_allocs=resource_data["num_wires"],
            depth=resource_data["depth"],
        )


def _specs_qjit_intermediate_passes(qjit, original_qnode, level, *args, **kwargs) -> tuple[
    SpecsResources | list[SpecsResources] | dict[str, SpecsResources | list[SpecsResources]],
    str | dict[int, str],
]:  # pragma: no cover

    # Note that this only gets transforms manually applied by the user
    compile_pipeline = original_qnode.compile_pipeline

    # This value is used to determine the last level which is a transform and not an MLIR pass
    num_tape_levels = get_last_tape_transform_level(compile_pipeline)
    if num_tape_levels != 0:
        # Account for the "Before Tape Transforms" tape at level 0
        num_tape_levels += 1

    # Map to convert back and forth between marker name and int level
    marker_to_level = get_marker_level_map(compile_pipeline)
    level_to_markers = defaultdict(list)  # Multiple markers can correspond to the same level
    for marker, lvl in marker_to_level.items():
        level_to_markers[lvl].append(marker)

    return_single_level: bool = isinstance(level, (int, str)) and level not in (
        "all",
        "all-mlir",
    )

    # Easier to assume level is always a sorted list of int levels
    level = preprocess_level_input(level, marker_to_level, len(compile_pipeline), num_tape_levels)
    level_to_name: dict[int, str] = {}

    tape_levels = [lvl for lvl in level if lvl < num_tape_levels]
    mlir_levels = [lvl for lvl in level if lvl >= num_tape_levels]

    resources = {}

    # Handle tape transforms
    if len(tape_levels) > 0:
        for tape_level in tape_levels:
            # User transforms always come first, so level and tape_level align correctly
            batch, _ = qp.workflow.construct_batch(original_qnode, level=tape_level)(
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

            trans_name = make_level_name_unique(trans_name, frozenset(level_to_name.values()))
            resources[trans_name] = res
            level_to_name[tape_level] = trans_name

    # Handle MLIR passes
    if len(mlir_levels) > 0:
        resources.update(
            resources_from_analysis_pass(
                qjit,
                original_qnode,
                mlir_levels,
                num_tape_levels,
                level_to_markers,
                level_to_name,
                *args,
                **kwargs,
            )
        )

    # Unpack dictionary to single item if only 1 level was given as input
    if return_single_level:
        resources = next(iter(resources.values()))
        level_to_name = next(iter(level_to_name.values()))

    return resources, level_to_name


def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> CircuitSpecs:  # pragma: no cover
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests

    if level is None:
        level = "device"

    # Unwrap the original QNode if any passes have been applied
    if isinstance(qjit.original_function, qp.QNode):
        original_qnode = qjit.original_function
    else:
        raise ValueError(
            "qp.specs can only be applied to a QNode or qjit'd QNode, instead got:",
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
    level: str | int | slice[int] | Iterable[int | str] | None = None,
    compute_depth: bool | None = None,
) -> Callable[..., CircuitSpecs]:
    r"""Provides the specifications of a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit after applying the specified transforms, expansions, and/or compilation passes.

    Args:
        qnode (:class:`~pennylane.QNode` | :class:`~catalyst.jit.QJIT`): the QNode to calculate the specifications for.
            ``functools.partial`` wrappers around supported callables are also accepted.

    Keyword Args:
        level (str | int | slice | iter[int | str] | None): An indication of which transforms, expansions, and passes to apply before
            computing the resource information. See :func:`~pennylane.workflow.get_compile_pipeline` for more details
            on the available levels without ``qjit``. For ``qjit``-compiled workflows, see the sections below for more information.
            When set to ``None`` (the default), this is treated as ``"device"`` for ``qjit``-compiled workflows or ``"gradient"`` otherwise.
        compute_depth (bool): Whether to compute the depth of the circuit. If ``False``, circuit
            depth will not be included in the output. By default, ``specs`` will always attempt to calculate circuit
            depth (behaves as ``True``), except where not available, such as in pass-by-pass analysis for ``qjit``-compiled workflows.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications, including gate and
        measurement data, wire allocations, device information, shots, and more.

    .. warning::

        Computing circuit depth is computationally expensive and can lead to slower ``specs`` calculations.
        If circuit depth is not needed, set ``compute_depth=False``.

    .. note::

        The available options for ``levels`` are different for circuits which have been compiled using Catalyst.
        There are two broad ways to use ``specs`` on ``qjit``-compiled QNodes:

        * Runtime resource tracking via mock circuit execution
        * Pass-by-pass resource collection for user applied compilation passes

        See related sections below for details regarding use with Catalyst.

    **Example**

    .. code-block:: python

        from pennylane import numpy as pnp

        dev = qp.device("default.qubit", wires=2)
        x = pnp.array([0.1, 0.2])
        Hamiltonian = qp.dot([1.0, 0.5], [qp.X(0), qp.Y(0)])
        gradient_kwargs = {"shifts": pnp.pi / 4}

        @qp.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
        def circuit(x, add_ry=True):
            qp.RX(x[0], wires=0)
            qp.CNOT(wires=(0,1))
            qp.TrotterProduct(Hamiltonian, time=1.0, n=4, order=2)
            if add_ry:
                qp.RY(x[1], wires=1)
            qp.TrotterProduct(Hamiltonian, time=1.0, n=4, order=4)
            return qp.probs(wires=(0,1))

    >>> print(qp.specs(circuit)(x, add_ry=False))
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

    The :class:`~.resource.SpecsResources` can be accessed using the ``.resources`` attribute, which provides more direct
    access to the data fields. For example:

    >>> qp.specs(circuit)(x, add_ry=False).resources.gate_counts
    {'RX': 1, 'CNOT': 1, 'Evolution': 96}

    .. details::
        :title: Specs with Tape Transforms

        Here you can see how the number of gates and their types change as we apply different amounts of transforms
        through the ``level`` argument:

        .. code-block:: python

            dev = qp.device("default.qubit")
            gradient_kwargs = {"shifts": pnp.pi / 4}

            @qp.transforms.merge_rotations
            @qp.transforms.undo_swaps
            @qp.transforms.cancel_inverses
            @qp.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
            def circuit(x):
                qp.RandomLayers(pnp.array([[1.0, 2.0]]), wires=(0, 1))
                qp.RX(x, wires=0)
                qp.RX(-x, wires=0)
                qp.SWAP((0, 1))
                qp.X(0)
                qp.X(0)
                return qp.expval(qp.X(0) + qp.Y(1))

        First, we can inspect the unmodified QNode by setting ``level=0``. Note that ``level="top"`` is equivalent:

        >>> print(qp.specs(circuit, level=0)(0.1).resources)
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

        >>> print(qp.specs(circuit, level=2)(0.1).resources)
        Wire allocations: 2
        Total gates: 3
        Gate counts:
        - RandomLayers: 1
        - RX: 2
        Measurements:
        - expval(Sum(num_wires=2, num_terms=2)): 1
        Depth: 3

        We can then check the resources after applying all user transforms with ``level="user"`` (which, in this particular example,
        would be equivalent to ``level=3``). The two rotations merge and cancel out, leaving us with only ``RandomLayers``:

        >>> print(qp.specs(circuit, level="user")(0.1).resources)
        Wire allocations: 2
        Total gates: 1
        Gate counts:
        - RandomLayers: 1
        Measurements:
        - expval(Sum(num_wires=2, num_terms=2)): 1
        Depth: 1

        After the user transforms, additional transforms for device compatibility and gradient support may be applied. To see the
        resources after all transforms are applied, we can use ``level="device"``. In this case, ``RandomLayers`` is not
        device-compatible and is further decomposed before handing the circuit off to the device:

        >>> print(qp.specs(circuit, level="device")(0.1).resources)
        Wire allocations: 2
        Total gates: 2
        Gate counts:
        - RY: 1
        - RX: 1
        Measurements:
        - expval(Sum(num_wires=2, num_terms=2)): 1
        Depth: 1

        If a QNode with a tape-splitting transform is supplied to the function, the output will provide
        resource information separately for each tape:

        .. code-block:: python

            dev = qp.device("default.qubit")
            H = qp.Hamiltonian([0.2, -0.543], [qp.X(0) @ qp.Z(1), qp.Z(0) @ qp.Y(2)])
            gradient_kwargs = {"shifts": pnp.pi / 4}

            @qp.transforms.split_non_commuting
            @qp.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
            def circuit():
                qp.RandomLayers(qp.numpy.array([[1.0, 2.0]]), wires=(0, 1))
                return qp.expval(H)

        >>> print(qp.specs(circuit, level="user")())
        Device: default.qubit
        Device wires: None
        Shots: Shots(total=None)
        Level: user
        <BLANKLINE>
        Batched tape a:
            Wire allocations: 2
            Total gates: 1
            Gate counts:
            - RandomLayers: 1
            Measurements:
            - expval(Prod(num_wires=2, num_terms=2)): 1
            Depth: 1
        <BLANKLINE>
        Batched tape b:
            Wire allocations: 3
            Total gates: 1
            Gate counts:
            - RandomLayers: 1
            Measurements:
            - expval(Prod(num_wires=2, num_terms=2)): 1
            Depth: 1

        In this case, the ``.resources`` attribute of the returned :class:`~.resource.CircuitSpecs` is a list containing a
        :class:`~.resource.SpecsResources` for each resulting tape:

        >>> qp.specs(circuit, level="user")().resources
        [SpecsResources(gate_types={'RandomLayers': 1}, gate_sizes={2: 1}, measurements={'expval(Prod(num_wires=2, num_terms=2))': 1}, num_allocs=2, depth=1),
         SpecsResources(gate_types={'RandomLayers': 1}, gate_sizes={2: 1}, measurements={'expval(Prod(num_wires=2, num_terms=2))': 1}, num_allocs=3, depth=1)]

    .. details::
        :title: Runtime Specs with Catalyst

        .. note::

            This functionality is specific to workflows with ``qjit``.

        **Runtime resource tracking** (specified by ``level="device"``) works by mock-executing the desired
        workflow and tracking the number of times a given gate has been applied. This mock-execution happens
        after all compilation steps, and should be highly accurate to the final gate counts of running on
        a real device.

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=3)

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

        >>> print(qp.specs(circuit, level="device")(1.23))
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

        .. note::

            The resources shown when using ``level="device"`` may reflect changes to the circuit beyond those applied
            by the user transforms added to the QNode. Theses changes are a result of additional passes applied to ensure
            compatibility with lowering to MLIR and/or execution on the chosen device.

    .. details::
        :title: Pass-by-pass Specs with Catalyst

        .. note::

            This functionality is specific to workflows with ``qjit``.

        **Pass-by-pass specs** analyze the intermediate representations of compiled circuits.
        This can be helpful for determining how circuit resources change after a given transform or compilation pass.

        .. warning::
            Some resource information from pass-by-pass specs may be estimated, since it is not always
            possible to determine exact resource usage from intermediate representations.
            For example, resources contained in a ``for`` loop with a non-static range or a ``while`` loop will be counted as if only one iteration occurred.
            Additionally, resources contained in conditional branches from ``if`` or ``switch`` statements will take a union of resources over all branches, providing a tight upper-bound.

            Due to similar technical limitations, depth computation is not available for pass-by-pass specs.

        Pass-by-pass specs can be obtained by providing one of the following values for the ``level`` argument:

        * An ``int``: the desired pass level of a user-applied pass, see the note below
        * A marker name (str): The name of an applied :func:`qp.marker <pennylane.marker>` pass
        * An iterable: A ``list``, ``tuple``, or similar containing ints and/or marker names. Should be sorted in
          ascending pass order with no duplicates
        * The string ``"all"``: To provide information at each stage of compilation with respect to user-specified transforms
        * The string ``"all-mlir"``: To provide information at each stage of compilation with respect to user-specified transforms exclusively at the MLIR level
        * The string ``"user"``: To provide information after all user-specified transforms have been applied

        .. note::
            The ``level`` argument is based on user-applied transforms and compilation passes.
            Level ``0`` always corresponds to the original circuit before any user-specified
            tape transforms or compilation passes have been applied,
            and incremental levels correspond to the aggregate of user-specified transforms and passes
            in the order in which they are applied.

            In addition to the user-passes, pass-by-pass inspection will indicate where the MLIR
            "lowering" occurs with the ``Before MLIR Passes`` stage. This will be placed after all tape
            transforms, but before all other MLIR passes. Note that this may be at level ``0`` if there are no tape transforms.
            In some cases, the pass to lower to MLIR will
            apply additional transforms to the circuit to ensure compatibility with the MLIR representation
            and/or with the device, so resources may change as a result of this pass.


        Consider the following circuit:

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=3)

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

        We can get a pass-by-pass overview of the resources using ``level="all"``:

        >>> all_specs = qp.specs(circuit, level="all")(1.23)
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
        - CNOT             |  1 |  1 |  1
        - PauliX           |  2 |  0 |  0
        - RX               |  2 |  2 |  1
        Measurements:      |
        - probs(all wires) |  1 |  1 |  1

        When invoked with an iterable of levels, or ``"all"`` as above, the resources at different levels can be
        accessed from the the returned :class:`~.resource.CircuitSpecs` object's ``.resources`` attribute, using
        the name of a pass or marker. For example:

        >>> print(all_specs.resources['merge-rotations'])
        Wire allocations: 3
        Total gates: 2
        Gate counts:
        - CNOT: 1
        - RX: 1
        Measurements:
        - probs(all wires): 1
        Depth: Not computed

        A shortcut to access the resources after all user-specified transforms and passes have been
        applied is to use the ``"user"`` level. For example, the following will also return the
        resources after the ``merge-rotations`` pass:

        >>> print(qp.specs(circuit, level="user")(1.23).resources)
        Wire allocations: 3
        Total gates: 2
        Gate counts:
        - CNOT: 1
        - RX: 1
        Measurements:
        - probs(all wires): 1
        Depth: Not computed

        .. warning::
            Certain transforms, like the ``split_non_commuting`` transform, can result in splitting a single execution
            into multiple executions. In this case, the resources for that level will be returned as a list of
            :class:`~.resource.SpecsResources` objects. When printed, these split tapes will be shown as individual columns.

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=3)

            @qp.qjit
            @qp.transforms.cancel_inverses
            @qp.transforms.split_non_commuting
            @qp.qnode(dev)
            def circuit():
                qp.X(0)
                qp.X(0)
                return qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliX(0))

        >>> print(qp.specs(circuit, level="all")())
        Device: lightning.qubit
        Device wires: 3
        Shots: Shots(total=None)
        Levels:
        - 0: Before Tape Transforms
        - 1: split_non_commuting
        - 2: Before MLIR Passes
        - 3: cancel-inverses
        <BLANKLINE>
        ↓Metric   Level→ |    0 |  1-a |  1-b |  2-a |  2-b |  3-a |  3-b
        -----------------------------------------------------------------
        Wire allocations |    1 |    1 |    1 |    3 |    3 |    3 |    3
        Total gates      |    2 |    2 |    2 |    2 |    2 |    0 |    0
        Gate counts:     |
        - PauliX         |    2 |    2 |    2 |    2 |    2 |    0 |    0
        Measurements:    |
        - expval(PauliZ) |    1 |    1 |    0 |    1 |    0 |    1 |    0
        - expval(PauliX) |    1 |    0 |    1 |    0 |    1 |    0 |    1

        Note that in the above example, the ``split_non_commuting`` transform results in two separate executions,
        which are labeled with the suffixes ``-a`` and ``-b`` in the output. The resources for these executions are
        returned and displayed separately, though the level name for both is the same, since they come from the same transform.

    .. details::
        :title: Symbolic Results for Pass-by-pass Specs with Catalyst

        In cases where the exact resources of a circuit are not easily obtained at compile time,
        ``specs`` may return resources which include expressions rather than exact values.
        This can occur when the resources depend on values that are not known at
        compile time, such as the number of iterations in a loop.
        In these cases, the resource information will be returned as a
        :class:`~.resource.SymbolicSpecsResources` including symbolic expressions,
        rather than a
        :class:`~.resource.SpecsResources` with concrete values.

        For example, consider the following circuit which contains a ``for`` loop with a
        non-static range:

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=1)

            @qp.qjit(autograph=True)
            @qp.qnode(dev)
            def circuit(x, z):
                qp.Hadamard(0)
                qp.PauliX(0)
                for _ in range(x):
                    qp.PauliX(0)
                for _ in range(z):
                    qp.PauliZ(0)
                return qp.expval(qp.PauliZ(0))

            specs_result = qp.specs(circuit, level=0)(5, 3)

        If we attempt to get pass-by-pass specs for this circuit, the resource information will be
        symbolic due to the dependence on the input parameters ``x`` and ``z``:

        >>> print(specs_result)
        Device: lightning.qubit
        Device wires: 1
        Shots: Shots(total=None)
        Level: Before MLIR Passes
        <BLANKLINE>
        Symbolic Variables: a, b
        Wire allocations: 1
        Total gates: b + a + 2
        Gate counts:
        - Hadamard: 1
        - PauliX: a + 1
        - PauliZ: b
        Measurements:
        - expval(PauliZ): 1
        Depth: Not computed

        You can estimate the concrete resource values using the ``.subs`` method of the
        returned :class:`~.resource.SymbolicSpecsResources` object, and providing keyword arguments
        which describe the mapping from each symbolic variable to an integer value:

        >>> res = specs_result.resources
        >>> print(res.subs(a=5, b=3))
        Wire allocations: 1
        Total gates: 10
        Gate counts:
        - Hadamard: 1
        - PauliX: 6
        - PauliZ: 3
        Measurements:
        - expval(PauliZ): 1
        Depth: Not computed

        These substitutions may also be provided as a dictionary, which can be helpful in
        programmatic contexts:

        >>> print(res.subs({"a": 5, "b": 3}))
        Wire allocations: 1
        Total gates: 10
        Gate counts:
        - Hadamard: 1
        - PauliX: 6
        - PauliZ: 3
        Measurements:
        - expval(PauliZ): 1
        Depth: Not computed
    """
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed

    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    specs_fn = _specs_qnode if isinstance(qnode, qp.QNode) else None

    if specs_fn is None:
        try:
            from ..qnn.torch import TorchLayer

            if isinstance(qnode, TorchLayer) and isinstance(qnode.qnode, qp.QNode):
                specs_fn = _specs_qnode
        except ImportError:  # pragma: no cover
            pass

    if specs_fn is None:
        try:  # pragma: no cover
            # This is tested by integration tests within the Catalyst frontend
            import catalyst

            if isinstance(qnode, catalyst.jit.QJIT):
                specs_fn = _specs_qjit
        except ImportError:  # pragma: no cover
            pass

    if specs_fn is not None:
        return apply_partial_args(
            partial(specs_fn, qnode, level, compute_depth), partial_args, partial_kwargs
        )

    raise ValueError("qp.specs can only be applied to a QNode or qjit'd QNode")
