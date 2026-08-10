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
    get_marker_level_map,
    preprocess_level_input,
    unwrap_partial,
)
from .mlir_specs import resources_from_analysis_pass
from .resource import CircuitSpecs, SpecsResources

# Used for device-level qjit resource tracking
_RESOURCE_TRACKING_PREFIX = "pennylane_specs_qjit_resources"


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
        # which will give resource usage information for after all transforms have completed
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
            counts=resource_data["gate_types"],
            measurement_processes=resource_data["measurements"],
            num_allocs=resource_data["num_wires"],
            circuit_depth=resource_data["depth"],
        )


def _specs_qjit_intermediate_passes(qjit, original_qnode, level, *args, **kwargs) -> tuple[
    SpecsResources | list[SpecsResources] | dict[str, SpecsResources | list[SpecsResources]],
    str | dict[int, str],
]:  # pragma: no cover

    # Note that this only gets transforms manually applied by the user
    compile_pipeline = original_qnode.compile_pipeline

    # Map to convert back and forth between marker name and int level
    marker_to_level = get_marker_level_map(compile_pipeline)
    level_to_markers = defaultdict(list)  # Multiple markers can correspond to the same level
    for marker, lvl in marker_to_level.items():
        level_to_markers[lvl].append(marker)

    return_single_level: bool = isinstance(level, (int, str)) and level != "all"

    # Easier to assume level is always a sorted list of int levels
    level = preprocess_level_input(level, marker_to_level, len(compile_pipeline))
    level_to_name: dict[int, str] = {}

    resources = {}

    # Handle MLIR passes
    resources.update(
        resources_from_analysis_pass(
            qjit,
            original_qnode,
            level,
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

    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    from catalyst import QJIT

    if level is None:
        level = "device"

    # Unwrap the original QNode if any transforms have been applied
    if isinstance(qjit, QJIT) and isinstance(qjit.original_function, qp.QNode):
        original_qnode = qjit.original_function
    else:
        raise ValueError(
            "qp.specs can only be applied to a QNode or qjit'd QNode, instead got: " f"{qjit}",
        )

    device = original_qnode.device

    if level == "device":
        resources = _specs_qjit_device_level_tracking(
            qjit, original_qnode, compute_depth, *args, **kwargs
        )

    elif isinstance(level, (int, tuple, list, range, str)):
        if compute_depth:
            warnings.warn(
                "Cannot calculate circuit depth before applying all transforms."
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
    about the circuit after applying the specified transforms.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the QNode to calculate the specifications for.
            ``functools.partial`` wrappers around supported callables are also accepted.

    Keyword Args:
        level (str | int | iter[int | str] | None): An indication of which transforms to apply before
            computing the resource information. See the sections below for more information about
            acceptable values.
        compute_depth (bool): Whether to compute the depth of the circuit. If ``False``, circuit
            depth will not be included in the output. By default, ``specs`` will always attempt
            to calculate circuit depth (behaves as ``True``), except where not available, such as
            in pass-by-pass analysis for ``qjit``-compiled workflows.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications, including
        gate and measurement data, wire allocations, device information, shots, and more.

    .. warning::

        Computing circuit depth is computationally expensive and can lead to slower ``specs`` calculations.
        If circuit depth is not needed, set ``compute_depth=False``.

    .. note::

        The available options for ``levels`` are:

        * ``"top"`` or ``0``: The original circuit before any transforms have been applied.
        * ``"user"``: The circuit after all user-specified transforms have been applied.
        * ``"device"``: The circuit after all user-specified transforms and device
          preprocessing transforms have been applied.
        * An ``int``: The circuit after the specified number of user-specified transforms have been applied.
        * A marker name (str): The circuit after the specified user-specified transform (and all before
          it) has been applied.
        * An iterable: A ``list``, ``tuple``, or similar containing ints and/or marker names.
          Should be sorted in ascending transform order with no duplicates. The output will provide
          resource information for each level.
        * The string ``"all"``: To provide information at each stage of compilation with respect to
          user-specified transforms.

    **Example**

    .. code-block:: python

        dev = qp.device("null.qubit", wires=2)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.CNOT(wires=(0,1))
            return qp.probs(wires=(0,1))

    >>> print(qp.specs(circuit, level="top")(1.23))
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: Before MLIR Passes
    <BLANKLINE>
    Quantum operations:
    - Total: 2
      - CNOT: 1
      - RX: 1
    Measurement processes:
    - probs(2 wires): 1
    Wire allocations: 2
    Circuit Depth: Not computed

    The :class:`~.resource.SpecsResources` can be accessed using the ``.resources`` attribute, which provides more direct
    access to the data fields. For example:

    >>> qp.specs(circuit)(1.23).resources.quantum_operations
    {'CNOT': 1, 'RX': 1}

    .. details::
        :title: Runtime Specs with Catalyst

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
        Quantum operations:
        - Total: 2
          - CNOT: 1
          - RX: 1
        Measurement processes:
        - probs(all wires): 1
        Wire allocations: 3
        Circuit Depth: 2

        .. note::

            The resources shown when using ``level="device"`` may reflect changes to the circuit
            beyond the transforms manually applied to the QNode. Theses changes are a result of
            additional "device preprocessing" transforms applied to ensure compatibility with
            lowering to MLIR and/or execution on the chosen device.

    .. details::
        :title: Pass-by-pass Specs with Catalyst

        **Pass-by-pass specs** analyze the intermediate representations of compiled circuits.
        This can be helpful for determining how circuit resources change after a given transform.

        .. warning::
            Some resource information from pass-by-pass specs may be estimated, since it is not always
            possible to determine exact resource usage from intermediate representations.
            For example, resources contained in a ``for`` loop with a non-static range or a ``while`` loop will be counted as if only one iteration occurred.
            Additionally, resources contained in conditional branches from ``if`` or ``switch`` statements will take a union of resources over all branches, providing a tight upper-bound.

            Due to similar technical limitations, depth computation is not available for pass-by-pass specs.

        Pass-by-pass specs can be obtained by providing one of the following values for the ``level`` argument:

        * An ``int``: the desired transform level of a user-applied transform, see the note below
        * A marker name (str): The name of an applied :func:`qp.marker <pennylane.marker>` transform
        * An iterable: A ``list``, ``tuple``, or similar containing ints and/or marker names. Should be sorted in
          ascending transform order with no duplicates
        * The string ``"all"``: To provide information at each stage of compilation with respect to user-specified transforms
        * The string ``"all-mlir"``: To provide information at each stage of compilation with respect to user-specified transforms exclusively at the MLIR level
        * The string ``"user"``: To provide information after all user-specified transforms have been applied

        .. note::
            The ``level`` argument is based on user-applied transforms.
            Level ``0`` always corresponds to the original circuit before any user-specified
            transforms have been applied,
            and incremental levels correspond to the aggregate of user-specified transforms
            in the order in which they are applied.

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
        ↓Metric         Level→ |  0 |  1 |  2
        -------------------------------------
        Quantum operations:    |
        - Total                |  5 |  3 |  2
          - CNOT               |  1 |  1 |  1
          - PauliX             |  2 |  0 |  0
          - RX                 |  2 |  2 |  1
        Measurement processes: |
        - probs(all wires)     |  1 |  1 |  1
        Wire allocations       |  3 |  3 |  3

        When invoked with an iterable of levels, or ``"all"`` as above, the resources at different levels can be
        accessed from the the returned :class:`~.resource.CircuitSpecs` object's ``.resources`` attribute, using
        the name of a transform or marker. For example:

        >>> print(all_specs.resources['merge-rotations'])
        Quantum operations:
        - Total: 2
          - CNOT: 1
          - RX: 1
        Measurement processes:
        - probs(all wires): 1
        Wire allocations: 3
        Circuit Depth: Not computed

        A shortcut to access the resources after all user-specified transforms have been
        applied is to use the ``"user"`` level. For example, the following will also return the
        resources after the ``merge-rotations`` transform:

        >>> print(qp.specs(circuit, level="user")(1.23).resources)
        Quantum operations:
        - Total: 2
          - CNOT: 1
          - RX: 1
        Measurement processes:
        - probs(all wires): 1
        Wire allocations: 3
        Circuit Depth: Not computed

        .. warning::
            Certain transforms, like the ``split-non-commuting`` transform, can result in splitting
            a single execution into multiple executions. In this case, the resources for that level
            will be returned as a list of :class:`~.resource.SpecsResources` objects. When printed,
            these split executions will be shown as individual columns.

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=3)

            @qp.qjit
            @qp.transforms.cancel_inverses
            @qp.transform(pass_name="split-non-commuting")
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
        - 0: Before MLIR Passes
        - 1: split-non-commuting
        - 2: cancel-inverses
        <BLANKLINE>
        ↓Metric         Level→ |    0 |  1-a |  1-b |  2-a |  2-b
        ---------------------------------------------------------
        Quantum operations:    |
        - Total                |    2 |    2 |    2 |    0 |    0
        - PauliX             |    2 |    2 |    2 |    0 |    0
        Measurement processes: |
        - expval(PauliX)       |    1 |    0 |    1 |    0 |    1
        - expval(PauliZ)       |    1 |    1 |    0 |    1 |    0
        Wire allocations       |    3 |    3 |    3 |    3 |    3

        Note that in the above example, the ``split-non-commuting`` transform results in two separate executions,
        which are labeled with the suffixes ``-a`` and ``-b`` in the output. The resources for these executions are
        returned and displayed separately, though the level name for both is the same, since they come from the same transform.

    .. details::
        :title: Symbolic Results for Pass-by-pass Specs with Catalyst

        In cases where the exact resources of a circuit are not easily obtained at compile time,
        ``specs`` may return resources which include expressions rather than exact values.
        This can occur when the resources depend on values that are not known at
        compile time, such as the number of iterations in a loop.
        In these cases, the resource information will be returned as a
        :class:`~.resource.SpecsResources` including symbolic expressions,
        rather than one with concrete values.
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
        Quantum operations:
        - Total: b + a + 2
          - Hadamard: 1
          - PauliX: a + 1
          - PauliZ: b
        Measurement processes:
        - expval(PauliZ): 1
        Wire allocations: 1
        Circuit Depth: Not computed

        You can estimate the concrete resource values using the ``.subs`` method of the
        returned :class:`~.resource.SpecsResources` object, and providing keyword arguments
        which describe the mapping from each symbolic variable to an integer value:

        >>> res = specs_result.resources
        >>> print(res.subs(a=5, b=3))
        Quantum operations:
        - Total: 10
          - Hadamard: 1
          - PauliX: 6
          - PauliZ: 3
        Measurement processes:
        - expval(PauliZ): 1
        Wire allocations: 1
        Circuit Depth: Not computed

        These substitutions may also be provided as a dictionary, which can be helpful in
        programmatic contexts:

        >>> print(res.subs({"a": 5, "b": 3}))
        Quantum operations:
        - Total: 10
          - Hadamard: 1
          - PauliX: 6
          - PauliZ: 3
        Measurement processes:
        - expval(PauliZ): 1
        Wire allocations: 1
        Circuit Depth: Not computed
    """
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    return apply_partial_args(
        partial(_specs_qjit, qnode, level, compute_depth), partial_args, partial_kwargs
    )
