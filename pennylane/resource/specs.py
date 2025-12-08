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

import copy
import json
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial

import pennylane as qml

from .resource import CircuitSpecs, SpecsResources, resources_from_tape

# Used for device-level qjit resource tracking
_RESOURCE_TRACKING_FILEPATH = "__qml_specs_qjit_resources.json"


def _specs_qnode(qnode, level, compute_depth, *args, **kwargs) -> CircuitSpecs:
    """Returns information on the structure and makeup of provided QNode.

    Returns:
        CircuitSpecs: result object that contains QNode specifications
    """

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
    qjit, original_qnode, pass_pipeline_wrapped, compute_depth, *args, **kwargs
) -> SpecsResources:  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    import catalyst
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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The device's shots value does not match "
        )
        if pass_pipeline_wrapped:
            new_qnode = original_qnode.update(device=spoofed_dev)

            def recursively_add_passes(pass_pipeline):
                if isinstance(pass_pipeline, catalyst.passes.pass_api.PassPipelineWrapper):
                    inner_fxn = recursively_add_passes(pass_pipeline.qnode)
                    new_pass_pipeline = catalyst.passes.pass_api.PassPipelineWrapper(
                        inner_fxn,
                        pass_pipeline.pass_name_or_pipeline,
                        *pass_pipeline.flags,
                        **pass_pipeline.valued_options,
                    )
                    return new_pass_pipeline
                return new_qnode

            pass_pipeline = recursively_add_passes(qjit.original_function)
            new_qjit = QJIT(pass_pipeline, copy.copy(qjit.compile_options))
        else:
            new_qnode = qjit.original_function.update(device=spoofed_dev)
            new_qjit = QJIT(new_qnode, copy.copy(qjit.compile_options))

    if os.path.exists(_RESOURCE_TRACKING_FILEPATH):
        # TODO: Warn that something has gone wrong here
        os.remove(_RESOURCE_TRACKING_FILEPATH)

    try:
        # Execute on null.qubit with resource tracking
        new_qjit(*args, **kwargs)

        with open(_RESOURCE_TRACKING_FILEPATH, encoding="utf-8") as f:
            resource_data = json.load(f)

        # TODO: Once measurements are tracked for runtime specs, include that data here
        warnings.warn(
            "Measurement resource tracking is not yet supported for qjit'd QNodes. "
            "The returned SpecsResources will have an empty measurements field.",
            UserWarning,
        )
        return SpecsResources(
            gate_types=resource_data["gate_types"],
            gate_sizes={int(k): v for (k, v) in resource_data["gate_sizes"].items()},
            measurements={},  # Not tracked at the moment
            num_allocs=resource_data["num_wires"],
            depth=resource_data["depth"],
        )
    finally:
        # Ensure we clean up the resource tracking file
        if os.path.exists(_RESOURCE_TRACKING_FILEPATH):
            os.remove(_RESOURCE_TRACKING_FILEPATH)


def _specs_qjit_intermediate_passes(
    qjit, original_qnode, level, *args, **kwargs
) -> (
    SpecsResources | list[SpecsResources] | dict[str, SpecsResources | list[SpecsResources]]
):  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    from catalyst.python_interface.inspection import mlir_specs

    # Note that this only gets transforms manually applied by the user
    trans_prog = original_qnode.transform_program

    single_level = isinstance(level, (int, str)) and not level in ("all", "all-mlir")

    # Levels where qml.marker transforms are applied, needed since markers may be applied before or
    # after the first MLIR transform, and ones after need to be incremented by 1 to account for the
    # extra lowering pass
    marker_to_level = {
        trans.kwargs["level"]: i + 1
        for i, trans in enumerate(trans_prog)
        if trans.transform == qml.marker.transform
    }
    level_to_marker = {v: k for k, v in marker_to_level.items()}

    # Easier to assume level is always a sorted list of int levels (if not "all" or "all-mlir")
    if level not in ("all", "all-mlir"):
        if single_level:
            level = [level]
        else:
            level = list(level)

        # Convert marker names to the associated level number
        for i, lvl in enumerate(level):
            if isinstance(lvl, str):
                if lvl not in marker_to_level:
                    raise ValueError(f"Transform name '{lvl}' not found in the transform program.")
                level[i] = marker_to_level[lvl]
            elif isinstance(lvl, int):
                if lvl < 0:
                    raise ValueError(
                        "The 'level' argument to qml.specs for QJIT'd QNodes must be non-negative, "
                        f"got {lvl}."
                    )

        level_sorted = sorted(level)
        if level != level_sorted:
            warnings.warn(
                "The 'level' argument to qml.specs for QJIT'd QNodes has been sorted to be in ascending order.",
                UserWarning,
            )
            level = level_sorted

    resources = {}

    if level != "all-mlir":
        if qml.capture.enabled():
            # If capture is enabled, find the seam where PLxPR transforms end and MLIR passes begin
            num_trans_levels = 0

            # If the pass name is None, it indicates a PLxPR transform which is not recognized by Catalyst
            for i, trans in reversed(list(enumerate(trans_prog))):
                if trans.pass_name is None:
                    num_trans_levels = i + 1
                    break

        else:
            # If capture is NOT enabled, all transforms are tape transforms
            num_trans_levels = len(trans_prog)

        num_trans_levels += 1  # Have to include the "before transforms" level

        if level != "all":
            # Account for off-by-one error
            # TODO: This is actually currently unused, since markers are tape transforms only
            level = [
                lvl + 1 if lvl in level_to_marker and lvl >= num_trans_levels else lvl
                for lvl in level
            ]

        # Handle tape transforms
        trans_levels = (
            list(range(num_trans_levels))
            if level == "all"
            else [lvl for lvl in level if lvl < num_trans_levels]
        )

        # Handle tape transforms
        for trans_level in trans_levels:
            # User transforms always come first, so level and trans_level align correctly
            batch, _ = qml.workflow.construct_batch(original_qnode, level=trans_level)(
                *args, **kwargs
            )
            res = [resources_from_tape(tape, False) for tape in batch]

            if len(res) == 1:
                res = res[0]

            if trans_level == 0:
                trans_name = "Before transforms"
            elif trans_level in level_to_marker:
                trans_name = level_to_marker[trans_level]
            else:
                trans_name = trans_prog[trans_level - 1].transform.__name__

            # If the same transform appears multiple times, append a suffix
            if trans_name in resources:
                rep = 2
                while f"{trans_name}-{rep}" in resources:
                    rep += 1
                trans_name += f"-{rep}"
            resources[trans_name] = res

    # Handle MLIR levels
    mlir_levels = (
        [lvl - num_trans_levels for lvl in level if lvl >= num_trans_levels]
        if level not in ("all", "all-mlir")
        else "all"
    )
    if mlir_levels == "all" or len(mlir_levels) > 0:
        try:
            results = mlir_specs(qjit, mlir_levels, *args, **kwargs)
        except ValueError as ve:
            levels = re.match("Requested specs levels (.*) not found in MLIR pass list.", str(ve))
            bad_levels = [str(int(lvl) + num_trans_levels) for lvl in levels[1].split(", ")]
            raise ValueError(
                f"Requested specs levels {', '.join(bad_levels)} not found in MLIR pass list."
            ) from ve

        for level_name, res in results.items():
            gate_sizes = defaultdict(int)
            for _, sizes in res.operations.items():
                for size, count in sizes.items():
                    gate_sizes[size] += count

            res_resources = SpecsResources(
                gate_types={r: sum(sizes.values()) for r, sizes in res.operations.items()},
                gate_sizes=dict(gate_sizes),
                measurements=dict(res.measurements),
                num_allocs=res.num_allocs,
                depth=None,  # Can't get depth for intermediate stages
            )
            resources[level_name] = res_resources

    # Unpack dictionary to single item if only 1 level was given as input
    if single_level:
        resources = next(iter(resources.values()))
        level = level[0]

    return resources


# NOTE: Some information is missing from specs_qjit compared to specs_qnode
def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> CircuitSpecs:  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests
    from catalyst.passes.pass_api import PassPipelineWrapper

    # Unwrap the original QNode if any passes have been applied
    pass_pipeline_wrapped = False
    if isinstance(qjit.original_function, PassPipelineWrapper):
        pass_pipeline_wrapped = True
        original_qnode = qjit.original_qnode
    elif isinstance(qjit.original_function, qml.QNode):
        original_qnode = qjit.original_function
    else:
        raise ValueError(
            "qml.specs can only be applied to a QNode or qjit'd QNode, instead got:",
            qjit.original_function,
        )

    device = original_qnode.device

    if level == "device":
        resources = _specs_qjit_device_level_tracking(
            qjit, original_qnode, pass_pipeline_wrapped, compute_depth, *args, **kwargs
        )

    elif isinstance(level, (int, tuple, list, range, str)):
        if compute_depth:
            warnings.warn(
                "Cannot calculate circuit depth for intermediate transformations or compilation passes."
                " To compute the depth, please use level='device'.",
                UserWarning,
            )
        resources = _specs_qjit_intermediate_passes(qjit, original_qnode, level, *args, **kwargs)

    else:
        raise NotImplementedError(f"Unsupported level argument '{level}' for QJIT'd code.")

    if isinstance(resources, dict):
        level = list(resources.keys())

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
    level: str | int | slice = "gradient",
    compute_depth: bool | None = None,
) -> Callable[..., CircuitSpecs]:
    r"""Provides the specifications of a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit after applying the specified amount of transforms/expansions first.

    Args:
        qnode (.QNode | .QJIT): the QNode to calculate the specifications for.

    Keyword Args:
        level (str | int | slice | iter[int]): An indication of what transforms to apply before
        computing the resource information.
        compute_depth (bool): Whether to compute the depth of the circuit. If ``False``, the depth
        will not be included in the returned information. Default: True where available.

    Returns:
        A function that has the same argument signature as ``qnode``. This function
        returns a :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications.

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
    Resource specifications:
      Total qubit allocations: 2
      Total gates: 98
      Circuit depth: 98
    <BLANKLINE>
      Gate types:
        RX: 1
        CNOT: 1
        Evolution: 96
    <BLANKLINE>
      Measurements:
        probs(all wires): 1

    .. details::
        :title: Usage Details

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

        First, we can check the resource information of the QNode without any modifications. Note that ``level=top`` would
        return the same results:

        >>> print(qml.specs(circuit, level=0)(0.1).resources)
        Total qubit allocations: 2
        Total gates: 6
        Circuit depth: 6
        <BLANKLINE>
        Gate types:
          RandomLayers: 1
          RX: 2
          SWAP: 1
          PauliX: 2
        <BLANKLINE>
        Measurements:
          expval(Sum(num_wires=2, num_terms=2)): 1

        We then check the resources after applying all transforms:

        >>> print(qml.specs(circuit, level="device")(0.1).resources)
        Total qubit allocations: 2
        Total gates: 2
        Circuit depth: 1
        <BLANKLINE>
        Gate types:
          RY: 1
          RX: 1
        <BLANKLINE>
        Measurements:
          expval(Sum(num_wires=2, num_terms=2)): 1

        We can also notice that ``SWAP`` and ``PauliX`` are not present in the circuit if we set ``level=2``:

        >>> print(qml.specs(circuit, level=2)(0.1).resources)
        Total qubit allocations: 2
        Total gates: 3
        Circuit depth: 3
        <BLANKLINE>
        Gate types:
          RandomLayers: 1
          RX: 2
        <BLANKLINE>
        Measurements:
          expval(Sum(num_wires=2, num_terms=2)): 1

        If a QNode with a tape-splitting transform is supplied to the function, with the transform included in the
        desired transforms, the specs output's resources field is instead returned as a list with a
        :class:`~.resource.CircuitSpecs` for each resulting tape:

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
