# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import inspect
import json
import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import pennylane as qml

from .resource import Resources, SpecsDict, specs_from_tape

_RESOURCE_TRACKING_FILEPATH = "__qml_specs_qjit_resources.json"


def _get_absolute_import_path(fn):
    return f"{inspect.getmodule(fn).__name__}.{fn.__name__}"


def _specs_qnode(qnode, level, compute_depth, *args, **kwargs) -> list[SpecsDict] | SpecsDict:
    """Returns information on the structure and makeup of provided QNode.

    Dictionary keys:
        * ``"num_operations"`` number of operations in the qnode
        * ``"num_observables"`` number of observables in the qnode
        * ``"resources"``: a :class:`~.resource.Resources` object containing resource quantities used by the qnode
        * ``"errors"``: combined algorithmic errors from the quantum operations executed by the qnode
        * ``"num_used_wires"``: number of wires used by the circuit
        * ``"num_device_wires"``: number of wires in device
        * ``"depth"``: longest path in directed acyclic graph representation
        * ``"device_name"``: name of QNode device
        * ``"gradient_options"``: additional configurations for gradient computations
        * ``"interface"``: autodiff framework to dispatch to for the qnode execution
        * ``"diff_method"``: a string specifying the differntiation method
        * ``"gradient_fn"``: executable to compute the gradient of the qnode

    Potential Additional Information:
        * ``"num_trainable_params"``: number of individual scalars that are trainable
        * ``"num_gradient_executions"``: number of times circuit will execute when
                calculating the derivative

    Returns:
        dict[str, Union[defaultdict,int]]: dictionaries that contain QNode specifications
    """

    infos = []
    batch, _ = qml.workflow.construct_batch(qnode, level=level)(*args, **kwargs)

    # These values are the same for the whole batch, so we can just get them once
    config = qml.workflow.construct_execution_config(qnode)(*args, **kwargs)
    gradient_fn = config.gradient_method

    for tape in batch:

        info = specs_from_tape(tape, compute_depth)
        info["num_device_wires"] = len(qnode.device.wires or tape.wires)
        info["num_tape_wires"] = tape.num_wires
        info["device_name"] = qnode.device.name
        info["level"] = level
        info["gradient_options"] = qnode.gradient_kwargs
        info["interface"] = qnode.interface
        info["diff_method"] = (
            _get_absolute_import_path(qnode.diff_method)
            if callable(qnode.diff_method)
            else qnode.diff_method
        )

        if isinstance(gradient_fn, qml.transforms.core.TransformDispatcher):
            info["gradient_fn"] = _get_absolute_import_path(gradient_fn)

            try:
                info["num_gradient_executions"] = len(gradient_fn(tape)[0])
            except Exception as e:  # pylint: disable=broad-except
                # In the case of a broad exception, we don't want the `qml.specs` transform
                # to fail. Instead, we simply indicate that the number of gradient executions
                # is not supported for the reason specified.
                info["num_gradient_executions"] = f"NotSupported: {str(e)}"
        else:
            info["gradient_fn"] = gradient_fn

        infos.append(info)

    return infos[0] if len(infos) == 1 else infos


# NOTE: Some information is missing from specs_qjit compared to specs_qnode
def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> SpecsDict:  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests

    from catalyst.jit import QJIT

    from ..devices import NullQubit

    # TODO: Determine if its possible to have batched QJIT code / how to handle it

    if not isinstance(qjit.original_function, qml.QNode):
        raise ValueError("qml.specs can only be applied to a QNode or qjit'd QNode")

    original_device = qjit.device
    info = SpecsDict()

    if level != "device":
        raise NotImplementedError(f"Unsupported level argument '{level}' for QJIT'd code.")

    # When running at the device level, execute on null.qubit directly with resource tracking,
    # which will give resource usage information for after all compiler passes have completed

    # TODO: Find a way to inherit all devices args from input
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

        info["resources"] = Resources(
            num_wires=resource_data["num_wires"],
            num_gates=resource_data["num_gates"],
            gate_types=defaultdict(int, resource_data["gate_types"]),
            gate_sizes=defaultdict(
                int, {int(k): v for (k, v) in resource_data["gate_sizes"].items()}
            ),
            depth=resource_data["depth"],
            shots=qjit.original_function.shots,  # TODO: Can this ever be overriden during compilation?
        )
    finally:
        # Ensure we clean up the resource tracking file
        if os.path.exists(_RESOURCE_TRACKING_FILEPATH):
            os.remove(_RESOURCE_TRACKING_FILEPATH)

    info["num_device_wires"] = len(original_device.wires)
    info["device_name"] = original_device.name
    info["level"] = level
    info["gradient_options"] = qjit.gradient_kwargs
    info["interface"] = qjit.original_function.interface
    info["diff_method"] = (
        _get_absolute_import_path(qjit.diff_method)
        if callable(qjit.diff_method)
        else qjit.diff_method
    )

    return info


def specs(
    qnode,
    level: None | Literal["top", "user", "device", "gradient"] | int | slice = "gradient",
    compute_depth: bool = True,
) -> Callable[..., list[dict[str, Any]] | dict[str, Any]]:
    r"""Resource information about a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit after applying the specified amount of transforms/expansions first.

    Args:
        qnode (.QNode | .QJIT): the QNode to calculate the specifications for.

    Keyword Args:
        level (None, str, int, slice): An indication of what transforms to apply before computing the resource information.
            Check :func:`~.workflow.get_transform_program` for more information on the allowed values and usage details of
            this argument.
        compute_depth (bool): Whether to compute the depth of the circuit. If ``False``, the depth will not be included in the returned information.

    Returns:
        A function that has the same argument signature as ``qnode``. This function
        returns a dictionary (or a list of dictionaries) of information about qnode structure.

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

    >>> from pprint import pprint
    >>> pprint(qml.specs(circuit)(x, add_ry=False))
    {'device_name': 'default.qubit',
    'diff_method': 'parameter-shift',
    'errors': {'SpectralNormError': SpectralNormError(0.42998560822421455)},
    'gradient_fn': 'pennylane.gradients.parameter_shift.param_shift',
    'gradient_options': {'shifts': 0.7853981633974483},
    'interface': 'auto',
    'level': 'gradient',
    'num_device_wires': 2,
    'num_gradient_executions': 2,
    'num_observables': 1,
    'num_tape_wires': 2,
    'num_trainable_params': 1,
    'resources': Resources(num_wires=2,
                            num_gates=98,
                            gate_types=defaultdict(<class 'int'>,
                                                {'CNOT': 1,
                                                    'Evolution': 96,
                                                    'RX': 1}),
                            gate_sizes=defaultdict(<class 'int'>, {1: 97, 2: 1}),
                            depth=98,
                            shots=Shots(total_shots=None, shot_vector=()))}

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

        >>> print(qml.specs(circuit, level=0)(0.1)["resources"])
        num_wires: 2
        num_gates: 6
        depth: 6
        shots: Shots(total=None)
        gate_types:
        {'RandomLayers': 1, 'RX': 2, 'SWAP': 1, 'PauliX': 2}
        gate_sizes:
        {2: 2, 1: 4}

        We then check the resources after applying all transforms:

        >>> print(qml.specs(circuit, level="device")(0.1)["resources"])
        num_wires: 2
        num_gates: 2
        depth: 1
        shots: Shots(total=None)
        gate_types:
        {'RY': 1, 'RX': 1}
        gate_sizes:
        {1: 2}

        We can also notice that ``SWAP`` and ``PauliX`` are not present in the circuit if we set ``level=2``:

        >>> print(qml.specs(circuit, level=2)(0.1)["resources"])
        num_wires: 2
        num_gates: 3
        depth: 3
        shots: Shots(total=None)
        gate_types:
        {'RandomLayers': 1, 'RX': 2}
        gate_sizes:
        {2: 1, 1: 2}

        If we attempt to apply only the ``merge_rotations`` transform, we end up with only one trainable object, which is in ``RandomLayers``:

        >>> qml.specs(circuit, level=slice(2, 3))(0.1)["num_trainable_params"]
        1

        However, if we apply all transforms, ``RandomLayers`` is decomposed into an ``RY`` and an ``RX``, giving us two trainable objects:

        >>> qml.specs(circuit, level="device")(0.1)["num_trainable_params"]
        2

        If a QNode with a tape-splitting transform is supplied to the function, with the transform included in the desired transforms, a dictionary
        is returned for each resulting tape:

        .. code-block:: python

            dev = qml.device("default.qubit")
            H = qml.Hamiltonian([0.2, -0.543], [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Y(2)])
            gradient_kwargs = {"shifts": pnp.pi / 4}

            @qml.transforms.split_non_commuting
            @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs=gradient_kwargs)
            def circuit():
                qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0, 1))
                return qml.expval(H)

        >>> len(qml.specs(circuit, level="user")())
        2
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
