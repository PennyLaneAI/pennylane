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
import warnings
from collections.abc import Callable
from functools import partial

import pennylane as qml

from .error import _compute_algo_error
from .resource import CircuitSpecs, SpecsResources, resources_from_tape

# Used for device-level qjit resource tracking
_RESOURCE_TRACKING_FILEPATH = "__qml_specs_qjit_resources.json"


def _specs_qnode(qnode, level, compute_depth, *args, **kwargs) -> CircuitSpecs:
    """Returns information on the structure and makeup of provided QNode.

    Returns:
        CircuitSpecs: result object that contains QNode specifications
    """

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


# NOTE: Some information is missing from specs_qjit compared to specs_qnode
def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> CircuitSpecs:  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    # Integration tests for this function are within the Catalyst frontend tests, it is not covered by unit tests

    from catalyst.jit import QJIT

    from ..devices import NullQubit

    if not isinstance(qjit.original_function, qml.QNode):
        raise ValueError("qml.specs can only be applied to a QNode or qjit'd QNode")

    original_device = qjit.device

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

        # TODO: Once measurements are tracked for runtime specs, include that data here
        warnings.warn(
            "Measurement resource tracking is not yet supported for qjit'd QNodes. "
            "The returned SpecsResources will have an empty measurements field.",
            UserWarning,
        )
        resources = SpecsResources(
            gate_types=resource_data["gate_types"],
            gate_sizes={int(k): v for (k, v) in resource_data["gate_sizes"].items()},
            measurements={},
            num_allocs=resource_data["num_wires"],
            depth=resource_data["depth"],
        )
    finally:
        # Ensure we clean up the resource tracking file
        if os.path.exists(_RESOURCE_TRACKING_FILEPATH):
            os.remove(_RESOURCE_TRACKING_FILEPATH)

    return CircuitSpecs(
        resources=resources,
        num_device_wires=len(qjit.original_function.device.wires),
        device_name=qjit.original_function.device.name,
        level=level,
        shots=qjit.original_function.shots,
    )


def specs(
    qnode,
    level: str | int | slice = "gradient",
    compute_depth: bool = True,
) -> Callable[..., CircuitSpecs]:
    r"""Provides the specifications of a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit after applying the specified amount of transforms/expansions first.

    Args:
        qnode (.QNode | .QJIT): the QNode to calculate the specifications for.

    Keyword Args:
        level (str | int | slice | iter[int]): An indication of which transforms to apply before computing the resource information.
        compute_depth (bool): Whether to compute the depth of the circuit. If ``False``, the depth will not be included in the returned information. Default: True

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


def _algo_error_qnode(qnode, level, *args, **kwargs) -> dict[str, "AlgorithmicError"]:
    """Returns the algorithmic error dictionary for the provided QNode.

    Returns:
        dict[str, AlgorithmicError]: dictionary with error type names as keys and combined error objects as values
    """

    batch, _ = qml.workflow.construct_batch(qnode, level=level)(*args, **kwargs)

    # Combine errors from all tapes in the batch
    combined_errors = {}
    for tape in batch:
        tape_errors = _compute_algo_error(tape)
        for error_name, error_obj in tape_errors.items():
            if error_name in combined_errors:
                combined_errors[error_name] = combined_errors[error_name].combine(error_obj)
            else:
                combined_errors[error_name] = error_obj

    return combined_errors


def algo_error(
    qnode,
    level: str | int | slice = "gradient",
) -> Callable[..., dict[str, "AlgorithmicError"]]:
    r"""Computes the algorithmic errors in a quantum circuit.

    This transform converts a QNode into a callable that returns a dictionary
    of algorithmic errors after applying the specified amount of transforms/expansions.

    Args:
        qnode (.QNode): the QNode to calculate the algorithmic errors for.

    Keyword Args:
        level (str | int | slice | iter[int]): An indication of which transforms to apply before computing the errors.

    Returns:
        A function that has the same argument signature as ``qnode``. This function
        returns a dictionary with error type names as keys (e.g., ``"SpectralNormError"``)
        and combined :class:`~.resource.AlgorithmicError` objects as values.

    **Example**

    Consider a circuit with operations that introduce algorithmic errors, such as
    :class:`~.TrotterProduct`:

    .. code-block:: python

        import pennylane as qml

        dev = qml.device("null.qubit", wires=2)
        Hamiltonian = qml.dot([1.0, 0.5], [qml.X(0), qml.Y(0)])

        @qml.qnode(dev)
        def circuit(time):
            qml.TrotterProduct(Hamiltonian, time=time, n=4, order=2)
            qml.TrotterProduct(Hamiltonian, time=time, n=4, order=4)
            return qml.state()

    We can compute the errors using ``algo_error``:

    >>> errors = qml.resource.algo_error(circuit)(time=1.0)
    >>> print(errors)
    {'SpectralNormError': SpectralNormError(...)}

    The error values can be accessed from the returned dictionary:

    >>> errors["SpectralNormError"].error
    np.float64(0.4299...)

    .. note::

        This function is the standard way to retrieve algorithm-specific error metrics
        from quantum circuits that use :class:`~.resource.ErrorOperation` subclasses.
        Operations like :class:`~.TrotterProduct` and :class:`~.QuantumPhaseEstimation`
        implement the ``error()`` method and will contribute to the returned error dictionary.

    .. seealso::
        :class:`~.resource.AlgorithmicError`, :class:`~.resource.SpectralNormError`,
        :class:`~.resource.ErrorOperation`, :class:`~.TrotterProduct`
    """
    if isinstance(qnode, qml.QNode):
        return partial(_algo_error_qnode, qnode, level)

    try:
        from ..qnn.torch import TorchLayer  # pylint: disable=import-outside-toplevel

        if isinstance(qnode, TorchLayer) and isinstance(qnode.qnode, qml.QNode):
            return partial(_algo_error_qnode, qnode.qnode, level)
    except ImportError:  # pragma: no cover
        pass

    raise ValueError("qml.resource.algo_error can only be applied to a QNode")
