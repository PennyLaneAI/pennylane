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
"""Code for tracking the resources of an executed circuit"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from ._utils import apply_partial_args, build_circuit_specs, unwrap_partial, unwrap_qjit_qnode
from .resource import CircuitSpecs, SpecsResources

# Used for device-level qjit resource tracking
_RESOURCE_TRACKING_PREFIX = "pennylane_track_resources"


def _run_with_resource_tracking(
    qjit, original_qnode, *args, compute_depth: bool, **kwargs
) -> tuple[Any, SpecsResources]:
    """Execute a qjit'd QNode on ``null.qubit`` with resource tracking enabled.

    Returns the result of the execution along with the resources counted while running it.
    """
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    from catalyst import QJIT

    from ..devices import NullQubit

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

        new_qnode = original_qnode.update(device=spoofed_dev)
        new_qjit = QJIT(new_qnode, copy.deepcopy(qjit.compile_options))

        # Execute on null.qubit with resource tracking
        results = new_qjit(*args, **kwargs)

        with filepath.open("r", encoding="utf-8") as f:
            resource_data = json.load(f)

        return results, SpecsResources(
            counts=resource_data["gate_types"],
            measurement_processes=resource_data["measurements"],
            num_wires=resource_data["num_wires"],
            circuit_depth=resource_data["depth"],
        )


def _track_qjit(qjit, level, *args, **kwargs) -> tuple[Any, CircuitSpecs]:
    """Execute a qjit'd QNode on ``null.qubit`` and return its result along with its specs."""
    original_qnode = unwrap_qjit_qnode(qjit, fn_name="qp.track")

    if level != "device":
        raise NotImplementedError(f"qp.track only supports level='device', instead got: {level!r}.")

    results, resources = _run_with_resource_tracking(
        qjit, original_qnode, *args, compute_depth=True, **kwargs
    )

    return results, build_circuit_specs(original_qnode, resources, level)


def track(
    qnode,
    level: str = "device",
) -> Callable[..., tuple[Any, CircuitSpecs]]:
    r"""Executes a quantum circuit and tracks the resources it uses.

    This transform converts a QNode into a callable that executes the circuit on
    ``null.qubit`` and returns both the result of that execution and the resource
    information gathered while running it.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the (qjit'd) QNode to execute and track.
            ``functools.partial`` wrappers around supported callables are also accepted.

    Keyword Args:
        level (str): The level at which to track resources. Only ``"device"`` is currently
            supported, meaning that resources are counted after all user-specified transforms
            and device preprocessing transforms have been applied.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        tuple containing the result of executing the circuit and a
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications,
        including gate and measurement data, total wires, device information, shots, and more.


    .. note::

        Resources are tracked by mock-executing the workflow on ``null.qubit``. For a QNode bound
        to any other device, the returned execution result therefore carries the shape and dtype
        of that device's result, but not its values.

    .. warning::

        ``null.qubit`` does not perform a true state-vector simulation, so mid-circuit measurement
        outcomes are not grounded in real measurement statistics. If the circuit contains a
        conditional whose branch depends on such an outcome, the reported circuit depth and any
        branch-dependent gate counts correspond to the branch taken during the mock execution, and
        should be treated as an estimate rather than exact counts.

    **Example**

    .. code-block:: python

        n_wires = 2
        dev = qp.device("lightning.qubit", wires=n_wires)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.CNOT(wires=(0,1))
            return qp.probs(wires=(0,1))

    >>> result, circuit_specs = qp.track(circuit)(1.23)
    >>> circuit_specs.resources.quantum_operations
    {'CNOT': 1, 'RX': 1}

    Since the circuit is executed on ``null.qubit``, the result has the shape and dtype of the
    ``lightning.qubit`` result, but not its values:

    >>> result
    Array([1., 0., 0., 0.], dtype=float64)
    >>> circuit(1.23)
    Array([0.66711886, 0.        , 0.        , 0.33288114], dtype=float64)

    The specifications are the same as the ones returned by :func:`~.specs` at the device level:

    >>> circuit_specs == qp.specs(circuit, level="device")(1.23)
    True

    Because the circuit is executed, control flow is unrolled with the given arguments and
    the resources are concrete numbers:

    .. code-block:: python

        @qp.qjit(autograph=True)
        @qp.qnode(dev)
        def circuit(n):
            for i in range(n):
                qp.Hadamard(wires=i % n_wires)
            qp.CNOT(wires=(0, 1))
            return qp.probs(wires=(0, 1))

    >>> _, circuit_specs = qp.track(circuit)(3)
    >>> circuit_specs.resources.quantum_operations
    {'CNOT': 1, 'Hadamard': 3}

    In contrast, the compile-time analysis reports the number of loop iterations
    symbolically, since it does not depend on the runtime value of ``n``:

    >>> qp.specs(circuit, level=0)(3).resources.quantum_operations
    {'CNOT': 1, 'Hadamard': Expression({('a',): 1})}
    """
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    return apply_partial_args(partial(_track_qjit, qnode, level), partial_args, partial_kwargs)
