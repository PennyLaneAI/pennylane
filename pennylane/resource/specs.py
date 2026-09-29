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

import warnings
from collections.abc import Callable, Iterable
from functools import partial

from pennylane.workflow import QNode

from ._utils import apply_partial_args, build_circuit_specs, unwrap_partial, unwrap_qjit_qnode
from .analyze import _run_resource_analysis
from .resource import CircuitSpecs
from .track import _run_with_resource_tracking


def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> CircuitSpecs:
    original_qnode = unwrap_qjit_qnode(qjit, fn_name="qp.specs")

    # Unwrap the original QNode if any transforms have been applied
    if isinstance(qjit, QJIT) and isinstance(qjit.original_function, qp.QNode):
        return qjit.original_function

    raise ValueError(f"{fn_name} can only be applied to a qjit'd QNode, instead got: {qjit}")


def _build_circuit_specs(original_qnode, resources, level) -> CircuitSpecs:
    """Assemble the ``CircuitSpecs`` describing a qjit'd QNode at a given level."""
    return CircuitSpecs(
        resources=resources,
        shots=original_qnode.shots,
        device_name=original_qnode.device.name,
        num_device_wires=(
            len(original_qnode.device.wires) if original_qnode.device.wires is not None else None
        ),
        level=level,
    )


def _specs_qjit(qjit, level, compute_depth, *args, **kwargs) -> CircuitSpecs:
    original_qnode = _unwrap_qjit_qnode(qjit, fn_name="qp.specs")

    if level is None:
        level = "device"

    if level == "device":
        # Tracking executes the circuit, but specs only reports the resources.
        if compute_depth is None:
            compute_depth = True
        _, resources = _run_with_resource_tracking(
            qjit, original_qnode, *args, compute_depth=compute_depth, **kwargs
        )

    elif isinstance(level, (int, tuple, list, range, str)):
        if compute_depth:
            warnings.warn(
                "Cannot calculate circuit depth before applying all transforms."
                " To compute the depth, please use level='device'.",
                UserWarning,
            )
        resources, level = _run_resource_analysis(qjit, original_qnode, level, *args, **kwargs)

    else:
        raise NotImplementedError(f"Unsupported level argument '{level}'.")

    return build_circuit_specs(original_qnode, resources, level)

    return results, _build_circuit_specs(original_qnode, resources, level)


def specs(
    qnode,
    level: str | int | Iterable[int | str] | None = None,
    compute_depth: bool | None = None,
) -> Callable[..., CircuitSpecs]:
    r"""Provides the specifications of a quantum circuit.

    This transform converts a QNode into a callable that provides resource information
    about the circuit after applying the specified transforms.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the (qjit'd) QNode to calculate the specifications for.
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
        :class:`~.resource.CircuitSpecs` object containing the ``qnode`` specifications, including gate and
        measurement data, total wires, device information, shots, and more.

    .. seealso:: :func:`~.analyze`, which provides the same pass-by-pass analysis, and
        :func:`~.track`, which returns the same device-level information along with the result
        of executing the circuit.

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
    Total wires: 2
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
        Total wires: 3
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

        >>> all_specs = qp.specs(circuit, level="all")(1.23) # doctest: +SKIP
        >>> print(all_specs) # doctest: +SKIP
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
        Total wires            |  3 |  3 |  3

        When invoked with an iterable of levels, or ``"all"`` as above, the resources at different levels can be
        accessed from the the returned :class:`~.resource.CircuitSpecs` object's ``.resources`` attribute, using
        the name of a transform or marker. For example:

        >>> print(all_specs.resources['merge-rotations']) # doctest: +SKIP
        Quantum operations:
        - Total: 2
          - CNOT: 1
          - RX: 1
        Measurement processes:
        - probs(all wires): 1
        Total wires: 3
        Circuit Depth: Not computed

        A shortcut to access the resources after all user-specified transforms have been
        applied is to use the ``"user"`` level. For example, the following will also return the
        resources after the ``merge-rotations`` transform:

        >>> print(qp.specs(circuit, level="user")(1.23).resources)# doctest: +SKIP
        Quantum operations:
        - Total: 2
          - CNOT: 1
          - RX: 1
        Measurement processes:
        - probs(all wires): 1
        Total wires: 3
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

        >>> print(qp.specs(circuit, level="all")()) # doctest: +SKIP
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
        - PauliX               |    2 |    2 |    2 |    0 |    0
        Measurement processes: |
        - expval(PauliX)       |    1 |    0 |    1 |    0 |    1
        - expval(PauliZ)       |    1 |    1 |    0 |    1 |    0
        Total wires            |    3 |    3 |    3 |    3 |    3

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

        >>> specs_result = qp.specs(circuit, level=0)(5, 3)

        If we attempt to get pass-by-pass specs for this circuit, the resource information will be
        symbolic due to the dependence on the input parameters ``x`` and ``z``:

        >>> print(specs_result) # doctest: +SKIP
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
        Total wires: 1
        Circuit Depth: Not computed

        You can estimate the concrete resource values using the ``.subs`` method of the
        returned :class:`~.resource.SpecsResources` object, and providing keyword arguments
        which describe the mapping from each symbolic variable to an integer value:

        >>> res = specs_result.resources # doctest: +SKIP
        >>> print(res.subs(a=5, b=3)) # doctest: +SKIP
        Quantum operations:
        - Total: 10
          - Hadamard: 1
          - PauliX: 6
          - PauliZ: 3
        Measurement processes:
        - expval(PauliZ): 1
        Total wires: 1
        Circuit Depth: Not computed

        These substitutions may also be provided as a dictionary, which can be helpful in
        programmatic contexts:

        >>> print(res.subs({"a": 5, "b": 3})) # doctest: +SKIP
        Quantum operations:
        - Total: 10
          - Hadamard: 1
          - PauliX: 6
          - PauliZ: 3
        Measurement processes:
        - expval(PauliZ): 1
        Total wires: 1
        Circuit Depth: Not computed
    """
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    if isinstance(qnode, QNode):
        raise ValueError(
            "qp.specs no longer supports being applied to a bare QNode; it must be applied to "
            "a qjit'd QNode. Instead, apply qp.qjit to the QNode first or consider "
            "using qp.workflow.construct_tape with qp.resource.resources_from_tape."
        )

    return apply_partial_args(
        partial(_specs_qjit, qnode, level, compute_depth), partial_args, partial_kwargs
    )


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

    .. seealso:: :func:`~.specs`, which returns the same information without the execution result,
        and supports levels other than ``"device"``.

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

        dev = qp.device("null.qubit", wires=2)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=0)
            qp.CNOT(wires=(0,1))
            return qp.probs(wires=(0,1))

    >>> result, circuit_specs = qp.track(circuit)(1.23)
    >>> result.shape
    (4,)
    >>> circuit_specs.resources.quantum_operations
    {'CNOT': 1, 'RX': 1}

    The specifications are the same as the ones returned by :func:`~.specs` at the device level:

    >>> circuit_specs == qp.specs(circuit, level="device")(1.23)
    True
    """
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    return apply_partial_args(partial(_track_qjit, qnode, level), partial_args, partial_kwargs)
