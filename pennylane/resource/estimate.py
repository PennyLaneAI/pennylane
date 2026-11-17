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
"""Code for projecting the resources of a circuit onto a target gate set"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable
from functools import partial

from pennylane import capture
from pennylane.core.transforms import CompilePipeline
from pennylane.decomposition import GateSet
from pennylane.transforms import decompose
from pennylane.workflow._levels import (
    device_level,
    is_int_level,
    is_single_level,
    level_to_number,
    pipeline_at_slice,
    validate_level,
)
from pennylane.workflow.get_compile_pipeline import unwrap_qjit_qnode

from ._utils import (
    apply_partial_args,
    build_circuit_specs,
    check_no_tape_transforms,
    unwrap_partial,
)
from .mlir_specs import resources_from_analysis_pass
from .resource import CircuitSpecs

_DECOMPOSE_PASS = "graph-decomposition"

# Passes of the transforms that estimate can skip without a warning: they only remove, merge or
# reorder gates, or split the measurements, so skipping them does not add gate types, though
# merge-rotations can merge Pauli product rotations into a larger angle.
# Any other transform may introduce new gate types.
# The pass names are not read from the transforms, which are plain functions in docs builds.
# TODO: let each transform declare whether it is safe to skip, instead of listing them here.
_SAFE_TO_SKIP_TRANSFORMS = frozenset(
    {"cancel-inverses", "merge-rotations", "combine-global-phases", "split-to-single-terms"}
)

_GRIDSYNTH_ROTATIONS = ("RZ", "PhaseShift", "Adjoint(RZ)", "Adjoint(PhaseShift)")


def _decompose_kwargs(bound_transform) -> dict:
    """Return the arguments of a decomposition that ``estimate`` uses."""
    return {
        key: bound_transform.kwargs.get(key)
        for key in ("gate_set", "fixed_decomps", "exclude_decomps")
    }


def _device_gate_set(qnode) -> dict[str, float]:
    """Return the gate set of the decomposition in the device preprocessing that Catalyst applies
    to ``qnode``."""
    # pylint: disable-next=import-outside-toplevel
    from catalyst.from_plxpr.device_utils import create_device_preprocessing_pipeline

    from pennylane.devices import ExecutionConfig  # pylint: disable=import-outside-toplevel

    # The gate set depends on the shots, but not on the execution config
    pipeline = create_device_preprocessing_pipeline(
        qnode.device, ExecutionConfig(), qnode.shots.total_shots or 0, warn=False
    )
    gate_set = next(
        (
            bound_transform.kwargs["gate_set"]
            for bound_transform in reversed(pipeline)
            if bound_transform.pass_name == _DECOMPOSE_PASS
        ),
        None,
    )
    if gate_set is None:
        raise NotImplementedError(
            f"qp.estimate could not find the gate set of the device {qnode.device.name}, since "
            "its preprocessing has no decomposition. Provide a gate set as target instead."
        )
    return gate_set


def _target_level(target, pipeline: CompilePipeline) -> int | None:
    """Return the level number of ``target`` in ``pipeline``, or ``None`` if ``target`` is a gate
    set. A marker label takes priority over a gate with the same name."""
    if target is None or isinstance(target, bool):
        raise ValueError(
            f"Invalid target={target!r}, expected a gate set or a level of the compile pipeline."
        )
    if isinstance(target, slice):
        raise ValueError(
            f"Invalid target={target!r}, which selects several levels of the compile pipeline."
        )
    if isinstance(target, str):
        if target in ("all-user", "all"):
            raise ValueError(
                f"Invalid target={target!r}, which selects several levels of the compile pipeline."
            )
        if target not in ("top", "user", "device", "gradient", "all-mlir") and (
            target not in pipeline.markers
        ):
            return None
    elif not is_int_level(target):
        return None
    try:
        return level_to_number(target, pipeline)
    except ValueError as exc:
        raise ValueError(f"Invalid target={target!r}: {exc}") from None


def _estimate_passes(qnode, target) -> tuple[CompilePipeline, dict | None, list[tuple]]:
    """Return the decompositions to apply to ``qnode`` before estimating the resources at
    ``target``, the arguments of the decomposition to estimate, and the skipped transforms that
    may change the gates.

    The arguments are ``None`` if no decomposition is estimated, so that the resources are those
    at level 0. Each skipped transform comes with whether the estimated resources are the gates
    that it would act on, which is the case if no decomposition comes after it."""
    pipeline = qnode.compile_pipeline
    target_level = _target_level(target, pipeline)
    if target_level is None:
        if any(
            bound_transform.kwargs.get("fixed_decomps")
            or bound_transform.kwargs.get("exclude_decomps")
            for bound_transform in pipeline
            if bound_transform.pass_name == _DECOMPOSE_PASS
        ):
            warnings.warn(
                "qp.estimate ignores the fixed_decomps and exclude_decomps of the qp.decompose "
                "transforms in the compile pipeline, since the target is a gate set. Use a level "
                'of the compile pipeline as target, such as target="user", to use them.',
                UserWarning,
            )
        return CompilePipeline(), {"gate_set": target}, []

    # As in qp.analyze, since a pipeline with tape transforms cannot be compiled
    check_no_tape_transforms(pipeline[:target_level])
    transforms = list(pipeline)[:target_level]
    is_device_level = target_level == device_level(pipeline)
    decompose_positions = [
        position
        for position, bound_transform in enumerate(transforms)
        if bound_transform.pass_name == _DECOMPOSE_PASS
    ]
    decompositions = [transforms[position] for position in decompose_positions]
    last_position = decompose_positions[-1] if decompose_positions else -1

    # Only the decompositions are applied or estimated, so the other transforms are skipped
    def skipped(device_decomposition: bool) -> list[tuple]:
        return [
            (bound_transform, position > last_position and not device_decomposition)
            for position, bound_transform in enumerate(transforms)
            if bound_transform.pass_name != _DECOMPOSE_PASS
            and bound_transform.pass_name not in _SAFE_TO_SKIP_TRANSFORMS
        ]

    if not decompositions:
        if is_device_level:
            return CompilePipeline(), {"gate_set": _device_gate_set(qnode)}, skipped(True)
        return CompilePipeline(), None, skipped(False)

    last_kwargs = _decompose_kwargs(decompositions[-1])
    if is_device_level:
        device_gate_set = _device_gate_set(qnode)
        # The device preprocessing decomposes any gate outside the device gate set
        if not set(GateSet(last_kwargs["gate_set"])) <= set(GateSet(device_gate_set)):
            return CompilePipeline(decompositions), {"gate_set": device_gate_set}, skipped(True)
    return CompilePipeline(decompositions[:-1]), last_kwargs, skipped(False)


# TODO: once estimate supports levels other than "top", suggest qp.estimate with that level
# instead of qp.analyze, here, in _missing_rule_message and in the docstring of estimate.
def _warn_skipped_transforms(gate_changing: list[tuple], target, resources) -> None:
    """Warn that the transforms in ``gate_changing``, which may change the gates, are skipped in
    the ``resources`` estimated for ``target``. For ``gridsynth``, the warning gives the number
    of rotations it would approximate, if the ``resources`` are the gates it would act on."""
    for bound_transform, acts_on_resources in gate_changing:
        name = bound_transform.pass_name
        if name != "gridsynth" or not acts_on_resources:
            warnings.warn(
                f"qp.estimate skips the transform {name} for target={target!r}, although it may "
                "change the gates. The resources do not include the gates that it introduces. "
                "Use qp.analyze at that level to include them.",
                UserWarning,
            )
            continue
        num_rotations = sum(
            specs.quantum_operations.get(rotation, 0)
            for specs in (resources if isinstance(resources, list) else [resources])
            for rotation in _GRIDSYNTH_ROTATIONS
        )
        if num_rotations:
            warnings.warn(
                f"qp.estimate skips the transform gridsynth for target={target!r}, so "
                f"{num_rotations} RZ and PhaseShift gates are not approximated in the "
                f"Clifford+T basis with epsilon={bound_transform.kwargs['epsilon']:g}. Use "
                "qp.analyze at that level to include the gates of gridsynth.",
                UserWarning,
            )


def _missing_rule_op(exc: Exception) -> str | None:
    """Return the operator of a compile error for a missing decomposition rule, if any."""
    missing_rule = re.search(r"Decomposition rule not found for operator '([^'{]+)", str(exc))
    return missing_rule.group(1) if missing_rule else None


def _failing_decomposition(
    compile_error: type[Exception], func, estimate_qnode, *args, **kwargs
) -> tuple[str, dict] | None:
    """Return the operator and the arguments of the first decomposition applied in
    ``estimate_qnode`` that cannot decompose it, if any. Each decomposition is compiled separately,
    so this is only done after a ``compile_error``."""
    for num_applied, bound_transform in enumerate(estimate_qnode.compile_pipeline, start=1):
        try:
            resources_from_analysis_pass(func, estimate_qnode, num_applied, {}, {}, *args, **kwargs)
        except compile_error as exc:
            if (op := _missing_rule_op(exc)) is not None:
                return op, _decompose_kwargs(bound_transform)
            raise
    return None


def _missing_rule_message(op: str, target, pipeline: CompilePipeline, gate_changing: list) -> str:
    """Return the error message for an operator ``op`` that cannot be decomposed into the gate set
    of ``target``."""
    if _target_level(target, pipeline) is not None:
        message = (
            f"Decomposition rule not found for operator {op}: it cannot be decomposed into the "
            f"gate set at target={target!r}."
        )
        if gate_changing:
            message += (
                f" qp.estimate skips the transform {gate_changing[0][0].pass_name}, which "
                "may be needed to reach it. Use qp.analyze at that level instead."
            )
        return message
    message = (
        f"Decomposition rule not found for operator {op}: it cannot be decomposed into the target "
        f"gate set. Add {op} to the target, or use a gate set it can be decomposed into."
    )
    if isinstance(target, str):
        message += (
            f" target={target!r} is not a level of the compile pipeline, so it is the name of the "
            "only gate in the target gate set."
        )
    elif levels := [name for name in target if _is_level_name(name, pipeline)]:
        message += (
            f" The target gate set contains {', '.join(map(repr, levels))}, which is a gate name "
            "here. To use a level of the compile pipeline as target, pass it on its own, such as "
            f"target={levels[0]!r}."
        )
    return message


def _is_level_name(name, pipeline: CompilePipeline) -> bool:
    """Return whether ``name`` is the name of a level or a marker label of ``pipeline``."""
    return isinstance(name, str) and (
        name in ("top", "user", "device", "all-user", "all") or name in pipeline.markers
    )


def _estimate_qjit(func, level, target, *args, **kwargs) -> CircuitSpecs:
    """Decompose a qjit'd QNode at ``level`` into the ``target`` gate set and return its specs."""
    original_qnode = unwrap_qjit_qnode(func, fn_name="qp.estimate")
    from catalyst.utils.exceptions import CompileError  # pylint: disable=import-outside-toplevel

    compile_pipeline = original_qnode.compile_pipeline
    validate_level(level, compile_pipeline)

    # The ``isinstance`` check avoids elementwise comparisons
    if isinstance(level, str) and level == "device":
        raise NotImplementedError("qp.estimate does not support level='device' yet.")
    if isinstance(level, slice):
        # A slice that selects no transforms is the same as level 0
        is_top = not pipeline_at_slice(compile_pipeline, level)
    else:
        is_top = is_single_level(level) and level_to_number(level, compile_pipeline) == 0
    if not is_top:
        raise NotImplementedError(
            f"qp.estimate only supports level='top' or level=0, instead got: {level!r}."
        )

    uses_capture = func.compile_options.capture
    if uses_capture == "global":
        uses_capture = capture.enabled()
    if not uses_capture:
        raise ValueError(
            "qp.estimate requires program capture. Compile the QNode with qp.qjit(capture=True)."
        )

    decompositions, decompose_kwargs, gate_changing = _estimate_passes(original_qnode, target)
    # The compile pipeline is replaced by the decompositions to apply before the estimate
    estimate_qnode = original_qnode.update_pipelines(
        decompositions,
        post_device_pipeline=CompilePipeline(original_qnode.post_device_pipeline),
    )
    if decompose_kwargs is None:
        resources = resources_from_analysis_pass(func, estimate_qnode, 0, {}, {}, *args, **kwargs)
    else:
        try:
            resources = resources_from_analysis_pass(
                func,
                estimate_qnode,
                len(decompositions),
                {},
                {},
                *args,
                analysis_pass_name="resource-estimate",
                analysis_pass_kwargs=decompose(**decompose_kwargs).kwargs,
                **kwargs,
            )
        except CompileError as exc:
            if (op := _missing_rule_op(exc)) is None:
                raise
            if failing := _failing_decomposition(
                CompileError, func, estimate_qnode, *args, **kwargs
            ):
                op, failing_kwargs = failing
                raise ValueError(
                    f"Decomposition rule not found for operator {op}: it cannot be decomposed "
                    f"into the gate set {failing_kwargs['gate_set']} of a qp.decompose transform "
                    f"before target={target!r}."
                ) from None
            raise ValueError(
                _missing_rule_message(op, target, compile_pipeline, gate_changing)
            ) from None

    resources = next(iter(resources.values()))
    _warn_skipped_transforms(gate_changing, target, resources)
    return build_circuit_specs(original_qnode, resources, "top")


def estimate(
    qnode,
    level: str | int | slice = "top",
    *,
    target: str | int | Iterable[type | str] | dict[type | str, float] = "device",
) -> Callable[..., CircuitSpecs]:
    r"""Provides a resource estimate of a quantum circuit with respect to the provided compilation target,
    via rapid analysis of decomposition pathways.

    .. note::

        Only QNodes which have been decorated with :func:`@qp.qjit(capture=True) <~pennylane.qjit>` are supported.

    The ``estimate`` function never executes the circuit. It only provides resource estimates.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): the qjit'd QNode for which to estimate resources.
        level (str | int | slice): The portion of the
            :class:`~.CompilePipeline` to apply before projecting the resources onto the
            ``target``. See :func:`~.workflow.get_compile_pipeline` for accepted values.
            Currently, only ``"top"``, ``0``, or the label of a marker placed before all
            transforms are supported. Defaults to ``"top"``.
        target (str | int | Iterable[type | str] | dict[type | str, float]): The gate set to
            project the resources onto, in any form accepted by the ``gate_set`` argument of
            :func:`~.decompose`, or a single level of the compile pipeline: ``"top"``,
            ``"user"``, ``"device"``, a marker label, or an ``int``. Any other ``str`` is the name
            of a gate, as in :func:`~.decompose`. Defaults to ``"device"``.

    Returns:
        A function that has the same argument signature as ``qnode``. This function returns a
        :class:`~.resource.CircuitSpecs` object containing the projected resources, including
        gate and measurement data, wire allocations, device information, shots, and more.

    .. seealso:: :func:`~.analyze`, :func:`~.track`

    **Example**

    Consider the following circuit.

    .. code-block:: python

        dev = qp.device("null.qubit", wires=2)

        @qp.qjit(capture=True)
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit(x):
            qp.X(0)
            qp.X(0)
            qp.CZ([0, 1])
            qp.RX(x, wires=1)
            return qp.probs()

    By calling ``estimate`` on this circuit with ``level="top"``, the :func:`~.cancel_inverses`
    pass is ignored. Subsequently, the operations in the original circuit will have their
    resources projected into the ``target`` gate set by analyzing decomposition pathways.

    >>> print(qp.estimate(circuit, target={"PauliX", "Hadamard", "CNOT", "RX"})(1.23))
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: top
    <BLANKLINE>
    Quantum operations:
    - Total: 6
      - CNOT: 1
      - Hadamard: 2
      - PauliX: 2
      - RX: 1
    Measurement processes:
    - probs(all wires): 1
    Total wires: 2
    Circuit Depth: Not computed

    Since :func:`~.cancel_inverses` is ignored, both ``PauliX`` gates are counted. The ``CZ`` gate
    is decomposed into ``Hadamard`` and ``CNOT`` gates.

    .. details::
        :title: Pipeline levels as target

        Instead of a gate set, ``target`` can be a single level of the compile pipeline, as given
        by :func:`~.workflow.get_compile_pipeline`. The resources are then projected onto the
        gates at that level:

        * Before the device level, the gates are those of the last :func:`~.decompose` transform
          before that level. If there is none, the resources are those at level 0, without any
          decomposition.
        * At the device level, as with the default ``target="device"``, the gates are those of
          the last :func:`~.decompose` transform if the device supports all of them. Otherwise,
          they are further decomposed into the gates supported by the QNode's device.

        Only the :func:`~.decompose` transforms are applied. The other transforms are skipped, as
        with ``level="top"``, and so are the device preprocessing transforms, such as
        :func:`~.transforms.split_non_commuting`. Therefore, the resources can differ from those
        of :func:`~.analyze` at that level:

        * Transforms that only remove, merge or reorder gates, such as :func:`~.cancel_inverses`,
          :func:`~.merge_rotations` or :func:`~.combine_global_phases`, are skipped without a
          warning. They do not add gate types, but the gate counts can be higher.
        * Other transforms are skipped with a warning, since they may introduce new gates. For
          example, the ``RZ`` and ``PhaseShift`` gates that :func:`~.transforms.gridsynth` would
          approximate in the Clifford+T basis are left as they are, and the warning gives their
          number and the ``epsilon`` of :func:`~.transforms.gridsynth`.

        To include the skipped transforms, use :func:`~.analyze` at that level instead.

        The resources can also differ from those of :func:`~.analyze` because ``estimate`` uses
        the resources that the decomposition rules declare, which can be upper bounds of the
        gates that they apply.

        A marker label takes priority over a gate with the same name.

        For example, ``lightning.qubit`` does not support ``QFT``:

        .. code-block:: python

            dev = qp.device("lightning.qubit", wires=2)

            @qp.qjit(capture=True)
            @qp.qnode(dev)
            def circuit():
                qp.QFT(wires=[0, 1])
                return qp.probs()

        >>> qp.estimate(circuit)().resources.quantum_operations
        {'ControlledPhaseShift': 1, 'Hadamard': 2, 'SWAP': 1}

        The following circuit has two :func:`~.decompose` transforms, followed by
        :func:`~.transforms.gridsynth`:

        .. code-block:: python

            @qp.qjit(capture=True)
            @qp.transforms.gridsynth
            @qp.decompose(gate_set={"RZ", "RY", "CNOT", "GlobalPhase"})
            @qp.decompose(gate_set={"ControlledPhaseShift", "Hadamard", "SWAP"})
            @qp.qnode(dev)
            def circuit():
                qp.QFT(wires=[0, 1])
                return qp.probs()

        ``target=0`` is the circuit before any transform, and each integer level uses the gate
        set of the :func:`~.decompose` transform before it:

        >>> qp.estimate(circuit, target=0)().resources.quantum_operations
        {'QFT': 1}
        >>> qp.estimate(circuit, target=1)().resources.quantum_operations
        {'ControlledPhaseShift': 1, 'Hadamard': 2, 'SWAP': 1}
        >>> qp.estimate(circuit, target=2)().resources.quantum_operations
        {'CNOT': 5, 'GlobalPhase': 3, 'RY': 2, 'RZ': 5}

        ``"user"`` comes after :func:`~.transforms.gridsynth`, which is skipped, so the gates are
        those of the last :func:`~.decompose` transform. The same holds for ``"device"``, since
        ``lightning.qubit`` supports all of them:

        >>> qp.estimate(circuit, target="user")().resources.quantum_operations
        {'CNOT': 5, 'GlobalPhase': 3, 'RY': 2, 'RZ': 5}
        >>> qp.estimate(circuit)().resources.quantum_operations
        {'CNOT': 5, 'GlobalPhase': 3, 'RY': 2, 'RZ': 5}

        Both raise a ``UserWarning`` that says how many ``RZ`` and ``PhaseShift`` gates are not
        approximated in the Clifford+T basis. :func:`~.analyze` at ``level="user"`` includes the
        gates of :func:`~.transforms.gridsynth`.

    .. details::
        :title: Custom decomposition rules

        If ``target`` is a level of the compile pipeline, ``estimate`` also uses the
        ``fixed_decomps`` and ``exclude_decomps`` of the :func:`~.decompose` transforms before
        that level, as in the compile pipeline. If ``target`` is a gate set, they are ignored, with
        a warning.

        For example, the following circuit fixes a decomposition rule for ``CZ`` that uses
        ``PauliZ`` and ``RY`` gates instead of ``Hadamard`` gates:

        .. code-block:: python

            @qp.register_resources({qp.Z: 2, qp.RY: 2, qp.CNOT: 1})
            def cz_with_ry(wires, **_):
                qp.Z(wires[1])
                qp.RY(np.pi / 2, wires[1])
                qp.CNOT(wires)
                qp.Z(wires[1])
                qp.RY(np.pi / 2, wires[1])

            qp.add_decomps(qp.CZ, cz_with_ry)

            @qp.qjit(capture=True)
            @qp.decompose(
                gate_set={"CNOT", "PauliZ", "RY"}, fixed_decomps={qp.CZ: "cz_with_ry"}
            )
            @qp.qnode(qp.device("null.qubit", wires=2))
            def circuit():
                qp.CZ([0, 1])
                return qp.probs()

        With ``target="user"``, the ``CZ`` gate is decomposed with this rule:

        >>> qp.estimate(circuit, target="user")().resources.quantum_operations
        {'CNOT': 1, 'PauliZ': 2, 'RY': 2}

        With a gate set as ``target``, the rule is ignored, and a ``UserWarning`` says so:

        >>> qp.estimate(circuit, target={"CNOT", "Hadamard"})().resources.quantum_operations
        {'CNOT': 1, 'Hadamard': 2}

    .. details::
        :title: Symbolic resource counts

        Since the circuit is not executed, the projected resources are subject to the same
        compile-time constraints as :func:`~.analyze`. In particular, operations inside a loop
        whose number of iterations depends on runtime values are counted symbolically.

        For example, the number of ``CZ`` gates in the following circuit, and therefore the number
        of gates they decompose into, depends on the value of ``n``:

        .. code-block:: python

            @qp.qjit(capture=True)
            @qp.qnode(qp.device("null.qubit", wires=2))
            def circuit(n):
                qp.Hadamard(0)

                @qp.for_loop(n)
                def loop(i):
                    qp.CZ([0, 1])

                loop()
                return qp.expval(qp.Z(0))

        >>> resources = qp.estimate(circuit, target={"Hadamard", "CNOT"})(5).resources
        >>> print(resources)
        Symbolic Variables: a
        Quantum operations:
        - Total: 3*a + 1
          - CNOT: a
          - Hadamard: 2*a + 1
        Measurement processes:
        - expval(PauliZ): 1
        Total wires: 2
        Circuit Depth: Not computed

        Concrete values can be estimated by substituting each symbolic variable with an integer,
        using the ``.subs`` method:

        >>> resources.subs(a=5).quantum_operations
        {'CNOT': 5, 'Hadamard': 11}

        .. note::

            Loops inside the decomposition rules used to reach the ``target`` gate set are counted
            symbolically as well, so the result can be symbolic even if the circuit has no dynamic
            control flow.
    """
    # TODO: [sc-133408] Add a qp.hint example to the docstring once hints survive decomposition
    qnode, partial_args, partial_kwargs = unwrap_partial(qnode)

    return apply_partial_args(
        partial(_estimate_qjit, qnode, level, target), partial_args, partial_kwargs
    )
