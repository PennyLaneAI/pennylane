# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains functions for preprocessing `QuantumTape` objects to ensure
that they are supported for execution by a device."""
# pylint: disable=protected-access, too-many-arguments

import os
import warnings
from collections.abc import Callable, Generator, Sequence
from copy import copy
from itertools import chain
from typing import Optional, Type

import pennylane as qml
from pennylane import Snapshot, transform
from pennylane.measurements import SampleMeasurement, StateMeasurement
from pennylane.operation import StatePrepBase
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn
from pennylane.wires import WireError

from .execution_config import MCMConfig


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def _operator_decomposition_gen(  # pylint: disable = too-many-positional-arguments
    op: qml.operation.Operator,
    acceptance_function: Callable[[qml.operation.Operator], bool],
    decomposer: Callable[[qml.operation.Operator], Sequence[qml.operation.Operator]],
    max_expansion: Optional[int] = None,
    current_depth=0,
    name: str = "device",
    error: Optional[Type[Exception]] = None,
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted."""
    if error is None:
        error = qml.DeviceError

    max_depth_reached = False
    if max_expansion is not None and max_expansion <= current_depth:
        max_depth_reached = True
    if acceptance_function(op) or max_depth_reached:
        yield op
    else:
        try:
            decomp = decomposer(op)
            current_depth += 1
        except qml.operation.DecompositionUndefinedError as e:
            raise error(
                f"Operator {op} not supported with {name} and does not provide a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(
                sub_op,
                acceptance_function,
                decomposer=decomposer,
                max_expansion=max_expansion,
                current_depth=current_depth,
                name=name,
                error=error,
            )


#######################


@transform
def no_sampling(
    tape: QuantumScript, name: str = "device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Raises an error if the tape has finite shots.

    Args:
        tape (QuantumTape or .QNode or Callable): a quantum circuit
        name (str): name to use in error message.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.


    This transform can be added to forbid finite shots. For example, ``default.qubit`` uses it for
    adjoint and backprop validation.
    """
    if tape.shots:
        raise qml.DeviceError(f"Finite shots are not supported with {name}")
    return (tape,), null_postprocessing


@transform
def validate_device_wires(
    tape: QuantumScript, wires: Optional[qml.wires.Wires] = None, name: str = "device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates that all wires present in the tape are in the set of provided wires. Adds the
    device wires to measurement processes like :class:`~.measurements.StateMP` that are broadcasted
    across all available wires.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        wires=None (Optional[Wires]): the allowed wires. Wires of ``None`` allows any wires
            to be present in the tape.
        name="device" (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        WireError: if the tape has a wire not present in the provided wires, or if abstract wires are present.
    """

    if any(qml.math.is_abstract(w) for w in tape.wires):
        raise WireError(
            f"Cannot run circuit(s) on {name} as abstract wires are present in the tape: {tape.wires}. "
            f"Abstract wires are not yet supported."
        )

    if wires:

        if any(qml.math.is_abstract(w) for w in wires):
            raise WireError(
                f"Cannot run circuit(s) on {name} as abstract wires are present in the device: {wires}. "
                f"Abstract wires are not yet supported."
            )

        if extra_wires := set(tape.wires) - set(wires):
            raise WireError(
                f"Cannot run circuit(s) on {name} as they contain wires "
                f"not found on the device: {extra_wires}"
            )

        modified = False
        new_ops = None
        for i, op in enumerate(tape.operations):
            if isinstance(op, qml.Snapshot):
                mp = op.hyperparameters["measurement"]
                if not mp.wires:
                    if not new_ops:
                        new_ops = list(tape.operations)
                    modified = True
                    new_mp = copy(mp)
                    new_mp._wires = wires  # pylint:disable=protected-access
                    new_ops[i] = qml.Snapshot(measurement=new_mp, tag=op.tag)
        if not new_ops:
            new_ops = tape.operations  # no copy in this case

        measurements = tape.measurements.copy()
        for m_idx, mp in enumerate(measurements):
            if not mp.obs and not mp.wires:
                modified = True
                new_mp = copy(mp)
                new_mp._wires = wires  # pylint:disable=protected-access
                measurements[m_idx] = new_mp
        if modified:
            tape = tape.copy(ops=new_ops, measurements=measurements)

    return (tape,), null_postprocessing


@transform
def mid_circuit_measurements(
    tape: QuantumScript,
    device,
    mcm_config=MCMConfig(),
    **kwargs,  # pylint: disable=unused-argument
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Provide the transform to handle mid-circuit measurements.

    If the tape or device uses finite-shot, use the native implementation (i.e. no transform),
    and use the ``qml.defer_measurements`` transform otherwise.
    """
    if isinstance(mcm_config, dict):
        mcm_config = MCMConfig(**mcm_config)
    mcm_method = mcm_config.mcm_method
    if mcm_method is None:
        mcm_method = "one-shot" if tape.shots else "deferred"

    if mcm_method == "one-shot":
        return qml.dynamic_one_shot(tape, postselect_mode=mcm_config.postselect_mode)
    if mcm_method == "tree-traversal":
        return (tape,), null_postprocessing
    return qml.defer_measurements(
        tape, allow_postselect=isinstance(device, qml.devices.DefaultQubit)
    )


@transform
def validate_multiprocessing_workers(
    tape: QuantumScript, max_workers: int, device
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates the number of workers for multiprocessing.

    Checks that the CPU is not oversubscribed and warns user if it is,
    making suggestions for the number of workers and/or the number of
    threads per worker.

    Args:
        tape (QuantumTape or .QNode or Callable): a quantum circuit.
        max_workers (int): Maximal number of multiprocessing workers
        device (pennylane.devices.Device): The device to be checked.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    """
    if max_workers is not None:
        threads_per_proc = os.cpu_count()  # all threads by default
        varname = "OMP_NUM_THREADS"
        varnames = ["MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        for var in varnames:
            if os.getenv(var):  # pragma: no cover
                varname = var
                threads_per_proc = int(os.getenv(var))
                break
        num_threads = threads_per_proc * max_workers
        num_cpu = os.cpu_count()
        num_threads_suggest = max(1, os.cpu_count() // max_workers)
        num_workers_suggest = max(1, os.cpu_count() // threads_per_proc)
        if num_threads > num_cpu:
            warnings.warn(
                f"""The device requested {num_threads} threads ({max_workers} processes
                times {threads_per_proc} threads per process), but the processor only has
                {num_cpu} logical cores. The processor is likely oversubscribed, which may
                lead to performance deterioration. Consider decreasing the number of processes,
                setting the device or execution config argument `max_workers={num_workers_suggest}`
                for example, or decreasing the number of threads per process by setting the
                environment variable `{varname}={num_threads_suggest}`.""",
                UserWarning,
            )

        if device._debugger and device._debugger.active:
            raise qml.DeviceError(
                "Debugging with ``Snapshots`` is not available with multiprocessing."
            )

        if any(isinstance(op, Snapshot) for op in tape.operations):
            raise RuntimeError(
                """ProcessPoolExecutor cannot execute a QuantumScript with
                a ``Snapshot`` operation. Change the value of ``max_workers``
                to ``None`` or execute the QuantumScript separately."""
            )

    return (tape,), null_postprocessing


@transform
def validate_adjoint_trainable_params(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Raises a warning if any of the observables is trainable, and raises an error if any
    trainable parameters belong to state-prep operations. Can be used in validating circuits
    for adjoint differentiation.
    """

    for op in tape.operations[: tape.num_preps]:
        if qml.operation.is_trainable(op):
            raise qml.QuantumFunctionError(
                "Differentiating with respect to the input parameters of state-prep operations "
                "is not supported with the adjoint differentiation method."
            )
    for m in tape.measurements:
        if m.obs and qml.operation.is_trainable(m.obs):
            warnings.warn(
                f"Differentiating with respect to the input parameters of {m.obs.name} "
                "is not supported with the adjoint differentiation method. Gradients are computed "
                "only with regards to the trainable parameters of the circuit.\n\n Mark the "
                "parameters of the measured observables as non-trainable to silence this warning.",
                UserWarning,
            )
    return (tape,), null_postprocessing


@transform
def decompose(  # pylint: disable = too-many-positional-arguments
    tape: QuantumScript,
    stopping_condition: Callable[[qml.operation.Operator], bool],
    stopping_condition_shots: Callable[[qml.operation.Operator], bool] = None,
    skip_initial_state_prep: bool = True,
    decomposer: Optional[
        Callable[[qml.operation.Operator], Sequence[qml.operation.Operator]]
    ] = None,
    name: str = "device",
    error: Optional[Type[Exception]] = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Decompose operations until the stopping condition is met.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        stopping_condition (Callable): a function from an operator to a boolean. If ``False``,
            the operator should be decomposed. If an operator cannot be decomposed and is not
            accepted by ``stopping_condition``, an ``Exception`` will be raised (of a type
            specified by the ``error`` keyword argument).

    Keyword Args:
        stopping_condition_shots (Callable): a function from an operator to a boolean. If
            ``False``, the operator should be decomposed. This replaces ``stopping_condition``
            if and only if the tape has shots.
        skip_initial_state_prep (bool): If ``True``, the first operator will not be decomposed if
            it inherits from :class:`~.StatePrepBase`. Defaults to ``True``.
        decomposer (Callable): an optional callable that takes an operator and implements the
            relevant decomposition. If ``None``, defaults to using a callable returning
            ``op.decomposition()`` for any :class:`~.Operator` .
        name (str): The name of the transform, process or device using decompose. Used in the
            error message. Defaults to "device".
        error (type): An error type to raise if it is not possible to obtain a decomposition that
            fulfills the ``stopping_condition``. Defaults to ``qml.DeviceError``.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    .. seealso:: This transform is intended for device developers. See :func:`qml.transforms.decompose <pennylane.transforms.decompose>` for a more user-friendly interface.

    Raises:
        Exception: Type defaults to ``qml.DeviceError`` but can be modified via keyword argument.
            Raised if an operator is not accepted and does not define a decomposition, or if
            the decomposition enters an infinite loop and raises a ``RecursionError``.

    **Example:**

    >>> def stopping_condition(obj):
    ...     return obj.name in {"CNOT", "RX", "RZ"}
    >>> tape = qml.tape.QuantumScript([qml.IsingXX(1.2, wires=(0,1))], [qml.expval(qml.Z(0))])
    >>> batch, fn = decompose(tape, stopping_condition)
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]),
    RX(1.2, wires=[0]),
    CNOT(wires=[0, 1]),
    expval(Z(0))]

    If an operator cannot be decomposed into a supported operation, an error is raised:

    >>> decompose(tape, lambda obj: obj.name == "S")
    qml.DeviceError: Operator CNOT(wires=[0, 1]) not supported on device and does not provide a decomposition.

    The ``skip_initial_state_prep`` specifies whether the device supports state prep operations
    at the beginning of the circuit.

    >>> tape = qml.tape.QuantumScript([qml.BasisState([1], wires=0), qml.BasisState([1], wires=1)])
    >>> batch, fn = decompose(tape, stopping_condition)
    >>> batch[0].circuit
    [BasisState(array([1]), wires=[0]),
    RZ(1.5707963267948966, wires=[1]),
    RX(3.141592653589793, wires=[1]),
    RZ(1.5707963267948966, wires=[1])]
    >>> batch, fn = decompose(tape, stopping_condition, skip_initial_state_prep=False)
    >>> batch[0].circuit
    [RZ(1.5707963267948966, wires=[0]),
    RX(3.141592653589793, wires=[0]),
    RZ(1.5707963267948966, wires=[0]),
    RZ(1.5707963267948966, wires=[1]),
    RX(3.141592653589793, wires=[1]),
    RZ(1.5707963267948966, wires=[1])]

    """

    if error is None:
        error = qml.DeviceError

    if decomposer is None:

        def decomposer(op):
            return op.decomposition()

    if stopping_condition_shots is not None and tape.shots:
        stopping_condition = stopping_condition_shots

    if tape.operations and isinstance(tape[0], StatePrepBase) and skip_initial_state_prep:
        prep_op = [tape[0]]
    else:
        prep_op = []

    if all(stopping_condition(op) for op in tape.operations[len(prep_op) :]):
        return (tape,), null_postprocessing
    try:

        new_ops = [
            final_op
            for op in tape.operations[len(prep_op) :]
            for final_op in _operator_decomposition_gen(
                op,
                stopping_condition,
                decomposer=decomposer,
                name=name,
                error=error,
            )
        ]
    except RecursionError as e:
        raise error(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
        ) from e

    tape = tape.copy(operations=prep_op + new_ops)

    return (tape,), null_postprocessing


@transform
def validate_observables(
    tape: QuantumScript,
    stopping_condition: Callable[[qml.operation.Operator], bool],
    stopping_condition_shots: Callable[[qml.operation.Operator], bool] = None,
    name: str = "device",
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates the observables and measurements for a circuit.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        stopping_condition (callable): a function that specifies whether an observable is accepted.
        stopping_condition_shots (callable): a function that specifies whether an observable is
            accepted in finite-shots mode. This replaces ``stopping_condition`` if and only if the
            tape has shots.
        name (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ~pennylane.DeviceError: if an observable is not supported

    **Example:**

    >>> def accepted_observable(obj):
    ...    return obj.name in {"PauliX", "PauliY", "PauliZ"}
    >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0) + qml.Y(0))])
    >>> validate_observables(tape, accepted_observable)
    qml.DeviceError: Observable Z(0) + Y(0) not supported on device

    """
    if bool(tape.shots) and stopping_condition_shots is not None:
        stopping_condition = stopping_condition_shots

    for m in tape.measurements:
        if m.obs is not None and not stopping_condition(m.obs):
            raise qml.DeviceError(f"Observable {repr(m.obs)} not supported on {name}")

    return (tape,), null_postprocessing


@transform
def validate_measurements(
    tape: QuantumScript, analytic_measurements=None, sample_measurements=None, name="device"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Validates the supported state and sample based measurement processes.

    Args:
        tape (QuantumTape, .QNode, Callable): a quantum circuit.
        analytic_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it is accepted in analytic simulations.
        sample_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it accepted for finite shot simulations.
        name (str): the name to use in error messages.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ~pennylane.DeviceError: if a measurement process is not supported.

    >>> def analytic_measurements(m):
    ...     return isinstance(m, qml.measurements.StateMP)
    >>> def shots_measurements(m):
    ...     return isinstance(m, qml.measurements.CountsMP)
    >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    qml.DeviceError: Measurement expval(Z(0)) not accepted for analytic simulation on device.
    >>> tape = qml.tape.QuantumScript([], [qml.sample()], shots=10)
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    qml.DeviceError: Measurement sample(wires=[]) not accepted with finite shots on device

    """
    if analytic_measurements is None:

        def analytic_measurements(m):
            return isinstance(m, StateMeasurement)

    if sample_measurements is None:

        def sample_measurements(m):
            return isinstance(m, SampleMeasurement)

    # Gather all the measurements present in the snapshot operations with the
    # exception of `qml.state` as this is supported for any supported simulator regardless
    # of its configuration
    snapshot_measurements = [
        meas
        for op in tape.operations
        if isinstance(op, qml.Snapshot)
        and not isinstance(meas := op.hyperparameters["measurement"], qml.measurements.StateMP)
    ]

    shots = qml.measurements.Shots(tape.shots)

    if shots.total_shots is not None:
        for m in chain(snapshot_measurements, tape.measurements):
            if not sample_measurements(m):
                raise qml.DeviceError(f"Measurement {m} not accepted with finite shots on {name}")

    else:
        for m in chain(snapshot_measurements, tape.measurements):
            if not analytic_measurements(m):
                raise qml.DeviceError(
                    f"Measurement {m} not accepted for analytic simulation on {name}."
                )

    return (tape,), null_postprocessing
