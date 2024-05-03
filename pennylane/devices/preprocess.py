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
from typing import Generator, Callable, Union, Sequence, Optional
from copy import copy
import warnings

import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
    StateMeasurement,
    SampleMeasurement,
)
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane import transform
from pennylane.wires import WireError

PostprocessingFn = Callable[[ResultBatch], Union[Result, ResultBatch]]


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def _operator_decomposition_gen(
    op: qml.operation.Operator,
    acceptance_function: Callable[[qml.operation.Operator], bool],
    decomposer: Callable[[qml.operation.Operator], Sequence[qml.operation.Operator]],
    max_expansion: Optional[int] = None,
    current_depth=0,
    name: str = "device",
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted."""
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
            raise DeviceError(
                f"Operator {op} not supported on {name} and does not provide a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(
                sub_op,
                acceptance_function,
                decomposer=decomposer,
                max_expansion=max_expansion,
                current_depth=current_depth,
                name=name,
            )


#######################


@transform
def no_sampling(
    tape: qml.tape.QuantumTape, name: str = "device"
) -> (Sequence[qml.tape.QuantumTape], Callable):
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
    tape: qml.tape.QuantumTape, wires: Optional[qml.wires.Wires] = None, name: str = "device"
) -> (Sequence[qml.tape.QuantumTape], Callable):
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
        WireError: if the tape has a wire not present in the provided wires.
    """
    if wires:
        if extra_wires := set(tape.wires) - set(wires):
            raise WireError(
                f"Cannot run circuit(s) on {name} as they contain wires "
                f"not found on the device: {extra_wires}"
            )
        measurements = tape.measurements.copy()
        modified = False
        for m_idx, mp in enumerate(measurements):
            if not mp.obs and not mp.wires:
                modified = True
                new_mp = copy(mp)
                new_mp._wires = wires  # pylint:disable=protected-access
                measurements[m_idx] = new_mp
        if modified:
            tape = type(tape)(tape.operations, measurements, shots=tape.shots)

    return (tape,), null_postprocessing


@transform
def mid_circuit_measurements(
    tape: qml.tape.QuantumTape, device
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Provide the transform to handle mid-circuit measurements.

    If the tape or device uses finite-shot, use the native implementation (i.e. no transform),
    and use the ``qml.defer_measurements`` transform otherwise.
    """

    if tape.shots:
        return qml.dynamic_one_shot(tape)
    return qml.defer_measurements(tape, device=device)


@transform
def validate_multiprocessing_workers(
    tape: qml.tape.QuantumTape, max_workers: int, device
) -> (Sequence[qml.tape.QuantumTape], Callable):
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
            raise DeviceError("Debugging with ``Snapshots`` is not available with multiprocessing.")

        if any(isinstance(op, Snapshot) for op in tape.operations):
            raise RuntimeError(
                """ProcessPoolExecutor cannot execute a QuantumScript with
                a ``Snapshot`` operation. Change the value of ``max_workers``
                to ``None`` or execute the QuantumScript separately."""
            )

    return (tape,), null_postprocessing


@transform
def validate_adjoint_trainable_params(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
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
def decompose(
    tape: qml.tape.QuantumTape,
    stopping_condition: Callable[[qml.operation.Operator], bool],
    stopping_condition_shots: Callable[[qml.operation.Operator], bool] = None,
    skip_initial_state_prep: bool = True,
    decomposer: Optional[
        Callable[[qml.operation.Operator], Sequence[qml.operation.Operator]]
    ] = None,
    max_expansion: Union[int, None] = None,
    name: str = "device",
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Decompose operations until the stopping condition is met.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        stopping_condition (Callable): a function from an operator to a boolean. If ``False``, the operator
            should be decomposed. If an operator cannot be decomposed and is not accepted by ``stopping_condition``,
            a ``DecompositionUndefinedError`` will be raised.
        stopping_condition_shots (Callable): a function from an operator to a boolean. If ``False``, the operator
            should be decomposed. If an operator cannot be decomposed and is not accepted by ``stopping_condition``,
            a ``DecompositionUndefinedError`` will be raised. This replaces stopping_condition if and only if the tape has shots.
        skip_initial_state_prep=True (bool): If ``True``, the first operator will not be decomposed if it inherits from :class:`~.StatePrepBase`.
        decomposer (Callable): an optional callable that takes an operator and implements the relevant decomposition.
            If None, defaults to using a callable returning ``op.decomposition()`` for any :class:`~.Operator` .
        max_expansion (int): The maximum depth of the expansion.


    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        DecompositionUndefinedError: if an operator is not accepted and does not define a decomposition

        DeviceError: If the decomposition enters and infinite loop and raises a ``RecursionError``.

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
    DeviceError: Operator CNOT(wires=[0, 1]) not supported on device and does not provide a decomposition.

    The ``skip_initial_state_prep`` specifies whether or not the device supports state prep operations
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
    if decomposer is None:

        def decomposer(op):
            return op.decomposition()

    if stopping_condition_shots is not None and tape.shots:
        stopping_condition = stopping_condition_shots

    if not all(stopping_condition(op) for op in tape.operations):
        try:
            # don't decompose initial operations if its StatePrepBase
            prep_op = (
                [tape[0]] if isinstance(tape[0], StatePrepBase) and skip_initial_state_prep else []
            )

            new_ops = [
                final_op
                for op in tape.operations[bool(prep_op) :]
                for final_op in _operator_decomposition_gen(
                    op,
                    stopping_condition,
                    decomposer=decomposer,
                    max_expansion=max_expansion,
                    name=name,
                )
            ]
        except RecursionError as e:
            raise DeviceError(
                "Reached recursion limit trying to decompose operations. "
                "Operator decomposition may have entered an infinite loop."
            ) from e
        tape = qml.tape.QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)

    return (tape,), null_postprocessing


@transform
def validate_observables(
    tape: qml.tape.QuantumTape,
    stopping_condition: Callable[[qml.operation.Operator], bool],
    name: str = "device",
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the observables and measurements for a circuit.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        stopping_condition (callable): a function that specifies whether or not an observable is accepted.
        name (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        DeviceError: if an observable is not supported

    **Example:**

    >>> def accepted_observable(obj):
    ...    return obj.name in {"PauliX", "PauliY", "PauliZ"}
    >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0) + qml.Y(0))])
    >>> validate_observables(tape, accepted_observable)
    DeviceError: Observable Z(0) + Y(0) not supported on device

    Note that if the observable is a :class:`~.Tensor`, the validation is run on each object in the
    ``Tensor`` instead.

    """
    for m in tape.measurements:
        if m.obs is not None:
            if isinstance(m.obs, Tensor):
                if any(not stopping_condition(o) for o in m.obs.obs):
                    raise DeviceError(f"Observable {repr(m.obs)} not supported on {name}")
            elif not stopping_condition(m.obs):
                raise DeviceError(f"Observable {repr(m.obs)} not supported on {name}")

    return (tape,), null_postprocessing


@transform
def validate_measurements(
    tape: qml.tape.QuantumTape, analytic_measurements=None, sample_measurements=None, name="device"
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the supported state and sample based measurement processes.

    Args:
        tape (QuantumTape, .QNode, Callable): a quantum circuit.
        analytic_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it is accepted in analytic simulations.
        sample_measurements (Callable[[MeasurementProcess], bool]): a function from a measurement process
            to whether or not it accepted for finite shot siutations
        name (str): the name to use in error messages.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:

        The unaltered input circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.

    Raises:
        DeviceError: if a measurement process is not supported.

    >>> def analytic_measurements(m):
    ...     return isinstance(m, qml.measurements.StateMP)
    >>> def shots_measurements(m):
    ...     return isinstance(m, qml.measurements.CountsMP)
    >>> tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    DeviceError: Measurement expval(Z(0)) not accepted for analytic simulation on device.
    >>> tape = qml.tape.QuantumScript([], [qml.sample()], shots=10)
    >>> validate_measurements(tape, analytic_measurements, shots_measurements)
    DeviceError: Measurement sample(wires=[]) not accepted with finite shots on device

    """
    if analytic_measurements is None:

        def analytic_measurements(m):
            return isinstance(m, StateMeasurement)

    if sample_measurements is None:

        def sample_measurements(m):
            return isinstance(m, SampleMeasurement)

    if tape.shots:
        for m in tape.measurements:
            if not sample_measurements(m):
                raise DeviceError(f"Measurement {m} not accepted with finite shots on {name}")

    else:
        for m in tape.measurements:
            if not analytic_measurements(m):
                raise DeviceError(
                    f"Measurement {m} not accepted for analytic simulation on {name}."
                )

    return (tape,), null_postprocessing
