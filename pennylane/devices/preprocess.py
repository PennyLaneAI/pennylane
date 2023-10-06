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
# pylint: disable=protected-access
import os
from typing import Generator, Callable, Union, Sequence
from copy import copy
import warnings

import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
    StateMeasurement,
    SampleMeasurement,
    ClassicalShadowMP,
    ShadowExpvalMP,
)
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane.transforms.core import transform
from pennylane.wires import WireError

PostprocessingFn = Callable[[ResultBatch], Union[Result, ResultBatch]]


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def _operator_decomposition_gen(
    op: qml.operation.Operator, acceptance_function: Callable[[qml.operation.Operator], bool]
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted by DefaultQubit."""
    if acceptance_function(op):
        yield op
    else:
        try:
            decomp = op.decomposition()
        except qml.operation.DecompositionUndefinedError as e:
            raise DeviceError(
                f"Operator {op} not supported on DefaultQubit. Must provide either a matrix or a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(sub_op, acceptance_function)


#######################


@transform
def validate_device_wires(
    tape: qml.tape.QuantumTape, device
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the device wires.

    Args:
        tape (QuantumTape): a quantum circuit.
        device (pennylane.devices.Device): The device to be checked.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """
    if device.wires:
        if extra_wires := set(tape.wires) - set(device.wires):
            raise WireError(
                f"Cannot run circuit(s) on {device.name} as they contain wires "
                f"not found on the device: {extra_wires}"
            )
        measurements = tape.measurements.copy()
        modified = False
        for m_idx, mp in enumerate(measurements):
            if not mp.obs and not mp.wires:
                modified = True
                new_mp = copy(mp)
                new_mp._wires = device.wires  # pylint:disable=protected-access
                measurements[m_idx] = new_mp
        if modified:
            tape = type(tape)(tape.operations, measurements, shots=tape.shots)

    return [tape], null_postprocessing


@transform
def validate_multiprocessing_workers(
    tape: qml.tape.QuantumTape, max_workers: int, device
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the number of workers for multiprocessing.

    Checks that the CPU is not oversubscribed and warns user if it is,
    making suggestions for the number of workers and/or the number of
    threads per worker.

    Args:
        tape (QuantumTape): a quantum circuit.
        max_workers (int): Maximal number of multiprocessing workers
        device (pennylane.devices.Device): The device to be checked.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
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

    return [tape], null_postprocessing


@transform
def warn_about_trainable_observables(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Raises a warning if any of the observables is trainable. Can be used in validating circuits
    for adjoint differentiation.
    """

    for k in tape.trainable_params:
        if hasattr(tape._par_info[k]["op"], "return_type"):
            warnings.warn(
                "Differentiating with respect to the input parameters of "
                f"{tape._par_info[k]['op'].name} is not supported with the "
                "adjoint differentiation method. Gradients are computed "
                "only with regards to the trainable parameters of the circuit.\n\n Mark "
                "the parameters of the measured observables as non-trainable "
                "to silence this warning.",
                UserWarning,
            )
    return (tape,), null_postprocessing


@transform
def decompose(
    tape: qml.tape.QuantumTape,
    stopping_condition: Callable[[qml.operation.Operator], bool],
    skip_initial_state_prep=True,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Decompose operations until the stopping condition is met.

    Args:
        tape (QuantumTape): a quantum circuit.
        stopping_condition (Callable): a function from an operator to a boolean. If ``False``, the operator
            should be decomposed. If an operator cannot be decomposed and is not accepted by ``stopping_condition``,
            a ``DecompositionUndefinedError`` will be raised.
        skip_initial_state_prep=True (bool): If ``True``, the first operator will not be decomposed if it inherits from :class:`~.StatePrepBase`.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.

    Raises:
        DecompositionUndefinedError: if an operator is not accepted and does not define a decomposition

        DeviceError: If the decomposition enters and infinite loop and raises a ``RecursionError``.
    """

    if not all(stopping_condition(op) for op in tape.operations):
        try:
            # don't decompose initial operations if its StatePrepBase
            prep_op = (
                [tape[0]] if isinstance(tape[0], StatePrepBase) and skip_initial_state_prep else []
            )

            new_ops = [
                final_op
                for op in tape.operations[bool(prep_op) :]
                for final_op in _operator_decomposition_gen(op, stopping_condition)
            ]
        except RecursionError as e:
            raise DeviceError(
                "Reached recursion limit trying to decompose operations. "
                "Operator decomposition may have entered an infinite loop."
            ) from e
        tape = qml.tape.QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)

    return [tape], null_postprocessing


@transform
def validate_measurements(
    tape: qml.tape.QuantumTape,
    observable_stopping_condition: Callable[[qml.operation.Operator], bool],
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Validates the observables and measurements for a circuit.

    Args:
        tape (QuantumTape): a quantum circuit.
        observable_stopping_condition (callable): a function that specifies whether or not an observable is accepted.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.

    Raises:
        DeviceError: if an observable is not supported, if not all the measurements are sample based
            when a finite shots is requested, or if not all measurements are state based when an analytic
            simulation is requested.

    """
    for observable in tape.observables:
        if isinstance(observable, Tensor):
            if any(not observable_stopping_condition(o) for o in observable.obs):
                raise DeviceError(f"Observable {observable} not supported on DefaultQubit")
        elif not observable_stopping_condition(observable):
            raise DeviceError(f"Observable {observable} not supported on DefaultQubit")

    if not tape.shots:
        for m in tape.measurements:
            if not isinstance(m, StateMeasurement):
                raise DeviceError(f"Analytic circuits must only contain StateMeasurements; got {m}")

    else:
        for m in tape.measurements:
            if not isinstance(m, (SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP)):
                raise DeviceError(
                    f"Circuits with finite shots must only contain SampleMeasurements, ClassicalShadowMP, or ShadowExpvalMP; got {m}"
                )

    return (tape,), null_postprocessing
