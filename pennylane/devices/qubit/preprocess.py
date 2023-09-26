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
from dataclasses import replace
import os
from typing import Generator, Callable, Tuple, Union, Sequence
from copy import copy
import warnings

import pennylane as qml
from pennylane import Snapshot
from pennylane.operation import Tensor, StatePrepBase
from pennylane.measurements import (
    MidMeasureMP,
    StateMeasurement,
    SampleMeasurement,
    ExpectationMP,
    ClassicalShadowMP,
    ShadowExpvalMP,
)
from pennylane.typing import ResultBatch, Result
from pennylane import DeviceError
from pennylane.transforms.core import transform, TransformProgram
from pennylane.wires import WireError

from pennylane.devices import ExecutionConfig, DefaultExecutionConfig

PostprocessingFn = Callable[[ResultBatch], Union[Result, ResultBatch]]

# Update observable list. Current list is same as supported observables for
# default.qubit.
_observables = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Hermitian",
    "Identity",
    "Projector",
    "SparseHamiltonian",
    "Hamiltonian",
    "Sum",
    "SProd",
    "Prod",
    "Exp",
    "Evolution",
}


### UTILITY FUNCTIONS FOR EXPANDING UNSUPPORTED OPERATIONS ###


def accepted_operator(op: qml.operation.Operator) -> bool:
    """Specify whether an input operator is supported on :class:`~.DefaultQubit`.

    Args:
        op (Operator): the input operator

    Returns:
        bool: whether the operator is supported
    """
    if op.name == "QFT" and len(op.wires) >= 6:
        return False
    if op.name == "GroverOperator" and len(op.wires) >= 13:
        return False
    if op.name == "Snapshot":
        return True
    if op.__class__.__name__ == "Pow" and qml.operation.is_trainable(op):
        return False

    return op.has_matrix


def _accepted_adjoint_operator(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Oeprator is supported by adjoint differentiation."""
    return op.num_params == 0 or op.num_params == 1 and op.has_generator


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

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

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

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing


@transform
def validate_and_expand_adjoint(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Function for validating that the operations and observables present in the input circuit
    are valid for adjoint differentiation.

    Args:
        circuit(.QuantumTape): the tape to validate

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """

    try:
        new_ops = [
            final_op
            for op in tape.operations[tape.num_preps :]
            for final_op in _operator_decomposition_gen(op, _accepted_adjoint_operator)
        ]
    except RecursionError as e:
        raise DeviceError(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
        ) from e

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

    # Check validity of measurements
    measurements = []
    for m in tape.measurements:
        if not isinstance(m, ExpectationMP):
            raise DeviceError(
                "Adjoint differentiation method does not support "
                f"measurement {m.__class__.__name__}."
            )

        if not m.obs.has_matrix:
            raise DeviceError(
                f"Adjoint differentiation method does not support observable {m.obs.name}."
            )

        measurements.append(m)

    new_ops = tape.operations[: tape.num_preps] + new_ops
    new_tape = qml.tape.QuantumScript(new_ops, measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


@transform
def validate_measurements(
    tape: qml.tape.QuantumTape, execution_config: ExecutionConfig = DefaultExecutionConfig
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Check that the circuit contains a valid set of measurements. A valid
    set of measurements is defined as:

    1. If circuit.shots is None (i.e., the execution is analytic), then
       the circuit must only contain ``StateMeasurements``.
    2. If circuit.shots is not None, then the circuit must only contain
       ``SampleMeasurements``.

    If the circuit has an invalid set of measurements, then an error is raised.

    Args:
        tape (.QuantumTape): the circuit to validate
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """
    if not tape.shots:
        for m in tape.measurements:
            if not isinstance(m, StateMeasurement):
                raise DeviceError(f"Analytic circuits must only contain StateMeasurements; got {m}")

    else:
        # check if an analytic diff method is used with finite shots
        if execution_config.gradient_method in ["adjoint", "backprop"]:
            raise DeviceError(
                f"Circuits with finite shots must be executed with non-analytic "
                f"gradient methods; got {execution_config.gradient_method}"
            )

        for m in tape.measurements:
            if not isinstance(m, (SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP)):
                raise DeviceError(
                    f"Circuits with finite shots must only contain SampleMeasurements, ClassicalShadowMP, or ShadowExpvalMP; got {m}"
                )

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing


@transform
def expand_fn(tape: qml.tape.QuantumTape, acceptance_function: Callable=accepted_operator) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Method for expanding or decomposing an input circuit.

    This method expands the tape if:

    - mid-circuit measurements are present,
    - any operations are not supported on the device.

    Args:
        tape (.QuantumTape): the circuit to expand.
        acceptance_function (callable): A function that returns a boolean indicating whether an
            input operation is supported. Defaults to :func:`~.accepted_operator`.

    Returns:
        pennylane.QNode or qfunc or Tuple[List[.QuantumTape], Callable]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.
    """
    if any(isinstance(o, MidMeasureMP) for o in tape.operations):
        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]

    if not all(acceptance_function(op) for op in tape.operations):
        try:
            # don't decompose initial operations if its StatePrepBase
            prep_op = [tape[0]] if isinstance(tape[0], StatePrepBase) else []

            new_ops = [
                final_op
                for op in tape.operations[bool(prep_op) :]
                for final_op in _operator_decomposition_gen(op, acceptance_function)
            ]
        except RecursionError as e:
            raise DeviceError(
                "Reached recursion limit trying to decompose operations. "
                "Operator decomposition may have entered an infinite loop."
            ) from e
        tape = qml.tape.QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)

    for observable in tape.observables:
        if isinstance(observable, Tensor):
            if any(o.name not in _observables for o in observable.obs):
                raise DeviceError(f"Observable {observable} not supported on DefaultQubit")
        elif observable.name not in _observables:
            raise DeviceError(f"Observable {observable} not supported on DefaultQubit")

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [tape], null_postprocessing


def _update_config(config: ExecutionConfig) -> ExecutionConfig:
    """Choose the "best" options for the configuration if they are left unspecified.

    Args:
        config (ExecutionConfig): the initial execution config

    Returns:
        ExecutionConfig: a new config with the best choices selected.
    """
    updated_values = {}
    if config.gradient_method == "best":
        updated_values["gradient_method"] = "backprop"
    if config.use_device_gradient is None:
        updated_values["use_device_gradient"] = config.gradient_method in {
            "best",
            "adjoint",
            "backprop",
        }
    if config.grad_on_execution is None:
        updated_values["grad_on_execution"] = config.gradient_method == "adjoint"
    return replace(config, **updated_values)


def preprocess(
    execution_config: ExecutionConfig = DefaultExecutionConfig,
) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn, ExecutionConfig]:
    """Preprocess a batch of :class:`~.QuantumTape` objects to make them ready for execution.

    This function validates a batch of :class:`~.QuantumTape` objects by transforming and expanding
    them to ensure all operators and measurements are supported by the execution device.

    Args:
        execution_config (ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        TransformProgram, ExecutionConfig: A transform program and a configuration with originally unset specifications
        filled in.
    """
    transform_program = TransformProgram()

    # Validate measurement
    transform_program.add_transform(validate_measurements, execution_config)

    # Circuit expand
    transform_program.add_transform(expand_fn)

    if execution_config.gradient_method == "adjoint":
        # Adjoint expand
        transform_program.add_transform(validate_and_expand_adjoint)
        ### Broadcast expand
        transform_program.add_transform(qml.transforms.broadcast_expand)

    return transform_program, _update_config(execution_config)
