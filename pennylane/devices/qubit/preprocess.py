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
from typing import Generator, Callable, Tuple, Union
import warnings
from functools import partial

import pennylane as qml

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

from ..experimental import ExecutionConfig, DefaultExecutionConfig

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


def _accepted_operator(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator object is supported by the device."""
    if op.name == "QFT" and len(op.wires) >= 6:
        return False
    if op.name == "GroverOperator" and len(op.wires) >= 13:
        return False
    return op.name == "Snapshot" or op.has_matrix


def _accepted_adjoint_operator(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Oeprator is supported by adjoint differentiation."""
    return op.num_params == 0 or op.num_params == 1 and op.has_generator


def _operator_decomposition_gen(
    op: qml.operation.Operator, acceptance_function: Callable[[qml.operation.Operator], bool]
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted by DefaultQubit2."""
    if acceptance_function(op):
        yield op
    else:
        try:
            decomp = op.decomposition()
        except qml.operation.DecompositionUndefinedError as e:
            raise DeviceError(
                f"Operator {op} not supported on DefaultQubit2. Must provide either a matrix or a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(sub_op, acceptance_function)


#######################


def validate_multiprocessing_workers(max_workers):
    """Validates the number of workers for multiprocessing.

    Checks that the CPU is not oversubscribed and warns user if it is,
    making suggestions for the number of workers and/or the number of
    threads per worker.

    Args:
        max_workers (int): Maximal number of multiprocessing workers
    """
    if max_workers is None:
        return
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


def validate_and_expand_adjoint(
    circuit: qml.tape.QuantumTape,
) -> Union[qml.tape.QuantumTape, DeviceError]:  # pylint: disable=protected-access
    """Function for validating that the operations and observables present in the input circuit
    are valid for adjoint differentiation.

    Args:
        circuit(.QuantumTape): the tape to validate

    Returns:
        Union[.QuantumTape, .DeviceError]: The expanded tape, such that it is supported by adjoint differentiation.
        If the circuit is invalid for adjoint differentiation, a DeviceError with an explanation is returned instead.
    """

    try:
        new_ops = [
            final_op
            for op in circuit._ops
            for final_op in _operator_decomposition_gen(op, _accepted_adjoint_operator)
        ]
    except RecursionError as e:
        raise DeviceError(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
        ) from e

    prep = circuit._prep[:1]

    for k in circuit.trainable_params:
        if hasattr(circuit._par_info[k]["op"], "return_type"):
            warnings.warn(
                "Differentiating with respect to the input parameters of "
                f"{circuit._par_info[k]['op'].name} is not supported with the "
                "adjoint differentiation method. Gradients are computed "
                "only with regards to the trainable parameters of the circuit.\n\n Mark "
                "the parameters of the measured observables as non-trainable "
                "to silence this warning.",
                UserWarning,
            )

    # Check validity of measurements
    measurements = []
    for m in circuit.measurements:
        if not isinstance(m, ExpectationMP):
            return DeviceError(
                "Adjoint differentiation method does not support "
                f"measurement {m.__class__.__name__}."
            )

        if not m.obs.has_matrix:
            return DeviceError(
                f"Adjoint differentiation method does not support observable {m.obs.name}."
            )

        measurements.append(m)

    return qml.tape.QuantumScript(new_ops, measurements, prep, circuit.shots)


def validate_measurements(
    circuit: qml.tape.QuantumTape, execution_config: ExecutionConfig = DefaultExecutionConfig
):
    """Check that the circuit contains a valid set of measurements. A valid
    set of measurements is defined as:

    1. If circuit.shots is None (i.e., the execution is analytic), then
       the circuit must only contain ``StateMeasurements``.
    2. If circuit.shots is not None, then the circuit must only contain
       ``SampleMeasurements``.

    If the circuit has an invalid set of measurements, then an error is raised.

    Args:
        circuit (.QuantumTape): the circuit to validate
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.
    """
    if not circuit.shots:
        for m in circuit.measurements:
            if not isinstance(m, StateMeasurement):
                raise DeviceError(f"Analytic circuits must only contain StateMeasurements; got {m}")
    else:
        # check if an analytic diff method is used with finite shots
        if execution_config.gradient_method in ["adjoint", "backprop"]:
            raise DeviceError(
                f"Circuits with finite shots must be executed with non-analytic "
                f"gradient methods; got {execution_config.gradient_method}"
            )

        for m in circuit.measurements:
            if not isinstance(m, (SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP)):
                raise DeviceError(
                    f"Circuits with finite shots must only contain SampleMeasurements, ClassicalShadowMP, or ShadowExpvalMP; got {m}"
                )


def expand_fn(circuit: qml.tape.QuantumScript) -> qml.tape.QuantumScript:
    """Method for expanding or decomposing an input circuit.

    This method expands the tape if:

    - mid-circuit measurements are present,
    - any operations are not supported on the device.

    Args:
        circuit (.QuantumTape): the circuit to expand.

    Returns:
        .QuantumTape: The expanded/decomposed circuit, such that the device
        will natively support all operations.
    """

    if any(isinstance(o, MidMeasureMP) for o in circuit.operations):
        circuit = qml.defer_measurements(circuit)

    if not all(_accepted_operator(op) for op in circuit.operations):
        try:
            # don't decompose initial operations if its StatePrepBase
            prep_op = [circuit[0]] if isinstance(circuit[0], StatePrepBase) else []

            new_ops = [
                final_op
                for op in circuit.operations[bool(prep_op) :]
                for final_op in _operator_decomposition_gen(op, _accepted_operator)
            ]
        except RecursionError as e:
            raise DeviceError(
                "Reached recursion limit trying to decompose operations. "
                "Operator decomposition may have entered an infinite loop."
            ) from e
        circuit = qml.tape.QuantumScript(
            prep_op + new_ops, circuit.measurements, shots=circuit.shots
        )

    for observable in circuit.observables:
        if isinstance(observable, Tensor):
            if any(o.name not in _observables for o in observable.obs):
                raise DeviceError(f"Observable {observable} not supported on DefaultQubit2")
        elif observable.name not in _observables:
            raise DeviceError(f"Observable {observable} not supported on DefaultQubit2")

    return circuit


def batch_transform(
    circuit: qml.tape.QuantumScript, execution_config: ExecutionConfig = DefaultExecutionConfig
) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn]:
    """Apply a differentiable batch transform for preprocessing a circuit
    prior to execution.

    By default, this method contains logic for generating multiple
    circuits, one per term, of a circuit that terminates in ``expval(Sum)``.

    .. warning::

        This method will be tracked by autodifferentiation libraries,
        such as Autograd, JAX, TensorFlow, and Torch. Please make sure
        to use ``qml.math`` for autodiff-agnostic tensor processing
        if required.

    Args:
        circuit (.QuantumTape): the circuit to preprocess
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
        the sequence of circuits to be executed, and a post-processing function
        to be applied to the list of evaluated circuit results.
    """
    # Check whether the circuit was broadcasted or if the diff method is anything other than adjoint
    if circuit.batch_size is None or execution_config.gradient_method != "adjoint":
        # If the circuit wasn't broadcasted, or if built-in PennyLane broadcasting
        # can be used, then no action required
        circuits = [circuit]

        def batch_fn(res: ResultBatch) -> Result:
            """A post-processing function to convert the results of a batch of
            executions into the result of a single executiion."""
            return res[0]

        return circuits, batch_fn

    # Expand each of the broadcasted circuits
    tapes, batch_fn = qml.transforms.broadcast_expand(circuit)

    return tapes, batch_fn


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
    circuits: Tuple[qml.tape.QuantumScript],
    execution_config: ExecutionConfig = DefaultExecutionConfig,
) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn, ExecutionConfig]:
    """Preprocess a batch of :class:`~.QuantumTape` objects to make them ready for execution.

    This function validates a batch of :class:`~.QuantumTape` objects by transforming and expanding
    them to ensure all operators and measurements are supported by the execution device.

    Args:
        circuits (Sequence[QuantumTape]): Batch of tapes to be processed.
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        Tuple[QuantumTape], Callable, ExecutionConfig: QuantumTapes that the device can natively execute,
        a postprocessing function to be called after execution, and a configuration with originally unset specifications filled in.
    """
    for c in circuits:
        validate_measurements(c, execution_config)

    circuits = tuple(expand_fn(c) for c in circuits)
    if execution_config.gradient_method == "adjoint":
        circuits = tuple(validate_and_expand_adjoint(c) for c in circuits)
        for circuit_or_error in circuits:
            if isinstance(circuit_or_error, DeviceError):
                raise circuit_or_error  # it's an error

    transform = partial(batch_transform, execution_config=execution_config)
    circuits, batch_fn = qml.transforms.map_batch_transform(transform, circuits)

    return circuits, batch_fn, _update_config(execution_config)
