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
from typing import Generator, Callable, Tuple, Union
import warnings

import pennylane as qml

from pennylane.operation import Tensor
from pennylane.measurements import MidMeasureMP, StateMeasurement, ExpectationMP
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
    return op.has_matrix


def _operator_decomposition_gen(
    op: qml.operation.Operator,
) -> Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted by DefaultQubit2."""
    if _accepted_operator(op):
        yield op
    else:
        try:
            decomp = op.decomposition()
        except qml.operation.DecompositionUndefinedError as e:
            raise DeviceError(
                f"Operator {op} not supported on DefaultQubit2. Must provide either a matrix or a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(sub_op)


#######################


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

    expanded_ops = []
    for op in circuit._ops:
        if op.num_params > 1:
            if not isinstance(op, qml.Rot):
                return DeviceError(
                    f"The {op} operation is not supported using "
                    'the "adjoint" differentiation method.'
                )
            ops = op.decomposition()
            expanded_ops.extend(ops)
        elif not isinstance(op, qml.operation.StatePrep):
            expanded_ops.append(op)

    prep = circuit._prep[:1]

    trainable_params = []
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
        else:
            trainable_params.append(k)

    expanded_tape = qml.tape.QuantumScript(expanded_ops, measurements, prep)
    expanded_tape.trainable_params = trainable_params

    return expanded_tape


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

    if len(circuit._prep) > 1:
        raise DeviceError("DefaultQubit2 accepts at most one state prep operation.")

    if not all(_accepted_operator(op) for op in circuit._ops):
        try:
            new_ops = [
                final_op for op in circuit._ops for final_op in _operator_decomposition_gen(op)
            ]
        except RecursionError as e:
            raise DeviceError(
                "Reached recursion limit trying to decompose operations. "
                "Operator decomposition may have entered an infinite loop."
            ) from e
        circuit = qml.tape.QuantumScript(new_ops, circuit.measurements, circuit._prep)

    for observable in circuit.observables:
        if isinstance(observable, Tensor):
            if any(o.name not in _observables for o in observable.obs):
                raise DeviceError(f"Observable {observable} not supported on DefaultQubit2")
        elif observable.name not in _observables:
            raise DeviceError(f"Observable {observable} not supported on DefaultQubit2")

    # change this once shots are supported
    for m in circuit.measurements:
        if not isinstance(m, StateMeasurement):
            raise DeviceError(f"Measurement process {m} is only useable with finite shots.")

    return circuit


def batch_transform(
    circuit: qml.tape.QuantumScript,
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

    Returns:
        tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
        the sequence of circuits to be executed, and a post-processing function
        to be applied to the list of evaluated circuit results.
    """
    # Check whether the circuit was broadcasted
    if circuit.batch_size is None:
        # If the circuit wasn't broadcasted, no action required
        circuits = [circuit]

        def batch_fn(res: ResultBatch) -> Result:
            """A post-processing function to convert the results of a batch of
            executions into the result of a single executiion."""
            return res[0]

        return circuits, batch_fn

    # Expand each of the broadcasted circuits
    tapes, batch_fn = qml.transforms.broadcast_expand(circuit)

    return tapes, batch_fn


def preprocess(
    circuits: Tuple[qml.tape.QuantumScript],
    execution_config: ExecutionConfig = DefaultExecutionConfig,
) -> Tuple[Tuple[qml.tape.QuantumScript], PostprocessingFn]:
    """Preprocess a batch of :class:`~.QuantumTape` objects to make them ready for execution.

    This function validates a batch of :class:`~.QuantumTape` objects by transforming and expanding
    them to ensure all operators and measurements are supported by the execution device.

    Args:
        circuits (Sequence[QuantumTape]): Batch of tapes to be processed.
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.

    Returns:
        Tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
        the sequence of circuits to be executed, and a post-processing function
        to be applied to the list of evaluated circuit results.
    """
    if execution_config.shots is not None:
        # Finite shot support will be added later
        raise DeviceError("The Python Device does not support finite shots.")

    circuits = tuple(expand_fn(c) for c in circuits)
    if execution_config.gradient_method == "adjoint":
        circuits = tuple(validate_and_expand_adjoint(c) for c in circuits)
        for circuit_or_error in circuits:
            if isinstance(circuit_or_error, DeviceError):
                raise circuit_or_error  # it's an error

    circuits, batch_fn = qml.transforms.map_batch_transform(batch_transform, circuits)

    return circuits, batch_fn
