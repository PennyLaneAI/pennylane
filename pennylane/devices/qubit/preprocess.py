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

"""This module contains functions for preprocessing `QuantumScript`s to ensure
that they are supported for execution by a device."""

import pennylane as qml

from pennylane.operation import Observable, Tensor
from pennylane.measurements import (
    MidMeasureMP,
    ExpectationMP,
    ShadowExpvalMP,
)
from pennylane import DeviceError

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


def _stopping_condition(op):
    """Specify whether or not an Operator object is supported by the device"""
    return getattr(op, "has_matrix", False) and qml.matrix(op).shape[-1] % 2 == 0


def _supports_observable(observable):
    """Checks if an observable is supported by this device.

    Args:
        observable (type or str): observable to be checked

    Returns:
        bool: ``True`` iff supplied observable is supported
    """
    if isinstance(observable, type) and issubclass(observable, Observable):
        observable = observable.__name__
    if isinstance(observable, str):
        return observable in _observables

    return False


def expand_fn(circuit, max_expansion=10):
    """Method for expanding or decomposing an input circuit.

    This method expands the tape if:

    - nested tapes are present,
    - any operations are not supported on the device.

    Args:
        circuit (.QuantumTape): the circuit to expand.
        max_expansion (int): The number of times the circuit should be
            expanded. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations
            in the decomposition remain unsupported by the device, another
            expansion occurs.

    Returns:
        .QuantumTape: The expanded/decomposed circuit, such that the device
        will natively support all operations.
    """

    if any(isinstance(o, MidMeasureMP) for o in circuit.operations):
        circuit = qml.defer_measurements(circuit)

    if not all(_stopping_condition(op) for op in circuit.operations):
        circuit = circuit.expand(depth=max_expansion, stop_at=_stopping_condition)

    return circuit


def batch_transform(circuit):
    """Apply a differentiable batch transform for preprocessing a circuit
    prior to execution.

    By default, this method contains logic for generating multiple
    circuits, one per term, of a circuit that terminates in ``expval(H)``,
    if the device requires finite shots.

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
    grouping_known = all(
        obs.grouping_indices is not None
        for obs in circuit.observables
        if isinstance(obs, qml.Hamiltonian)
    )

    hamiltonian_in_obs = any(isinstance(obs, qml.Hamiltonian) for obs in circuit.observables)
    expval_sum_in_obs = any(
        isinstance(m.obs, qml.Sum) and isinstance(m, ExpectationMP) for m in circuit.measurements
    )

    is_shadow = any(isinstance(m, ShadowExpvalMP) for m in circuit.measurements)

    if hamiltonian_in_obs and (not is_shadow or grouping_known):
        # If the observable contains a Hamiltonian and if the simulation uses finite shots,
        # or if the Hamiltonian explicitly specifies an observable grouping, split tape
        # into multiple tapes of diagonalizable known observables.
        try:
            circuits, hamiltonian_fn = qml.transforms.hamiltonian_expand(circuit, group=False)
        except ValueError as e:
            raise ValueError(
                "Can only return the expectation of a single Hamiltonian observable"
            ) from e
    elif expval_sum_in_obs and not is_shadow:
        circuits, hamiltonian_fn = qml.transforms.sum_expand(circuit)

    else:
        # otherwise, return the output of an identity transform
        circuits = [circuit]

        def hamiltonian_fn(res):
            return res[0]

    # Check whether the circuit was broadcasted
    if circuit.batch_size is None:
        # If the circuit wasn't broadcasted, no action required
        return circuits, hamiltonian_fn

    # Expand each of the broadcasted Hamiltonian-expanded circuits
    expanded_tapes, expanded_fn = qml.transforms.map_batch_transform(
        qml.transforms.broadcast_expand, circuits
    )

    # Chain the postprocessing functions of the broadcasted-tape expansions and the Hamiltonian
    # expansion. Note that the application order is reversed compared to the expansion order,
    # i.e. while we first applied `hamiltonian_expand` to the tape, we need to process the
    # results from the broadcast expansion first.
    def total_processing(results):
        return hamiltonian_fn(expanded_fn(results))

    return expanded_tapes, total_processing


def check_validity(tape):
    """Checks whether the operations and observables in queue are all supported by the device.

    Args:
        tape (.QuantumTape): tape from which to validate operations and observables

    Raises:
        DeviceError: if there are operations in the queue or observables that the device does
            not support
    """
    for o in tape.operations:
        operation_name = o.name

        if not _stopping_condition(o):
            raise DeviceError(f"Gate {operation_name} not supported on Python Device")

    for o in tape.observables:
        if isinstance(o, qml.measurements.MeasurementProcess) and o.obs is not None:
            o = o.obs

        if isinstance(o, Tensor):
            for i in o.obs:
                if not _supports_observable(i.name):
                    raise DeviceError(f"Observable {i.name} not supported on Python Device")
        else:
            observable_name = o.name

            if not _supports_observable(observable_name):
                raise DeviceError(f"Observable {observable_name} not supported on Python Device")


def preprocess(tapes, execution_config=None, max_expansion=10):
    """Preprocess a batch of `QuantumTape` objects to make them ready for execution.

    This function validates a batch of `QuantumTape` objects by transforming and expanding
    them to ensure all operators and measurements are supported by the execution device.

    Args:
        tapes (Sequence[QuantumTape]): Batch of tapes to be processed.
        execution_config (.ExecutionConfig): execution configuration with configurable
            options for the execution.
        max_expansion (int): The number of times the circuit should be
            expanded. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations
            in the decomposition remain unsupported by the device, another
            expansion occurs.

    Returns:
        Tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
        the sequence of circuits to be executed, and a post-processing function
        to be applied to the list of evaluated circuit results.
    """
    # Finite shot support will be added later
    if execution_config and getattr(execution_config, "shots", None) is not None:
        raise DeviceError("The Python Device does not support finite shots.")

    tapes, batch_fn = qml.transforms.map_batch_transform(batch_transform, tapes)

    for i, tape in enumerate(tapes):
        tapes[i] = expand_fn(tape, max_expansion=max_expansion)
        check_validity(tapes[i])

    return tapes, batch_fn
