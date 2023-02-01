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

# from pennylane.tape import QuantumScript
from pennylane.operation import Operation, Observable, Tensor
from pennylane.measurements import (
    MidMeasureMP,
    ExpectationMP,
    ShadowExpvalMP,
    SampleMP,
    ProbabilityMP,
    CountsMP,
)
from pennylane import DeviceError


def _stopping_condition(op):
    """Specify whether or not an Operator is supported"""
    return getattr(op, "has_matrix", False)


def _supports_observable(dev, observable):
    """Checks if an observable is supported by this device. Raises a ValueError,
        if not a subclass or string of an Observable was passed.

    Args:
        dev (.Device): device to check
        observable (type or str): observable to be checked

    Raises:
        ValueError: if `observable` is not a :class:`~.Observable` class or string

    Returns:
        bool: ``True`` iff supplied observable is supported
    """
    if isinstance(observable, type) and issubclass(observable, Observable):
        return observable.__name__ in dev.observables
    if isinstance(observable, str):
        # This check regards observables that are also operations
        if observable.endswith(".inv"):
            return dev.supports_operation(observable[:-4])

        return observable in dev.observables

    raise ValueError(
        "The given observable must either be a pennylane.Observable class or a string."
    )


def expand_fn(circuit, dev, max_expansion=10):
    """Method for expanding or decomposing an input circuit. Can be the default or
    a custom expansion method, see :meth:`.Device.custom_expand` for more details.

    By default, this method expands the tape if:

    - nested tapes are present,
    - any operations are not supported on the device, or
    - multiple observables are measured on the same wire.

    Args:
        circuit (.QuantumTape): the circuit to expand.
        dev (.Device): the device to execute circuit(s) on.
        max_expansion (int): The number of times the circuit should be
            expanded. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations
            in the decomposition remain unsupported by the device, another
            expansion occurs.

    Returns:
        .QuantumTape: The expanded/decomposed circuit, such that the device
        will natively support all operations.
    """
    # pylint: disable=protected-access

    if dev.custom_expand_fn is not None:
        return dev.custom_expand_fn(circuit, max_expansion=max_expansion)

    comp_basis_sampled_multi_measure = (
        len(circuit.measurements) > 1 and circuit.samples_computational_basis
    )
    obs_on_same_wire = len(circuit._obs_sharing_wires) > 0 or comp_basis_sampled_multi_measure
    obs_on_same_wire &= not any(isinstance(o, qml.Hamiltonian) for o in circuit._obs_sharing_wires)

    ops_not_supported = not all(_stopping_condition(op) for op in circuit.operations)

    if ops_not_supported or obs_on_same_wire:
        circuit = circuit.expand(depth=max_expansion, stop_at=_stopping_condition)

    return circuit


def batch_transform(self, circuit, dev):
    """Apply a differentiable batch transform for preprocessing a circuit
    prior to execution. This method is called directly by the QNode, and
    should be overwritten if the device requires a transform that
    generates multiple circuits prior to execution.

    By default, this method contains logic for generating multiple
    circuits, one per term, of a circuit that terminates in ``expval(H)``,
    if the underlying device does not support Hamiltonian expectation values,
    or if the device requires finite shots.

    .. warning::

        This method will be tracked by autodifferentiation libraries,
        such as Autograd, JAX, TensorFlow, and Torch. Please make sure
        to use ``qml.math`` for autodiff-agnostic tensor processing
        if required.

    Args:
        circuit (.QuantumTape): the circuit to preprocess
        dev (.Device): the device to execute circuit on

    Returns:
        tuple[Sequence[.QuantumTape], callable]: Returns a tuple containing
        the sequence of circuits to be executed, and a post-processing function
        to be applied to the list of evaluated circuit results.
    """
    supports_hamiltonian = _supports_observable(dev, "Hamiltonian")
    finite_shots = dev.shots is not None
    grouping_known = all(
        obs.grouping_indices is not None
        for obs in circuit.observables
        if isinstance(obs, qml.Hamiltonian)
    )
    # device property present in braket plugin
    use_grouping = getattr(self, "use_grouping", True)

    hamiltonian_in_obs = any(isinstance(obs, qml.Hamiltonian) for obs in circuit.observables)
    expval_sum_in_obs = any(
        isinstance(m.obs, qml.Sum) and isinstance(m, ExpectationMP) for m in circuit.measurements
    )

    is_shadow = any(isinstance(m, ShadowExpvalMP) for m in circuit.measurements)

    hamiltonian_unusable = not supports_hamiltonian or (finite_shots and not is_shadow)

    if hamiltonian_in_obs and (hamiltonian_unusable or (use_grouping and grouping_known)):
        # If the observable contains a Hamiltonian and the device does not
        # support Hamiltonians, or if the simulation uses finite shots, or
        # if the Hamiltonian explicitly specifies an observable grouping,
        # split tape into multiple tapes of diagonalizable known observables.
        try:
            circuits, hamiltonian_fn = qml.transforms.hamiltonian_expand(circuit, group=False)
        except ValueError as e:
            raise ValueError(
                "Can only return the expectation of a single Hamiltonian observable"
            ) from e
    elif expval_sum_in_obs and not is_shadow:
        circuits, hamiltonian_fn = qml.transforms.sum_expand(circuit)

    elif (
        len(circuit._obs_sharing_wires) > 0  # pylint: disable=protected-access
        and not hamiltonian_in_obs
        and all(
            not isinstance(m, (SampleMP, ProbabilityMP, CountsMP)) for m in circuit.measurements
        )
    ):
        # Check for case of non-commuting terms and that there are no Hamiltonians
        # TODO: allow for Hamiltonians in list of observables as well.
        circuits, hamiltonian_fn = qml.transforms.split_non_commuting(circuit)

    else:
        # otherwise, return the output of an identity transform
        circuits = [circuit]

        def hamiltonian_fn(res):
            return res[0]

    # Check whether the circuit was broadcasted (then the Hamiltonian-expanded
    # ones will be as well) and whether broadcasting is supported
    if circuit.batch_size is None or self.capabilities().get("supports_broadcasting"):
        # If the circuit wasn't broadcasted or broadcasting is supported, no action required
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


def check_validity(queue, dev, observables):
    """Checks whether the operations and observables in queue are all supported by the device.
    Includes checks for inverse operations.

    Args:
        queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
            to be applied on the device
        dev (.Device): device for which to validate queue
        observables (Iterable[~.operation.Observable]): observables which are intended
            to be evaluated on the device

    Raises:
        DeviceError: if there are operations in the queue or observables that the device does
            not support
    """

    for o in queue:
        operation_name = o.name

        if isinstance(o, MidMeasureMP) and not dev.capabilities().get(
            "supports_mid_measure", False
        ):
            raise DeviceError(
                f"Mid-circuit measurements are not natively supported on device {dev.short_name}. "
                "Apply the @qml.defer_measurements decorator to your quantum function to "
                "simulate the application of mid-circuit measurements on this device."
            )

        if getattr(o, "inverse", False):
            # TODO: update when all capabilities keys changed to "supports_inverse_operations"
            supports_inv = dev.capabilities().get(
                "supports_inverse_operations", False
            ) or dev.capabilities().get("inverse_operations", False)
            if not supports_inv:
                raise DeviceError(
                    f"The inverse of gates are not supported on device {dev.short_name}"
                )
            operation_name = o.base_name

        if not dev.stopping_condition(o):
            raise DeviceError(f"Gate {operation_name} not supported on device {dev.short_name}")

    for o in observables:
        if isinstance(o, qml.measurements.MeasurementProcess) and o.obs is not None:
            o = o.obs

        if isinstance(o, Tensor):
            # TODO: update when all capabilities keys changed to "supports_tensor_observables"
            supports_tensor = dev.capabilities().get(
                "supports_tensor_observables", False
            ) or dev.capabilities().get("tensor_observables", False)
            if not supports_tensor:
                raise DeviceError(f"Tensor observables not supported on device {dev.short_name}")

            for i in o.obs:
                if not _supports_observable(dev, i.name):
                    raise DeviceError(
                        f"Observable {i.name} not supported on device {dev.short_name}"
                    )
        else:
            observable_name = o.name

            if issubclass(o.__class__, Operation) and o.inverse:
                # TODO: update when all capabilities keys changed to "supports_inverse_operations"
                supports_inv = dev.capabilities().get(
                    "supports_inverse_operations", False
                ) or dev.capabilities().get("inverse_operations", False)
                if not supports_inv:
                    raise DeviceError(
                        f"The inverse of gates are not supported on device {dev.short_name}"
                    )
                observable_name = o.base_name

            if not _supports_observable(dev, observable_name):
                raise DeviceError(
                    f"Observable {observable_name} not supported on device {dev.short_name}"
                )


# pylint: disable=unused-argument, missing-function-docstring
def preprocess(tapes, dev, execution_config=None):
    # Combine check_validity, _expand_fn and batch_transform in this function:
    # Check that tapes are valid
    # Expand tapes
    # Split tapes as needed
    # Create wrapper function that combines results of expanded and batched tapes as expected
    # Return tapes, wrapper
    # Returns Sequence[QuantumScript], callable
    pass
