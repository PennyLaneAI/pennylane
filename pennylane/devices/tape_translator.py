# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod, abstractproperty
from pyclbr import Function

import pennylane as qml
from pennylane._device import DeviceError
from pennylane.operation import Operation, Observable


class TapeTranslator(ABC):
    @abstractmethod
    def expand(self, tape: qml.tape.QuantumTape, max_expansion: int = 10) -> qml.tape.QuantumTape:
        pass

    @abstractmethod
    def batch_transform(self, tape: qml.tape.QuantumTape):
        """
        Returns:
            list[QuantumTape], list[function]
        """
        pass


class DefaultTranslator(TapeTranslator):
    def __init__(self, ops: set, obs: set, short_name=None, **capabilities):

        self.device_short_name = short_name

        self._operations = ops
        self._observables = obs
        self._capabilities = capabilities

    @property
    def _supports_inverses(self):
        return self._capabilities.get("supports_inverse_operations", False)

    @property
    def _supports_mid_measure(self):
        return self._capabilities.get("supports_mid_measure", False)

    @property
    def _supports_tensor_observables(self):
        return self._capabilities.get("supports_tensor_observables", False)

    @property
    def _supports_broadcasting(self):
        return self._capabiltiies.get("supports_broadcasting", False)

    def _supports_operation(self, operation):

        if isinstance(operation, type) and issubclass(operation, Operation):
            return operation.__name__ in self._obs
        if isinstance(operation, str):

            if operation.endswith(".inv"):
                in_ops = operation[:-4] in self._operations
                return in_ops and self._supports_inverses

            return operation in self._operations

        raise ValueError(
            "The given operation must either be a pennylane.Operation class or a string."
        )

    def _supports_observable(self, observable):
        if isinstance(observable, type) and issubclass(observable, Observable):
            return observable.__name__ in self._observables
        if isinstance(observable, str):

            # This check regards observables that are also operations
            if observable.endswith(".inv"):
                return self.supports_operation(observable[:-4])

            return observable in self.observables

        raise ValueError(
            "The given observable must either be a pennylane.Observable class or a string."
        )

    @property
    def _stopping_condition(self):
        return qml.BooleanFn(
            lambda obj: not isinstance(obj, qml.tape.QuantumTape)
            and self.supports_operation(obj.name)
        )

    def _check_validity(self, queue, observables):
        """Checks whether the operations and observables in queue are all supported by the device.
        Includes checks for inverse operations.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            observables (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device

        Raises:
            DeviceError: if there are operations in the queue or observables that the device does
                not support
        """
        for o in queue:

            operation_name = o.name

            if (
                getattr(o, "return_type", None) == qml.measurements.MidMeasure
                and not self._supports_mid_measure
            ):
                raise qml._device.DeviceError(
                    f"Mid-circuit measurements are not natively supported on device {self.device_short_name}. "
                    "Apply the @qml.defer_measurements decorator to your quantum function to "
                    "simulate the application of mid-circuit measurements on this device."
                )

            if o.inverse:
                if not self._supports_inverses:
                    raise qml._device.DeviceError(
                        f"The inverse of gates are not supported on device {self.device_short_name}"
                    )
                operation_name = o.base_name

            if not self.supports_operation(operation_name):
                raise qml._device.DeviceError(
                    f"Gate {operation_name} not supported on device {self.device_short_name}"
                )

        for o in observables:
            if isinstance(o, qml.measurements.MeasurementProcess) and o.obs is not None:
                o = o.obs

            if isinstance(o, qml.operation.Tensor):
                # TODO: update when all capabilities keys changed to "supports_tensor_observables"
                if not self._supports_tensor_observables:
                    raise DeviceError(
                        f"Tensor observables not supported on device {self.device_short_name}"
                    )

                for i in o.obs:
                    if not self._supports_observable(i.name):
                        raise DeviceError(
                            f"Observable {i.name} not supported on device {self.device_short_name}"
                        )
            else:
                observable_name = o.name

                if issubclass(o.__class__, Operation) and o.inverse:
                    if not self._supports_inverses:
                        raise DeviceError(
                            f"The inverse of gates are not supported on device {self.device_short_name}"
                        )
                    observable_name = o.base_name

                if not self.supports_observable(observable_name):
                    raise DeviceError(
                        f"Observable {observable_name} not supported on device {self.device_short_name}"
                    )

    def expand(self, tape: qml.tape.QuantumTape, max_expansion: int = 10) -> qml.tape.QuantumTape:

        obs_on_same_wire = len(tape._obs_sharing_wires) > 0
        obs_on_same_wire &= not any(isinstance(o, qml.Hamiltonian) for o in tape._obs_sharing_wires)

        ops_not_supported = not all(self._stopping_condition(op) for op in tape.operations)

        if ops_not_supported or obs_on_same_wire:
            new_tape = tape.expand(depth=max_expansion, stop_at=self._stopping_condition)

        self._check_validity(new_tape.operations, new_tape.observables)

        return new_tape

    def batch_transform(self, tape: qml.tape.QuantumTape):

        # find a good way to retrieve this information
        finite_shots = False

        supports_hamiltonian = self._supports_observable("Hamiltonian")

        grouping_known = all(
            obs.grouping_indices is not None
            for obs in tape.observables
            if obs.name == "Hamiltonian"
        )

        hamiltonian_in_obs = "Hamiltonian" in [obs.name for obs in tape.observables]

        return_types = [m.return_type for m in tape.observables]

        if hamiltonian_in_obs and ((not supports_hamiltonian or finite_shots) or grouping_known):
            # If the observable contains a Hamiltonian and the device does not
            # support Hamiltonians, or if the simulation uses finite shots, or
            # if the Hamiltonian explicitly specifies an observable grouping,
            # split tape into multiple tapes of diagonalizable known observables.
            try:
                tapes, hamiltonian_fn = qml.transforms.hamiltonian_expand(tape, group=False)

            except ValueError as e:
                raise ValueError(
                    "Can only return the expectation of a single Hamiltonian observable"
                ) from e
        elif (
            len(tape._obs_sharing_wires) > 0
            and not hamiltonian_in_obs
            and not qml.measurements.Sample in return_types
            and not qml.measurements.Probability in return_types
        ):
            # Check for case of non-commuting terms and that there are no Hamiltonians
            # TODO: allow for Hamiltonians in list of observables as well.
            tapes, hamiltonian_fn = qml.transforms.split_non_commuting(tape)

        else:
            # otherwise, return the output of an identity transform
            tapes, hamiltonian_fn = [tape], lambda res: res[0]

        # Check whether the tape was broadcasted (then the Hamiltonian-expanded
        # ones will be as well) and whether broadcasting is supported
        if tape.batch_size is None or self._supports_broadcasting:
            # If the tape wasn't broadcasted or broadcasting is supported, no action required
            return tapes, hamiltonian_fn

        # Expand each of the broadcasted Hamiltonian-expanded tapes
        expanded_tapes, expanded_fn = qml.transforms.map_batch_transform(
            qml.transforms.broadcast_expand, tapes
        )

        # Chain the postprocessing functions of the broadcasted-tape expansions and the Hamiltonian
        # expansion. Note that the application order is reversed compared to the expansion order,
        # i.e. while we first applied `hamiltonian_expand` to the tape, we need to process the
        # results from the broadcast expansion first.
        total_processing = lambda results: hamiltonian_fn(expanded_fn(results))

        return expanded_tapes, total_processing
