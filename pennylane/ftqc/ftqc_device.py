# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains an implementation of a PennyLane frontend (Device) for an FTQC/MBQC based
hardware device (or emulator)
"""
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, validate_mcm_method
from pennylane.devices.device_api import Device, _default_mcm_method
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane.devices.preprocess import (
    measurements_from_samples,
    no_analytic,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.ftqc import convert_to_mbqc_formalism, convert_to_mbqc_gateset, diagonalize_mcms
from pennylane.tape.qscript import QuantumScript
from pennylane.transforms import combine_global_phases, split_non_commuting


class FTQCQubit(Device):
    """An experimental PennyLane device frontend for processing circuits and executing them on an
    FTQC/MBQC based backend.

    Args:
        wires (int, Iterable[Number, str]): Number of logical wires present on the device, or iterable that
            contains consecutive integers starting at 0 to be used as wire labels (i.e., ``[0, 1, 2]``).
            This device allows for ``wires`` to be unspecified at construction time. The number of wires
            will be limited by the capabilities of the backend.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device. Note that during execution, shots
            are pulled from the circuit, not from the device.
        backend (?????): A backend that circuits will be executed on.

    """

    name = "ftqc.qubit"
    config_filepath = Path(__file__).parent / "ftqc_device.toml"

    def __init__(self, wires=None, backend=None):
        if backend is None:
            raise RuntimeError

        super().__init__(wires=wires)

        self.backend = backend
        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def preprocess_transforms(self, execution_config=DefaultExecutionConfig):

        program = qml.transforms.core.TransformProgram()

        # validate the initial circuit is one we can process + update all observables to have wires
        program.add_transform(no_analytic)
        program.add_transform(validate_device_wires, wires=self.wires, name=self.name)
        program.add_transform(
            validate_observables,
            lambda obs: self.capabilities.supports_observable(obs.name),
            name=self.name,
        )

        # convert to mbqc formalism
        program.add_transform(split_non_commuting)
        program.add_transform(measurements_from_samples)

        program.add_transform(convert_to_mbqc_gateset)
        program.add_transform(combine_global_phases)
        program.add_transform(convert_to_mbqc_formalism)

        # validate that conversion didn't use too many wires
        program.add_transform(
            validate_device_wires, wires=self.backend.wires, name=f"{self.name}.{self.backend.name}"
        )

        # set up for backend execution (including MCM handling)
        if self.backend.diagonalize_mcms:
            program.add_transform(diagonalize_mcms)
        backend_program, _ = self.backend.device.preprocess(
            execution_config
        )  # adds mcm execution method if relevant

        # we skip gradient preprocess transforms, as the device does not support derivatives

        return program + backend_program

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        """Sets up an ``ExecutionConfig`` that configures the execution behaviour.

        The execution config stores information on how the device should perform the execution,
        as well as how PennyLane should interact with the device. See :class:`ExecutionConfig`
        for all available options and what they mean.

        An ``ExecutionConfig`` is constructed from arguments passed to the ``QNode``, and this
        method allows the device to update the config object based on device-specific requirements
        or preferences. See :ref:`execution_config` for more details.

        Args:
            config (ExecutionConfig): The initial ExecutionConfig object that describes the
                parameters needed to configure the execution behaviour.
            circuit (QuantumScript): The quantum circuit to customize the execution config for.

        Returns:
            ExecutionConfig: The updated ExecutionConfig object

        """

        if config is None:
            config = ExecutionConfig()

        if self.supports_derivatives(config) and config.gradient_method in ("best", None):
            return replace(config, gradient_method="device")

        default_mcm_method = _default_mcm_method(self.backend.capabilities, shots_present=True)
        new_mcm_config = replace(config.mcm_config, mcm_method=default_mcm_method)
        config = replace(config, mcm_config=new_mcm_config)
        validate_mcm_method(
            self.backend.capabilities, config.mcm_config.mcm_method, shots_present=True
        )

        return config

    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return self.backend.execute(circuits, execution_config)


class LightningQubitBackend:

    name = "lightning"
    config_filepath = Path(__file__).parent / "lightning_backend.toml"

    def __init__(self):
        self.diagonalize_mcms = True
        self.wires = qml.wires.Wires(range(25))
        self.device = qml.device("lightning.qubit")
        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def execute(self, circuits, execution_config):
        return self.device.execute(circuits, execution_config)


class NullQubitBackend:

    name = "null"
    config_filepath = Path(__file__).parent / "null_backend.toml"

    def __init__(self):
        self.diagonalize_mcms = False
        self.wires = qml.wires.Wires(range(1000))
        self.device = qml.device("null.qubit")

        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def execute(self, circuits, execution_config):
        return self.device.execute(circuits, execution_config)
