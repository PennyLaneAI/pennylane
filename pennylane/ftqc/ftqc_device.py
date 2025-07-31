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

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, validate_mcm_method
from pennylane.devices.device_api import Device, _default_mcm_method
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane.devices.preprocess import (
    measurements_from_samples,
    no_analytic,
    validate_device_wires,
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
        backend: A backend that circuits will be executed on.

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
            validate_device_wires,
            wires=self.backend.wires,
            name=f"{self.name}.{self.backend.name}",
        )

        # set up for backend execution (including MCM handling)
        if self.backend.diagonalize_mcms:
            program.add_transform(diagonalize_mcms)
        # backend preprocess will include mcm execution method if relevant
        backend_program, _ = self.backend.device.preprocess(execution_config)

        # we skip gradient preprocess transforms, not worrying about derivatives for this prototype

        return program + backend_program

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        """Sets up an ``ExecutionConfig`` that configures the execution behaviour.

        In this case, the only modification compared to the standard `ExecutionConfig` is
        that it gets the MCM method for the backend toml file (either "device" or "one-shot"),
        so that it can be included when building the transform program for the backend.
        uses the PennyLane device API.

        Args:
            config (ExecutionConfig): The initial ExecutionConfig object that describes the
                parameters needed to configure the execution behaviour.
            circuit (QuantumScript): The quantum circuit to customize the execution config for.

        Returns:
            ExecutionConfig: The updated ExecutionConfig object

        """

        if config is None:
            config = ExecutionConfig()

        # get mcm method - "device" if its an option, otherwise "one-shot"
        default_mcm_method = _default_mcm_method(self.backend.capabilities, shots_present=True)
        new_mcm_config = replace(config.mcm_config, mcm_method=default_mcm_method)
        config = replace(config, mcm_config=new_mcm_config)
        validate_mcm_method(
            self.backend.capabilities, config.mcm_config.mcm_method, shots_present=True
        )

        return config

    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        """Execution method for the frontend. To be expanded to orchestrate executing
        in chunks with feedback for corrections before non-Clifford gates. Currently
        just feeds into the backend execution"""
        return self.backend.execute(circuits, execution_config)


# pylint: disable=too-few-public-methods
class LightningQubitBackend:
    """Wrapper for using lightning.qubit as a backend for the ftqc.qubit device"""

    name = "lightning"
    config_filepath = Path(__file__).parent / "lightning_backend.toml"

    def __init__(self):
        self.diagonalize_mcms = True
        self.wires = qml.wires.Wires(range(25))
        self.device = qml.device("lightning.qubit")
        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def execute(self, circuits, execution_config):
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        mcmc = {
            "mcmc": self.device._mcmc,
            "kernel_name": self.device._kernel_name,
            "num_burnin": self.device._num_burnin,
        }
        results = []
        for circuit in circuits:
            if self.device._wire_map is not None:
                [circuit], _ = qml.map_wires(circuit, self.device._wire_map)
            results.append(
                self.simulate(
                    self.device.dynamic_wires_from_circuit(circuit),
                    self.device._statevector,
                    mcmc=mcmc,
                    postselect_mode=execution_config.mcm_config.postselect_mode,
                )
            )

        return tuple(results)

    def simulate(
        self,
        circuit: QuantumScript,
        state: LightningStateVector,
        mcmc: dict = None,
        postselect_mode: str = None,
    ) -> Result:
        """Simulate a single quantum script.

        Args:
            circuit (QuantumTape): The single circuit to simulate
            state (LightningStateVector): handle to Lightning state vector
            mcmc (dict): Dictionary containing the Markov Chain Monte Carlo
                parameters: mcmc, kernel_name, num_burnin. Descriptions of
                these fields are found in :class:`~.LightningQubit`.
            postselect_mode (str): Configuration for handling shots with mid-circuit measurement
                postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
                keep the same number of shots. Default is ``None``.

        Returns:
            Tuple[TensorLike]: The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.
        """
        if mcmc is None:
            mcmc = {}

        results = []
        aux_circ = qml.tape.QuantumScript(
            circuit.operations,
            circuit.measurements,
            shots=[1],
            trainable_params=circuit.trainable_params,
        )
        for _ in range(circuit.shots.total_shots):
            state.reset_state()
            mid_measurements = {}
            final_state = state.get_final_state(
                aux_circ, mid_measurements=mid_measurements, postselect_mode=postselect_mode
            )
            results.append(
                self.device.LightningMeasurements(final_state, **mcmc).measure_final_state(
                    aux_circ, mid_measurements=mid_measurements
                )
            )
        return tuple(results)


class NullQubitBackend:
    """Wrapper for using null.qubit as a backend for the ftqc.qubit device"""

    name = "null"
    config_filepath = Path(__file__).parent / "null_backend.toml"

    def __init__(self):
        self.diagonalize_mcms = False
        self.wires = qml.wires.Wires(range(1000))
        self.device = qml.device("null.qubit")

        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def execute(self, circuits, execution_config):
        """Probably not in need of any modification, since its mocked."""
        return self.device.execute(circuits, execution_config)
