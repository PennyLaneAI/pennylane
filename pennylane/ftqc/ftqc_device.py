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

# ToDo: see if we can get rid of this in favour of backend.device.preprocess
from pennylane_lightning.lightning_qubit.lightning_qubit import stopping_condition_shots

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, validate_mcm_method
from pennylane.devices.device_api import Device, _default_mcm_method
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane.devices.preprocess import (
    measurements_from_samples,
    no_analytic,
    null_postprocessing,
    validate_device_wires,
    validate_observables,
)
from pennylane.ftqc import (
    RotXZX,
    convert_to_mbqc_formalism,
    convert_to_mbqc_gateset,
    diagonalize_mcms,
)
from pennylane.ops import RZ
from pennylane.tape.qscript import QuantumScript
from pennylane.transforms import combine_global_phases, split_non_commuting
from pennylane.typing import Result

from .quantum_script_sequence import QuantumScriptSequence


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
        program.add_transform(split_at_non_clifford_gates)
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

        # backend_program, _ = self.backend.device.preprocess()

        program.add_transform(
            qml.devices.preprocess.mid_circuit_measurements,
            device=self,
            mcm_config=execution_config.mcm_config,
        )

        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=stopping_condition_shots,
            skip_initial_state_prep=True,
            name=self.name,
        )

        return program

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

        if config.mcm_config.mcm_method is None:
            # get mcm method - "device" if its an option, otherwise "one-shot"
            default_mcm_method = _default_mcm_method(self.backend.capabilities, shots_present=True)
            new_mcm_config = replace(config.mcm_config, mcm_method=default_mcm_method)
            config = replace(config, mcm_config=new_mcm_config)
        validate_mcm_method(
            self.backend.capabilities, config.mcm_config.mcm_method, shots_present=True
        )

        if config.mcm_config.mcm_method == "deferred":
            raise ValueError("Please no deferred measurements, thanks")

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

        # pylint: disable=protected-access
        mcmc = {
            "mcmc": self.device._mcmc,
            "kernel_name": self.device._kernel_name,
            "num_burnin": self.device._num_burnin,
        }
        postselect_mode = execution_config.mcm_config.postselect_mode

        results = []
        for circuit in circuits:
            if isinstance(circuit, QuantumScript):
                res = self._qscript_execute(circuit, mcmc, postselect_mode)
            elif isinstance(circuit, QuantumScriptSequence):
                res = self._sequence_execute(circuit, mcmc, postselect_mode)
            else:
                raise RuntimeError("That's not a QuantumScript or a QuantumScriptSequence")
            results.append(tuple(res))
        return tuple(results)

    def _qscript_execute(self, qscript, mcmc, postselect_mode):
        """Execute a circuit defined by a QuantumScript"""

        aux_circ = qml.tape.QuantumScript(
            qscript.operations,
            qscript.measurements,
            shots=[1],
            trainable_params=qscript.trainable_params,
        )
        circ_res = []
        for _ in range(qscript.shots.total_shots):
            # calling dynamic_wires_from_circuit for each call of simulate
            # resets the statevector
            res = self.simulate(
                self.device.dynamic_wires_from_circuit(aux_circ),
                self.device._statevector,  # pylint: disable=protected-access
                mcmc=mcmc,
                postselect_mode=postselect_mode,
            )
            # simulate has already cast it to an array, but we are
            # doing shots in a different order than it expects, so
            # it messes up array indexing in post-processing. Casting
            # to a list is a hack to fix this.
            circ_res.append(list(res))
        return circ_res

    def _sequence_execute(self, sequence, mcmc, postselect_mode):
        """Execute a QuantumTapeSequence, containing several tapes to be executed
        in succession without resetting the state, followed by terminal measurements
        performed at the end of the sequence"""
        # this resets the statevector and makes it the correct size
        # We need to do it once for the full sequence, then reset for each shot
        sequence = self.device.dynamic_wires_from_circuit(sequence)

        circ_res = []
        # pylint: disable=protected-access
        for _ in range(sequence.shots.total_shots):
            self.device._statevector.reset_state()

            for segment in sequence.intermediate_tapes:

                _ = self.simulate(
                    segment,
                    self.device._statevector,
                    mcmc=mcmc,
                    postselect_mode=postselect_mode,
                )

            res = self.simulate(
                sequence.final_tape,
                self.device._statevector,
                mcmc=mcmc,
                postselect_mode=postselect_mode,
            )
            circ_res.append(list(res))
        return circ_res

    def simulate(
        self,
        circuit: QuantumScript,
        state: "LightningStateVector",
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

        if circuit.shots.total_shots != 1:
            raise RuntimeError("Lightning backend expects single-shot for simulation")

        mid_measurements = {}
        final_state = state.get_final_state(
            circuit, mid_measurements=mid_measurements, postselect_mode=postselect_mode
        )

        return self.device.LightningMeasurements(final_state, **mcmc).measure_final_state(
            circuit, mid_measurements=mid_measurements
        )


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

        assert isinstance(
            circuits[0], QuantumScriptSequence
        ), "something is amiss - this device uses QuantumScriptSequence"

        tapes_to_execute = []
        for circuit in circuits:
            shots = circuit.shots.total_shots
            final_tape = circuit.final_tape
            tapes_to_execute.append(final_tape.copy(shots=shots))
        return tuple(
            self.device.execute(tapes_to_execute, execution_config),
        )


@qml.transform
def split_at_non_clifford_gates(tape):
    """The most basic implementation to ensure that we flush the buffer before
    each non-Clifford gate. Pays no attention to wires/commutation, and splits
    the tapes up more than necessary, but the logic is very simple."""
    all_operations = [[]]

    for op in tape.operations:
        # if its a non-Clifford gate, and there are already ops in the list, add a new list
        if isinstance(op, (RotXZX, RZ)) and all_operations[-1]:
            all_operations.append([])
        all_operations[-1].append(op)

    tapes = []
    for ops_list in all_operations[:-1]:
        tapes.append(tape.copy(operations=ops_list, measurements=[]))
    tapes.append(tape.copy(operations=all_operations[-1]))

    return [
        QuantumScriptSequence(tapes),
    ], null_postprocessing
