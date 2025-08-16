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
device
"""
from dataclasses import replace
from pathlib import Path

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, validate_mcm_method
from pennylane.devices.device_api import Device, _default_mcm_method
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig, MCMConfig
from pennylane.devices.preprocess import (
    measurements_from_samples,
    no_analytic,
    validate_device_wires,
    validate_observables,
)
from pennylane.ftqc import (
    GraphStatePrep,
    convert_to_mbqc_formalism,
    convert_to_mbqc_gateset,
    diagonalize_mcms,
)
from pennylane.tape.qscript import QuantumScript
from pennylane.transforms import combine_global_phases, split_non_commuting
from pennylane.typing import Result

from .quantum_script_sequence import QuantumScriptSequence, split_at_non_clifford_gates


class FTQCQubit(Device):
    """An experimental PennyLane device frontend for processing circuits and executing them on an
    FTQC/MBQC based backend.

    Args:
        wires (int, Iterable[Number, str]): Number of logical wires present on the device, or iterable that
            contains consecutive integers starting at 0 to be used as wire labels (i.e., ``[0, 1, 2]``).
            The number of wires will be limited by the capabilities of the backend.
        backend: A backend that circuits will be executed on.

    """

    name = "ftqc.qubit"
    config_filepath = Path(__file__).parent / "ftqc_device.toml"

    def __init__(self, wires, backend):

        super().__init__(wires=wires)

        self._backend = backend
        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def preprocess_transforms(self, execution_config=DefaultExecutionConfig):
        """Returns the transform program to preprocess a circuit for execution.

        Args:
            execution_config (ExecutionConfig): The execution configuration object

        Returns:
            TransformProgram: A transform program that is called before execution

        The transform program is composed of a list of individual transforms. For this device,
        it processes a general circuit to one that is expressed in the gate-set needed for
        conversion to the textbook MBQC formalism, including (but not limited to):

        * Splitting a circuit with measurements of non-commuting observables or Hamiltonians into multiple executions.
        * Diagonalizing and converting all measurements to be derived from sampling the computational basis.
        * Decomposition of operations and measurements to the MBQC gate-set
        * Validation of wires, measurements, and observables.
        """

        program = qml.transforms.core.TransformProgram()

        # validate the initial circuit is one we can process + update all observables to have wires
        program.add_transform(no_analytic)
        program.add_transform(validate_device_wires, wires=self.wires, name=self.name)
        program.add_transform(
            validate_observables,
            lambda obs: self.capabilities.supports_observable(obs.name),
            name=self.name,
        )

        # convert to mbqc gate set
        program.add_transform(split_non_commuting)
        program.add_transform(measurements_from_samples)

        program.add_transform(convert_to_mbqc_gateset)
        program.add_transform(combine_global_phases)
        program.add_transform(split_at_non_clifford_gates)

        return program

    @property
    def backend(self):
        """The backend device circuits will be sent to for execution"""
        return self._backend

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

        # get mcm method - "device" if its an option, otherwise "one-shot"
        default_mcm_method = _default_mcm_method(self.backend.capabilities, shots_present=True)
        assert default_mcm_method in ["device", "one-shot"]

        if config is None:
            config = ExecutionConfig(mcm_config=MCMConfig(mcm_method=default_mcm_method))
        else:
            if config.mcm_config.mcm_method is None:
                new_mcm_config = replace(config.mcm_config, mcm_method=default_mcm_method)
                config = replace(config, mcm_config=new_mcm_config)

        validate_mcm_method(
            self.backend.capabilities, config.mcm_config.mcm_method, shots_present=True
        )

        return config

    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        """Execute a circuit or a batch of circuits and turn it into results. For this
        device, each circuit is expected to the expressed as QuantumScriptSequence. This
        is ensured by the transform program in device preprocessing.

        Args:
            circuits (Sequence[QuantumScriptSequence]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.

        """
        return self.backend.execute(circuits, execution_config)


# pylint: disable=too-few-public-methods
class LightningQubitBackend:
    """Wrapper for using lightning.qubit as a backend for the ftqc.qubit device"""

    name = "lightning"
    config_filepath = Path(__file__).parent / "lightning_backend.toml"

    def preprocess_transforms(self, execution_config=DefaultExecutionConfig):
        """Returns the transform program to preprocess a circuit for execution on the backend.

        Args:
            execution_config (ExecutionConfig): The execution configuration object

        Returns:
            TransformProgram: A transform program that is called before execution

        The transform program is composed of a list of individual transforms. For this device,
        it assumes the circuit is expressed in the MBQC formalism. Transforms for backend include:

        * Any handling needed for the backed to process mid-circuit measurements
        * Decomposition of graph state operations to their component parts
        * Validation of wires
        """

        program = qml.transforms.core.TransformProgram()

        # validate that conversion didn't use too many wires
        program.add_transform(
            validate_device_wires,
            wires=self.wires,
            name=f"{self.name} FTQC backend",
        )

        program.add_transform(diagonalize_mcms)
        program.add_transform(
            qml.devices.preprocess.mid_circuit_measurements,
            device=self,
            mcm_config=execution_config.mcm_config,
        )
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=lambda op: not isinstance(op, GraphStatePrep),
            skip_initial_state_prep=True,
            name=self.name,
        )

        return program

    def __init__(self):
        self.diagonalize_mcms = True
        self.wires = qml.wires.Wires(range(25))
        self.device = qml.device("lightning.qubit")
        self.capabilities = DeviceCapabilities.from_toml_file(self.config_filepath)

    def execute(self, sequences, execution_config=DefaultExecutionConfig):
        """Execute a sequence or a batch of sequences and turn it into results.

        Args:
            sequences (Sequence[QuantumScriptSequence]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        # before we process the tape any further, we need to get the online and offline corrections
        # from the pauli tracker based on the circuit expressed in the MBQC gate set
        from functools import partial

        from pennylane.ftqc.decomposition import convert_to_mbqc_formalism_with_pauli_tracker

        mbqc_sequences = []
        online_corrections = []
        # ToDo: I believe this this could be applied to the full set of sequences in one go
        # if we get it from a transform that just processes the tapes and returns the tapes and postprocessing
        # that would also move us closer to putting it the correct place in the pipeline if we can sort out the
        # diagonalization bug that's stopping us from appending measurement
        offline_corrections = []
        for sequence in sequences:
            new_seq, online, offline = convert_to_mbqc_formalism_with_pauli_tracker(sequence)
            mbqc_sequences.append(new_seq)
            online_corrections.append(online)
            offline_corrections.append(offline)

        # sequences, _ = convert_to_mbqc_formalism(sequences)

        preprocess_transforms = self.preprocess_transforms(execution_config)
        mbqc_sequences, postprocess_fns = preprocess_transforms(mbqc_sequences)

        results = []
        for sequence, byops_fns, correction_fn in zip(
            mbqc_sequences, online_corrections, offline_corrections
        ):
            assert isinstance(
                sequence, QuantumScriptSequence
            ), "something is amiss - this device uses QuantumScriptSequence"
            res = self._execute_sequence(sequence, execution_config, byops_fns)
            res = correction_fn(res)
            results.append(res)

        return postprocess_fns(results)

    def _execute_sequence(self, sequence, execution_config, online_corrections):
        """Execute a single QuantumScriptSecuence.

        Args:
            circuits (QuantumScriptSequence): the sequence of quantum scripts to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        # pylint: disable=protected-access
        mcmc = {
            "mcmc": self.device._mcmc,
            "kernel_name": self.device._kernel_name,
            "num_burnin": self.device._num_burnin,
        }
        postselect_mode = execution_config.mcm_config.postselect_mode

        sequence = self.device.dynamic_wires_from_circuit(sequence)

        results = []
        # pylint: disable=protected-access
        for _ in range(sequence.shots.total_shots):
            self.device._statevector.reset_state()

            for segment, corr in zip(sequence.intermediate_tapes, online_corrections):

                mcms = self.simulate(
                    segment,
                    self.device._statevector,
                    mcmc=mcmc,
                    postselect_mode=postselect_mode,
                )

                ops = corr(mid_meas=mcms)  # ToDo: are these in the correct order?
                corrections = QuantumScript(ops, shots=1)

                _ = self.simulate(
                    corrections,
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

            results.append(list(res))
        return tuple(results)

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

    def preprocess_transforms(self, execution_config=DefaultExecutionConfig):
        """Returns the transform program to preprocess a circuit for execution on the backend.

        Args:
            execution_config (ExecutionConfig): The execution configuration object

        Returns:
            TransformProgram: A transform program that is called before execution

        The transform program is composed of a list of individual transforms. For this device,
        it assumes the circuit is expressed in the MBQC formalism. Transforms for backend include:

        * Any handling needed for the backed to process mid-circuit measurements
        * Decomposition of graph state operations to their component parts
        * Validation of wires
        """

        program = qml.transforms.core.TransformProgram()

        # validate that conversion didn't use too many wires
        program.add_transform(
            validate_device_wires,
            wires=self.wires,
            name=f"{self.name} FTQC backend",
        )

        program = qml.transforms.core.TransformProgram()
        program.add_transform(
            qml.devices.preprocess.mid_circuit_measurements,
            device=self,
            mcm_config=execution_config.mcm_config,
        )
        program.add_transform(
            qml.devices.preprocess.decompose,
            stopping_condition=lambda op: not isinstance(op, GraphStatePrep),
            skip_initial_state_prep=True,
            name=self.name,
        )

        return program

    def execute(self, sequences, execution_config):
        """Mock execution of a sequence or a batch of sequences and turn it into null results.

        Args:
            sequences (Sequence[QuantumScriptSequence]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        tapes_to_execute = []

        preprocess_transforms = self.preprocess_transforms(execution_config)
        sequences, mcm_corrections = convert_to_mbqc_formalism(sequences)
        sequences, postprocess_fns = preprocess_transforms(sequences)

        for sequence in sequences:
            assert isinstance(
                sequence, QuantumScriptSequence
            ), "something is amiss - this device uses QuantumScriptSequence"

            shots = sequence.shots.total_shots
            final_tape = sequence.final_tape
            tapes_to_execute.append(final_tape.copy(shots=shots))

        result = tuple(self.device.execute(tapes_to_execute, execution_config))

        # ToDo: this doesn't work yet, I'm getting a shape wrong somewhere
        # fix it or drop the NullQubit backend
        result = mcm_corrections(result)

        return postprocess_fns(result)
