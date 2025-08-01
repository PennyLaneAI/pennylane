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
from functools import cached_property
import copy

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
from pennylane.ftqc import (
    RotXZX,
    convert_to_mbqc_formalism,
    convert_to_mbqc_gateset,
    diagonalize_mcms,
)
from pennylane.measurements import MidMeasureMP
from pennylane.ops import RZ
from pennylane.tape.qscript import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import combine_global_phases, split_non_commuting
from pennylane.typing import Result


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
        postselect_mode = execution_config.mcm_config.postselect_mode

        results = []
        for circuit in circuits:
            if isinstance(circuit, QuantumScript):
                results.append(self._qscript_execute(circuit, mcmc, postselect_mode))
            elif isinstance(circuit, QuantumScriptSequence):
                results.append(self._sequence_execute(circuit, mcmc, postselect_mode))
            else:
                raise RuntimeError("That's not a QuantumScript or a QuantumScriptSequence")
        return tuple(results)


    def _qscript_execute(self, qscript, mcmc, postselect_mode):
        """Execute a circuit defined by a QuantumScript"""

        if self.device._wire_map is not None:
            [circuit], _ = qml.map_wires(circuit, self.device._wire_map)

        aux_circ = qml.tape.QuantumScript(
            qscript.operations,
            qscript.measurements,
            shots=[1],
            trainable_params=qscript.trainable_params,
        )
        circ_res = []
        for _ in range(qscript.shots.total_shots):
            if self.device._statevector is not None:
                self.device._statevector.reset_state()
            circ_res.append(
                self.simulate(
                    self.device.dynamic_wires_from_circuit(aux_circ),
                    self.device._statevector,
                    mcmc=mcmc,
                    postselect_mode=postselect_mode,
                )
            )
        return tuple(circ_res)


    def _sequence_execute(self, sequence, mcmc, postselect_mode):

            # this resets the statevector (and makes it the correct size), actually, so we need to do it once for the full sequence 
            sequence = self.device.dynamic_wires_from_circuit(sequence)

            circ_res = []
            for _ in range(sequence.shots.total_shots):
                if self.device._statevector is not None:
                    self.device._statevector.reset_state()

                for segment in sequence.intermediate_tapes:
                    if self.device._wire_map is not None:
                        [segment], _ = qml.map_wires(segment, self.device._wire_map)

                    _ = self.simulate(
                            segment,
                            self.device._statevector,
                            mcmc=mcmc,
                            postselect_mode=postselect_mode,
                        )
                    
                circ_res.append(
                    self.simulate(
                        sequence.final_tape,
                        self.device._statevector,
                        mcmc=mcmc,
                        postselect_mode=postselect_mode,
                    )
                )
            return tuple(circ_res)

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
        return self.device.execute(circuits, execution_config)


class QuantumScriptSequence:
    """A sequence of tapes meant to be executed in order without resetting the system state.
    Intermediate tapes may return mid-circuit measurements, or nothing. This is not currently 
    validated. The final tape returns terminal measurements."""

    def __init__(self, tapes, shots=None):
        
        if shots is None:
            shots = [tape.shots for tape in tapes]
            if len(set(shots)) != 1:
                raise RuntimeError("All scripts in a QuantumScriptSequence must have the same shots")
            shots=shots[0]
        self._shots = shots

        self._tapes = []

        for tape in tapes:
            aux_tape = qml.tape.QuantumScript(
                tape.operations,
                tape.measurements,
                shots=[1],
            )
            self._tapes.append(aux_tape)

    @property
    def tapes(self):
        return self._tapes

    @property
    def final_tape(self):
        return self._tapes[-1]

    @property
    def intermediate_tapes(self):
        return self._tapes[:-1]

    @property
    def measurements(self):
        return self.final_tape.measurements

    @property
    def intermediate_measurements(self):
        return [tape.measurements for tape in self.intermediate_tapes]

    @property
    def operations(self):
        return [tape.operations for tape in self.tapes]

    @cached_property
    def wires(self) -> qml.wires.Wires:
        """Returns the wires used in the quantum script process

        Returns:
            ~.Wires: wires in quantum script process
        """
        wires = self.tapes[0].wires
        for tape in self.tapes[1:]:
            wires += tape.wires
        return wires
    
    @property
    def num_wires(self) -> int:
        """Returns the number of wires in the quantum script process

        Returns:
            int: number of wires in quantum script process
        """
        return len(self.wires)
    
    @property
    def shots(self):
        return self._shots

    def __repr__(self):
        return f"<QuantumScriptSequence: wires={list(self.wires)}>"
    
    def map_to_standard_wires(self) -> "QuantumScriptSequence":
        """
        Wrapper to apply qscript.map_to_standard_wires to each segment contained in the Sequence
        """
        wire_map = self._get_standard_wire_map()
        if wire_map is None:
            return self
        new_tapes = []
        for tape in self.tapes:
            tapes, fn = qml.map_wires(tape, wire_map)
            new_tapes.append(fn(tapes))

        return self.copy(tapes=new_tapes)

    def _get_standard_wire_map(self) -> dict:
        """Helper function to produce the wire map for map_to_standard_wires. Wire map
        is the same as if the sequence were a flat tape"""
        flat_ops = []
        for ops in self.operations:
            flat_ops.extend(ops)

        as_tape = qml.tape.QuantumScript(flat_ops, self.measurements)
        return as_tape._get_standard_wire_map()
    
    def copy(self, copy_operations: bool = False, **update):
        """Make it copy-able as if it were a tape where possible. Do not allow 
        modifications to operations or trainable parameters, because any transform 
        or function modifying operations on a tape will not work on a sequence of 
        tapes. Allow updating tapes as a whole as an alternative for 
        QuantumScriptSquence-specific functions to deal with modifying operations 
        on tapes.
        
        This is not able to support trainable parameters and almost certainly also 
        has other flaws. It is not a thorough implementation of the desired behaviour."""
        if copy_operations is True:
            raise RuntimeError("Can't use copy_operations when copying a QuantumScriptSequence")
        
        if update:
            if "ops" in update:
                update["operations"] = update["ops"]
            for k in update:
                if k not in ["tapes", "measurements", "shots"]:
                    raise TypeError(
                        f"{self.__class__}.copy() cannot update '{k}'"
                    )
            if "tapes" in update and "measurements" in update:
                raise RuntimeError("Can't update tapes and measurements at the same time, as tapes include measurements")

        _tapes = update.get("tapes", [copy.copy(tape) for tape in self.tapes])
        _shots = update.get("shots", self.shots)

        if "measurements" in update:
            old_final_tape = _tapes.pop()
            _new_final_tape = old_final_tape.copy(measurements=update["measurements"])
            _tapes.append(_new_final_tape)

        new_sequence = QuantumScriptSequence(
            tapes = _tapes,
            shots=_shots,
        )

        return new_sequence



def split_at_non_clifford_gates(tape):
    """The most basic implementation to ensure that we flush the buffer before 
    each non-Clifford gate. Pays no attention to wires/commutation, and splits 
    the tapes up more than necessary, but the logic is very simple."""
    all_operations = [[]]

    for op in tape.operations:
        if isinstance(op, (RotXZX, RZ)) and all_operations[-1] != []:
            all_operations.append([])
        all_operations[-1].append(op)

    tapes = []
    for ops_list in all_operations[:-1]:
        tapes.append(tape.copy(operations=ops_list, measurements=[]))
    tapes.append(tape.copy(operations=all_operations[-1]))

    return QuantumScriptSequence(tapes)
