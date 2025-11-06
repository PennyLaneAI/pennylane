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
r"""
The ``default.mixed`` device is PennyLane's standard qubit simulator for mixed-state computations.

It implements some built-in qubit :doc:`operations </introduction/operations>`,
providing a simple mixed-state simulation of qubit-based quantum circuits.

"""
import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace

import pennylane as qml
from pennylane.devices.qubit_mixed import simulate
from pennylane.exceptions import DeviceError
from pennylane.logging import debug_logger, debug_logger_init
from pennylane.math import Interface
from pennylane.ops.channel import __qubit_channels__ as channels
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from . import Device
from .execution_config import ExecutionConfig
from .modifiers import simulator_tracking, single_tape_support
from .preprocess import (
    decompose,
    no_sampling,
    null_postprocessing,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

observables = {
    "Hadamard",
    "Hermitian",
    "Identity",
    "PauliX",
    "PauliY",
    "PauliZ",
    "Prod",
    "Projector",
    "SparseHamiltonian",
    "SProd",
    "Sum",
}

operations = {
    "Identity",
    "Snapshot",
    "BasisState",
    "StatePrep",
    "QubitDensityMatrix",
    "QubitUnitary",
    "ControlledQubitUnitary",
    "BlockEncode",
    "MultiControlledX",
    "DiagonalQubitUnitary",
    "SpecialUnitary",
    "PauliX",
    "PauliY",
    "PauliZ",
    "MultiRZ",
    "Hadamard",
    "S",
    "T",
    "SX",
    "CNOT",
    "SWAP",
    "ISWAP",
    "CSWAP",
    "Toffoli",
    "CCZ",
    "CY",
    "CZ",
    "CH",
    "PhaseShift",
    "PCPhase",
    "ControlledPhaseShift",
    "CPhaseShift00",
    "CPhaseShift01",
    "CPhaseShift10",
    "RX",
    "RY",
    "RZ",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "AmplitudeDamping",
    "GeneralizedAmplitudeDamping",
    "PhaseDamping",
    "DepolarizingChannel",
    "BitFlip",
    "PhaseFlip",
    "PauliError",
    "ResetError",
    "QubitChannel",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "QubitCarry",
    "QubitSum",
    "OrbitalRotation",
    "FermionicSWAP",
    "QFT",
    "ThermalRelaxationError",
    "ECR",
    "ParametrizedEvolution",
    "GlobalPhase",
}


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether an observable is accepted by DefaultQubitMixed."""
    if obs.name in {"Prod", "Sum"}:
        return all(observable_stopping_condition(observable) for observable in obs.operands)
    if obs.name == "LinearCombination":
        return all(observable_stopping_condition(observable) for observable in obs.terms()[1])
    if obs.name == "SProd":
        return observable_stopping_condition(obs.base)

    return obs.name in observables


def stopping_condition(op: qml.operation.Operator) -> bool:
    """Specify whether an Operator object is supported by the device."""
    expected_set = operations | {"Snapshot"} | channels
    return op.name in expected_set


@qml.transform
def warn_readout_error_state(
    tape: qml.tape.QuantumTape,
) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
    """If a measurement in the QNode is an analytic state or density_matrix, warn that readout error will not be applied.

    Args:
        tape (QuantumTape, .QNode, Callable): a quantum circuit.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:
        The unaltered input circuit.
    """
    if not tape.shots:
        for m in tape.measurements:
            if isinstance(m, qml.measurements.StateMP):
                warnings.warn(f"Measurement {m} is not affected by readout error.")

    return (tape,), null_postprocessing


@simulator_tracking
@single_tape_support
class DefaultMixed(Device):
    r"""A PennyLane Python-based device for mixed-state qubit simulation.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['auxiliary', 'q1', 'q2']``).
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator, jax.random.PRNGKey]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
            If a ``jax.random.PRNGKey`` is passed as the seed, a JAX-specific sampling function using
            ``jax.random.choice`` and the ``PRNGKey`` will be used for sampling rather than
            ``numpy.random.default_rng``.
        r_dtype (numpy.dtype): Real datatype to use for computations. Default is np.float64.
        c_dtype (numpy.dtype): Complex datatype to use for computations. Default is np.complex128.
        readout_prob (float): Probability of readout error for qubit measurements. Must be in :math:`[0,1]`.
    """

    _device_options = ("rng", "prng_key")  # tuple of string names for all the device options.

    @property
    def name(self):
        """The name of the device."""
        return "default.mixed"

    @debug_logger_init
    def __init__(
        self,
        wires=None,
        shots=None,
        seed="global",
        readout_prob=None,
    ) -> None:

        if isinstance(wires, int) and wires > 23:
            raise ValueError(
                "This device does not currently support computations on more than 23 wires"
            )

        self.readout_err = readout_prob
        # Check that the readout error probability, if entered, is either integer or float in [0,1]
        if self.readout_err is not None:
            if not isinstance(self.readout_err, float) and not isinstance(self.readout_err, int):
                raise TypeError(
                    "The readout error probability should be an integer or a floating-point number in [0,1]."
                )
            if self.readout_err < 0 or self.readout_err > 1:
                raise ValueError("The readout error probability should be in the range [0,1].")
        super().__init__(wires=wires, shots=shots)

        # Seed setting
        seed = qml.math.random.randint(0, high=10000000) if seed == "global" else seed
        if qml.math.get_interface(seed) == "jax":
            self._prng_key = seed
            self._rng = qml.math.random.default_rng(None)
        else:
            self._prng_key = None
            self._rng = qml.math.random.default_rng(seed)

        self._debugger = None

    @debug_logger
    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQubitMixed`` supports backpropagation derivatives with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information.

        """
        if execution_config is None or execution_config.gradient_method in {"backprop", "best"}:
            return circuit is None or not circuit.shots
        return False

    @debug_logger
    def execute(
        self,
        circuits: QuantumScript,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        if execution_config is None:
            execution_config = ExecutionConfig()
        return tuple(
            simulate(
                c,
                rng=self._rng,
                prng_key=self._prng_key,
                debugger=self._debugger,
                interface=execution_config.interface,
                readout_errors=self.readout_err,
            )
            for c in circuits
        )

    def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
        """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig): an unprocessed execution config.

        Returns:
            ExecutionConfig: a preprocessed execution config.
        """
        updated_values = {}

        # Add gradient related
        if execution_config.gradient_method == "best":
            updated_values["gradient_method"] = "backprop"
        updated_values["use_device_gradient"] = execution_config.gradient_method in {
            "backprop",
            "best",
        }
        updated_values["grad_on_execution"] = False
        updated_values["interface"] = Interface(execution_config.interface)

        # Add device options
        updated_values["device_options"] = dict(execution_config.device_options)  # copy

        for option in execution_config.device_options:
            if option not in self._device_options:
                raise DeviceError(f"device option {option} not present on {self}")

        for option in self._device_options:
            if option not in updated_values["device_options"]:
                updated_values["device_options"][option] = getattr(self, f"_{option}")
        return replace(execution_config, **updated_values)

    @debug_logger
    def preprocess(
        self,
        execution_config: ExecutionConfig = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device
        configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure
                describing the parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns
            ``QuantumTape`` objects that the device can natively execute, as well as a postprocessing
            function to be called after execution, and a configuration with unset
            specifications filled in.

        This device:

        * Supports any qubit operations that provide a matrix
        * Supports any qubit channel that provides Kraus matrices

        """
        execution_config = execution_config or ExecutionConfig()
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        # Defer first since it addes wires to the device
        transform_program.add_transform(qml.defer_measurements, allow_postselect=False)
        transform_program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            name=self.name,
        )

        # TODO: If the setup_execution_config method becomes circuit-dependent in the future,
        # we should handle this case directly within setup_execution_config. This would
        # eliminate the need for the no_sampling transform in this section.
        if config.gradient_method == "backprop":
            transform_program.add_transform(no_sampling, name="backprop + default.mixed")

        if self.readout_err is not None:
            transform_program.add_transform(warn_readout_error_state)

        # Add the validate section
        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(
            validate_measurements,
            analytic_measurements=qml.devices.default_qubit.accepted_analytic_measurement,
            sample_measurements=qml.devices.default_qubit.accepted_sample_measurement,
            name=self.name,
        )
        transform_program.add_transform(
            validate_observables, stopping_condition=observable_stopping_condition, name=self.name
        )

        return transform_program, config
