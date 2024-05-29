# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains the default.tensor device to perform tensor network simulations of quantum circuits using ``quimb``.
"""
import copy
from dataclasses import replace
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

import pennylane as qml
from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.measurements import ExpectationMP, MeasurementProcess, StateMeasurement, VarianceMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch, TensorLike

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

has_quimb = True
try:
    import quimb.tensor as qtn
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_quimb = False

_operations = frozenset(
    {
        "Identity",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "GlobalPhase",
        "Hadamard",
        "S",
        "T",
        "SX",
        "CNOT",
        "SWAP",
        "ISWAP",
        "PSWAP",
        "SISWAP",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CY",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "CPhase",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
        "BlockEncode",
    }
)
# The set of supported operations.


_observables = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Projector",
        "SparseHamiltonian",
        "Hamiltonian",
        "LinearCombination",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }
)
# The set of supported observables.

_methods = frozenset({"mps"})
# The set of supported methods.


def accepted_methods(method: str) -> bool:
    """A function that determines whether or not a method is supported by ``default.tensor``."""
    return method in _methods


def stopping_condition(op: qml.operation.Operator) -> bool:
    """A function that determines if an operation is supported by ``default.tensor``."""
    return op.name in _operations


def accepted_observables(obs: qml.operation.Operator) -> bool:
    """A function that determines if an observable is supported by ``default.tensor``."""
    return obs.name in _observables


@simulator_tracking
@single_tape_support
class DefaultTensor(Device):
    """A PennyLane device to perform tensor network simulations of quantum circuits using
    `quimb <https://github.com/jcmgray/quimb/>`_.

    This device is designed to simulate large-scale quantum circuits using tensor networks. For small circuits, other devices like ``default.qubit`` may be more suitable.

    The backend uses the ``quimb`` library to perform the tensor network operations, and different methods can be used to simulate the quantum circuit.
    Currently, only the Matrix Product State (MPS) method is supported, based on the ``quimb``'s ``CircuitMPS`` class.

    This device does not currently support finite shots, derivatives, or vector-Jacobian products.
    The currently supported measurement types are expectation values and variances.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['aux_wire', 'q1', 'q2']``).
        method (str): Supported method. Currently, only ``"mps"`` is supported.
        dtype (type): Data type for the tensor representation. Must be one of ``np.complex64`` or ``np.complex128``.
        **kwargs: keyword arguments for the device, passed to the ``quimb`` backend.

    Keyword Args:
        max_bond_dim (int): Maximum bond dimension for the MPS method.
            It corresponds to the maximum number of Schmidt coefficients retained at the end of the SVD algorithm when applying gates. Default is ``None``.
        cutoff (float): Truncation threshold for the Schmidt coefficients in the MPS method. Default is the machine limit for the given tensor data type,
            retrieved with the ``numpy.finfo`` function.
        contract (str): The contraction method for applying gates in the MPS method. It can be either ``auto-mps`` or ``nonlocal``.
            ``nonlocal`` turns each gate into a Matrix Product Operator (MPO) and applies it directly to the MPS,
            while ``auto-mps`` swaps nonlocal qubits in 2-qubit gates to be next to each other before applying the gate,
            then swaps them back. Default is ``auto-mps``.

    **Example:**

    The following code shows how to create a simple short-depth quantum circuit with 100 qubits using the ``default.tensor`` device.
    Depending on the machine, the execution time for this circuit is around 0.3 seconds:

    .. code-block:: python

        import pennylane as qml
        import numpy as np

        num_qubits = 100

        dev = qml.device("default.tensor", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(num_qubits):
            for qubit in range(0, num_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
                qml.X(wires=[qubit])
                qml.Z(wires=[qubit + 1])
            return qml.expval(qml.Z(0))

    >>> circuit(num_qubits)
    tensor(-1., requires_grad=True)


    .. details::
            :title: Usage Details

            We can provide additional keyword arguments to the device to customize the simulation. These are passed to the ``quimb`` backend.

            In the following example, we consider a slightly more complex circuit. We use the ``default.tensor`` device with the MPS method,
            setting the maximum bond dimension to 100 and the cutoff to 1e-16. We set ``"auto-mps"`` as the contraction technique to apply gates.

            .. code-block:: python

                import pennylane as qml
                import numpy as np

                theta = 0.5
                phi = 0.1
                num_qubits = 50
                device_kwargs = {"max_bond_dim": 100, "cutoff": 1e-16, "contract": "auto-mps"}

                dev = qml.device("default.tensor", wires=num_qubits, **device_kwargs)

                @qml.qnode(dev)
                def circuit(theta, phi, num_qubits):
                    for qubit in range(num_qubits - 4):
                        qml.X(wires=qubit)
                        qml.RX(theta, wires=qubit + 1)
                        qml.CNOT(wires=[qubit, qubit + 1])
                        qml.DoubleExcitation(phi, wires=[qubit, qubit + 1, qubit + 3, qubit + 4])
                        qml.CSWAP(wires=[qubit + 1, qubit + 3, qubit + 4])
                        qml.RY(theta, wires=qubit + 1)
                        qml.Toffoli(wires=[qubit + 1, qubit + 3, qubit + 4])
                    return [
                        qml.expval(qml.Z(0)),
                        qml.expval(qml.Hamiltonian([np.pi, np.e], [qml.Z(15) @ qml.Y(25), qml.Hadamard(40)])),
                        qml.var(qml.Y(20)),
                    ]

            >>> circuit(theta, phi, num_qubits)
            [-0.9953099539219951, 0.0036631029671767208, 0.9999999876072984]

            After the first execution, the time to run this circuit for 50 qubits is around 0.5 seconds depending on the machine.
            Increasing the number of qubits to 500 brings the execution time to approximately 15 seconds, and for 1000 qubits to around 50 seconds.

            The time complexity and the accuracy of the results also depend on the chosen keyword arguments for the device, such as the maximum bond dimension.
            The specific structure of the circuit significantly affects how the time complexity and accuracy of the simulation scale with these parameters.
    """

    # pylint: disable=too-many-instance-attributes

    # So far we just consider the options for MPS simulator
    _device_options = (
        "contract",
        "cutoff",
        "dtype",
        "method",
        "max_bond_dim",
    )

    def __init__(
        self,
        wires,
        *,
        method="mps",
        dtype=np.complex128,
        **kwargs,
    ) -> None:

        if wires is None:
            raise TypeError("Wires must be provided for the default.tensor device.")

        if not has_quimb:
            raise ImportError(
                "This feature requires quimb, a library for tensor network manipulations. "
                "It can be installed with:\n\npip install quimb"
            )  # pragma: no cover

        if not accepted_methods(method):
            raise ValueError(
                f"Unsupported method: {method}. The only currently supported method is mps."
            )

        if dtype not in [np.complex64, np.complex128]:
            raise TypeError(
                f"Unsupported type: {dtype}. Supported types are np.complex64 and np.complex128."
            )

        super().__init__(wires=wires, shots=None)

        self._method = method
        self._dtype = dtype

        # options for MPS
        self._max_bond_dim = kwargs.get("max_bond_dim", None)
        self._cutoff = kwargs.get("cutoff", np.finfo(self._dtype).eps)
        self._contract = kwargs.get("contract", "auto-mps")

        device_options = self._setup_execution_config().device_options

        self._init_state_opts = {
            "binary": "0" * (len(self._wires) if self._wires else 1),
            "dtype": self._dtype.__name__,
            "tags": [str(l) for l in self._wires.labels] if self._wires else None,
        }

        self._gate_opts = {
            "parametrize": None,
            "contract": device_options["contract"],
            "cutoff": device_options["cutoff"],
            "max_bond": device_options["max_bond_dim"],
        }

        self._expval_opts = {
            "dtype": self._dtype.__name__,
            "simplify_sequence": "ADCRS",
            "simplify_atol": 0.0,
        }

        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

        for arg in kwargs:
            if arg not in self._device_options:
                raise TypeError(
                    f"Unexpected argument: {arg} during initialization of the default.tensor device."
                )

    @property
    def name(self):
        """The name of the device."""
        return "default.tensor"

    @property
    def method(self):
        """Supported method."""
        return self._method

    @property
    def dtype(self):
        """Tensor complex data type."""
        return self._dtype

    def _reset_state(self) -> None:
        """
        Reset the MPS.

        This method modifies the tensor state of the device.
        """
        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    def _initial_mps(self) -> "qtn.MatrixProductState":
        r"""
        Return an initial state to :math:`\ket{0}`.

        Internally, it uses `quimb`'s `MPS_computational_state` method.

        Returns:
            MatrixProductState: The initial MPS of a circuit.
        """
        return qtn.MPS_computational_state(**self._init_state_opts)

    def _setup_execution_config(
        self, config: Optional[ExecutionConfig] = DefaultExecutionConfig
    ) -> ExecutionConfig:
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        # TODO: add options for gradients next quarter

        updated_values = {}

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        return replace(config, **updated_values, device_options=new_device_options)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns :class:`~.QuantumTape`'s that the
            device can natively execute as well as a postprocessing function to be called after execution, and a configuration
            with unset specifications filled in.

        This device currently:

        * Does not support finite shots.
        * Does not support derivatives.
        * Does not support vector-Jacobian products.
        """

        config = self._setup_execution_config(execution_config)

        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self._wires, name=self.name)
        program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            skip_initial_state_prep=True,
            name=self.name,
        )
        program.add_transform(qml.transforms.broadcast_expand)

        return program, config

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed.
            execution_config (ExecutionConfig): a data structure with additional information required for execution.

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

        results = []
        for circuit in circuits:
            # we need to check if the wires of the circuit are compatible with the wires of the device
            # since the initial tensor state is created with the wires of the device
            if not self.wires.contains_wires(circuit.wires):
                raise AttributeError(
                    f"Circuit has wires {circuit.wires.tolist()}. "
                    f"Tensor on device has wires {self.wires.tolist()}"
                )
            circuit = circuit.map_to_standard_wires()
            results.append(self.simulate(circuit))

        return tuple(results)

    def simulate(self, circuit: QuantumScript) -> Result:
        """Simulate a single quantum script. This function assumes that all operations provide matrices.

        Args:
            circuit (QuantumScript): The single circuit to simulate.

        Returns:
            Tuple[TensorLike]: The results of the simulation.
        """

        self._reset_state()

        for op in circuit.operations:
            self._apply_operation(op)

        if not circuit.shots:
            if len(circuit.measurements) == 1:
                return self.measurement(circuit.measurements[0])
            return tuple(self.measurement(mp) for mp in circuit.measurements)

        raise NotImplementedError  # pragma: no cover

    def _apply_operation(self, op: qml.operation.Operator) -> None:
        """Apply a single operator to the circuit, keeping the state always in a MPS form.

        Internally it uses `quimb`'s `apply_gate` method. This method modifies the tensor state of the device.

        Args:
            op (Operator): The operation to apply.
        """

        self._circuitMPS.apply_gate(op.matrix().astype(self._dtype), *op.wires, **self._gate_opts)

    def measurement(self, measurementprocess: MeasurementProcess) -> TensorLike:
        """Measure the measurement required by the circuit over the MPS.

        Args:
            measurementprocess (MeasurementProcess): measurement to apply to the state.

        Returns:
            TensorLike: the result of the measurement.
        """

        return self._get_measurement_function(measurementprocess)(measurementprocess)

    def _get_measurement_function(
        self, measurementprocess: MeasurementProcess
    ) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
        """Get the appropriate method for performing a measurement.

        Args:
            measurementprocess (MeasurementProcess): measurement process to apply to the state

        Returns:
            Callable: function that returns the measurement result
        """
        if isinstance(measurementprocess, StateMeasurement):
            if isinstance(measurementprocess, ExpectationMP):
                return self.expval

            if isinstance(measurementprocess, VarianceMP):
                return self.var

        raise NotImplementedError

    def expval(self, measurementprocess: MeasurementProcess) -> float:
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the MPS.

        Returns:
            Expectation value of the observable.
        """

        obs = measurementprocess.obs

        result = self._local_expectation(obs.matrix(), tuple(obs.wires))

        return result

    def var(self, measurementprocess: MeasurementProcess) -> float:
        """Variance of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply to the MPS.

        Returns:
            Variance of the observable.
        """

        obs = measurementprocess.obs

        obs_mat = obs.matrix()
        expect_op = self.expval(measurementprocess)
        expect_squar_op = self._local_expectation(obs_mat @ obs_mat.conj().T, tuple(obs.wires))

        return expect_squar_op - np.square(expect_op)

    def _local_expectation(self, matrix, wires) -> float:
        """Compute the local expectation value of a matrix on the MPS.

        Internally, it uses `quimb`'s `local_expectation` method.

        Args:
            matrix (array): the matrix to compute the expectation value of.
            wires (tuple[int]): the wires the matrix acts on.

        Returns:
            Local expectation value of the matrix on the MPS.
        """

        # We need to copy the MPS to avoid modifying the original state
        qc = copy.deepcopy(self._circuitMPS)

        exp_val = qc.local_expectation(
            matrix,
            wires,
            **self._expval_opts,
        )

        return float(np.real(exp_val))

    # pylint: disable=unused-argument
    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[qml.tape.QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.

        """
        return False

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate the Jacobian of either a single or a batch of circuits on the device.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits to calculate derivatives for.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple: The Jacobian for each trainable parameter.
        """
        raise NotImplementedError(
            "The computation of derivatives has yet to be implemented for the default.tensor device."
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Compute the results and Jacobians of circuits at the same time.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuits or batch of circuits.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            tuple: A numeric result of the computation and the gradient.
        """
        raise NotImplementedError(
            "The computation of derivatives has yet to be implemented for the default.tensor device."
        )

    # pylint: disable=unused-argument
    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector-Jacobian product.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information.
        """
        return False

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        r"""The vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, ``cotangents`` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            tensor-like: A numeric result of computing the vector-Jacobian product.
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the default.tensor device."
        )

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Calculate both the results and the vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector-Jacobian product
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the default.tensor device."
        )
