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
# pylint: disable=protected-access
import copy
import warnings
from collections.abc import Callable
from dataclasses import replace
from functools import singledispatch
from numbers import Number
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.exceptions import DeviceError, WireError
from pennylane.measurements import (
    ExpectationMP,
    MeasurementProcess,
    StateMeasurement,
    StateMP,
    VarianceMP,
)
from pennylane.operation import Operation, Operator
from pennylane.ops import LinearCombination, Prod, SProd, Sum
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.templates.subroutines.time_evolution.trotter import _recursive_expression
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch, TensorLike

has_quimb = True

warnings.filterwarnings("ignore", message=".*kahypar")

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
        "ECR",
        "BlockEncode",
        "PauliRot",
        "MultiRZ",
        "TrotterProduct",
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
        "LinearCombination",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }
)
# The set of supported observables.

_methods = frozenset({"mps", "tn"})
# The set of supported methods.


# The following sets are used to determine if a gate contraction method is supported by the device.
# These should be updated if `quimb` adds new options or changes the existing ones.

_gate_contract_mps = frozenset({"auto-mps", "swap+split", "nonlocal"})
# The set of supported gate contraction methods for the MPS method.

_gate_contract_tn = frozenset(
    {"auto-split-gate", "split-gate", "reduce-split", "swap-split-gate", "split", True, False}
)
# The set of supported gate contraction methods for the TN method.
_PAULI_MATRICES = {
    "I": qml.Identity(0).matrix(),
    "X": qml.PauliX(0).matrix(),
    "Y": qml.PauliY(0).matrix(),
    "Z": qml.PauliZ(0).matrix(),
}


def accepted_methods(method: str) -> bool:
    """A function that determines whether or not a method is supported by ``default.tensor``."""
    return method in _methods


def stopping_condition(op: qml.operation.Operator) -> bool:
    """A function that determines if an operation is supported by ``default.tensor``."""
    return op.name in _operations


def accepted_observables(obs: qml.operation.Operator) -> bool:
    """A function that determines if an observable is supported by ``default.tensor``."""
    return obs.name in _observables


def _accepted_gate_contract(contract: str, method: str) -> bool:
    """A function that determines if a gate contraction option is supported by the device."""
    if method == "mps":
        return contract in _gate_contract_mps
    if method == "tn":
        return contract in _gate_contract_tn
    return False  # pragma: no cover


def _warn_unused_kwarg_tn(max_bond_dim: None, cutoff: None):
    """A function that warns the user about unused keyword arguments for the TN method."""
    if max_bond_dim is not None:
        warnings.warn("The keyword argument 'max_bond_dim' is not used for the 'tn' method. ")
    if cutoff is not None:
        warnings.warn("The keyword argument 'cutoff' is not used for the 'tn' method. ")


@simulator_tracking
@single_tape_support
class DefaultTensor(Device):
    """A PennyLane device to perform tensor network simulations of quantum circuits using
    `quimb <https://github.com/jcmgray/quimb/>`_.

    This device is designed to simulate large-scale quantum circuits using tensor networks. For small circuits, other devices like ``default.qubit`` may be more suitable.

    The backend uses the ``quimb`` library to perform the tensor network operations, and different methods can be used to simulate the quantum circuit.
    The supported methods are Matrix Product State (MPS) and Tensor Network (TN).

    This device does not currently support finite-shots or differentiation with ``diff_method`` set to ``"backprop"``, ``"adjoint"``, or ``"device"``. `Other differentiation methods <https://docs.pennylane.ai/en/stable/code/qml_gradients.html>`_ such as
    ``parameter-shift`` and ``hadamard_grad`` are compatible with all devices, including ``default.tensor``.
    At present, the supported measurement types are expectation values, variances, and state measurements.
    Finally, ``UserWarnings`` from the ``cotengra`` package may appear when using this device.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (e.g., ``[-1, 0, 2]``) or strings
            (e.g., ``['aux_wire', 'q1', 'q2']``).
        method (str): Supported method. The supported methods are ``"mps"`` (Matrix Product State) and ``"tn"`` (Tensor Network).
        c_dtype (type): Complex data type for the tensor representation. Must be one of ``numpy.complex64`` or ``numpy.complex128``.
        **kwargs: Keyword arguments for the device, passed to the ``quimb`` backend.

    Keyword Args:
        max_bond_dim (int): Maximum bond dimension for the MPS method.
            It corresponds to the maximum number of Schmidt coefficients (singular values) retained at the end of the SVD algorithm when applying gates. Default is ``None`` (i.e. unlimited).
        cutoff (float): Truncation threshold for the Schmidt coefficients in the MPS method. Default is ``None`` (which is equivalent to retaining all coefficients).
        contract (str): The contraction method for applying gates. The possible options depend on the method chosen.
            For the MPS method, the options are ``"auto-mps"``, ``"swap+split"`` and ``"nonlocal"``. For a description of these options, see the
            `quimb's CircuitMPS documentation <https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/index.html#quimb.tensor.CircuitMPS>`_.
            Default is ``"auto-mps"``.
            For the TN method, the options are ``"auto-split-gate"``, ``"split-gate"``, ``"reduce-split"``, ``"swap-split-gate"``, ``"split"``, ``True``, and ``False``.
            For details, see the `quimb's tensor_core documentation <https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_core/index.html#quimb.tensor.tensor_core.tensor_network_gate_inds>`_.
            Default is ``"auto-split-gate"``.
        contraction_optimizer (str): The contraction path optimizer to use for the computation of local expectation values.
            For more information on the optimizer options accepted by ``quimb``, see the
            `quimb's tensor_contract documentation <https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_core/index.html#quimb.tensor.tensor_core.tensor_contract>`_.
            Default is ``"auto-hq"``.
        local_simplify (str): The simplification sequence to apply to the tensor network for computing local expectation values.
            At present, this argument can only be provided when the TN method is used. For a complete list of available simplification options,
            see the `quimb's full_simplify documentation <https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_core/index.html#quimb.tensor.tensor_core.TensorNetwork.full_simplify>`_.
            Default is ``"ADCRS"``.


    **Example:**

    The following code shows how to create a simple short-depth quantum circuit with 100 qubits using the ``default.tensor`` device.
    Depending on the machine, the execution time for this circuit is around 0.3 seconds:

    .. code-block:: python

        import pennylane as qml

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

    We can provide additional keyword arguments to the device to customize the simulation. These are passed to the ``quimb`` backend.

    .. note::

        Be aware that ``quimb`` uses multi-threading with `numba <https://numba.pydata.org/numba-doc/dev/user/threading-layer.html>`_
        as well as for linear algebra operations with
        `numpy.linalg <https://numpy.org/doc/stable/reference/routines.linalg.html#linear-algebra-numpy-linalg>`_. Proper setting of
        the corresponding environment variables (e.g. ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``NUMBA_NUM_THREADS`` etc.)
        depending on your hardware is highly recommended and will have a strong impact on the device's performance.

        To avoid a slowdown in performance for circuits with more than 10 wires, we recommend setting the environment variable relevant
        for your BLAS library backend (e.g. ``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1`` or ``MKL_NUM_THREADS=1``), depending on your
        NumPy package and associated libraries. Alternatively, you can use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to
        limit the threads within your executing script. For optimal performance you can adjust the number of threads to find the best fit
        for your workload.

    .. details::
            :title: Usage with MPS Method

            In the following example, we consider a slightly more complex circuit. We use the ``default.tensor`` device with the MPS method,
            setting the maximum bond dimension to 100 and the cutoff to the machine epsilon.

            We set ``"auto-mps"`` as the contraction technique to apply gates. With this option, ``quimb`` turns 3-qubit gates and 4-qubit gates
            into Matrix Product Operators (MPO) and applies them directly to the MPS. On the other hand, qubits involved in 2-qubit gates may be
            temporarily swapped to adjacent positions before applying the gate and then returned to their original positions.

            .. code-block:: python

                import pennylane as qml
                import numpy as np

                theta = 0.5
                phi = 0.1
                num_qubits = 50
                device_kwargs_mps = {
                    "max_bond_dim": 100,
                    "cutoff": np.finfo(np.complex128).eps,
                    "contract": "auto-mps",
                }

                dev = qml.device("default.tensor", wires=num_qubits, method="mps", **device_kwargs_mps)

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

            After the first execution, the time to run this circuit for 50 qubits is around 0.5 seconds on a standard laptop.
            Increasing the number of qubits to 500 brings the execution time to approximately 15 seconds, and for 1000 qubits to around 50 seconds.

            The time complexity and the accuracy of the results also depend on the chosen keyword arguments for the device, such as the maximum bond dimension.
            The specific structure of the circuit significantly affects how the time complexity and accuracy of the simulation scale with these parameters.

    .. details::
            :title: Usage with TN Method

            We can also simulate quantum circuits using the Tensor Network (TN) method. This can be particularly useful for circuits that build up entanglement.
            The following example shows how to execute a quantum circuit with the TN method and configurable depth using ``default.tensor``.

            We set the contraction technique to ``"auto-split-gate"``. With this option, each gate is lazily added to the tensor network
            and nothing is initially contracted, but the gate is automatically split if this results in a rank reduction.


            .. code-block:: python

                import pennylane as qml

                phi = 0.1
                depth = 10
                num_qubits = 100

                dev = qml.device("default.tensor", method="tn", contract="auto-split-gate")

                @qml.qnode(dev)
                def circuit(phi, depth, num_qubits):
                    for qubit in range(num_qubits):
                        qml.X(wires=qubit)
                    for _ in range(depth):
                        for qubit in range(num_qubits - 1):
                            qml.CNOT(wires=[qubit, qubit + 1])
                        for qubit in range(num_qubits):
                            qml.RX(phi, wires=qubit)
                    for qubit in range(num_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                    return qml.expval(qml.Z(0))

            >>> circuit(phi, depth, num_qubits)
            -0.9511499466743283

            The execution time for this circuit with the above parameters is around 0.8 seconds on a standard laptop.

            The tensor network method can be faster than MPS and state vector methods in some cases.
            As a comparison, the time for the exact calculation (i.e., with ``max_bond_dim = None``) of the same circuit
            using the ``MPS`` method of the ``default.tensor`` device is approximately three orders of magnitude slower.
            Similarly, using the ``default.qubit`` device results in a much slower simulation.
    """

    # pylint: disable=too-many-instance-attributes

    _device_options = (
        "contract",
        "contraction_optimizer",
        "cutoff",
        "c_dtype",
        "local_simplify",
        "max_bond_dim",
        "method",
    )

    def __init__(
        self,
        wires=None,
        method="mps",
        c_dtype=np.complex128,
        **kwargs,
    ) -> None:
        if not has_quimb:
            raise ImportError(
                "This feature requires quimb, a library for tensor network manipulations. "
                "It can be installed with:\n\npip install quimb"
            )  # pragma: no cover

        if not accepted_methods(method):
            raise ValueError(
                f"Unsupported method: {method}. Supported methods are 'mps' (Matrix Product State) and 'tn' (Exact Tensor Network)."
            )

        if c_dtype not in [np.complex64, np.complex128]:
            raise TypeError(
                f"Unsupported type: {c_dtype}. Supported types are numpy.complex64 and numpy.complex128."
            )

        super().__init__(wires=wires, shots=None)

        self._method = method
        self._c_dtype = c_dtype

        # options for MPS
        self._max_bond_dim = kwargs.get("max_bond_dim", None)
        self._cutoff = kwargs.get("cutoff", None)

        # options for TN
        self._local_simplify = kwargs.get("local_simplify", "ADCRS")

        # options for both MPS and TN
        self._contraction_optimizer = kwargs.get("contraction_optimizer", "auto-hq")
        self._contract = None

        if method == "mps":
            self._contract = kwargs.get("contract", "auto-mps")
        elif method == "tn":
            self._contract = kwargs.get("contract", "auto-split-gate")
            _warn_unused_kwarg_tn(self._max_bond_dim, self._cutoff)
        else:
            raise ValueError  # pragma: no cover

        # The `quimb` circuit is a class attribute so that we can implement methods
        # that access it as soon as the device is created before running a circuit.
        self._quimb_circuit = self._initial_quimb_circuit(self.wires)

        shots = kwargs.pop("shots", None)
        if shots is not None:
            raise DeviceError("default.tensor only supports analytic simulations with shots=None.")

        for arg in kwargs:
            if arg not in self._device_options:
                raise TypeError(
                    f"Unexpected argument: {arg} during initialization of the default.tensor device."
                )

    @property
    def name(self) -> str:
        """The name of the device."""
        return "default.tensor"

    @property
    def method(self) -> str:
        """Method used by the device."""
        return self._method

    @property
    def c_dtype(self) -> type:
        """Tensor complex data type."""
        return self._c_dtype

    def _initial_quimb_circuit(
        self, wires: qml.wires.Wires, psi0=None
    ) -> Union["qtn.CircuitMPS", "qtn.Circuit"]:
        """
        Initialize the quimb circuit according to the method chosen.

        Internally, it uses ``quimb``'s ``CircuitMPS`` or ``Circuit`` class.

        Args:
            wires (Wires): The wires to initialize the quimb circuit.

        Returns:
            CircuitMPS or Circuit: The initial quimb instance of a circuit.
        """

        if not _accepted_gate_contract(self._contract, self.method):
            raise ValueError(
                f"Unsupported gate contraction option: '{self._contract}' for '{self.method}' method. "
                "Please refer to the documentation for the supported options."
            )
        if psi0 is None:
            psi0 = self._initial_mps(wires)

        if self.method == "mps":
            return qtn.CircuitMPS(
                psi0=psi0,
                max_bond=self._max_bond_dim,
                gate_contract=self._contract,
                cutoff=self._cutoff,
            )

        if self.method == "tn":
            return qtn.Circuit(
                psi0=psi0.column_reduce(),
                gate_contract=self._contract,
                tags=[str(l) for l in wires.labels] if wires else None,
            )

        raise NotImplementedError  # pragma: no cover

    def _initial_mps(self, wires: qml.wires.Wires, basis_state=None) -> "qtn.MatrixProductState":
        r"""
        Return a MPS object in the :math:`\ket{0}` state.

        Internally, it uses ``quimb``'s ``MPS_computational_state`` method.

        Args:
            wires (Wires): The wires to initialize the MPS.
            basis_state (str, None): prepares the basis state :math:`\ket{n}`, where ``n`` is a
                string of integers from the set :math:`\{0, 1\}`, i.e.,
                if ``n = "010"``, prepares the state :math:`|010\rangle`.

        Returns:
            MatrixProductState: The initial MPS of a circuit.
        """
        if basis_state is None:
            basis_state = "0" * (len(wires) if wires else 1)
        return qtn.MPS_computational_state(
            binary=basis_state,
            dtype=self._c_dtype.__name__,
            tags=[str(l) for l in wires.labels] if wires else None,
        )

    def draw(self, color="auto", **kwargs):
        """
        Draw the current state (wavefunction) associated with the circuit using ``quimb``'s functionality.

        Internally, it uses ``quimb``'s ``draw`` method.

        Args:
            color (str): The color of the tensor network diagram. Default is ``"auto"``.
            **kwargs: Additional keyword arguments for the ``quimb``'s ``draw`` function. For more information, see the
                `quimb's draw documentation <https://quimb.readthedocs.io/en/latest/tensor-drawing.html>`_.

        **Example**

        Here is a minimal example of how to draw the current state of the circuit:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device("default.tensor", method="mps", wires=15)

            dev.draw()

        We can also customize the appearance of the tensor network diagram by passing additional keyword arguments:

        .. code-block:: python

            dev = qml.device("default.tensor", method="tn", contract=False)

            @qml.qnode(dev)
            def circuit(num_qubits):
                for i in range(num_qubits):
                    qml.Hadamard(wires=i)
                for _ in range(1, num_qubits - 1):
                    for i in range(0, num_qubits, 2):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(10):
                        qml.RZ(1.234, wires=i)
                    for i in range(1, num_qubits - 1, 2):
                        qml.CZ(wires=[i, i + 1])
                    for i in range(num_qubits):
                        qml.RX(1.234, wires=i)
                for i in range(num_qubits):
                    qml.Hadamard(wires=i)
                return qml.expval(qml.Z(0))

            num_qubits = 12

            result = circuit(num_qubits)

            dev.draw(color="auto", show_inds=True)
        """

        color = kwargs.pop("color", [f"I{w}" for w in range(len(self._quimb_circuit.psi.tensors))])
        edge_color = kwargs.pop("edge_color", "black")
        show_tags = kwargs.pop("show_tags", False)
        show_inds = kwargs.pop("show_inds", False)

        return self._quimb_circuit.psi.draw(
            color=color,
            edge_color=edge_color,
            show_tags=show_tags,
            show_inds=show_inds,
            **kwargs,
        )

    def _setup_execution_config(self, config: ExecutionConfig) -> ExecutionConfig:
        """
        Update the execution config with choices for how the device should be used and the device options.
        """
        # TODO: add options for gradients next quarter
        updated_values = {}

        new_device_options = dict(config.device_options)
        for option in self._device_options:
            if option not in new_device_options:
                new_device_options[option] = getattr(self, f"_{option}", None)

        if config.mcm_config.mcm_method not in {None, "deferred"}:
            raise DeviceError(
                f"{self.name} only supports the deferred measurement principle, not {config.mcm_config.mcm_method}"
            )

        return replace(config, **updated_values, device_options=new_device_options)

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
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
        if execution_config is None:
            execution_config = ExecutionConfig()

        config = self._setup_execution_config(execution_config)

        program = TransformProgram()

        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(validate_observables, accepted_observables, name=self.name)
        program.add_transform(validate_device_wires, self._wires, name=self.name)
        program.add_transform(qml.defer_measurements, allow_postselect=False)
        program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            skip_initial_state_prep=True,
            name=self.name,
            device_wires=self.wires,
            target_gates=_operations,
        )
        program.add_transform(qml.transforms.broadcast_expand)

        return program, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed.
            execution_config (ExecutionConfig): a data structure with additional information required for execution.

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        results = []
        for circuit in circuits:
            if self.wires is not None and not self.wires.contains_wires(circuit.wires):
                # quimb raises a cryptic error if the circuit has wires that are not in the device,
                # so we raise a more informative error here
                raise WireError(
                    "Mismatch between circuit and device wires. "
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

        # The state is reset every time a new circuit is executed, and number of wires
        # is established at runtime to match the circuit if not provided.
        wires = circuit.wires if self.wires is None else self.wires
        operations = copy.deepcopy(circuit.operations)
        if operations and isinstance(operations[0], qml.BasisState):
            op = operations.pop(0)
            self._quimb_circuit = self._initial_quimb_circuit(
                wires,
                psi0=self._initial_mps(
                    op.wires,
                    basis_state="".join(
                        str(int(b)) for b in op.parameters[0].astype(self._c_dtype)
                    ),
                ),
            )
        elif operations and isinstance(operations[0], qml.StatePrep):
            op = operations.pop(0)
            self._quimb_circuit = self._initial_quimb_circuit(
                wires,
                psi0=qtn.MatrixProductState.from_dense(
                    op.state_vector(wire_order=wires).astype(self._c_dtype)
                ),
            )
        else:
            self._quimb_circuit = self._initial_quimb_circuit(wires)

        for op in operations:
            self._apply_operation(op)

        if not circuit.shots:
            if len(circuit.measurements) == 1:
                return self.measurement(circuit.measurements[0])
            return tuple(self.measurement(mp) for mp in circuit.measurements)

        raise NotImplementedError  # pragma: no cover

    def _apply_operation(self, op: qml.operation.Operator) -> None:
        """Apply a single operator to the circuit.

        Internally it uses ``quimb``'s ``apply_gate`` method. This method modifies the tensor state of the device.

        Args:
            op (Operator): The operation to apply.
        """
        apply_operation_core(op, self)

    def measurement(self, measurementprocess: MeasurementProcess) -> TensorLike:
        """Measure the measurement required by the circuit.

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
            measurementprocess (MeasurementProcess): measurement process to apply to the state.

        Returns:
            Callable: function that returns the measurement result.
        """
        if isinstance(measurementprocess, StateMeasurement):
            if isinstance(measurementprocess, ExpectationMP):
                return self.expval

            if isinstance(measurementprocess, StateMP):
                return self.state

            if isinstance(measurementprocess, VarianceMP):
                return self.var

        raise NotImplementedError(
            f"Measurement process {measurementprocess} currently not supported by default.tensor."
        )

    def expval(self, measurementprocess: MeasurementProcess) -> float:
        """Expectation value of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply.

        Returns:
            Expectation value of the observable.
        """

        obs = measurementprocess.obs
        return expval_core(obs, self)

    def state(self, measurementprocess: MeasurementProcess):  # pylint: disable=unused-argument
        """Returns the state vector."""
        return self._quimb_circuit.psi.to_dense().ravel()

    def var(self, measurementprocess: MeasurementProcess) -> float:
        """Variance of the supplied observable contained in the MeasurementProcess.

        Args:
            measurementprocess (StateMeasurement): measurement to apply.

        Returns:
            Variance of the observable.
        """

        obs = measurementprocess.obs

        obs_mat = qml.matrix(obs)
        expect_op = self.expval(measurementprocess)
        expect_squar_op = self._local_expectation(obs_mat @ obs_mat.conj().T, tuple(obs.wires))

        return expect_squar_op - np.square(expect_op)

    def _local_expectation(self, matrix, wires) -> float:
        """Compute the local expectation value of a matrix.

        Internally, it uses ``quimb``'s ``local_expectation`` method.

        Args:
            matrix (array): the matrix to compute the expectation value of.
            wires (tuple[int]): the wires the matrix acts on.

        Returns:
            Local expectation value of the matrix.
        """

        # We need to copy the quimb circuit since `local_expectation` modifies it.
        # If there is only one measurement and we don't want to keep track of the state
        # after the execution, we could avoid copying the circuit.
        qc = self._quimb_circuit.copy()

        if self.method == "mps":
            exp_val = qc.local_expectation(
                matrix,
                wires,
                dtype=self._c_dtype.__name__,
                optimize=self._contraction_optimizer,
            )
        else:
            exp_val = qc.local_expectation(
                matrix,
                wires,
                dtype=self._c_dtype.__name__,
                optimize=self._contraction_optimizer,
                simplify_sequence=self._local_simplify,
                simplify_atol=0.0,
            )

        return float(np.real(exp_val))

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: qml.tape.QuantumTape | None = None,
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
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
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
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
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

    def supports_vjp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
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
        circuits: QuantumScriptOrBatch,
        cotangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
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
        circuits: QuantumScriptOrBatch,
        cotangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        """Calculate both the results and the vector-Jacobian product used in reverse-mode differentiation.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits to be executed.
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit.
            execution_config (ExecutionConfig): a data structure with all additional information required for execution.

        Returns:
            Tuple, Tuple: the result of executing the scripts and the numeric result of computing the vector-Jacobian product.
        """
        raise NotImplementedError(
            "The computation of vector-Jacobian product has yet to be implemented for the default.tensor device."
        )


@singledispatch
def apply_operation_core(ops: Operation, device):
    """Dispatcher for _apply_operation."""
    if not isinstance(ops, qml.Identity):
        device._quimb_circuit.apply_gate(
            qml.matrix(ops).astype(device._c_dtype), *ops.wires, parametrize=None
        )


@apply_operation_core.register
def apply_operation_core_global_phase(ops: qml.GlobalPhase, device):
    """Dispatcher for _apply_operation."""
    device._quimb_circuit._psi *= qml.math.exp(-1j * ops.data[0])


@apply_operation_core.register
def apply_operation_core_multirz(ops: qml.MultiRZ, device):
    """Dispatcher for _apply_operation."""
    apply_operation_core(qml.PauliRot(ops.parameters[0], "Z" * len(ops.wires), ops.wires), device)


@apply_operation_core.register
def apply_operation_core_paulirot(ops: qml.PauliRot, device):
    """Apply a Pauli rotation operation in the form of a Matrix Product Operator (MPO)."""

    theta = ops.parameters[0]
    pauli_string = ops._hyperparameters["pauli_word"]

    arrays = []
    sites = list(ops.wires)
    for i, P in enumerate(pauli_string):

        if len(sites) == 1:
            # Special case for a single-qubit Pauli rotation
            arr = qml.math.zeros((1, 1, 2, 2), dtype=complex)
            arr[0, 0] = _PAULI_MATRICES[P] * (-1j) * qml.math.sin(theta / 2)
            arr[0, 0] += qml.math.eye(2, dtype=complex) * qml.math.cos(theta / 2)

        # Multi-qubit Pauli rotations are implemented with an MPO chain. Each tensor
        # in this chain has the shape of (in_dim, out_dim, 2, 2), where the last two
        # dimensions are the physical dimensions, i.e., the dimensions of the operator
        # acting on a single site.
        elif i == 0:
            # The first tensor has an in-dimension of 1, and an out-dimension of 2.
            arr = qml.math.zeros((1, 2, 2, 2), dtype=complex)
            arr[0, 0] = _PAULI_MATRICES[P]
            arr[0, 1] = qml.math.eye(2, dtype=complex)

        elif i == len(sites) - 1:
            # The last tensor has an out-dimension of 1, and an in-dimension of 2.
            arr = qml.math.zeros((2, 1, 2, 2), dtype=complex)
            arr[0, 0] = _PAULI_MATRICES[P] * (-1j) * qml.math.sin(theta / 2)
            arr[1, 0] = qml.math.eye(2, dtype=complex) * qml.math.cos(theta / 2)

        else:
            # The middle tensors maintain connectivity with the previous and next tensors.
            arr = qml.math.zeros((2, 2, 2, 2), dtype=complex)
            arr[0, 0] = _PAULI_MATRICES[P]
            arr[1, 1] = qml.math.eye(2, dtype=complex)

        arrays.append(arr)

    mpo = qtn.MatrixProductOperator(arrays=arrays, sites=sites)
    mpo = mpo.fill_empty_sites()
    device._quimb_circuit._psi = mpo.apply(
        device._quimb_circuit.psi,
        max_bond=device._max_bond_dim,
        cutoff=device._cutoff,
    )


@apply_operation_core.register
def apply_operation_core_trotter_product(ops: qml.TrotterProduct, device):
    """Dispatcher for _apply_operation."""
    time = ops.data[-1]
    n = ops._hyperparameters["n"]
    order = ops._hyperparameters["order"]
    ops = ops._hyperparameters["base"].operands
    decomp = _recursive_expression(time / n, order, ops)[::-1] * n
    for o in decomp:
        device._quimb_circuit.apply_gate(
            qml.matrix(o).astype(device._c_dtype), *o.wires, parametrize=None
        )


@singledispatch
def expval_core(obs: Operator, device) -> float:
    """Dispatcher for expval."""
    return device._local_expectation(qml.matrix(obs), tuple(obs.wires))


@expval_core.register
def expval_core_prod(obs: Prod, device) -> float:
    """Computes the expval of a Prod."""
    ket = device._quimb_circuit.copy()
    for op in obs:
        ket.apply_gate(qml.matrix(op).astype(device._c_dtype), *op.wires, parametrize=None)
    return np.real((device._quimb_circuit.psi.H & ket.psi).contract(all, output_inds=()))


@expval_core.register
def expval_core_sprod(obs: SProd, device) -> float:
    """Computes the expval of a SProd."""
    return obs.scalar * expval_core(obs.base, device)


@expval_core.register
def expval_core_sum(obs: Sum, device) -> float:
    """Computes the expval of a Sum."""
    return sum(expval_core(m, device) for m in obs)


@expval_core.register
def expval_core_linear_combination(obs: LinearCombination, device) -> float:
    """Computes the expval of a LinearCombination."""
    return sum(expval_core(m, device) for m in obs)
