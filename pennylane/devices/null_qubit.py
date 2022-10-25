# Copyright 2022 Xanadu Quantum Technologies Inc.

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
The null.qubit device is a no-op device for benchmarking PennyLane's auxiliary functionality outside direct circuit evaluations.
"""
from collections import defaultdict

from pennylane import QubitDevice
from pennylane import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis

from .._version import __version__


# pylint: disable=unused-argument, no-self-use
class NullQubit(QubitDevice):
    """Null qubit device for PennyLane. This device performs no operations involved in numerical calculations.
       Instead the time spent in execution is dominated by support (or setting up) operations, like tape creation etc.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['auxiliary', 'q1', 'q2']``). Default 1 if not specified.
    """

    name = "Null qubit PennyLane plugin"
    short_name = "null.qubit"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

    operations = {
        "Identity",
        "Snapshot",
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "PSWAP",
        "Adjoint(ISWAP)",
        "SISWAP",
        "Adjoint(SISWAP)",
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
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
    }

    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Projector",
        "SparseHamiltonian",
        "Hamiltonian",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }

    def __init__(self, wires, *args, **kwargs):
        defaultKwargs = {"shots": None, "r_dtype": np.float64, "c_dtype": np.complex128}
        kwargs = defaultKwargs | kwargs

        self._operation_calls = defaultdict(int)
        super().__init__(
            wires, shots=kwargs["shots"], r_dtype=kwargs["r_dtype"], c_dtype=kwargs["c_dtype"]
        )
        self._debugger = None

        # Create the initial state. The state will always be None.
        self._state = self._create_basis_state(0)  # pylint: disable=assignment-from-none
        self._pre_rotated_state = self._state

        self._apply_ops = {
            "PauliX": self._apply_x,
            "PauliY": self._apply_y,
            "PauliZ": self._apply_z,
            "Hadamard": self._apply_hadamard,
            "S": self._apply_s,
            "T": self._apply_t,
            "SX": self._apply_sx,
            "CNOT": self._apply_cnot,
            "SWAP": self._apply_swap,
            "CZ": self._apply_cz,
            "Toffoli": self._apply_toffoli,
        }

    # pylint: disable=arguments-differ
    def apply(self, operations, *args, **kwargs):
        for op in operations:
            self._apply_operation(self._state, op)

    def _apply_operation(self, state, operation):
        self._operation_calls[operation.base_name] += 1

        if operation.__class__.__name__ in self._apply_ops:
            return self._apply_ops[operation.base_name](state, axes=None, inverse=operation.inverse)

        wires = operation.wires
        if operation in diagonal_in_z_basis:
            return self._apply_diagonal_unitary(state, None, wires)
        if len(wires) <= 2:
            # Einsum is faster for small gates
            return self._apply_unitary_einsum(state, None, wires)
        return self._apply_unitary(state, None, wires)

    def _apply_x(self, state, axes, **kwargs):
        return [0.0]

    def _apply_y(self, state, axes, **kwargs):
        return [0.0]

    def _apply_z(self, state, axes, **kwargs):
        return [0.0]

    def _apply_hadamard(self, state, axes, **kwargs):
        return [0.0]

    def _apply_s(self, state, axes, inverse=False):
        return [0.0]

    def _apply_t(self, state, axes, inverse=False):
        return [0.0]

    def _apply_sx(self, state, axes, inverse=False):
        return [0.0]

    def _apply_cnot(self, state, axes, **kwargs):
        return [0.0]

    def _apply_swap(self, state, axes, **kwargs):
        return [0.0]

    def _apply_cz(self, state, axes, **kwargs):
        return [0.0]

    def _apply_toffoli(self, state, axes, **kwargs):
        return [0.0]

    def _apply_phase(self, state, axes, parameters, inverse=False):
        return [0.0]

    def expval(self, observable, shot_range=None, bin_size=None):
        return [0.0]

    def var(self, observable, shot_range=None, bin_size=None):
        return [0.0]

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            supports_broadcasting=False,
            returns_state=True,
            passthru_devices={
                "tf": "null.qubit",
                "torch": "null.qubit",
                "autograd": "null.qubit",
                "jax": "null.qubit",
            },
        )
        return capabilities

    @staticmethod
    def _create_basis_state(index):
        return [0.0]

    @property
    def state(self):
        return [0.0]

    def density_matrix(self, wires):
        return [0.0]

    def _apply_state_vector(self, state, device_wires):
        return [0.0]

    def _apply_basis_state(self, state, wires):
        return [0.0]

    def _apply_unitary(self, state, mat, wires):
        return [0.0]

    def _apply_unitary_einsum(self, state, mat, wires):
        return [0.0]

    def _apply_diagonal_unitary(self, state, phases, wires):
        return [0.0]

    def reset(self):
        self._operation_calls = defaultdict(int)

    def analytic_probability(self, wires=None):
        return [0.0]

    def generate_samples(self):
        """Returns the computational basis samples generated for all wires.
        In the _qubit_device.py, the function calls for analytic_probability for its operations."""
        self.analytic_probability()

    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        return [0.0]

    def operation_calls(self):
        """Statistics of operation calls"""
        return self._operation_calls

    def execute(self, circuit, **kwargs):
        self.apply(circuit.operations, rotations=circuit.diagonalizing_gates, **kwargs)

        if self.tracker.active:
            self.tracker.update(executions=1, shots=self._shots)
            self.tracker.record()
        return [0.0]

    def batch_execute(self, circuits, **kwargs):
        res = []
        for c in circuits:
            res.append(self.execute(c))
        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()
        return res

    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        return [0.0]
