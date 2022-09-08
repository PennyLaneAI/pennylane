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

from pennylane.ops.qubit.attributes import diagonal_in_z_basis

from pennylane import QubitDevice  # , DeviceError, QubitStateVector, BasisState, Snapshot
from pennylane import numpy as np
from .._version import __version__

# pylint: disable=unused-argument
class NullQubit(QubitDevice):
    """Null qubit device for PennyLane.

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
        defaultKwargs = {
            "shots": None,
            "analytic": None,
            "r_dtype": np.float64,
            "c_dtype": np.complex128,
        }
        kwargs = {**defaultKwargs, **kwargs}

        self._operation_calls = defaultdict(int)
        super().__init__(
            wires,
            shots=kwargs["shots"],
            r_dtype=kwargs["r_dtype"],
            c_dtype=kwargs["c_dtype"],
            analytic=kwargs["analytic"],
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
            self._apply_operation(None, op)

    def _apply_operation(self, state, operation):
        if operation.__class__.__name__ in self._apply_ops:
            return self._apply_ops[operation.base_name](state, axes=None, inverse=operation.inverse)

        self._operation_calls[operation.name] += 1

        wires = operation.wires
        if operation in diagonal_in_z_basis:
            return self._apply_diagonal_unitary(state, None, wires)
        if len(wires) <= 2:
            # Einsum is faster for small gates
            return self._apply_unitary_einsum(state, None, wires)
        return self._apply_unitary(state, None, wires)

    def _apply_x(self, state, axes, **kwargs):
        self._operation_calls["PauliX"] += 1

    def _apply_y(self, state, axes, **kwargs):
        self._operation_calls["PauliY"] += 1

    def _apply_z(self, state, axes, **kwargs):
        self._operation_calls["PauliZ"] += 1

    def _apply_hadamard(self, state, axes, **kwargs):
        self._operation_calls["Hadamard"] += 1

    def _apply_s(self, state, axes, inverse=False):
        self._operation_calls["S"] += 1

    def _apply_t(self, state, axes, inverse=False):
        self._operation_calls["T"] += 1

    def _apply_sx(self, state, axes, inverse=False):
        self._operation_calls["SX"] += 1

    def _apply_cnot(self, state, axes, **kwargs):
        self._operation_calls["CNOT"] += 1

    def _apply_swap(self, state, axes, **kwargs):
        self._operation_calls["SWAP"] += 1

    def _apply_cz(self, state, axes, **kwargs):
        self._operation_calls["CZ"] += 1

    def _apply_toffoli(self, state, axes, **kwargs):
        self._operation_calls["Toffoli"] += 1

    def _apply_phase(self, state, axes, parameters, inverse=False):
        pass

    def expval(self, observable, shot_range=None, bin_size=None):
        pass

    def var(self, observable, shot_range=None, bin_size=None):
        pass

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            supports_broadcasting=True,
            returns_state=True,
            passthru_devices={
                "tf": "null.qubit.tf",
                "torch": "null.qubit.torch",
                "autograd": "null.qubit.autograd",
                "jax": "null.qubit.jax",
            },
        )
        return capabilities

    @staticmethod
    def _create_basis_state(index):
        return None

    @property
    def state(self):
        return None

    def density_matrix(self, wires):
        return None

    def _apply_state_vector(self, state, device_wires):
        pass

    def _apply_basis_state(self, state, wires):
        pass

    def _apply_unitary(self, state, mat, wires):
        pass

    def _apply_unitary_einsum(self, state, mat, wires):
        pass

    def _apply_diagonal_unitary(self, state, phases, wires):
        pass

    def reset(self):
        pass

    def analytic_probability(self, wires=None):
        pass

    def generate_samples(self):
        """Returns the computational basis samples generated for all wires.
        In the _qubit_device.py, the function calls for analytic_probability for its operations."""
        self.analytic_probability()

    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        pass

    def operation_calls(self):
        """Statistics of operation calls"""
        return self._operation_calls

    def execute(self, circuit, **kwargs):
        self.apply(circuit.operations)

    def batch_execute(self, circuits, **kwargs):
        res = []
        for c in circuits:
            res.append(self.execute(c))
        return res
