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

from pennylane.devices import DefaultQubit
from .._version import __version__

# pylint: disable=unused-argument
class NullQubit(DefaultQubit):
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

    def __init__(self, wires, *args, **kwargs):
        defaultKwargs = {"shots": None}
        kwargs = {**defaultKwargs, **kwargs}

        self._operation_calls = defaultdict(int)
        self._shots = kwargs["shots"]
        self._shot_vector = None
        self.custom_expand_fn = None
        super().__init__(wires=wires, shots=self._shots)

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

    def _create_basis_state(self, index):
        pass

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
