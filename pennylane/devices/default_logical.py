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
The default.qubit device is PennyLane's standard qubit-based device.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.
"""
import itertools
import functools
from string import ascii_letters as ABC

import numpy as np
from scipy.sparse import csr_matrix

import pennylane as qml
from . import DefaultQubit
from pennylane import QubitDevice, QubitStateVector, BasisState, Snapshot
from pennylane.wires import WireError
from .._version import __version__

ABC_ARRAY = np.array(list(ABC))

# tolerance for numerical errors
tolerance = 1e-10


class StabilizerCode:

    mult_map = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 1,
        (1, 1): 0,
        (1, 2): 3,
        (1, 3): 2,
        (2, 0): 2,
        (2, 1): 3,
        (2, 2): 0,
        (2, 3): 1,
        (3, 0): 3,
        (3, 1): 2,
        (3, 2): 1,
        (3, 3): 0
    }

    def __init__(self, generators):
        self.generators = generators

        self.n = len(self.generators[0])  # number of physical qubits
        self.k = self.n - len(self.generators)  # number of logical qubits

        self.stabilizer = self.compute_stabilizer()
        self.normalizer = self.compute_normalizer()

        self.z_gens = None
        self.x_gens = None
        self.zero = None

    @staticmethod
    def pauli_mult(pauli1, pauli2):
        return [StabilizerCode.mult_map[(p1, p2)] for p1, p2 in zip(pauli1, pauli2)]

    @staticmethod
    def op_from_pauli(pauli):
        ops = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]])
        ]

        return functools.reduce(lambda a, b: np.kron(a, ops[b]), pauli, 1)

    @staticmethod
    def do_commute(pauli1, pauli2):
        diff_count = 0
        for p1, p2 in zip(pauli1, pauli2):
            if 0 in (p1, p2):
                continue

            if p1 == p2:
                continue

            diff_count += 1

        return diff_count % 2 == 0

    def compute_stabilizer(self):
        group = []
        for mask in itertools.product(*([[0, 1]] * (self.n - self.k))):
            masked_gens = [g for g, m in zip(self.generators, mask) if m == 1]
            group.append(functools.reduce(self.pauli_mult, masked_gens, [0] * self.n))

        return group

    def compute_normalizer(self):
        normalizer = []
        for pauli in itertools.product(*([[0, 1, 2, 3]] * self.n)):
            for gen in self.generators:
                if not self.do_commute(pauli, gen):
                    break
            else:
                normalizer.append(pauli)

        return normalizer

    def compute_norm_stab_cosets(self):
        cosets = []
        for pauli in self.normalizer:
            for coset in cosets:
                if self.pauli_mult(pauli, coset[0]) in self.stabilizer:
                    coset.append(pauli)
                    break
            else:
                cosets.append([pauli])

        return cosets

    def compute_ops_and_states(self):
        if self.z_gens is None:
            cosets = self.compute_norm_stab_cosets()

            # choose z and x generators
            z_gens = []
            x_gens = []
            for _ in range(self.k):
                for coset in cosets[1:]:
                    # find a coset that commutes with everything so far
                    if coset not in z_gens + x_gens:
                        for other in z_gens + x_gens:
                            if not self.do_commute(coset[0], other):
                                break
                        else:
                            z_gens.append(coset[0])
                            break
                else:
                    raise ValueError("Something went wrong!")

                for coset in cosets[1:]:
                    # find a coset that commutes with everything except the most recent
                    if coset not in z_gens + x_gens and not self.do_commute(coset[0], z_gens[-1]):
                        for other in z_gens[:-1] + x_gens:
                            if not self.do_commute(coset[0], other):
                                break
                        else:
                            x_gens.append(coset[0])
                            break
                else:
                    raise ValueError("Something went wrong!")

            z_gens = self.generators + z_gens

            # determine the simultaneous eigenspace of the z generators
            proj = np.eye(2 ** self.n)
            for gen in z_gens:
                proj = (np.eye(2 ** self.n) + self.op_from_pauli(gen)) @ proj / 2

            zero = proj @ np.ones(2 ** self.n)

            self.z_gens, self.x_gens, self.zero = z_gens[self.n - self.k:], x_gens, zero

        return self.z_gens, self.x_gens, self.zero


# pylint: disable=unused-argument
class DefaultLogical(QubitDevice):
    """Default qubit device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
    """

    name = "Default qubit PennyLane plugin"
    short_name = "default.qubit"
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

    def __init__(
        self, code, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None

        self.code = code

        # right now we only support one encoded set of qubits
        assert len(self.wires) == self.code.k

        # Create the initial state. Internally, we store the
        # state as an array of dimension [2]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    @property
    def stopping_condition(self):
        def accepts_obj(obj):
            if getattr(obj, "has_matrix", False):
                # pow operations dont work with backprop or adjoint without decomposition
                # use class name string so we don't need to use isisntance check
                return not (obj.__class__.__name__ == "Pow" and qml.operation.is_trainable(obj))
            return obj.name in self.observables.union(self.operations)

        return qml.BooleanFn(accepts_obj)

    @functools.lru_cache()
    def map_wires(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        try:
            mapped_wires = [self.wire_map[w] for w in wires]
        except KeyError as e:
            raise WireError(
                f"Did not find some of the wires {wires.labels} on device with wires {self.wires.labels}."
            ) from e

        return mapped_wires

    def define_wire_map(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        consecutive_wires = range(self.num_wires)
        wire_map = zip(wires, consecutive_wires)
        return dict(wire_map)

    # pylint: disable=arguments-differ
    def _get_batch_size(self, tensor, expected_shape, expected_size):
        """Determine whether a tensor has an additional batch dimension for broadcasting,
        compared to an expected_shape."""
        size = self._size(tensor)
        if self._ndim(tensor) > len(expected_shape) or size > expected_size:
            return size // expected_size

        return None

    # pylint: disable=arguments-differ
    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            if isinstance(operation, (QubitStateVector, BasisState)):
                # this is also pretty straightforward to implement
                raise NotImplementedError("Applying basis states currently not supported")
            elif isinstance(operation, Snapshot):
                if self._debugger and self._debugger.active:
                    state_vector = np.array(self._flatten(self._state))
                    if operation.tag:
                        self._debugger.snapshots[operation.tag] = state_vector
                    else:
                        self._debugger.snapshots[len(self._debugger.snapshots)] = state_vector
            else:
                self._state = self._apply_operation(self._state, operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._state = self._apply_operation(self._state, operation)

    def _apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (array[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            array[complex]: output state
        """
        if operation.__class__.__name__ == "Identity":
            return state
        wires = operation.wires

        matrix = self._asarray(qml.matrix(operation), dtype=self.C_DTYPE)
        return self._apply_unitary(state, matrix, wires)

    def _apply_unitary(self, state, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state

        Note: This function does not support simultaneously broadcasted states and matrices yet.
        """
        expanded_mat = qml.math.expand_matrix(mat, wires, self.wires)

        pauli_ops = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]])
        ]

        z_gens, x_gens, _ = self.code.compute_ops_and_states()
        logical_ops = [
            [np.eye(2 ** self.code.n)] * len(self.wires),
            [self.code.op_from_pauli(x) for x in x_gens],
            [1j * (self.code.op_from_pauli(x) @ self.code.op_from_pauli(z)) for x, z in zip(x_gens, z_gens)],
            [self.code.op_from_pauli(z) for z in z_gens]
        ]

        new_state = 0
        for pauli in itertools.product(*([[0, 1, 2, 3]] * len(self.wires))):
            coeff = self._einsum('ij,ji', expanded_mat, self.code.op_from_pauli(pauli)) / (2 ** len(self.wires))

            curr_state = self._state
            for i, p in enumerate(pauli):
                curr_state = logical_ops[p][i] @ curr_state

            new_state = coeff * curr_state + new_state

        return new_state

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            supports_broadcasting=True,
            returns_state=True
        )
        return capabilities

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[2]*self.num_wires``
            representing the statevector of the basis state

        Note: This function does not support broadcasted inputs yet.
        """
        self.code.compute_ops_and_states()
        state = self.code.zero
        x_gens = self.code.compute_ops_and_states()[1]

        mask = ((1 << np.arange(len(self.wires))[::-1]) & index) > 0

        for i, m in enumerate(mask):
            if m:
                state = self.code.op_from_pauli(x_gens[i]) @ state

        return state

    @property
    def state(self):
        return self._state

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        probs = []
        for i in range(2 ** self.num_wires):
            basis_state = self._create_basis_state(i)
            probs.append(self._abs(self._dot(basis_state, self._state)) ** 2)

        return self._stack(probs)
