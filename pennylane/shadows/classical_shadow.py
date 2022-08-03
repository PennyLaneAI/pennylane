# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classical Shadows baseclass with processing functions"""
import warnings
from collections.abc import Iterable
import pennylane.numpy as np
import pennylane as qml


class ClassicalShadow:
    """TODO: docstring"""

    def __init__(self, bitstrings, recipes):
        self.bitstrings = bitstrings
        self.recipes = recipes

        assert len(bitstrings) == len(recipes)
        self.snapshots = len(bitstrings)

        self.unitaries = [
            qml.matrix(qml.Hadamard(0)),
            (
                qml.matrix(qml.Hadamard(0)) @ qml.matrix(qml.PhaseShift(np.pi / 2, wires=0))
            ),  # .conj().T,
            qml.matrix(qml.Identity(0)),
        ]

    def local_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x n x 2 x 2 local snapshots
        i.e. compute :math:`3 U_i^\dagger |b_i \rangle \langle b_i| U_i - 1` for each qubit and each snapshot
        """
        bitstrings = self.bitstrings
        recipes = self.recipes

        if isinstance(snapshots, int):
            pick_snapshots = np.random.rand(self.snapshots, size=(snapshots))
            bitstrings = bitstrings[pick_snapshots]
            recipes = recipes[pick_snapshots]

        if isinstance(wires, Iterable):
            bitstrings = bitstrings[:, wires]
            recipes = recipes[:, wires]

        # unitaries as a class attirbute to allow for custom sets
        unitaries = self.unitaries

        # Computational basis states as density matrices
        zero = np.zeros((2, 2), dtype="complex")
        zero[0, 0] = 1.0
        one = np.zeros((2, 2), dtype="complex")
        one[1, 1] = 1.0

        T, n = bitstrings.shape

        # This vectorized approach is relying on clever broadcasting and might need some rework to make it compatible with all interfaces

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, _ in enumerate(unitaries):
            U[np.where(recipes == i)] = unitaries[i]

        state = np.empty((T, n, 2, 2), dtype="complex")
        state[np.where(bitstrings == 0)] = zero
        state[np.where(bitstrings == 1)] = one

        return 3 * qml.math.transpose(
            qml.math.conj(U), axes=(0, 1, 3, 2)
        ) @ state @ U - qml.math.reshape(np.eye(2), (1, 1, 2, 2))

    def global_snapshots(self, wires=None, snapshots=None):
        """Compute the T x 2**n x 2**n global snapshots"""

        local_snapshot = self.local_snapshots(wires, snapshots)

        if local_snapshot.shape[1] > 16:
            warnings.warn(
                "Querying density matrices for n_wires > 16 is not recommended, operation will take a long time",
                UserWarning,
            )

        global_snapshots = []
        for T_snapshot in local_snapshot:
            tensor_product = T_snapshot[0]
            for n_snapshot in T_snapshot[1:]:
                tensor_product = np.kron(tensor_product, n_snapshot)
            global_snapshots.append(tensor_product)

        return np.array(global_snapshots)

    def expval_observable_global(self, observable, k):
        """redundant method but keep for comparison, very slow because it unnecessarily computes the full density matrix for each snapshot"""
        global_snapshots = self.global_snapshots(wires=observable.wires)

        step = len(global_snapshots) // k
        if isinstance(observable, qml.operation.Observable):
            obs_m = qml.matrix(observable)

        return np.median(
            [
                np.mean(np.tensordot(global_snapshots[i : i + step], obs_m, axes=([1, 2], [0, 1])))
                for i in range(0, len(global_snapshots), step)
            ]
        )

    def _expval_observable(self, observable, k):
        """Compute expectation values of Pauli-string type observables"""
        # TODO: Use clever matching with modulo, i.e. Edward's formula!
        if isinstance(observable, qml.operation.Tensor):
            os = np.asarray([qml.matrix(o) for o in observable.obs])
        else:
            os = np.asarray([qml.matrix(observable)])  # wont work with other interfaces

        # Picking only the wires with non-trivial Pauli strings avoids computing unnecessary tr(rho_i 1)=1
        local_snapshots = self.local_snapshots(wires=observable.wires)

        means = []
        step = self.snapshots // k

        for i in range(0, self.snapshots, step):
            # Compute expectation value of snapshot = sum_t prod_i tr(rho_i^t P_i)
            # res_i^t = tr(rho_i P_i)
            res = np.trace(local_snapshots[i : i + step] @ os, axis1=-1, axis2=-2)
            # res^t = prod_i res_i^t
            res = np.prod(res, axis=-1)
            # res = sum_t res^t / T
            means.append(np.mean(res))
        return np.median(means)

    def expval(self, H, k):
        """Compute expectation values of Observables"""
        # TODO: allow for list of Hamiltonians
        if isinstance(H, qml.Hamiltonian):
            return np.sum([self._expval_observable(observable, k) for observable in H.ops])

        if isinstance(H, Iterable):
            return [self._expval_observable(observable, k) for observable in H]

        return self._expval_observable(H, k)
