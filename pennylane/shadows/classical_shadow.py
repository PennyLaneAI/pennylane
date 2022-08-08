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
"""Classical Shadows base class with processing functions"""
import warnings
from collections.abc import Iterable

import pennylane.numpy as np
import pennylane as qml
from pennylane.shadows.utils import median_of_means, pauli_expval


class ClassicalShadow:
    """TODO: docstring"""

    def __init__(self, bitstrings, recipes):
        self.bitstrings = bitstrings
        self.recipes = recipes

        assert bitstrings.shape == recipes.shape
        self.snapshots = len(bitstrings)

        self.observables = [
            qml.matrix(qml.PauliX(0)),
            qml.matrix(qml.PauliY(0)),
            qml.matrix(qml.PauliZ(0)),
        ]

    def local_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x n x 2 x 2 local snapshots
        i.e. compute :math:`3 U_i^\dagger |b_i \rangle \langle b_i| U_i - 1` for each qubit and each snapshot
        """
        bitstrings = self.bitstrings
        recipes = self.recipes

        if isinstance(snapshots, int):
            pick_snapshots = np.random.rand(self.snapshots, size=snapshots)
            bitstrings = bitstrings[pick_snapshots]
            recipes = recipes[pick_snapshots]

        if isinstance(wires, Iterable):
            bitstrings = bitstrings[:, wires]
            recipes = recipes[:, wires]

        T, n = bitstrings.shape

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, u in enumerate(self.observables):
            U[np.where(recipes == i)] = u

        state = ((1 - 2 * bitstrings[:, :, None, None]) * U + np.eye(2)) / 2

        return 3 * state - np.eye(2)[None, None, :, :]

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

    def _convert_to_pauli_words(self, observable):
        """Given an observable, obtain a list of coefficients and Pauli words, the
        sum of which is equal to the observable"""

        num_wires = self.bitstrings.shape[1]
        obs_to_recipe_map = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}

        def pauli_list_to_word(obs):
            word = [-1] * num_wires
            for ob in obs:
                if ob.name not in obs_to_recipe_map:
                    raise ValueError("Observable must be a combination of Pauli observables")

                word[ob.wires[0]] = obs_to_recipe_map[ob.name]

            return word

        if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
            word = pauli_list_to_word([observable])
            return [(1, word)]

        if isinstance(observable, qml.operation.Tensor):
            word = pauli_list_to_word(observable.obs)
            return [(1, word)]

        # TODO: cases for new operator arithmetic

        if isinstance(observable, qml.Hamiltonian):
            coeffs_and_words = []
            for coeff, op in zip(observable.data, observable.ops):
                coeffs_and_words.extend(
                    [(coeff * c, w) for c, w in self._convert_to_pauli_words(op)]
                )
            return coeffs_and_words

    def expval(self, H, k):
        coeffs_and_words = self._convert_to_pauli_words(H)

        expval = 0
        for coeff, word in coeffs_and_words:
            expvals = pauli_expval(self.bitstrings, self.recipes, np.array(word))
            expval += coeff * median_of_means(expvals, k)

        return expval

    ######### OLDER FUNCTIONS ################
    # These should be deprecated since they use matrix multiplication instead
    # of properties of Pauli observables

    def expval_observable_global_old(self, observable, k):
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

    def _expval_observable_old(self, observable, k):
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

    def expval_old(self, H, k):
        """Compute expectation values of Observables"""
        # TODO: allow for list of Hamiltonians
        if isinstance(H, qml.Hamiltonian):
            return np.sum([self._expval_observable(observable, k) for observable in H.ops])

        if isinstance(H, Iterable):
            return [self._expval_observable(observable, k) for observable in H]

        return self._expval_observable(H, k)
