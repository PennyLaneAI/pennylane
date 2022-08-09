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
from string import ascii_letters as ABC

import pennylane.numpy as np
import pennylane as qml
from pennylane.shadows.utils import median_of_means, pauli_expval


class ClassicalShadow:
    r"""Class for classical shadow post-processing

    A ``ClassicalShadow`` is a classical description of a quantum state that is capable of reproducing expectation values of local Pauli observables, see `2002.08953 <https://arxiv.org/abs/2002.08953>`_.
    The idea is to capture :math:`T` (``shots``) local snapshots of the state by performing measurements in random Pauli bases at each qubit.
    The measurement outcomes, denoted ``bitstrings``, as well as the choices of measurement bases, ``recipes``, are recorded in two ``(T, len(wires))`` integer tensors.

    From the :math:`t`-th measurement, we can reconstruct the ``local_snapshots``

    .. math:: \rho^{(t)} = \bigotimes_{i=1}^{n} 3 U^\dagger_i |b_i \rangle \langle b_i | U_i - \mathbb{I},

    where :math:`U_i` is the rotation corresponding to the measurement of qubit :math:`i` at snapshot :math:`t` and :math:`|b_i\rangle = (1 - b_i, b_i)`
    the corresponding computational basis state given the output bit :math:`b_i`.

    From these local snapshots, one can compute expectation values of local Pauli strings :math:`\langle P_p \otimes P_q \otimes P_r \rangle` (:math:`P_i \in \{X, Y, Z\}`
    are Pauli operators).

    The accuracy of the procedure is determined by the number of measurements :math:`T` (``shots``).
    To target an error :math:`\epsilon`, one needs of order :math:`T = \mathcal{O}\left( \log(M) 4^\ell/\epsilon^2 \right)` measurements to determine :math:`M` different,
    :math:`\ell`-local observables.

    One can in principle also reconstruct the global state :math:`\sum_t \rho^{(t)}/T`, though it is not advisable nor practical for larger systems due to the exponential scaling

    Args:
        bitstrings (tensor): recorded measurement outcomes in random Pauli bases.
        recipes (tensor): recorded measurement bases.
    
    .. seealso:: `PennyLane demo on Classical Shadows <https://pennylane.ai/qml/demos/tutorial_classical_shadows.html>`_, :func:`~.pennylane.classical_shadows`
    
    **Example**

    We obtain the ``bitstrings`` and ``recipes`` via :func:`~.pennylane.classical_shadows`:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=range(2), shots=1000)
        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(0)
            qml.CNOT((0,1))
            qml.RX(x, wires=0)
            return classical_shadow(wires=range(2))

        bitstrings, recipes = qnode(0)
        shadow = ClassicalShadow(bitstrings, recipes)
    
    After recording these ``T=1000`` quantum measurements, we can post-process the results to arbitrary local expectation values of Pauli strings.
    For example, we can compute the expectation value of a Pauli string
    
    >>> shadow.expval(qml.PauliX(0) @ qml.PauliX(1), k=1)
    (1.0079999999999998+0j)

    or of a Hamiltonian
    >>> H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliX(0)@qml.PauliX(1)])
    >>> shadow.expval(H, k=1)
    (2.2319999999999998+0j)

    The parameter ``k`` is used to estimate the expectation values via the `median of means` algorithm. The case ``k=1`` corresponds to simply taking the mean
    value over all local snapshots. ``k>1`` corresponds to splitting the ``T`` local snapshots into ``k`` equal parts, and taking the median of their individual means.
    """

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
            pick_snapshots = np.random.choice(
                np.arange(snapshots, dtype=np.int64), size=snapshots, replace=False
            )
            pick_snapshots = qml.math.convert_like(pick_snapshots, bitstrings)
            bitstrings = qml.math.gather(bitstrings, pick_snapshots)
            recipes = qml.math.gather(recipes, pick_snapshots)

        if isinstance(wires, Iterable):
            bitstrings = bitstrings[:, wires]
            recipes = recipes[:, wires]

        T, n = bitstrings.shape

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, u in enumerate(self.observables):
            U[np.where(recipes == i)] = u

        state = (
            qml.math.cast((1 - 2 * bitstrings[:, :, None, None]), np.complex64) * U + np.eye(2)
        ) / 2

        return 3 * state - np.eye(2)[None, None, :, :]

    @staticmethod
    def _obtain_global_snapshots(local_snapshot):
        T, n = local_snapshot.shape[:2]

        transposed_snapshots = np.transpose(local_snapshot, axes=(1, 0, 2, 3))

        old_indices = [f"a{ABC[1 + 2 * i: 3 + 2 * i]}" for i in range(n)]
        new_indices = f"a{ABC[1:2 * n + 1:2]}{ABC[2:2 * n + 1:2]}"

        return np.einsum(f'{",".join(old_indices)}->{new_indices}', *transposed_snapshots).reshape(
            T, 2**n, 2**n
        )

    def global_snapshots(self, wires=None, snapshots=None):
        """Compute the T x 2**n x 2**n global snapshots"""

        local_snapshot = self.local_snapshots(wires, snapshots)

        if local_snapshot.shape[1] > 16:
            warnings.warn(
                "Querying density matrices for n_wires > 16 is not recommended, operation will take a long time",
                UserWarning,
            )

        return self._obtain_global_snapshots(local_snapshot)

    def _convert_to_pauli_words(self, observable):
        """Given an observable, obtain a list of coefficients and Pauli words, the
        sum of which is equal to the observable"""

        num_wires = self.bitstrings.shape[1]
        obs_to_recipe_map = {"PauliX": 0, "PauliY": 1, "PauliZ": 2, "Identity": -1}

        def pauli_list_to_word(obs):
            word = [-1] * num_wires
            for ob in obs:
                if ob.name not in obs_to_recipe_map:
                    raise ValueError("Observable must be a linear combination of Pauli observables")

                word[ob.wires[0]] = obs_to_recipe_map[ob.name]

            return word

        if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity)):
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
        """
        Get the expectation of an observable
        """
        coeffs_and_words = self._convert_to_pauli_words(H)

        expval = 0
        for coeff, word in coeffs_and_words:
            expvals = pauli_expval(self.bitstrings, self.recipes, np.array(word))
            expval += coeff * median_of_means(expvals, k)

        return expval
