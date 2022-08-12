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


class ClassicalShadow:
    r"""Class for classical shadow post-processing

    A ``ClassicalShadow`` is a classical description of a quantum state that is capable of reproducing expectation values of local Pauli observables, see `2002.08953 <https://arxiv.org/abs/2002.08953>`_.

    The idea is to capture :math:`T` local snapshots (given by the ``shots`` set in the device) of the state by performing measurements in random Pauli bases at each qubit.
    The measurement outcomes, denoted ``bits``, as well as the choices of measurement bases, ``recipes``, are recorded in two ``(T, len(wires))`` integer tensors, respectively.

    From the :math:`t`-th measurement, we can reconstruct the ``local_snapshots`` (see methods)

    .. math:: \rho^{(t)} = \bigotimes_{i=1}^{n} 3 U^\dagger_i |b_i \rangle \langle b_i | U_i - \mathbb{I},

    where :math:`U_i` is the rotation corresponding to the measurement (e.g. :math:`U_i=H` for measurement in :math:`X`) of qubit :math:`i` at snapshot :math:`t` and :math:`|b_i\rangle = (1 - b_i, b_i)`
    the corresponding computational basis state given the output bit :math:`b_i`.

    From these local snapshots, one can compute expectation values of local Pauli strings, where locality refers to the number of non-Identity operators.
    The accuracy of the procedure is determined by the number of measurements :math:`T` (``shots``).
    To target an error :math:`\epsilon`, one needs of order :math:`T = \mathcal{O}\left( \log(M) 4^\ell/\epsilon^2 \right)` measurements to determine :math:`M` different,
    :math:`\ell`-local observables.

    One can in principle also reconstruct the global state :math:`\sum_t \rho^{(t)}/T`, though it is not advisable nor practical for larger systems due to its exponential scaling.

    Args:
        bits (tensor): recorded measurement outcomes in random Pauli bases.
        recipes (tensor): recorded measurement bases.

    .. seealso:: `PennyLane demo on Classical Shadows <https://pennylane.ai/qml/demos/tutorial_classical_shadows.html>`_, :func:`~.pennylane.classical_shadows`

    **Example**

    We obtain the ``bits`` and ``recipes`` via :func:`~.pennylane.classical_shadow` measurement:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=range(2), shots=1000)
        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(0)
            qml.CNOT((0,1))
            qml.RX(x, wires=0)
            return classical_shadow(wires=range(2))

        bits, recipes = qnode(0)
        shadow = ClassicalShadow(bits, recipes)

    After recording these ``T=1000`` quantum measurements, we can post-process the results to arbitrary local expectation values of Pauli strings.
    For example, we can compute the expectation value of a Pauli string

    >>> shadow.expval(qml.PauliX(0) @ qml.PauliX(1), k=1)
    (1.0079999999999998+0j)

    or of a Hamiltonian:

    >>> H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliX(0)@qml.PauliX(1)])
    >>> shadow.expval(H, k=1)
    (2.2319999999999998+0j)

    The parameter ``k`` is used to estimate the expectation values via the `median of means` algorithm (see `2002.08953 <https://arxiv.org/abs/2002.08953>`_). The case ``k=1`` corresponds to simply taking the mean
    value over all local snapshots. ``k>1`` corresponds to splitting the ``T`` local snapshots into ``k`` equal parts, and taking the median of their individual means.
    """

    def __init__(self, bits, recipes):
        self.bits = bits
        self.recipes = recipes

        assert bits.shape == recipes.shape
        self.snapshots = len(bits)

        self.observables = [
            qml.matrix(qml.PauliX(0)),
            qml.matrix(qml.PauliY(0)),
            qml.matrix(qml.PauliZ(0)),
        ]

    def local_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x n x 2 x 2 local snapshots

        For each qubit and each snapshot, compute :math:`3 U_i^\dagger |b_i \rangle \langle b_i| U_i - 1`

        Args:
            wires (Iterable[int]): The wires over which to compute the snapshots. For ``wires=None``(default) all ``n`` qubits are used.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None``(default), all local snapshots are taken. In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.

        Returns:
            tensor: The local snapshots tensor of shape ``(T, n, 2, 2)`` containing the local local density matrices for each snapshot and each qubit.
        """
        bits = self.bits
        recipes = self.recipes

        if snapshots is not None:
            if isinstance(snapshots, int):
                # choose the given number of random snapshots
                pick_snapshots = np.random.choice(
                    np.arange(snapshots, dtype=np.int64), size=snapshots, replace=False
                )
            else:
                # snapshots is an iterable that determines the indices
                pick_snapshots = snapshots

            pick_snapshots = qml.math.convert_like(pick_snapshots, bits)
            bits = qml.math.gather(bits, pick_snapshots)
            recipes = qml.math.gather(recipes, pick_snapshots)

        if isinstance(wires, Iterable):
            bits = bits[:, wires]
            recipes = recipes[:, wires]

        T, n = bits.shape

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, u in enumerate(self.observables):
            U[np.where(recipes == i)] = u

        state = (qml.math.cast((1 - 2 * bits[:, :, None, None]), np.complex64) * U + np.eye(2)) / 2

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
        r"""Compute the T x 2**n x 2**n global snapshots

        .. warning::

            Classical shadows are not intended to reconstruct global quantum states.
            This method requires exponential scaling of measurements for accurate representations. Further, the output scales exponentially in the output dimension,
            and is therefore not practical for larger systems. A warning is raised for systems of sizes ``n>16``.
        
        Args:
            wires (Iterable[int]): The wires over which to compute the snapshots. For ``wires=None``(default) all ``n`` qubits are used.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None``(default), all local snapshots are taken. In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.

        Returns:
            tensor: The global snapshots tensor of shape ``(T, 2**n, 2**n)`` containing the density matrices for each snapshot measurement.

        **Example**

        We can approximately reconstruct a Bell state:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=range(2), shots=1000)
            @qml.qnode(dev)
            def qnode():
                qml.Hadamard(0)
                qml.CNOT((0,1))
                return classical_shadow(wires=range(2))

            bits, recipes = qnode()
            shadow = ClassicalShadow(bits, recipes)
            shadow_state = np.mean(shadow.global_snapshots(), axis=0)

            bell_state = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])

            >>> np.allclose(bell_state, shadow_state, atol=1e-1)
            True

        """

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

        num_wires = self.bits.shape[1]
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
        r"""Compute expectation value of an observable :math:`H`.

        The canonical way of computing expectation values is to simply average the expectation values for each local snapshot, :math:`\langle O \rangle = \sum_t \text{tr}(\rho^{(t)}O) / T`.
        This corresponds to the case ``k=1``. However, it is often desirable for better accuracy to split the ``T`` measurements into ``k`` equal parts to compute the median of means, see `2002.08953 <https://arxiv.org/abs/2002.08953>`_.

        One of the main perks of classical shadows is being able to compute many different expectation values by classically post-processing the same measurements. This is helpful in general as it may help
        save quantum circuit executions.

        Args:
            H (:class:`~.pennylane.Hamiltonian` or :class:`~.pennylane.operation.Tensor`): Observable to compute the expectation value over.
            k (int): Number of equal parts to split the shadow's measurements to compute the median of means. ``k=1`` corresponds to simply taking the mean over all measurements.

        Args:
            H (qml.Observable): Observable to compute the expectation value
            k (int): Split the snapshots into ``k`` equal parts to compute the median of means.

        Returns:
            float: expectation value estimate.

        **Example**

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=range(2), shots=1000)
            @qml.qnode(dev)
            def qnode(x):
                qml.Hadamard(0)
                qml.CNOT((0,1))
                qml.RX(x, wires=0)
                return classical_shadow(wires=range(2))

            bits, recipes = qnode(0)
            shadow = ClassicalShadow(bits, recipes)

        Compute Pauli string observables

        >>> shadow.expval(qml.PauliX(0) @ qml.PauliX(1), k=1)
        (1.0079999999999998+0j)

        or of a Hamiltonian using `the same` measurement results

        >>> H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliX(0)@qml.PauliX(1)])
        >>> shadow.expval(H, k=1)
        (2.2319999999999998+0j)
        """
        coeffs_and_words = self._convert_to_pauli_words(H)

        expval = 0
        for coeff, word in coeffs_and_words:
            expvals = pauli_expval(self.bits, self.recipes, np.array(word))
            expval += coeff * median_of_means(expvals, k)

        return expval


# Util functions
def median_of_means(arr, num_batches):
    r"""
    The median of means of the given array.

    The array is split into the specified number of batches. The mean value
    of each batch is taken, then the median of the mean values is returned.

    Args:
        arr (tensor-like[float]): The 1-D array for which the median of means
            is determined
        num_batches (int): The number of batches to split the array into

    Returns:
        float: The median of means
    """
    means = []
    batch_size = int(np.ceil(arr.shape[0] / num_batches))

    for i in range(num_batches):
        means.append(qml.math.mean(arr[i * batch_size : (i + 1) * batch_size], 0))

    return np.median(means)


def pauli_expval(bits, recipes, word):
    r"""
    The approximate expectation value of a Pauli word given the bits and recipes
    from a classical shadow measurement.

    Args:
        bits (tensor-like[int]): An array with shape ``(T, n)``, where ``T`` is the
            number of snapshots and ``n`` is the number of measured qubits. Each
            entry must be either ``0`` or ``1`` depending on the sample for the
            corresponding snapshot and qubit.
        recipes (tensor-like[int]): An array with shape ``(T, n)``. Each entry
            must be either ``0``, ``1``, or ``2`` depending on the selected Pauli
            measurement for the corresponding snapshot and qubit. ``0`` corresponds
            to PauliX, ``1`` to PauliY, and ``2`` to PauliZ.
        word (tensor-like[int]): An array with shape ``(n,)``. Each entry must be
            either ``0``, ``1``, ``2``, or ``-1`` depending on the Pauli observable
            on each qubit. For example, when ``n=3``, the observable ``PauliY(0) @ PauliX(2)``
            corresponds to the word ``np.array([1 -1 0])``.

    Returns:
        tensor-like[float]: An array with shape ``(T,)`` containing the value
            of the Pauli observable for each snapshot. The expectation can be
            found by averaging across the snapshots.
    """
    T = recipes.shape[0]

    word = qml.math.convert_like(qml.math.cast_like(word, bits), bits)

    # -1 in the word indicates an identity observable on that qubit
    id_mask = word == -1

    # nothing to do if every observable is the identity
    if qml.math.allequal(id_mask, True):
        return np.ones(T)

    # determine snapshots and qubits that match the word
    # indices = recipes == word
    indices = qml.math.equal(recipes, word)
    indices = np.logical_or(indices, np.tile(id_mask, (T, 1)))
    indices = np.all(indices, axis=1)

    non_id_bits = qml.math.where(np.logical_not(id_mask))
    bits = qml.math.T(qml.math.gather(qml.math.T(bits), non_id_bits))

    # this reshape is necessary since the interfaces have different gather behaviours
    bits = qml.math.reshape(bits, (T, np.count_nonzero(np.logical_not(id_mask))))

    bits = qml.math.sum(bits, axis=1) % 2

    return np.where(indices, 1 - 2 * bits, 0) * 3 ** np.count_nonzero(np.logical_not(id_mask))
