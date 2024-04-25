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
# pylint: disable = too-many-arguments
import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC

import numpy as np
import pennylane as qml


class ClassicalShadow:
    r"""Class for classical shadow post-processing expectation values, approximate states, and entropies.

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

    .. note:: As per `arXiv:2103.07510 <https://arxiv.org/abs/2103.07510>`_, when computing multiple expectation values it is advisable to directly estimate the desired observables by simultaneously measuring
        qubit-wise-commuting terms. One way of doing this in PennyLane is via :class:`~pennylane.Hamiltonian` and setting ``grouping_type="qwc"``. For more details on this topic, see our demo
        on `estimating expectation values with classical shadows <https://pennylane.ai/qml/demos/tutorial_diffable_shadows.html>`_.

    Args:
        bits (tensor): recorded measurement outcomes in random Pauli bases.
        recipes (tensor): recorded measurement bases.
        wire_map (list[int]): list of the measured wires in the order that
            they appear in the columns of ``bits`` and ``recipes``. If None, defaults
            to ``range(n)``, where ``n`` is the number of measured wires.

    .. seealso:: Demo on `Estimating observables with classical shadows in the Pauli basis <https://pennylane.ai/qml/demos/tutorial_diffable_shadows.html>`_, :func:`~.pennylane.classical_shadow`

    **Example**

    We obtain the ``bits`` and ``recipes`` via :func:`~.pennylane.classical_shadow` measurement:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=range(2), shots=1000)
        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(0)
            qml.CNOT((0,1))
            qml.RX(x, wires=0)
            return qml.classical_shadow(wires=range(2))

        bits, recipes = qnode(0)
        shadow = qml.ClassicalShadow(bits, recipes)

    After recording these ``T=1000`` quantum measurements, we can post-process the results to arbitrary local expectation values of Pauli strings.
    For example, we can compute the expectation value of a Pauli string

    >>> shadow.expval(qml.X(0) @ qml.X(1), k=1)
    array(0.972)

    or of a Hamiltonian:

    >>> H = qml.Hamiltonian([1., 1.], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.X(1)])
    >>> shadow.expval(H, k=1)
    array(1.917)

    The parameter ``k`` is used to estimate the expectation values via the `median of means` algorithm (see `2002.08953 <https://arxiv.org/abs/2002.08953>`_). The case ``k=1`` corresponds to simply taking the mean
    value over all local snapshots. ``k>1`` corresponds to splitting the ``T`` local snapshots into ``k`` equal parts, and taking the median of their individual means. For the case of measuring only in the Pauli basis,
    there is no advantage expected from setting ``k>1``.
    """

    def __init__(self, bits, recipes, wire_map=None):
        self.bits = bits
        self.recipes = recipes

        # the wires corresponding to the columns of bitstrings
        if wire_map is None:
            self.wire_map = list(range(bits.shape[1]))
        else:
            self.wire_map = wire_map

        if bits.shape != recipes.shape:
            raise ValueError(
                f"Bits and recipes but have the same shape, got {bits.shape} and {recipes.shape}."
            )

        if bits.shape[1] != len(self.wire_map):
            raise ValueError(
                f"The 1st axis of bits must have the same size as wire_map, got {bits.shape[1]} and {len(self.wire_map)}."
            )

        self.observables = [
            qml.matrix(qml.X(0)),
            qml.matrix(qml.Y(0)),
            qml.matrix(qml.Z(0)),
        ]

    @property
    def snapshots(self):
        """
        The number of snapshots in the classical shadow measurement.
        """
        return len(self.bits)

    def local_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x n x 2 x 2 local snapshots

        For each qubit and each snapshot, compute :math:`3 U_i^\dagger |b_i \rangle \langle b_i| U_i - 1`

        Args:
            wires (Iterable[int]): The wires over which to compute the snapshots. For ``wires=None`` (default) all ``n`` qubits are used.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None`` (default), all local snapshots are taken.
                In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.

        Returns:
            tensor: The local snapshots tensor of shape ``(T, n, 2, 2)`` containing the local local density matrices for each snapshot and each qubit.
        """
        if snapshots is not None:
            if isinstance(snapshots, int):
                # choose the given number of random snapshots
                pick_snapshots = np.random.choice(
                    np.arange(snapshots, dtype=np.int64), size=snapshots, replace=False
                )
            else:
                # snapshots is an iterable that determines the indices
                pick_snapshots = snapshots

            pick_snapshots = qml.math.convert_like(pick_snapshots, self.bits)
            bits = qml.math.gather(self.bits, pick_snapshots)
            recipes = qml.math.gather(self.recipes, pick_snapshots)
        else:
            bits = self.bits
            recipes = self.recipes

        if isinstance(wires, Iterable):
            wires = qml.math.convert_like(wires, bits)
            bits = qml.math.T(qml.math.gather(qml.math.T(bits), wires))
            recipes = qml.math.T(qml.math.gather(qml.math.T(recipes), wires))

        T, n = bits.shape

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, u in enumerate(self.observables):
            U[np.where(recipes == i)] = u

        state = (qml.math.cast((1 - 2 * bits[:, :, None, None]), np.complex64) * U + np.eye(2)) / 2

        return 3 * state - np.eye(2)[None, None, :, :]

    def global_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x 2**n x 2**n global snapshots

        .. warning::

            Classical shadows are not intended to reconstruct global quantum states.
            This method requires exponential scaling of measurements for accurate representations. Further, the output scales exponentially in the output dimension,
            and is therefore not practical for larger systems. A warning is raised for systems of sizes ``n>16``.

        Args:
            wires (Iterable[int]): The wires over which to compute the snapshots. For ``wires=None`` (default) all ``n`` qubits are used.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None`` (default), all local snapshots are taken.
                In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.

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

        T, n = local_snapshot.shape[:2]

        transposed_snapshots = np.transpose(local_snapshot, axes=(1, 0, 2, 3))

        old_indices = [f"a{ABC[1 + 2 * i: 3 + 2 * i]}" for i in range(n)]
        new_indices = f"a{ABC[1:2 * n + 1:2]}{ABC[2:2 * n + 1:2]}"

        return np.reshape(
            np.einsum(f'{",".join(old_indices)}->{new_indices}', *transposed_snapshots),
            (T, 2**n, 2**n),
        )

    def _convert_to_pauli_words_with_pauli_rep(self, pr, num_wires):
        """Convert to recipe using pauli representation"""
        pr_to_recipe_map = {"X": 0, "Y": 1, "Z": 2, "I": -1}

        coeffs_and_words = []
        for pw, c in pr.items():
            word = [-1] * num_wires
            for i, s in pw.items():
                word[self.wire_map.index(i)] = pr_to_recipe_map[s]

            coeffs_and_words.append((c, word))

        return coeffs_and_words

    def _convert_to_pauli_words(self, observable):
        """Given an observable, obtain a list of coefficients and Pauli words, the
        sum of which is equal to the observable"""

        num_wires = self.bits.shape[1]

        # Legacy support for old opmath
        obs_to_recipe_map = {"PauliX": 0, "PauliY": 1, "PauliZ": 2, "Identity": -1}

        def pauli_list_to_word(obs):
            word = [-1] * num_wires
            for ob in obs:
                if ob.name not in obs_to_recipe_map:
                    raise ValueError("Observable must be a linear combination of Pauli observables")

                word[self.wire_map.index(ob.wires[0])] = obs_to_recipe_map[ob.name]

            return word

        if isinstance(observable, (qml.X, qml.Y, qml.Z, qml.Identity)):
            word = pauli_list_to_word([observable])
            return [(1, word)]

        if isinstance(observable, qml.operation.Tensor):
            word = pauli_list_to_word(observable.obs)
            return [(1, word)]

        if isinstance(observable, qml.ops.Hamiltonian):
            coeffs_and_words = []
            for coeff, op in zip(observable.data, observable.ops):
                coeffs_and_words.extend(
                    [(coeff * c, w) for c, w in self._convert_to_pauli_words(op)]
                )
            return coeffs_and_words

        # Support for all operators with a valid pauli_rep
        if (pr := observable.pauli_rep) is not None:
            return self._convert_to_pauli_words_with_pauli_rep(pr, num_wires)

        raise ValueError(
            f"Observable must have a valid pauli representation. Received {observable} with observable.pauli_rep = {pr}"
        )

    def expval(self, H, k=1):
        r"""Compute expectation value of an observable :math:`H`.

        The canonical way of computing expectation values is to simply average the expectation values for each local snapshot, :math:`\langle O \rangle = \sum_t \text{tr}(\rho^{(t)}O) / T`.
        This corresponds to the case ``k=1``. In the original work, `2002.08953 <https://arxiv.org/abs/2002.08953>`_, it has been proposed to split the ``T`` measurements into ``k`` equal
        parts to compute the median of means. For the case of Pauli measurements and Pauli observables, there is no advantage expected from setting ``k>1``.

        One of the main perks of classical shadows is being able to compute many different expectation values by classically post-processing the same measurements. This is helpful in general as it may help
        save quantum circuit executions.

        Args:
            H (qml.Observable): Observable to compute the expectation value
            k (int): Number of equal parts to split the shadow's measurements to compute the median of means. ``k=1`` (default) corresponds to simply taking the mean over all measurements.

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
                return qml.classical_shadow(wires=range(2))

            bits, recipes = qnode(0)
            shadow = qml.ClassicalShadow(bits, recipes)

        Compute Pauli string observables

        >>> shadow.expval(qml.X(0) @ qml.X(1), k=1)
        array(1.116)

        or of a Hamiltonian using `the same` measurement results

        >>> H = qml.Hamiltonian([1., 1.], [qml.Z(0) @ qml.Z(1), qml.X(0) @ qml.X(1)])
        >>> shadow.expval(H, k=1)
        array(1.9980000000000002)
        """
        if not isinstance(H, (list, tuple)):
            H = [H]

        coeffs_and_words = [self._convert_to_pauli_words(h) for h in H]
        expvals = pauli_expval(
            self.bits, self.recipes, np.array([word for cw in coeffs_and_words for _, word in cw])
        )
        expvals = median_of_means(expvals, k, axis=0)
        expvals = expvals * np.array([coeff for cw in coeffs_and_words for coeff, _ in cw])

        start = 0
        results = []
        for i in range(len(H)):
            results.append(np.sum(expvals[start : start + len(coeffs_and_words[i])]))
            start += len(coeffs_and_words[i])

        return qml.math.squeeze(results)

    def entropy(self, wires, snapshots=None, alpha=2, k=1, base=None):
        r"""Compute entropies from classical shadow measurements.

        Compute general Renyi entropies of order :math:`\alpha` for a reduced density matrix :math:`\rho` in terms of

        .. math:: S_\alpha(\rho) = \frac{1}{1-\alpha} \log\left(\text{tr}\left[\rho^\alpha \right] \right).

        There are two interesting special cases: In the limit :math:`\alpha \rightarrow 1`, we find the von Neumann entropy

        .. math:: S_{\alpha=1}(\rho) = -\text{tr}(\rho \log(\rho)).

        In the case of :math:`\alpha = 2`, the Renyi entropy becomes the logarithm of the purity of the reduced state

        .. math:: S_{\alpha=2}(\rho) = - \log\left(\text{tr}(\rho^2) \right).

        Since density matrices reconstructed from classical shadows can have negative eigenvalues, we use the algorithm described in
        `1106.5458 <https://arxiv.org/abs/1106.5458>`_ to project the estimator to the closest valid state.

        .. warning::

            Entropies are non-linear functions of the quantum state. Accuracy bounds on entropies with classical shadows are not known exactly,
            but scale exponentially in the subsystem size. It is advisable to only compute entropies for small subsystems of a few qubits.
            Further, entropies as post-processed by this class method are currently not automatically differentiable.

        Args:
            wires (Iterable[int]): The wires over which to compute the entropy of their reduced state. Note that the computation scales exponentially in the
                number of wires for the reduced state.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None`` (default), all local snapshots are taken.
                In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.
            alpha (float): order of the Renyi-entropy. Defaults to ``alpha=2``, which corresponds to the purity of the reduced state. This case is straight forward to compute.
                All other cases ``alpha!=2`` necessitate computing the eigenvalues of the reduced state and thus may lead to longer computations times.
                Another special case is ``alpha=1``, which corresponds to the von Neumann entropy.
            k (int): Allow to split the snapshots into ``k`` equal parts and estimate the snapshots in a median of means fashion. There is no known advantage to do this for entropies.
                Thus, ``k=1`` is default and advised.
            base (float): Base to the logarithm used for the entropies.

        Returns:
            float: Entropy of the chosen subsystem.

        **Example**

        For the maximally entangled state of ``n`` qubits, the reduced state has two constant eigenvalues :math:`\frac{1}{2}`. For constant distributions, all Renyi entropies are
        equivalent:

        .. code-block:: python3

            wires = 4
            dev = qml.device("default.qubit", wires=range(wires), shots=1000)

            @qml.qnode(dev)
            def max_entangled_circuit():
                qml.Hadamard(wires=0)
                for i in range(1, wires):
                    qml.CNOT(wires=[0, i])
                return qml.classical_shadow(wires=range(wires))

            bits, recipes = max_entangled_circuit()
            shadow = qml.ClassicalShadow(bits, recipes)

            entropies = [shadow.entropy(wires=[0], alpha=alpha) for alpha in [1., 2., 3.]]

        >>> np.isclose(entropies, entropies[0], atol=1e-2)
        [ True,  True,  True]

        For non-uniform reduced states that is not the case anymore and the entropy differs for each order ``alpha``:

        .. code-block:: python3

            @qml.qnode(dev)
            def qnode(x):
                for i in range(wires):
                    qml.RY(x[i], wires=i)

                for i in range(wires - 1):
                    qml.CNOT((i, i + 1))

                return qml.classical_shadow(wires=range(wires))

            x = np.linspace(0.5, 1.5, num=wires)
            bitstrings, recipes = qnode(x)
            shadow = qml.ClassicalShadow(bitstrings, recipes)

        >>> [shadow.entropy(wires=wires, alpha=alpha) for alpha in [1., 2., 3.]]
        [1.5419292874423107, 1.1537924276625828, 0.9593638767763727]

        """

        global_snapshots = self.global_snapshots(wires=wires, snapshots=snapshots)
        rdm = median_of_means(global_snapshots, k, axis=0)

        # Allow for different log base
        div = np.log(base) if base else 1

        evs_nonzero = _project_density_matrix_spectrum(rdm)
        if alpha == 1:
            # Special case of von Neumann entropy
            return qml.math.entr(evs_nonzero) / div

        # General Renyi-alpha entropy
        return qml.math.log(qml.math.sum(evs_nonzero**alpha)) / (1.0 - alpha) / div


def _project_density_matrix_spectrum(rdm):
    """Project the estimator density matrix rdm with possibly negative eigenvalues onto the closest true density matrix in L2 norm"""
    # algorithm below eq. (16) in https://arxiv.org/pdf/1106.5458.pdf
    evs = qml.math.eigvalsh(rdm)[::-1]  # order from largest to smallest
    d = len(rdm)
    a = 0.0
    for i in range(d - 1, -1, -1):
        if evs[i] + a / (i + 1) > 0:
            break
        a += evs[i]

    lambdas = evs[: i + 1] + a / (i + 1)
    return lambdas[::-1]


# Util functions
def median_of_means(arr, num_batches, axis=0):
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
    batch_size = int(np.ceil(arr.shape[0] / num_batches))
    means = [
        qml.math.mean(arr[i * batch_size : (i + 1) * batch_size], 0) for i in range(num_batches)
    ]

    return np.median(means, axis=axis)


def pauli_expval(bits, recipes, word):
    r"""
    The approximate expectation value of a Pauli word given the bits and recipes
    from a classical shadow measurement.

    The expectation value can be computed using

    .. math::

        \alpha = \frac{1}{|T_{match}|}\sum_{T_{match}}\left(1 - 2\left(\sum b \text{  mod }2\right)\right)

    where :math:`T_{match}` denotes the snapshots with recipes that match the Pauli word,
    and the right-most sum is taken over all bits in the snapshot where the observable
    in the Pauli word for that bit is not the identity.

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
            on each qubit. For example, when ``n=3``, the observable ``Y(0) @ X(2)``
            corresponds to the word ``np.array([1 -1 0])``.

    Returns:
        tensor-like[float]: An array with shape ``(T,)`` containing the value
        of the Pauli observable for each snapshot. The expectation can be
        found by averaging across the snapshots.
    """
    T, n = recipes.shape
    b = word.shape[0]

    bits = qml.math.cast(bits, np.int64)
    recipes = qml.math.cast(recipes, np.int64)

    word = qml.math.convert_like(qml.math.cast_like(word, bits), bits)

    # -1 in the word indicates an identity observable on that qubit
    id_mask = word == -1

    # determine snapshots and qubits that match the word
    indices = qml.math.equal(
        qml.math.reshape(recipes, (T, 1, n)), qml.math.reshape(word, (1, b, n))
    )
    indices = np.logical_or(indices, qml.math.tile(qml.math.reshape(id_mask, (1, b, n)), (T, 1, 1)))
    indices = qml.math.all(indices, axis=2)

    # mask identity bits (set to 0)
    bits = qml.math.where(id_mask, 0, qml.math.tile(qml.math.expand_dims(bits, 1), (1, b, 1)))

    bits = qml.math.sum(bits, axis=2) % 2

    expvals = qml.math.where(indices, 1 - 2 * bits, 0) * 3 ** np.count_nonzero(
        np.logical_not(id_mask), axis=1
    )
    return qml.math.cast(expvals, np.float64)
