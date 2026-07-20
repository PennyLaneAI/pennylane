# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the ``alias_sampling_thc`` quantum function, used as the coefficient
oracle (``PREPARE``) in tensor hypercontraction (THC) qubitization."""

import math

import numpy as np

import pennylane as qml
from pennylane.labs.templates import LeftQuantumComparator
from pennylane.templates.subroutines.arithmetic.out_square import OutSquare
from pennylane.wires import Wires


def _build_alias_tables(M, N, zeta, t_ell):
    r"""Build the classical Walker alias tables for the THC coefficients.

    The valid index set is

    .. math::

        \mathcal{S} = \{(\mu, \nu) : \mu \le \nu < M\} \cup \{(\mu, M) : \mu < N/2\},

    of size :math:`d = N/2 + M(M+1)/2`. Each entry is assigned the weight
    :math:`\zeta_{\mu\nu}` (halved on the diagonal :math:`\mu = \nu`) for the two-body
    block, and :math:`t_\ell` for the one-body block (the sentinel column
    :math:`\nu = M`). Walker's method turns the resulting target distribution into,
    for every index ``s``:

    * ``keep_prob[s]``: probability of keeping the original ``(mu, nu)``,
    * ``(mu_alt, nu_alt)``: the alternate pair used otherwise,
    * ``sign`` / ``alt_sign``: sign bits of the original and alternate weights.

    Args:
        M (int): the THC rank.
        N (int): the number of spin orbitals.
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``.
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``.

    Returns:
        list[dict]: one entry per valid pair, sorted lexicographically by ``(mu, nu)``.
    """
    n_half = N // 2
    d = n_half + M * (M + 1) // 2

    weights = {}
    # Two-body block: mu <= nu, both in [0, M - 1] (0-indexed).
    for nu in range(M):
        for mu in range(nu + 1):
            w = zeta[mu, nu]
            if mu == nu:
                w = w / 2.0
            weights[(mu, nu)] = w
    # One-body block: sentinel column nu = M, mu in [0, N/2 - 1].
    for ell in range(n_half):
        weights[(ell, M)] = t_ell[ell]

    entries = sorted(weights.keys())
    if len(entries) != d:
        raise ValueError(f"Expected {d} valid pairs, built {len(entries)}.")

    total_w = sum(abs(weights[k]) for k in entries)
    probs = {k: abs(weights[k]) / total_w for k in entries}
    signs = {k: (1 if weights[k] >= 0 else -1) for k in entries}

    # Walker's alias method.
    small, large = [], []
    prob_table, alias_of = {}, {}
    scaled = {k: probs[k] * d for k in entries}
    for k in entries:
        (small if scaled[k] < 1.0 else large).append(k)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob_table[s] = scaled[s]
        alias_of[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        (small if scaled[l] < 1.0 else large).append(l)

    for k in small + large:
        # Kept with probability ~1; the tiny offset avoids rounding
        # int(keep_prob * 2 ** aleph) up to the out-of-range value 2 ** aleph.
        prob_table[k] = 1.0 - 2.0 ** (-30)
        alias_of[k] = k

    table = []
    for key in entries:
        alt = alias_of[key]
        table.append(
            {
                "mu": key[0],
                "nu": key[1],
                "keep_prob": prob_table[key],
                "mu_alt": alt[0],
                "nu_alt": alt[1],
                "sign": signs[key],
                "alt_sign": signs[alt],
                "alt_edge": 1 if alt[1] == M else 0,
            }
        )
    return table


def _build_qrom_data(M, N, zeta, t_ell, num_index_wires, aleph):
    r"""Pack the alias tables into the bitstrings consumed by ``qml.QROM``.

    The QROM is addressed by the contiguous two-body index
    ``s = mu + nu (nu + 1) / 2`` (matching :func:`_first_arithmetic_op`). Each
    row concatenates, in order: ``sign``, ``alt_sign``, ``mu_alt`` (``num_index_wires``
    bits), ``nu_alt`` (``num_index_wires`` bits), the ``aleph``-bit keep threshold,
    and the ``alt_edge`` flag.

    Args:
        M, N, zeta, t_ell: as in :func:`_build_alias_tables`.
        num_index_wires (int): number of wires per index register (``len(mu_wires)``).
        aleph (int): number of bits used for the keep-probability comparison.

    Returns:
        list[list[int]]: the QROM data, one bitstring (list of ints) per address.
    """
    table = _build_alias_tables(M, N, zeta, t_ell)
    data = [[] for _ in range(len(table))]
    for entry in table:
        s = entry["mu"] + (entry["nu"] ** 2 + entry["nu"]) // 2
        row = (
            [(1 - entry["sign"]) // 2]
            + [(1 - entry["alt_sign"]) // 2]
            + [int(b) for b in f"{int(entry['mu_alt']):0{num_index_wires}b}"]
            + [int(b) for b in f"{int(entry['nu_alt']):0{num_index_wires}b}"]
            + [int(b) for b in f"{int(entry['keep_prob'] * 2 ** aleph):0{aleph}b}"]
            + [entry["alt_edge"]]
        )
        data[s] = row
    return data


def _first_arithmetic_op(M, N, mu_wires, nu_wires, work_wires):
    r"""Compute the contiguous address ``s = mu + nu (nu + 1) / 2`` into ``work_wires``.

    Uses ``nu (nu + 1) / 2 = (nu^2 + nu) / 2`` via ``OutSquare`` (``nu^2``) followed by
    ``SemiAdder`` (``+ nu``), a right shift by one bit (division by two, implemented
    with SWAPs), and a final ``SemiAdder`` (``+ mu``). The result lands on the first
    ``n_d`` work wires, which double as the QROM control register.
    """
    n_d = int(np.ceil(np.log2(N // 2 + (M * (M + 1) / 2)))) + 1
    OutSquare(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d], output_wires_zeroed=True)
    qml.SemiAdder(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d])
    for i in reversed(range(n_d - 1)):
        qml.SWAP(wires=[work_wires[i], work_wires[i + 1]])
    qml.SemiAdder(mu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d])


def alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, work_wires, aleph):
    r"""Coefficient oracle for tensor hypercontraction (THC) qubitization via
    coherent alias (Walker) sampling.

    Given the uniform superposition over the valid THC index pairs
    :math:`\mathcal{S}` (as prepared by :class:`~pennylane.labs.templates.SuperpositionTHC`),
    this quantum function reweights the amplitudes to the target distribution set by
    the THC coefficients:

    .. math::

        \frac{1}{\sqrt{d}} \sum_{(\mu, \nu) \in \mathcal{S}} \lvert \mu \rangle \lvert \nu \rangle
        \;\longmapsto\;
        \sum_{(\mu, \nu)} \sqrt{p_{\mu\nu}}\; (-1)^{s_{\mu\nu}} \lvert \mu \rangle \lvert \nu \rangle ,

    where :math:`p_{\mu\nu} \propto \lvert \zeta_{\mu\nu} \rvert` (two-body) or
    :math:`\lvert t_\ell \rvert` (one-body). The construction follows the alias-sampling
    ``PREPARE`` of `Lee et al. (2021), Fig. 3 <https://arxiv.org/abs/2011.03494>`_ and
    the inequality-test primitive of
    `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_.

    The keep-probabilities are represented with ``aleph`` bits, so the prepared
    distribution matches the target up to a discretization error that decreases as
    ``aleph`` grows.

    .. note::

        This is the ``PREPARE`` step *after* the index superposition. Prepare the input
        superposition first (e.g. with :class:`~pennylane.labs.templates.SuperpositionTHC`,
        or a ``qml.StatePrep`` over the valid pairs for testing).

    Args:
        M (int): the THC rank.
        N (int): the number of spin orbitals. Requires ``N / 2 <= M + 1``.
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``.
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``.
        mu_wires (Sequence[int]): the ``n`` wires storing the first THC index
            :math:`\mu`. Requires ``M <= 2 ** n - 1``.
        nu_wires (Sequence[int]): the ``n`` wires storing the second THC index
            :math:`\nu`. Must have the same length as ``mu_wires``.
        work_wires (Sequence[int]): the auxiliary wires. At least
            ``n_d + 2 * n + 3 * aleph + 5`` zeroed work wires are required, where
            ``n = len(mu_wires)`` and ``n_d = ceil(log2(N / 2 + M (M + 1) / 2)) + 1``.
        aleph (int): the number of bits used to encode the keep-probabilities.

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qml
        from pennylane.labs.templates import alias_sampling_thc

        M, N, n, aleph = 2, 2, 2, 6
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) / 2))) + 1
        num_work = n_d + 2 * n + 3 * aleph + 5
        work_wires = list(range(2 * n, 2 * n + num_work))

        # The valid index pairs, uniformly superposed on the input registers.
        pairs = [(mu, nu) for nu in range(M) for mu in range(nu + 1)]
        pairs += [(ell, M) for ell in range(N // 2)]
        vec = np.zeros(2 ** (2 * n))
        for a, b in pairs:
            vec[a * 2 ** n + b] = 1.0

        dev = qml.device("lightning.qubit", wires=2 * n + num_work)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(vec, normalize=True, wires=mu_wires + nu_wires)
            alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, work_wires, aleph)
            return qml.probs(wires=mu_wires + nu_wires)
    """
    mu_wires = list(Wires(mu_wires))
    nu_wires = list(Wires(nu_wires))
    work_wires = list(Wires(work_wires))
    n = len(mu_wires)

    if len(nu_wires) != n:
        raise ValueError(
            f"mu_wires and nu_wires must contain the same number of wires, "
            f"but got {n} and {len(nu_wires)}."
        )
    if N / 2 > M + 1:
        raise ValueError("N / 2 must be less than or equal to M + 1.")
    if M > 2**n - 1:
        raise ValueError(
            f"mu_wires and nu_wires each need at least ceil(log2(M + 1)) wires. "
            f"Got M={M} with {n} wires, which allows M up to {2**n - 1}."
        )
    if aleph < 1:
        raise ValueError(f"aleph must be a positive integer, got {aleph}.")

    n_d = int(np.ceil(np.log2(N // 2 + (M * (M + 1) / 2)))) + 1
    min_work = n_d + 2 * n + 3 * aleph + 5
    if len(work_wires) < min_work:
        raise ValueError(
            f"At least {min_work} work_wires (n_d + 2 * len(mu_wires) + 3 * aleph + 5) "
            f"should be provided, but only {len(work_wires)} were given."
        )

    # Wire layout on the work register (b is the base of the flag block).
    b = n_d + 2 * n + 2 * aleph
    keep_thresh = work_wires[n_d + 2 * n + 2 : n_d + 2 * n + aleph + 2]  # QROM keep prob
    sample_reg = work_wires[n_d + 2 * n + aleph + 2 : n_d + 2 * n + 2 * aleph + 2]
    keep_flag = work_wires[b + 2]  # comparator target: keep original pair
    swap_flag = work_wires[b + 3]  # symmetrization (mu <-> nu) control
    edge_flag = work_wires[b + 4]  # nu register holds sentinel value M
    alt_edge_flag = work_wires[b + 5]  # QROM-loaded alt_edge bit
    cmp_work = work_wires[b + 6 : n_d + 2 * n + 3 * aleph + 5]
    qrom_work = work_wires[b + 6 :]

    # 1. Flag the one-body sentinel column (nu register in state |M>).
    qml.BasisState(M, wires=nu_wires)
    for w in nu_wires:
        qml.X(w)
    qml.ctrl(qml.X(edge_flag), control=nu_wires, work_wires=work_wires[b + 6 :])
    for w in nu_wires:
        qml.X(w)
    qml.BasisState(M, wires=nu_wires)

    # 2. Compute the contiguous QROM address s = mu + nu (nu + 1) / 2.
    _first_arithmetic_op(M, N, mu_wires, nu_wires, work_wires)

    # 3. Load the alias data (signs, alternate indices, keep threshold, alt_edge).
    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
    qml.QROM(
        data,
        control_wires=work_wires[:n_d],
        target_wires=work_wires[n_d : n_d + 2 * n + aleph + 2] + [alt_edge_flag],
        work_wires=qrom_work,
    )

    # 4. Draw a uniform aleph-bit sample and compare it against the keep threshold.
    for w in sample_reg:
        qml.Hadamard(wires=w)

    LeftQuantumComparator(
        keep_thresh, sample_reg, keep_flag, work_wires=cmp_work, comparator="<"
    )

    # 5. Phase the sign of the kept / alternate entries onto the amplitudes.
    qml.CZ([keep_flag, work_wires[n_d + 1]])  # alt_sign, applied when keeping
    qml.X(keep_flag)
    qml.CZ([keep_flag, work_wires[n_d]])  # sign, applied when swapping to the alternate
    qml.X(keep_flag)

    # 6. If we do not keep, swap in the alternate (mu_alt, nu_alt) and alt_edge.
    for i in range(n):
        qml.CSWAP([keep_flag, mu_wires[i], work_wires[n_d + 2 + i]])
    for i in range(n):
        qml.CSWAP([keep_flag, nu_wires[i], work_wires[n_d + 2 + n + i]])
    qml.CSWAP([keep_flag, edge_flag, alt_edge_flag])

    # 7. Uncompute the comparator, leaving the keep decision imprinted on the state.
    qml.adjoint(LeftQuantumComparator)(
        keep_thresh, sample_reg, keep_flag, work_wires=cmp_work, comparator="<"
    )
    qml.H(swap_flag)

    # 8. Symmetrize: on the flagged subspace, swap the mu and nu registers so the
    #    prepared distribution covers both (mu, nu) and (nu, mu).
    qml.X(edge_flag)
    for i in range(n):
        qml.ctrl(
            qml.SWAP([mu_wires[i], nu_wires[i]]),
            control=[swap_flag, edge_flag],
            control_values=[1, 1],
        )
