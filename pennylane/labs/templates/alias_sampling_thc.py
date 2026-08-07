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

import numpy as np

import pennylane as qp
from pennylane.labs.templates import LeftQuantumComparator
from pennylane.templates.subroutines.arithmetic.out_square import OutSquare
from pennylane.wires import Wires


def _build_alias_tables(probs, mu):
    r"""Compute the classical alias-sampling tables ``alt`` and ``keep``.

    O(L) iterative matching (Walker/Vose) for the coherent alias sampling of
    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_. Returns integers
    :math:`\mathrm{alt}_\ell \in [0, L)` and :math:`\mathrm{keep}_\ell \in [0, 2^\mu)`
    satisfying the normalization constraint (Eq. requirekl):

    .. math::

        \frac{\mathrm{keep}_\ell + \sum_{k \,:\, \mathrm{alt}_k = \ell}
        (2^\mu - \mathrm{keep}_k)}{2^\mu L} = \widetilde{\rho}_\ell .

    Args:
        probs (Sequence[float]): non-negative weights (normalized internally).
        mu (int): number of bits for ``keep`` and the ``sigma`` register.

    Returns:
        tuple[list[int], list[int]]: ``(alt, keep)``, each of length ``L``.

    .. note::

        ``keep_l`` holds :math:`\mu` bits (range :math:`[0, 2^\mu - 1]`). Columns
        not touched by the matching loop keep their defaults ``alt_l = l`` and a
        full ``keep``; these are self-aliased, so the ``keep`` value cancels in the
        constraint above and capping at :math:`2^\mu - 1` is exact.
    """
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    L = len(probs)

    total = probs.sum()
    if total <= 0:
        raise ValueError("probs must sum to a positive value")

    n = 2**mu
    scaled = (L * probs / total).astype(float)
    alt = list(range(L))
    keep = [n] * L  # default: self-aliased, full keep (covers leftover columns)

    small_mask = scaled < 1.0
    small = np.where(small_mask)[0].tolist()
    large = np.where(~small_mask)[0].tolist()

    while small and large:
        s = small.pop()
        g = large.pop()
        keep[s] = int(round(scaled[s] * n))
        alt[s] = g
        scaled[g] += scaled[s] - 1.0
        if scaled[g] < 1.0:
            small.append(g)
        else:
            large.append(g)

    keep = np.clip(keep, 0, n - 1).tolist()
    return alt, keep


def _build_thc_pairs(M, N, zeta, t_ell):
    r"""Enumerate the valid THC index pairs and their (signed) weights.

    The valid index set is

    .. math::

        \mathcal{S} = \{(\mu, \nu) : \mu \le \nu < M\} \cup \{(\mu, M) : \mu < N/2\},

    of size :math:`d = N/2 + M(M+1)/2`. Each entry is assigned the weight
    :math:`\zeta_{\mu\nu}` (halved on the diagonal :math:`\mu = \nu`) for the two-body
    block, and :math:`t_\ell` for the one-body block (the sentinel column
    :math:`\nu = M`).

    Args:
        M (int): the THC rank.
        N (int): the number of spin orbitals.
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``.
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``.

    Returns:
        tuple[list[tuple[int, int]], list[float]]: the pairs sorted lexicographically
        by ``(mu, nu)`` and their (signed) weights, aligned index-by-index.
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

    return entries, [weights[k] for k in entries]


def _build_qrom_data(
    M, N, zeta, t_ell, num_index_wires, aleph
):  # pylint: disable=too-many-arguments
    r"""Pack the alias tables into the bitstrings consumed by ``qp.QROM``.

    The QROM is addressed by the contiguous two-body index
    ``s = mu + nu (nu + 1) / 2`` (matching :func:`_first_arithmetic_op`). Each
    row concatenates, in order: ``sign``, ``alt_sign``, ``mu_alt`` (``num_index_wires``
    bits), ``nu_alt`` (``num_index_wires`` bits), the ``aleph``-bit keep threshold,
    and the ``alt_edge`` flag.

    The keep threshold and alternate index are produced by the classical
    :func:`_build_alias_tables` (Walker/Vose, ``mu = aleph`` bits); signs and the
    ``alt_edge`` sentinel are derived from the THC pair enumeration.

    Args:
        M, N, zeta, t_ell: as in :func:`_build_thc_pairs`.
        num_index_wires (int): number of wires per index register (``len(mu_wires)``).
        aleph (int): number of bits used for the keep-probability comparison.

    Returns:
        list[list[int]]: the QROM data, one bitstring (list of ints) per address.
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    probs = [abs(w) for w in weights]
    signs = [1 if w >= 0 else -1 for w in weights]

    # Classical alias matching on the magnitudes; aleph bits for the keep register.
    alt, keep = _build_alias_tables(probs, aleph)

    data = [[] for _ in range(len(entries))]
    for i, (mu, nu) in enumerate(entries):
        s = mu + (nu**2 + nu) // 2
        alt_i = alt[i]
        mu_alt, nu_alt = entries[alt_i]
        row = (
            [(1 - signs[i]) // 2]
            + [(1 - signs[alt_i]) // 2]
            + [int(b) for b in f"{int(mu_alt):0{num_index_wires}b}"]
            + [int(b) for b in f"{int(nu_alt):0{num_index_wires}b}"]
            + [int(b) for b in f"{int(keep[i]):0{aleph}b}"]
            + [1 if nu_alt == M else 0]
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
    qp.SemiAdder(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d])
    for i in reversed(range(n_d - 1)):
        qp.SWAP(wires=[work_wires[i], work_wires[i + 1]])
    qp.SemiAdder(mu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d])


def alias_sampling_thc(  # pylint: disable=too-many-arguments
    M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph
):
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

        This is the ``PREPARE`` step *after* the index superposition. The input
        superposition must be prepared first with
        :class:`~pennylane.labs.templates.SuperpositionTHC`, which also produces the
        one-body sentinel flag (its ``work_wires[3]``, true when :math:`\nu = M`)
        passed here as ``edge_flag``. This routine does not recompute that flag.

    Args:
        M (int): the THC rank.
        N (int): the number of spin orbitals. Requires ``N / 2 <= M + 1``.
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``.
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``.
        mu_wires (Sequence[int]): the ``n`` wires storing the first THC index
            :math:`\mu`. Requires ``M <= 2 ** n - 1``.
        nu_wires (Sequence[int]): the ``n`` wires storing the second THC index
            :math:`\nu`. Must have the same length as ``mu_wires``.
        edge_flag (Sequence[int]): the wire holding the one-body sentinel flag
            (true when the ``nu`` register is in state :math:`\lvert M \rangle`), as
            produced by :class:`~pennylane.labs.templates.SuperpositionTHC`.
        work_wires (Sequence[int]): the auxiliary wires. At least
            ``n_d + 2 * n + 3 * aleph + 4`` zeroed work wires are required, where
            ``n = len(mu_wires)`` and ``n_d = ceil(log2(N / 2 + M (M + 1) / 2)) + 1``.
        aleph (int): the number of bits used to encode the keep-probabilities.

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        from pennylane.labs.templates import SuperpositionTHC, alias_sampling_thc

        M, N, n, aleph = 2, 2, 2, 6
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))

        # SuperpositionTHC prepares the uniform superposition and the one-body flag.
        sup_work = list(range(2 * n, 2 * n + 3 * n + 5))
        edge_flag = sup_work[3]  # nu register in state |M>

        n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) / 2))) + 1
        num_work = n_d + 2 * n + 3 * aleph + 4
        work_wires = list(range(sup_work[-1] + 1, sup_work[-1] + 1 + num_work))

        dev = qp.device("lightning.qubit", wires=work_wires[-1] + 1)

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
            alias_sampling_thc(
                M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph
            )
            return qp.probs(wires=mu_wires + nu_wires)
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
    min_work = n_d + 2 * n + 3 * aleph + 4
    if len(work_wires) < min_work:
        raise ValueError(
            f"At least {min_work} work_wires (n_d + 2 * len(mu_wires) + 3 * aleph + 4) "
            f"should be provided, but only {len(work_wires)} were given."
        )

    # The one-body sentinel flag (nu register in state |M>) is supplied by the
    # input state preparation (e.g. work_wires[3] of SuperpositionTHC); it is not
    # recomputed here.
    edge_flag = Wires(edge_flag)[0]

    # Wire layout on the work register (b is the base of the flag block).
    b = n_d + 2 * n + 2 * aleph
    keep_thresh = work_wires[n_d + 2 * n + 2 : n_d + 2 * n + aleph + 2]  # QROM keep prob
    sample_reg = work_wires[n_d + 2 * n + aleph + 2 : n_d + 2 * n + 2 * aleph + 2]
    keep_flag = work_wires[b + 2]  # comparator target: keep original pair
    swap_flag = work_wires[b + 3]  # symmetrization (mu <-> nu) control
    alt_edge_flag = work_wires[b + 4]  # QROM-loaded alt_edge bit
    cmp_work = work_wires[b + 5 : n_d + 2 * n + 3 * aleph + 4]
    qrom_work = work_wires[b + 5 :]

    # 1. Compute the contiguous QROM address s = mu + nu (nu + 1) / 2.
    _first_arithmetic_op(M, N, mu_wires, nu_wires, work_wires)

    # 2. Load the alias data (signs, alternate indices, keep threshold, alt_edge).
    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
    qp.QROM(
        data,
        control_wires=work_wires[:n_d],
        target_wires=work_wires[n_d : n_d + 2 * n + aleph + 2] + [alt_edge_flag],
        work_wires=qrom_work,
    )

    # 3. Draw a uniform aleph-bit sample and compare it against the keep threshold.
    for w in sample_reg:
        qp.Hadamard(wires=w)

    LeftQuantumComparator(keep_thresh, sample_reg, keep_flag, work_wires=cmp_work, comparator="<")

    # 4. Phase the sign of the kept / alternate entries onto the amplitudes.
    qp.CZ([keep_flag, work_wires[n_d + 1]])  # alt_sign, applied when keeping
    qp.X(keep_flag)
    qp.CZ([keep_flag, work_wires[n_d]])  # sign, applied when swapping to the alternate
    qp.X(keep_flag)

    # 5. If we do not keep, swap in the alternate (mu_alt, nu_alt) and alt_edge.
    for i in range(n):
        qp.CSWAP([keep_flag, mu_wires[i], work_wires[n_d + 2 + i]])
    for i in range(n):
        qp.CSWAP([keep_flag, nu_wires[i], work_wires[n_d + 2 + n + i]])
    qp.CSWAP([keep_flag, edge_flag, alt_edge_flag])

    # 6. Uncompute the comparator, leaving the keep decision imprinted on the state.
    qp.adjoint(LeftQuantumComparator)(
        keep_thresh, sample_reg, keep_flag, work_wires=cmp_work, comparator="<"
    )
    qp.H(swap_flag)

    # 7. Symmetrize: on the flagged subspace, swap the mu and nu registers so the
    #    prepared distribution covers both (mu, nu) and (nu, mu). The one-body block
    #    (edge_flag == 1) is excluded via a zero-control, leaving edge_flag untouched
    #    so it can be uncomputed by the adjoint of the input state preparation.
    for i in range(n):
        qp.ctrl(
            qp.SWAP([mu_wires[i], nu_wires[i]]),
            control=[swap_flag, edge_flag],
            control_values=[1, 0],
        )
