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
"""Contains the ``PREPARE`` template for tensor hypercontraction (THC) qubitization."""

import pennylane as qp
from pennylane.labs.templates.alias_sampling import _build_alias_tables
from pennylane.labs.templates.left_quantum_comparator import LeftQuantumComparator
from pennylane.wires import Wires


def _num_index_wires(M):
    r"""Number of wires per index register: exactly ``ceil(log2(M + 1))``."""
    return int(qp.math.ceil_log2(M + 1))


def _num_address_wires(M, N):
    r"""Number of wires holding the contiguous QROM address ``s``.

    One wire more than ``ceil(log2(d))``: the register must transiently hold
    :math:`\nu^2 + \nu \le M(M + 1)`, i.e. up to twice the largest address, before the
    division by two of :func:`_compute_contiguous_register`. The final address
    satisfies :math:`s < d \le 2^{n_d - 1}`, so the leading wire is back to
    :math:`\lvert 0 \rangle` and is *not* used to control the QROM.
    """
    d = N // 2 + M * (M + 1) // 2
    return int(qp.math.ceil_log2(d)) + 1


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
        M (int): the THC rank
        N (int): the number of spin orbitals
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``

    Returns:
        tuple[list[tuple[int, int]], list[float]]: the pairs sorted lexicographically
        by ``(mu, nu)`` and their (signed) weights, aligned index-by-index

    Raises:
        ValueError: if ``zeta`` is not of shape ``(M, M)`` or ``t_ell`` is not of
            shape ``(N // 2,)``
    """
    n_half = N // 2
    d = n_half + M * (M + 1) // 2

    zeta_shape = tuple(qp.math.shape(zeta))
    if zeta_shape != (M, M):
        raise ValueError(f"zeta must be of shape ({M}, {M}), got {zeta_shape}.")
    t_shape = tuple(qp.math.shape(t_ell))
    if t_shape != (n_half,):
        raise ValueError(f"t_ell must be of shape ({n_half},), got {t_shape}.")

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


def _lcu_signs(M, entries, weights):
    r"""Signs of the LCU coefficients that multiply the ``SELECT`` unitaries.

    The stored sign is *not* the sign of the raw weight. ``SELECT`` applies products of
    the reflections :math:`V_\mu = I - 2 c^\dagger_\mu c_\mu`, so the Hamiltonian is
    rewritten with :math:`c^\dagger_\mu c_\mu = (I - V_\mu) / 2`: the two-body block
    carries a *product* of two number operators and the one-body block a single one,

    .. math::

        \zeta_{\mu\nu} n_\mu n_\nu \to +\tfrac{1}{4} \zeta_{\mu\nu} V_\mu V_\nu ,
        \qquad t_\ell n_\ell \to -\tfrac{1}{2} t_\ell V_\ell ,

    so the one-body column (:math:`\nu = M`) enters with an extra minus sign while the
    two-body column does not. Everything else (the identity terms dropped above) only
    shifts the block encoded operator by a multiple of the identity.

    Args:
        M (int): the THC rank
        entries (Sequence[tuple[int, int]]): the THC index pairs
        weights (Sequence[float]): the aligned (signed) weights

    Returns:
        list[int]: one sign bit per entry, ``0`` for a positive LCU coefficient and ``1``
        for a negative one
    """
    # Flip only the one-body column. Negate both branches to block encode ``-H`` instead.
    coeffs = [-w if nu == M else w for w, (_, nu) in zip(weights, entries)]
    return [0 if c >= 0 else 1 for c in coeffs]


def _build_qrom_data(
    M, N, zeta, t_ell, num_index_wires, aleph, include_sign=True
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    r"""Pack the alias tables into the bitstrings consumed by ``qp.QROM``.

    The QROM is addressed by the contiguous two-body index
    ``s = mu + nu (nu + 1) / 2`` (matching :func:`_compute_contiguous_register`). Each
    row concatenates, in order: ``sign`` and ``alt_sign`` (only when ``include_sign``),
    ``mu_alt`` (``num_index_wires`` bits), ``nu_alt`` (``num_index_wires`` bits), the
    ``aleph``-bit keep threshold, and the ``alt_edge`` flag.

    The keep threshold and alternate index are produced by the classical
    ``_build_alias_tables`` of :func:`~pennylane.labs.templates.alias_sampling`
    (Walker/Vose, ``mu = aleph`` bits); the signs (see :func:`_lcu_signs`) and the
    ``alt_edge`` sentinel are derived from the THC pair enumeration.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``
        num_index_wires (int): number of wires per index register (``len(mu_wires)``)
        aleph (int): number of bits used for the keep-probability comparison
        include_sign (bool): whether to emit the ``sign`` and ``alt_sign`` columns

    Returns:
        list[list[int]]: the QROM data, one bitstring (list of ints) per address
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    probs = [abs(w) for w in weights]
    signs = _lcu_signs(M, entries, weights) if include_sign else None

    # Classical alias matching on the magnitudes; aleph bits for the keep register.
    alt, keep = _build_alias_tables(probs, aleph)

    data = [[] for _ in range(len(entries))]
    for i, (mu, nu) in enumerate(entries):
        s = mu + (nu**2 + nu) // 2
        alt_i = alt[i]
        mu_alt, nu_alt = entries[alt_i]
        row = (
            ([signs[i], signs[alt_i]] if include_sign else [])
            + [int(b) for b in f"{int(mu_alt):0{num_index_wires}b}"]
            + [int(b) for b in f"{int(nu_alt):0{num_index_wires}b}"]
            + [int(b) for b in f"{int(keep[i]):0{aleph}b}"]
            + [1 if nu_alt == M else 0]
        )
        data[s] = row
    return data


def _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires):
    r"""Compute the contiguous address ``s = mu + nu (nu + 1) / 2`` into ``work_wires``.

    Uses ``nu (nu + 1) / 2 = (nu^2 + nu) / 2`` via ``OutSquare`` (``nu^2``) followed by
    ``SemiAdder`` (``+ nu``), a right shift by one bit (division by two, implemented
    with SWAPs), and a final ``SemiAdder`` (``+ mu``). The result lands on the first
    ``n_d`` work wires.
    """
    n_d = _num_address_wires(M, N)
    qp.OutSquare(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d], output_wires_zeroed=True)
    qp.SemiAdder(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d])
    for i in reversed(range(n_d - 1)):
        qp.SWAP(wires=[work_wires[i], work_wires[i + 1]])
    qp.SemiAdder(mu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d])


def alias_sampling_thc_wires(M, N, aleph, include_sign=True):
    r"""Return the wire counts required by :func:`alias_sampling_thc`.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals. Requires ``N // 2 <= M + 1``
        aleph (int): the number of bits used to encode the keep-probabilities
        include_sign (bool): whether :func:`alias_sampling_thc` outputs the sign of the
            selected coefficient. Costs two extra work wires

    Returns:
        dict: ``{"mu_wires": n, "nu_wires": n, "superposition_work_wires": 3 * n + 5,
        "work_wires": n_d + 2 * n + 3 * aleph + 2 + 2 * include_sign, "sign_wire": n_d}``,
        where ``n = ceil(log2(M + 1))`` and
        ``n_d = ceil(log2(N // 2 + M (M + 1) // 2)) + 1``

        * ``mu_wires`` / ``nu_wires``: the two index registers, exact
        * ``superposition_work_wires``: the work register of
          :class:`~pennylane.labs.templates.SuperpositionTHC`; its entry at index ``3``
          is the one-body sentinel flag to pass as ``edge_flag``
        * ``work_wires``: the minimum scratch register of :func:`alias_sampling_thc`.
          Additional wires are forwarded to the internal ``qp.QROM``, which uses them
          for a ``SelectSwap`` decomposition that lowers the T-gate count.
        * ``sign_wire``: the *index into* ``work_wires`` of the wire left holding the
          sign bit, i.e. the wire to apply the single ``qp.Z`` to between ``PREPARE``
          and ``SELECT``. ``None`` when ``include_sign=False``.

    **Example**

    >>> from pennylane.labs.templates import alias_sampling_thc_wires
    >>> alias_sampling_thc_wires(M=2, N=2, aleph=6)
    {'mu_wires': 2, 'nu_wires': 2, 'superposition_work_wires': 11, 'work_wires': 29, 'sign_wire': 3}
    >>> alias_sampling_thc_wires(M=2, N=2, aleph=6, include_sign=False)
    {'mu_wires': 2, 'nu_wires': 2, 'superposition_work_wires': 11, 'work_wires': 27, 'sign_wire': None}
    """
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError(f"M must be a positive integer, got {M!r}.")
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError(f"N must be a positive integer, got {N!r}.")
    if isinstance(aleph, bool) or not isinstance(aleph, int) or aleph < 1:
        raise ValueError(f"aleph must be a positive integer, got {aleph!r}.")
    if N // 2 > M + 1:
        raise ValueError("N // 2 must be less than or equal to M + 1.")

    n = _num_index_wires(M)
    n_d = _num_address_wires(M, N)
    return {
        "mu_wires": n,
        "nu_wires": n,
        "superposition_work_wires": 3 * n + 5,
        "work_wires": n_d + 2 * n + 3 * aleph + 2 + 2 * bool(include_sign),
        "sign_wire": n_d if include_sign else None,
    }


def alias_sampling_thc(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    M,
    N,
    zeta,
    t_ell,
    mu_wires,
    nu_wires,
    edge_flag,
    work_wires,
    aleph,
    include_sign=True,
):
    r"""Coefficient oracle for tensor hypercontraction (THC) qubitization via
    coherent alias (Walker) sampling.

    Given the uniform superposition over the valid THC index pairs
    :math:`\mathcal{S}` (as prepared by :class:`~pennylane.labs.templates.SuperpositionTHC`),
    this quantum function reweights the amplitudes to the target distribution set by
    the THC coefficients and symmetrizes the two-body block:

    .. math::

        \frac{1}{\sqrt{d}} \sum_{(\mu, \nu) \in \mathcal{S}}
        \lvert \mu \rangle \lvert \nu \rangle \lvert 0 \rangle \lvert 0 \rangle
        \;\longmapsto\;
        \sum_{\substack{(\mu, \nu) \in \mathcal{S} \\ \nu < M}}
        \sqrt{\frac{p_{\mu\nu}}{2}}
        \Big( \lvert \mu \rangle \lvert \nu \rangle \lvert 0 \rangle
        + \lvert \nu \rangle \lvert \mu \rangle \lvert 1 \rangle \Big)
        \lvert s_{\mu\nu} \rangle
        \;+\; \sum_{\ell < N/2} \sqrt{p_{\ell M}}\;
        \lvert \ell \rangle \lvert M \rangle \lvert + \rangle \lvert s_{\ell M} \rangle ,

    where the third register is the single symmetrization flag and the fourth is the sign
    bit :math:`s` of the selected coefficient (only present when ``include_sign``, see the
    note below). Both are left in this entangled state for the subsequent ``SELECT``, and

    .. math::

        p_{\mu\nu} \propto \begin{cases}
        \lvert \zeta_{\mu\nu} \rvert & \mu < \nu < M \\
        \lvert \zeta_{\mu\mu} \rvert / 2 & \mu = \nu < M \\
        \lvert t_\ell \rvert & \nu = M .
        \end{cases}

    The :math:`1/\sqrt{2}` above is the symmetrization factor: it splits every
    off-diagonal two-body weight evenly between the orderings :math:`(\mu, \nu)` and
    :math:`(\nu, \mu)`, while the diagonal and the one-body column
    (:math:`\nu = M`) are not split. This is exactly why :math:`\zeta_{\mu\mu}` is
    halved classically, so that the marginal on the index registers is

    .. math::

        \Pr(\mu, \nu) = \frac{\lvert \zeta_{\mu\nu} \rvert}{\lambda}, \qquad
        \Pr(\ell, M) = \frac{2 \lvert t_\ell \rvert}{\lambda}, \qquad
        \lambda = \sum_{\mu, \nu = 0}^{M - 1} \lvert \zeta_{\mu\nu} \rvert
        + 2 \sum_{\ell} \lvert t_\ell \rvert ,

    uniformly over all :math:`M^2` ordered two-body pairs (up to the ``aleph``-bit
    discretization). The construction follows the alias-sampling
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

    .. note::

        **The sign is an output register, not a phase on the amplitude.** With
        ``include_sign=True`` the wire ``work_wires[sign_wire]`` is left holding the sign
        bit :math:`s` of the coefficient that was actually selected, and the caller must
        apply a single ``qp.Z(work_wires[sign_wire])`` *between* ``PREPARE`` and
        ``SELECT``, where ``sign_wire`` comes from
        :func:`~pennylane.labs.templates.alias_sampling_thc_wires`. That same ``Z`` is
        what puts :math:`(-1)^s` on the amplitudes if you want to look at the prepared
        state on its own, so there is nothing else to switch on. Phasing
        the amplitude inside ``PREPARE`` instead would do nothing: the phase appears once
        in the ket and once in the bra of
        :math:`\langle 0 \rvert \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot
        \mathrm{PREPARE} \lvert 0 \rangle` and squares to :math:`+1`, so the block encoded
        operator would be :math:`\sum_\ell \lvert w_\ell \rvert U_\ell`. The single ``Z``
        sits between the two and contributes :math:`(-1)^s` exactly once. Because ``Z`` is
        diagonal it does not disturb the auxiliary registers, so
        ``qp.adjoint(alias_sampling_thc)`` still uncomputes exactly the same garbage.

        Use ``include_sign=False`` for a non-negative coefficient vector (all
        :math:`\zeta_{\mu\nu} \ge 0` and :math:`t_\ell \le 0`), or whenever only the
        prepared distribution matters. It saves two work wires and the ``QROM`` columns
        that carry them; no ``Z`` is then needed.

        The stored bit is the sign of the *LCU coefficient*, not of the raw weight: the
        one-body column carries an extra minus sign because ``SELECT`` applies the
        reflections :math:`V = I - 2 c^\dagger c` and the one-body block is linear in
        :math:`c^\dagger c` while the two-body block is quadratic (see the
        ``_lcu_signs`` helper). With this convention the block encoded operator is
        :math:`+H / \lambda`, matching Eq. (26) of `Lee et al. (2021)
        <https://arxiv.org/abs/2011.03494>`_.

    .. seealso:: :func:`~pennylane.labs.templates.alias_sampling_thc_wires`, which
        returns every register size for a given ``(M, N, aleph)``.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals. Requires ``N // 2 <= M + 1``
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``
        t_ell (tensor_like): the one-body eigenvalues, shape ``(N // 2,)``
        mu_wires (WiresLike): the ``n`` wires storing the first THC index
            :math:`\mu`. Requires exactly ``n = ceil(log2(M + 1))`` wires
        nu_wires (WiresLike): the ``n`` wires storing the second THC index
            :math:`\nu`. Must have the same length as ``mu_wires``
        edge_flag (WiresLike): the single wire holding the one-body sentinel flag
            (true when the ``nu`` register is in state :math:`\lvert M \rangle`), as
            produced by :class:`~pennylane.labs.templates.SuperpositionTHC`
        work_wires (WiresLike): the auxiliary wires. At least
            ``n_d + 2 * n + 3 * aleph + 2 + 2 * include_sign`` zeroed work wires are
            required, where ``n = ceil(log2(M + 1))`` and
            ``n_d = ceil(log2(N // 2 + M (M + 1) // 2)) + 1``
        aleph (int): the number of bits used to encode the keep-probabilities
        include_sign (bool): if ``True`` (default), the wire at index
            ``alias_sampling_thc_wires(M, N, aleph)["sign_wire"]`` of ``work_wires`` is
            left holding the sign bit of the selected coefficient, to be consumed by a
            single ``qp.Z`` between ``PREPARE`` and ``SELECT``. If ``False``, all
            coefficients are taken to be non-negative and two work wires are saved

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        from pennylane.labs.templates import (
            SuperpositionTHC,
            alias_sampling_thc,
            alias_sampling_thc_wires,
        )

        M, N, aleph = 2, 2, 6
        np.random.seed(3)
        zeta = np.random.randn(M, M)
        zeta = (zeta + zeta.T) / 2
        t_ell = np.random.randn(N // 2)

        sizes = alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        sign_wire_index = sizes["sign_wire"]

        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))

        # SuperpositionTHC prepares the uniform superposition and the one-body flag.
        sup_work = list(range(2 * n, 2 * n + sizes["superposition_work_wires"]))
        edge_flag = sup_work[3]  # nu register in state |M>

        start = sup_work[-1] + 1
        work_wires = list(range(start, start + sizes["work_wires"]))

        dev = qp.device("lightning.qubit", wires=work_wires[-1] + 1)

        @qp.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
            alias_sampling_thc(
                M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph
            )
            # The sign of the selected coefficient is consumed here, between PREPARE and
            # SELECT, by a single Z. This is also what puts the signs on the amplitudes
            # if you want to inspect the prepared state on its own with qp.state().
            qp.Z(work_wires[sign_wire_index])
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
    if N // 2 > M + 1:
        raise ValueError("N // 2 must be less than or equal to M + 1.")
    # The index registers must hold the one-body sentinel value ``M`` and nothing
    # more, so their size is fixed at ``ceil(log2(M + 1))``.
    if n != _num_index_wires(M):
        raise ValueError(
            f"mu_wires and nu_wires must each contain exactly ceil(log2(M + 1)) wires. "
            f"Got M={M} with {n} wires, but {_num_index_wires(M)} are required."
        )
    if isinstance(aleph, bool) or not isinstance(aleph, int) or aleph < 1:
        raise ValueError(f"aleph must be a positive integer, got {aleph!r}.")
    n_d = _num_address_wires(M, N)
    n_sign = 2 * bool(include_sign)  # QROM sign columns: ``sign`` and ``alt_sign``
    min_work = n_d + 2 * n + 3 * aleph + 2 + n_sign
    if len(work_wires) < min_work:
        raise ValueError(
            f"At least {min_work} work_wires (n_d + 2 * len(mu_wires) + 3 * aleph + 2 "
            f"+ 2 * include_sign) should be provided, but only {len(work_wires)} were given."
        )

    # The one-body sentinel flag (nu register in state |M>) is supplied by the
    # input state preparation (e.g. work_wires[3] of SuperpositionTHC); it is not
    # recomputed here.
    edge_flag = Wires(edge_flag)[0]

    # Wire layout on the work register, in order (``q`` is the base of the QROM index
    # block, ``f`` the base of the flag block). The two sign wires are absent when
    # ``include_sign=False``, and everything after them shifts down by two:
    #   [0 : n_d]                        contiguous QROM address ``s``
    #                                    ([0] is the spare high wire, see below)
    #   [n_d]                            QROM: ``sign``, ends on the *selected* entry
    #   [n_d + 1]                        QROM: ``alt_sign``, ends on the other entry
    #   [q : q + n]                      QROM: ``mu_alt``
    #   [q + n : q + 2 n]                QROM: ``nu_alt``
    #   [q + 2 n : q + 2 n + aleph]      QROM: ``keep`` threshold
    #   [f - aleph : f]                  uniform ``aleph``-bit sample ``sigma``
    #   [f], [f + 1], [f + 2]            alt_flag, swap_flag, alt_edge_flag
    #   [f + 3 : ]                       comparator work wires, then any extra wires
    #                                    forwarded to ``qp.QROM``
    q = n_d + n_sign
    f = q + 2 * n + 2 * aleph
    sign_wire = work_wires[n_d] if include_sign else None
    alt_sign_wire = work_wires[n_d + 1] if include_sign else None
    keep_thresh = work_wires[q + 2 * n : q + 2 * n + aleph]
    sample_reg = work_wires[q + 2 * n + aleph : f]
    # ``alt_flag == 1`` means the inequality test failed, i.e. the original pair is
    # *discarded* and the QROM-loaded alternate is used instead.
    alt_flag = work_wires[f]
    swap_flag = work_wires[f + 1]  # symmetrization (mu <-> nu) control
    alt_edge_flag = work_wires[f + 2]  # QROM-loaded ``alt_edge`` bit
    # ``qp.QROM`` restores its work wires to |0>, so the comparator safely reuses them.
    cmp_work = work_wires[f + 3 : f + aleph + 2]
    qrom_work = work_wires[f + 3 :]

    # 1. Compute the contiguous QROM address s = mu + nu (nu + 1) / 2.
    _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires)

    # 2. Load the alias data (signs, alternate indices, keep threshold, alt_edge).
    #    Only ``n_d - 1 = ceil(log2(d))`` control wires are needed: the address is
    #    ``s < d`` for every valid pair, so ``work_wires[0]`` (the most significant wire,
    #    only needed to hold ``nu ** 2 + nu`` before the division by two) is always |0>.
    #    Including it would double the QROM address space and its gate cost for nothing.
    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph, include_sign=include_sign)
    qp.QROM(
        data,
        control_wires=work_wires[1:n_d],
        target_wires=work_wires[n_d : q + 2 * n + aleph] + [alt_edge_flag],
        work_wires=qrom_work,
    )

    # 3. Draw a uniform aleph-bit sample sigma and test ``keep <= sigma``: the original
    #    pair is kept when the test fails, i.e. with probability keep / 2 ** aleph, which
    #    is the normalization the classical ``keep`` table is built for (Eq. (39) of
    #    `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_). This is the same
    #    comparator used by :func:`~pennylane.labs.templates.alias_sampling`; a strict
    #    ``"<"`` would keep with probability (keep + 1) / 2 ** aleph and bias every entry
    #    towards the uniform distribution.
    for w in sample_reg:
        qp.Hadamard(wires=w)

    LeftQuantumComparator(keep_thresh, sample_reg, alt_flag, work_wires=cmp_work, comparator="<=")

    # 4. If the original pair is discarded, swap in the alternate (mu_alt, nu_alt), its
    #    alt_edge flag and its sign bit.
    for i in range(n):
        qp.CSWAP([alt_flag, mu_wires[i], work_wires[q + i]])
    for i in range(n):
        qp.CSWAP([alt_flag, nu_wires[i], work_wires[q + n + i]])
    qp.CSWAP([alt_flag, edge_flag, alt_edge_flag])
    if include_sign:
        # The sign must reach ``SELECT`` as a register, not as a phase on the amplitude:
        # a phase applied here would show up once in the ket and once in the bra of
        # ``<0| PREPARE^dag SELECT PREPARE |0>`` and square to +1. This swap leaves
        # ``sign_wire`` holding the sign bit of whichever entry was actually selected,
        # ready for the caller's single ``qp.Z`` between ``PREPARE`` and ``SELECT``.
        qp.CSWAP([alt_flag, sign_wire, alt_sign_wire])

    # 5. Uncompute the inequality test. The comparator inputs (``keep_thresh`` and
    #    ``sample_reg``) are untouched by step 4, so the same ``comparator="<="`` returns
    #    ``alt_flag`` and ``cmp_work`` to |0>; any other comparator would leave
    #    ``alt_flag`` entangled with the sample register.
    qp.adjoint(LeftQuantumComparator)(
        keep_thresh, sample_reg, alt_flag, work_wires=cmp_work, comparator="<="
    )
    qp.H(swap_flag)

    # 6. Symmetrize: on the flagged subspace, swap the mu and nu registers so the
    #    prepared distribution covers both (mu, nu) and (nu, mu). The one-body block
    #    (edge_flag == 1) is excluded via a zero-control, leaving edge_flag untouched
    #    so it can be uncomputed by the adjoint of the input state preparation.
    for i in range(n):
        qp.ctrl(
            qp.SWAP([mu_wires[i], nu_wires[i]]),
            control=[swap_flag, edge_flag],
            control_values=[1, 0],
        )
