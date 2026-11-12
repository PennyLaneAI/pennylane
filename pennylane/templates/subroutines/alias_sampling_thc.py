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

import numpy as np

from pennylane.core.operator import Operator2
from pennylane.math import ceil_log2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


def _num_index_wires(M):
    r"""Number of wires per index register: exactly ``ceil(log2(M + 1))``."""
    return ceil_log2(M + 1)


def _num_address_wires(M, N):
    r"""Number of wires holding the contiguous QROM address ``s``.

    One wire more than ``ceil(log2(d))``: the register must transiently hold
    :math:`\nu^2 + \nu \le M(M + 1)`, i.e. up to twice the largest address, before the
    division by two of :func:`_compute_contiguous_register`. The final address
    satisfies :math:`s < d \le 2^{n_d - 1}`, so the leading wire is back to
    :math:`\lvert 0 \rangle` and is *not* used to control the QROM.
    """
    d = N // 2 + M * (M + 1) // 2
    return ceil_log2(d) + 1


def _validate_zeta(zeta):
    """Require ``zeta`` to already be hashable nested tuples (compilable static data)."""
    if not isinstance(zeta, tuple):
        raise ValueError(
            "zeta must be a tuple of tuples of floats, because it is compile-time static "
            f"data and has to be hashable; got {type(zeta).__name__}. Convert an array with "
            "tuple(map(tuple, arr))."
        )
    if not all(isinstance(row, tuple) for row in zeta):
        raise ValueError(
            "zeta must be a tuple of tuples of floats, because it is compile-time static "
            "data and has to be hashable; got a tuple whose rows are "
            f"{sorted({type(row).__name__ for row in zeta})}. Convert an array with "
            "tuple(map(tuple, arr))."
        )


def _validate_t_ell(t_ell):
    """Require ``t_ell`` to already be a hashable tuple (compilable static data)."""
    if not isinstance(t_ell, tuple):
        raise ValueError(
            "t_ell must be a tuple of floats, because it is compile-time static "
            f"data and has to be hashable; got {type(t_ell).__name__}. Convert an array with "
            "tuple(arr)."
        )
    if any(isinstance(x, (tuple, list)) for x in t_ell):
        raise ValueError(
            "t_ell must be a tuple of floats, but it contains nested sequences; "
            "a flat tuple was expected. Convert an array with tuple(arr)."
        )


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
        zeta (tuple[tuple[float]]): the THC central tensor, shape ``(M, M)``
        t_ell (tuple[float]): the one-body eigenvalues, shape ``(N // 2,)``

    Returns:
        tuple[list[tuple[int, int]], list[float]]: the pairs sorted lexicographically
        by ``(mu, nu)`` and their (signed) weights, aligned index-by-index

    Raises:
        ValueError: if ``zeta`` is not of shape ``(M, M)`` or ``t_ell`` is not of
            shape ``(N // 2,)``
    """
    n_half = N // 2

    zeta = np.asarray(zeta, dtype=float)
    t_ell = np.asarray(t_ell, dtype=float)
    zeta_shape = tuple(zeta.shape)
    if zeta_shape != (M, M):
        raise ValueError(f"zeta must be of shape ({M}, {M}), got {zeta_shape}.")
    t_shape = tuple(t_ell.shape)
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
    return entries, [weights[k] for k in entries]


def alias_sampling_thc_wires(M, N, aleph):
    r"""Return the wire counts required by :class:`~.AliasSamplingTHC`.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals. Requires ``N // 2 <= M + 1``
        aleph (int): the number of bits used to encode the keep-probabilities

    Returns:
        dict: ``{"mu_wires": n, "nu_wires": n, "superposition_work_wires": 3 * n + 5,
        "work_wires": n_d + 2 * n + 3 * aleph + 4 + max(n_d - aleph - 1, 0),
        "sign_wire": n_d}``, where ``n = ceil(log2(M + 1))`` and
        ``n_d = ceil(log2(N // 2 + M (M + 1) // 2)) + 1``

        * ``mu_wires`` / ``nu_wires``: the two index registers, exact
        * ``superposition_work_wires``: the work register of
          :class:`~.SuperpositionTHC`; its entry at index ``3``
          is the one-body sentinel flag to pass as ``edge_flag``
        * ``work_wires``: the minimum auxiliary register of :class:`~.AliasSamplingTHC`.
          Most of these wires retain data and are uncomputed by
          ``qp.adjoint(AliasSamplingTHC(...))``. Additional wires are forwarded to the internal
          :class:`~.QROM`, which uses them for a ``SelectSwap`` decomposition that lowers the
          T-gate count.
        * ``sign_wire``: the *index into* ``work_wires`` of the wire holding the sign bit
          of the selected coefficient.

    **Example**

    >>> qp.alias_sampling_thc_wires(M=2, N=2, aleph=6)
    {'mu_wires': 2, 'nu_wires': 2, 'superposition_work_wires': 11, 'work_wires': 29, 'sign_wire': 2}
    """
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError(f"M must be a positive integer, got {M!r}.")
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError(f"N must be a positive integer, got {N!r}.")
    if isinstance(aleph, bool) or not isinstance(aleph, int) or aleph < 1:
        raise ValueError(f"aleph must be a positive integer, got {aleph!r}.")
    if N // 2 > M + 1:
        raise ValueError("N // 2 must be less than or equal to M + 1.")

    n = ceil_log2(M + 1)
    n_d = _num_address_wires(M, N)
    # The compare stage needs aleph-1 work wires for the comparator, and one work wire for CSWAPs
    # The QROM has n_d-1 control wires and thus needs at least n_d-2 work wires for unary iteration
    qrom_and_compare = max((aleph - 1) + 1, n_d - 2)
    return {
        "mu_wires": n,
        "nu_wires": n,
        "superposition_work_wires": 3 * n + 5,
        "work_wires": n_d + 2 * n + 2 * aleph + 4 + qrom_and_compare,
        "sign_wire": n_d - 1,
    }


class AliasSamplingTHC(Operator2):
    r"""Coefficient oracle for tensor hypercontraction (THC) qubitization via
    coherent alias (Walker) sampling.

    Given the uniform superposition over the valid THC index pairs
    :math:`\mathcal{S}` (as prepared by :class:`~.SuperpositionTHC`),
    this template reweights the amplitudes to the target distribution set by
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
    bit :math:`s` of the selected coefficient. Both
    are left in this entangled state for the subsequent ``SELECT``, and

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
        :class:`~.SuperpositionTHC`, which also produces the
        one-body sentinel flag (its ``work_wires[3]``, true when :math:`\nu = M`)
        passed here as ``edge_flag``. This template does not recompute that flag.

    .. warning::

        Most ``work_wires`` are left entangled with the index registers and are not
        returned to :math:`\lvert 0\rangle`. In a prepare/select/prepare pattern,
        ``qp.adjoint(AliasSamplingTHC(...))`` uncomputes them.

    .. seealso:: :func:`~.alias_sampling_thc_wires`, which
        returns every register size for a given ``(M, N, aleph)``.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals. Requires ``N // 2 <= M + 1``
        zeta (tuple[tuple[float]]): the THC central tensor of shape ``(M, M)``,
            provided as a nested tuple (use ``tuple(map(tuple, arr))`` to convert an array).
        t_ell (tuple[float]): the one-body eigenvalues as a tuple of length ``N // 2``.
        mu_wires (WiresLike): the ``n`` wires storing the first THC index
            :math:`\mu`. Requires exactly ``n = ceil(log2(M + 1))`` wires
        nu_wires (WiresLike): the ``n`` wires storing the second THC index
            :math:`\nu`. Must have the same length as ``mu_wires``
        edge_flag (WiresLike): the single wire holding the one-body sentinel flag
            (true when the ``nu`` register is in state :math:`\lvert M \rangle`), as
            produced by :class:`~.SuperpositionTHC`
        work_wires (WiresLike): the auxiliary wires used by the operator.
            Let :math:`n_d=\lceil \log_2(N/2 + M(M+1)/2)\rceil + 1` and
            :math:`n=\lceil\log_2(M+1)\rceil` as above.
            The wires ``work_wires[:n_d+2*n+2*aleph+3]`` retain data until the adjoint of this
            template is applied; this includes the sampling register, which the conditional
            swaps correlate with the index registers. The wires
            ``work_wires[n_d+2*n+2*aleph+3:]`` are returned to the zero state. The required
            number is
            :math:`n_d + 2n + 2\aleph + 4 + \max(\aleph, n_d-2)`, computed as ``"work_wires"``
            entry in :func:`~.alias_sampling_thc_wires`. Excess wires are forwarded to
            the internal :class:`~.QROM`; every work wire must be initialized
            in :math:`\lvert 0\rangle`.
        aleph (int): the number of bits used to encode the keep-probabilities

    **Example**

    The index superposition is prepared first with
    :class:`~.SuperpositionTHC`. Use
    :func:`~.alias_sampling_thc_wires` for every register size, including the
    ``SuperpositionTHC`` work register whose entry at index ``3`` is ``edge_flag``.

    .. code-block:: python

        import pennylane as qp

        M, N, aleph = 2, 2, 6
        zeta = ((1.0, 0.0), (0.0, 1.0))
        t_ell = (1.0,)

        sizes = qp.alias_sampling_thc_wires(M, N, aleph)
        n = sizes["mu_wires"]
        mu_wires = list(range(n))
        nu_wires = list(range(n, 2 * n))
        sup_work = list(range(2 * n, 2 * n + sizes["superposition_work_wires"]))
        edge_flag = sup_work[3]
        start = sup_work[-1] + 1
        work_wires = list(range(start, start + sizes["work_wires"]))

        qp.SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
        qp.AliasSamplingTHC(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.AliasSamplingTHC
            :stacked:
            :symbols: :math:`n = \lceil \log_2(M+1) \rceil` wires per index register and
                :math:`\aleph` = ``aleph`` bits, and :math:`M` is the THC rank.
            :example: :math:`M=1`, :math:`N=2`, :math:`\aleph=1` (:math:`n=1`)

            .. decomposition-rule:: _alias_sampling_thc_decomp
                :image: /_static/decomposition_thumbnails/alias_sampling_thc_walker.svg

                :class:`~.OutSquare` | :math:`1` | 1
                :class:`~.SemiAdder` | :math:`2` | 2
                :class:`~.QROM` | :math:`1` | 1
                :class:`~.Hadamard` | :math:`\aleph + 1` | 2
                :class:`~.LeftQuantumComparator` | :math:`1` | 1
                :class:`Adjoint(LeftQuantumComparator) <.LeftQuantumComparator>` | :math:`1` | 1
                :class:`~.CSWAP` | :math:`3n + 2` | 5
                :class:`~.TemporaryAND` | :math:`1` | 1
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`1` | 1
    """

    wire_argnames = ("mu_wires", "nu_wires", "edge_flag", "work_wires")
    compilable_argnames = ("M", "N", "zeta", "t_ell", "aleph")
    arg_specs = {
        "mu_wires": Wire[-1],
        "nu_wires": Wire[-1],
        "edge_flag": Wire[1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        M,
        N,
        zeta,
        t_ell,
        mu_wires: WiresLike,
        nu_wires: WiresLike,
        edge_flag: WiresLike,
        work_wires: WiresLike,
        aleph,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        if isinstance(M, bool) or not isinstance(M, int) or M < 1:
            raise ValueError(f"M must be a positive integer, got {M!r}.")
        if isinstance(N, bool) or not isinstance(N, int) or N < 1:
            raise ValueError(f"N must be a positive integer, got {N!r}.")
        if isinstance(aleph, bool) or not isinstance(aleph, int) or aleph < 1:
            raise ValueError(f"aleph must be a positive integer, got {aleph!r}.")
        if N // 2 > M + 1:
            raise ValueError("N // 2 must be less than or equal to M + 1.")

        _validate_zeta(zeta)
        _validate_t_ell(t_ell)
        _build_thc_pairs(M, N, zeta, t_ell)

        mu_wires = Wires(mu_wires)
        nu_wires = Wires(nu_wires)
        edge_flag = Wires(edge_flag)
        work_wires = Wires([] if work_wires is None else work_wires)

        n = len(mu_wires)
        if len(nu_wires) != n:
            raise ValueError(
                f"mu_wires and nu_wires must contain the same number of wires, "
                f"but got {n} and {len(nu_wires)}."
            )
        expected_n = ceil_log2(M + 1)
        if n != expected_n:
            raise ValueError(
                f"mu_wires and nu_wires must each contain exactly ceil(log2(M + 1)) wires. "
                f"Got M={M} with {n} wires, but {expected_n} are required."
            )
        req = alias_sampling_thc_wires(M, N, aleph)
        if len(work_wires) < req["work_wires"]:
            raise ValueError(
                f"At least {req['work_wires']} work_wires (the \"work_wires\" entry of "
                f"alias_sampling_thc_wires({M}, {N}, {aleph})) should be provided, but only "
                f"{len(work_wires)} were given."
            )
        validate_no_wire_overlaps(
            {
                "mu_wires": mu_wires,
                "nu_wires": nu_wires,
                "edge_flag": edge_flag,
                "work_wires": work_wires,
            }
        )

        super().__init__(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.mu_wires + self.nu_wires + self.edge_flag + self.work_wires
