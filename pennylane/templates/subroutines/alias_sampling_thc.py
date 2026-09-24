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

from collections import defaultdict
from functools import lru_cache

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.math import ceil_log2
from pennylane.ops import SWAP, Hadamard, Z, adjoint, ctrl
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .alias_sampling import _apply_hadamards, _build_alias_tables
from .arithmetic.left_quantum_comparator import LeftQuantumComparator
from .arithmetic.out_square import OutSquare
from .arithmetic.semi_adder import SemiAdder
from .arithmetic.temporary_and import TemporaryAND
from .qrom import QROM


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
    coeffs = [-w if nu == M else w for w, (_, nu) in zip(weights, entries, strict=True)]
    return [int(c < 0) for c in coeffs]


@lru_cache(maxsize=16)
def _build_qrom_data(
    M, N, zeta, t_ell, num_index_wires, aleph
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    r"""Pack the alias tables into the bitstrings consumed by :class:`~.QROM`.

    The QROM is addressed by the contiguous two-body index
    ``s = mu + nu (nu + 1) / 2`` (matching :func:`_compute_contiguous_register`). Each
    row concatenates, in order: ``sign`` and ``alt_sign``, ``mu_alt``
    (``num_index_wires`` bits), ``nu_alt`` (``num_index_wires`` bits), the ``aleph``-bit
    keep threshold, and the ``alt_edge`` flag.

    The keep threshold and alternate index are produced by the classical
    ``_build_alias_tables`` of :class:`~.AliasSampling`
    (Walker/Vose, ``mu = aleph`` bits); the signs (see :func:`_lcu_signs`) and the
    ``alt_edge`` sentinel are derived from the THC pair enumeration.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals
        zeta (tuple[tuple[float]]): the THC central tensor, shape ``(M, M)``
        t_ell (tuple[float]): the one-body eigenvalues, shape ``(N // 2,)``
        num_index_wires (int): number of wires per index register (``len(mu_wires)``)
        aleph (int): number of bits used for the keep-probability comparison

    Returns:
        list[list[int]]: the QROM data, one bitstring (list of ints) per address
    """
    entries, weights = _build_thc_pairs(M, N, zeta, t_ell)
    probs = [abs(w) for w in weights]
    signs = _lcu_signs(M, entries, weights)

    alt, keep = _build_alias_tables(probs, aleph)

    n_bits = 2 + 2 * num_index_wires + aleph + 1
    data = [[0] * n_bits for _ in range(len(entries))]
    for i, (mu, nu) in enumerate(entries):
        s = mu + (nu**2 + nu) // 2
        alt_i = alt[i]
        mu_alt, nu_alt = entries[alt_i]
        data[s] = (
            [signs[i], signs[alt_i]]
            + math.int_to_binary(mu_alt, num_index_wires).tolist()
            + math.int_to_binary(nu_alt, num_index_wires).tolist()
            + math.int_to_binary(keep[i], aleph).tolist()
            + [1 if nu_alt == M else 0]
        )
    return data


def _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires):
    r"""Compute the contiguous address ``s = mu + nu (nu + 1) / 2`` into ``work_wires``.

    Uses ``nu (nu + 1) / 2 = (nu^2 + nu) / 2`` via ``OutSquare`` (``nu^2``) followed by
    ``SemiAdder`` (``+ nu``), a right shift by one bit (division by two, implemented
    with SWAPs), and a final ``SemiAdder`` (``+ mu``). The result lands on the first
    ``n_d`` work wires.
    """
    n_d = _num_address_wires(M, N)
    OutSquare(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d], output_wires_zeroed=True)
    SemiAdder(nu_wires, work_wires[:n_d], work_wires[n_d : 2 * n_d - 1])
    SemiAdder(mu_wires, work_wires[: n_d - 1], work_wires[n_d : 2 * n_d - 2])


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


def _cswap_pair(flag, left, right, work_wires):
    """CSWAP each pair of wires in ``left`` / ``right`` controlled on ``flag``."""
    n = min(len(left), len(right))
    if n == 0:
        return

    if compiler.active() or capture.enabled():
        left = math.array(left, like="jax")
        right = math.array(right, like="jax")

    @for_loop(n)
    def _loop(i):
        ctrl(
            SWAP(wires=[left[i], right[i]]),
            control=[flag],
            work_wires=work_wires,
            work_wire_type="zeroed",
        )

    _loop()  # pylint: disable=no-value-for-parameter


def _symmetrize(mu_wires, nu_wires, swap_flag, edge_flag, work_wires):
    """Swap ``mu`` and ``nu`` when ``swap_flag`` is 1 and ``edge_flag`` is 0."""
    n = len(mu_wires)
    if n == 0:
        return

    joint_flag = work_wires[0]
    _cswap_work = work_wires[1:]
    TemporaryAND([swap_flag, edge_flag, joint_flag], control_values=(1, 0))
    _cswap_pair(joint_flag, mu_wires, nu_wires, _cswap_work)
    adjoint(TemporaryAND([swap_flag, edge_flag, joint_flag], control_values=(1, 0)))


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
        apply_sign (bool): if ``True`` (default), the sign of the selected coefficient is
            applied here, so the prepared state carries it on its amplitudes. Set to
            ``False`` when using only positive coefficients.

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
    """

    wire_argnames = ("mu_wires", "nu_wires", "edge_flag", "work_wires")
    compilable_argnames = ("M", "N", "zeta", "t_ell", "aleph", "apply_sign")
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
        apply_sign: bool = True,
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

        super().__init__(
            M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph, apply_sign
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.mu_wires + self.nu_wires + self.edge_flag + self.work_wires


def _alias_sampling_thc_resources(
    M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph, apply_sign
):  # pylint: disable=too-many-arguments,unused-argument
    n = len(mu_wires)
    n_d = _num_address_wires(M, N)
    n_work = len(work_wires)
    # ``f`` and the QROM work pool are spelled exactly as in the decomposition below, where the
    # pool is ``work_wires[f + 2 * aleph + 3:]``. Declaring a different size hands the graph a
    # resource rep that no emitted QROM ever matches, and an op with no node in the graph
    # silently bypasses fixed_decomps and falls back to QROM.decomposition().
    f = n_d + 1 + 2 * n
    n_qrom_target = 2 + 2 * n + aleph + 1
    n_qrom_work = max(n_work - (f + 2 * aleph + 3), 0)
    assert n_qrom_work >= n_d - 2  # TODO: remove me
    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
    qrom = QROM(
        data,
        control_wires=Wire[n_d - 1],
        target_wires=Wire[n_qrom_target],
        work_wires=Wire[n_qrom_work],
        clean=True,
    )
    out_sq = OutSquare(Wire[n], Wire[n_d], Wire[n_d], output_wires_zeroed=True)
    adder_0 = SemiAdder(Wire[n], Wire[n_d], Wire[n_d - 1])
    adder_1 = SemiAdder(Wire[n], Wire[n_d - 1], Wire[n_d - 2])

    # The comparator does not restore its work wires, so the keep-value swaps only get the
    # tail of the pool; the symmetrization swaps run after the adjoint comparator and get
    # all of it but the joint flag. Both slices are spelled as in the decomposition below.
    n_cmp_work = min(aleph - 1, n_qrom_work)
    lqc = LeftQuantumComparator(
        Wire[aleph],
        Wire[aleph],
        Wire[1],
        Wire[n_cmp_work],
        comparator="<=",
    )
    keep_cswap = ctrl(
        SWAP(wires=Wire[2]),
        control=Wire[1],
        work_wires=Wire[n_qrom_work - n_cmp_work],
        work_wire_type="zeroed",
    )
    sym_cswap = ctrl(
        SWAP(wires=Wire[2]),
        control=Wire[1],
        work_wires=Wire[max(n_qrom_work - 1, 0)],
        work_wire_type="zeroed",
    )
    resources = defaultdict(int)
    resources[out_sq] += 1
    resources[adder_0] += 1
    resources[adder_1] += 1
    resources[qrom] += 1
    resources[Hadamard] += aleph + 1
    resources[lqc] += 1
    resources[adjoint(lqc)] += 1
    resources[TemporaryAND] += 1
    resources[adjoint(TemporaryAND(Wire[3]))] += 1
    resources[keep_cswap] += 2 * n + 2
    resources[sym_cswap] += n
    if apply_sign:
        resources[Z] += 1
    return dict(resources)


@register_resources(_alias_sampling_thc_resources)
def _alias_sampling_thc_decomp(
    M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph, apply_sign, **_
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    n = len(mu_wires)
    n_d = _num_address_wires(M, N)
    work_wires = list(work_wires)
    mu_wires = list(mu_wires)
    nu_wires = list(nu_wires)
    edge_flag = Wires(edge_flag)[0]

    # Work wires are used as follows (order is changed, compared to Fig.4 in Lee et al)
    # The following entries store a value by the end of the template and are not reset
    # [:n_d-1]        : ν(ν+1)//2 + μ after _compute_contiguous_register
    # n_d-1           : QROM loads the sign θ_s
    # n_d             : QROM loads the alternate sign {θ_alt}_s
    # [n_d+1:n_d+1+n] : QROM loads the alternate {μ_alt}_s
    # [n_d+1+n:n_d+1+2n] : QROM loads the alternate {ν_alt}_s
    # Call f = n_d+1+2n
    # [f:f+ℵ]         : QROM loads the keep values
    # [f+ℵ]           : alternate qubit for the input edge flag (not in Fig.4)
    # [f+ℵ+1]         : flag for symmetrization SWAPs
    # [f+ℵ+2:f+2ℵ+2]  : Sampling register to compare keep values against.
    # The following register is reset to zero
    # [f+2ℵ+2]        : The comparator flag for sampling keep values
    # The following registers are reset to zero, and overlap partially
    # [f+2ℵ+3:]       : Work wires for QROM
    # [f+2ℵ+3:f+3ℵ+2] : Work wires for keep value comparator, dirty until its adjoint runs
    # [f+3ℵ+2:]       : Work wires for keep value CSWAPs, the part the comparator leaves zeroed
    # [f+2ℵ+3:]       : Work wires for symmetrization CSWAPs, once the comparator is undone

    contiguous_register = work_wires[: n_d - 1]
    sign_wire = work_wires[n_d - 1]
    alt_sign_wire = work_wires[n_d]
    alt_mu_wires = work_wires[n_d + 1 : n_d + 1 + n]
    alt_nu_wires = work_wires[n_d + 1 + n : (f := n_d + 1 + 2 * n)]
    keep_wires = work_wires[f : f + aleph]
    alt_edge_flag = work_wires[f + aleph]
    symmetrize_flag = work_wires[f + aleph + 1]

    # Reset to zero and disjoint
    sample_reg = work_wires[f + aleph + 2 : f + 2 * aleph + 2]
    sample_flag = work_wires[f + 2 * aleph + 2]

    # Reset to zero and overlapping
    qrom_work = work_wires[f + 2 * aleph + 3 :]
    cmp_work = qrom_work[: aleph - 1]
    keep_cswap_work = qrom_work[aleph - 1 :]
    sym_cswap_work = qrom_work

    # work_wires includes the output contiguous_register and additional zeroed work wires that
    # are returned to zero, so we do not need to account for them explicitly.
    _compute_contiguous_register(M, N, mu_wires, nu_wires, work_wires)

    data = _build_qrom_data(M, N, zeta, t_ell, n, aleph)
    QROM(
        data,
        control_wires=contiguous_register,
        target_wires=work_wires[n_d - 1 : f + aleph] + [alt_edge_flag],
        work_wires=qrom_work,
    )

    _apply_hadamards(sample_reg)
    LeftQuantumComparator(keep_wires, sample_reg, sample_flag, work_wires=cmp_work, comparator="<=")

    _cswap_pair(sample_flag, mu_wires, alt_mu_wires, keep_cswap_work)
    _cswap_pair(sample_flag, nu_wires, alt_nu_wires, keep_cswap_work)
    _cswap_pair(
        sample_flag, [edge_flag, sign_wire], [alt_edge_flag, alt_sign_wire], keep_cswap_work
    )

    adjoint(
        LeftQuantumComparator(
            keep_wires, sample_reg, sample_flag, work_wires=cmp_work, comparator="<="
        )
    )

    Hadamard(symmetrize_flag)
    _symmetrize(mu_wires, nu_wires, symmetrize_flag, edge_flag, sym_cswap_work)

    if apply_sign:
        Z(sign_wire)


add_decomps(AliasSamplingTHC, _alias_sampling_thc_decomp)
