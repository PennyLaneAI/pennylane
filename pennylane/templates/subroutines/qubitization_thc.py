# Copyright 2026 Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qubitization walk operator for the tensor hypercontracted (THC) Hamiltonian."""

import numpy as np

from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import GlobalPhase, Hadamard, Z, adjoint, ctrl
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .alias_sampling_thc import (
    AliasSamplingTHC,
    _build_thc_pairs,
    _num_address_wires,
    _num_index_wires,
    _validate_t_ell,
    _validate_zeta,
    alias_sampling_thc_wires,
)
from .flip_sign import FlipSign
from .select_thc import SelectTHC, _validate_select_data, select_thc_wires
from .superposition_thc import SuperpositionTHC


def _alias_wire_split(M, N, aleph):
    r"""Split :class:`~.AliasSamplingTHC`'s work register into garbage and clean wires.

    That template takes a single ``work_wires`` register, but only part of it is garbage:
    with ``f = n_d + 2 + 2 n + 2 aleph`` its layout ends in ``alt_flag`` at ``f``,
    ``swap_flag`` at ``f + 1``, ``alt_edge_flag`` at ``f + 2``, and the comparator/QROM
    scratch from ``f + 3``. The adjoint inequality test returns ``alt_flag`` and the
    comparator scratch to :math:`\lvert 0 \rangle`, so those wires do *not* have to live
    inside the reflected register and can be drawn from the shared clean pool instead.
    Everything below ``f``, plus ``swap_flag`` and ``alt_edge_flag``, stays entangled and
    must be reflected.

    Returns:
        tuple[int, int]: ``(first_clean, num_garbage)``, the index ``f`` of the first clean
        wire and the number of garbage wires ``f + 2``
    """
    first_clean = _num_address_wires(M, N) + 2 + 2 * _num_index_wires(M) + 2 * aleph
    return first_clean, first_clean + 2


def qubitization_thc_wires(M, N, aleph, beth, num_batches=1):
    r"""Return the wire counts required by :class:`~.QubitizationTHC`.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals. Requires ``N // 2 <= M + 1``
        aleph (int): the number of bits used to encode the ``PREPARE``
            keep-probabilities
        beth (int): bits of precision per Givens angle in ``SELECT``
        num_batches (int): the number of batches the Givens angles are loaded in

    Returns:
        dict: ``{"system_wires": N, "index_wires": 2 * n, "prep_garbage_wires": n_prep,
        "gradient_wires": beth + 1, "work_wires": n_work}``

        * ``system_wires`` (``N``): the spin orbitals, the ``N/2`` spin-down spatial
          orbitals followed by the ``N/2`` spin-up ones
        * ``index_wires`` (``2 * ceil(log2(M + 1))``): the LCU index :math:`\mu` followed by
          :math:`\nu`, i.e. the register ``PREPARE`` writes the coefficient amplitudes on and
          ``SELECT`` reads to pick which :math:`V` to apply
        * ``prep_garbage_wires``: *not* a state preparation register. These are the wires
          ``PREPARE`` leaves entangled with the index and that therefore have to be reflected
          along with it, laid out as the :class:`~.SuperpositionTHC` work register, then
          whatever :class:`~.AliasSamplingTHC` garbage it cannot supply, then the two spin
          flags. The spin flags are the exception to the name: they are genuine LCU index
          wires, placed here because ``index_wires`` is sized exactly ``2 n`` by
          :func:`~.select_thc_wires`. Exact: the reflection acts on
          ``index_wires + prep_garbage_wires``, so a spare wire here changes the walk operator
        * ``gradient_wires`` (``beth + 1``): the phase gradient register, which must be
          prepared by the caller and is left unchanged
        * ``work_wires``: the shared clean pool, returned to :math:`\lvert 0 \rangle`. This
          is a minimum; extra wires are forwarded to ``SELECT``, to ``PREPARE``'s ``QROM``
          and to the multi-controlled :math:`Z` of the reflection to lower their T counts

    .. note::

        Every zeroed auxiliary wire is shared rather than duplicated, which is what keeps
        ``prep_garbage_wires`` well below the sum of the sub-template registers:

        * :class:`~.SuperpositionTHC` returns all but three of its ``3 n + 5`` work wires to
          :math:`\lvert 0 \rangle` (the exceptions are its indices ``0``, ``3`` and ``6``),
          so those ``3 n + 2`` wires are recycled as :class:`~.AliasSamplingTHC` garbage
        * of :class:`~.AliasSamplingTHC`'s work register, only
          ``n_d + 2 n + 2 aleph + 4`` wires stay entangled; the rest (its ``alt_flag``, the
          comparator scratch and the ``QROM`` scratch) are restored, so they are taken from
          ``work_wires`` instead of ``prep_garbage_wires``. This both saves qubits and
          removes controls from the reflection
        * ``work_wires`` is idle during ``PREPARE`` and clean again during ``SELECT``, so
          the same pool serves ``PREPARE``'s comparator and ``QROM``, ``SELECT``, and the
          reflection's multi-controlled :math:`Z`

        The remaining ``prep_garbage_wires`` are genuine garbage and must stay inside the
        reflection. Note that dropping them from it does *not* change the
        :math:`\lvert 0 \rangle` block: for any reflection whose fixed subspace contains
        :math:`\lvert 0 \rangle`, :math:`\langle 0 \rvert \mathcal{R} = \langle 0 \rvert`, so
        a single walk looks correct either way. What breaks is the Chebyshev property
        :math:`\langle 0 \rvert W^k \lvert 0 \rangle = T_k(H / \lambda)` from :math:`k = 2`
        on, since

        .. math::

            \langle 0 \rvert U \Pi U \lvert 0 \rangle = B^2
            \quad \text{but} \quad
            \langle 0 \rvert U \Pi_{\text{idx}} U \lvert 0 \rangle
            = B^2 + \sum_{g \neq 0} \langle 0 \rvert U \lvert 0, g \rangle
              \langle 0, g \rvert U \lvert 0 \rangle

        with :math:`U =` ``PREPARE``:math:`^\dagger \cdot` ``SELECT`` :math:`\cdot`
        ``PREPARE``, :math:`B = \langle 0 \rvert U \lvert 0 \rangle` and :math:`\Pi` the
        projector onto :math:`\lvert 0 \rangle` on *all* of ``index_wires +
        prep_garbage_wires``. The extra terms are the branches that are
        :math:`\lvert 0 \rangle` on the index but not on the garbage; reflecting only the
        index gives them :math:`+1` instead of :math:`-1`. The failure is therefore silent
        until phase estimation.

    **Example**

    >>> import pennylane as qp
    >>> qp.qubitization_thc_wires(M=2, N=2, aleph=1, beth=1)
    {'system_wires': 2, 'index_wires': 4, 'prep_garbage_wires': 18, 'gradient_wires': 2, 'work_wires': 2}
    """
    select_sizes = select_thc_wires(M, N, beth, num_batches)
    alias_sizes = alias_sampling_thc_wires(M, N, aleph)
    n_sup = alias_sizes["superposition_work_wires"]
    _, n_garbage = _alias_wire_split(M, N, aleph)

    return {
        "system_wires": N,
        "index_wires": select_sizes["index_wires"],
        # SuperpositionTHC work + the alias garbage it cannot supply + the two spin flags.
        "prep_garbage_wires": n_sup + max(0, n_garbage - (n_sup - 3)) + 2,
        "gradient_wires": select_sizes["gradient_wires"],
        # One shared clean pool: SELECT's scratch, the alias wires that AliasSamplingTHC
        # restores (its alt_flag, the comparator scratch and the QROM scratch, which is
        # what ``alias_sizes["work_wires"] - n_garbage`` counts), and at least one zeroed
        # auxiliary wire for the reflection's multi-controlled Z.
        "work_wires": max(select_sizes["work_wires"], alias_sizes["work_wires"] - n_garbage, 1),
    }


def _prepare_registers(
    M, N, aleph, index_wires, prep_garbage_wires, work_wires
):  # pylint: disable=too-many-arguments
    r"""Split the input registers into the sub-registers of the three THC oracles.

    The layout is the one documented by :func:`qubitization_thc_wires`. The returned
    ``alias_work`` interleaves garbage and clean wires: :class:`~.AliasSamplingTHC` takes a
    single work register that mixes the two, so the garbage slots are fed from the
    reflected register and the restored slots from the shared clean pool, leaving no wire
    in ``prep_garbage_wires`` that ends in :math:`\lvert 0 \rangle`.

    Returns:
        dict: the registers each sub-template receives, plus the three flags that
        ``PREPARE`` hands to ``SELECT`` and the register the reflection acts on
    """
    n = _num_index_wires(M)
    index = list(index_wires)
    garbage = list(prep_garbage_wires)
    clean = list(work_wires)

    n_sup = alias_sampling_thc_wires(M, N, aleph)["superposition_work_wires"]
    first_clean, n_garbage = _alias_wire_split(M, N, aleph)

    superposition_work = garbage[:n_sup]
    spin_wires = garbage[-2:]

    # SuperpositionTHC returns every work wire to |0> except its three flags (indices 0, 3
    # and 6), so the rest are reflected wires that are free to carry alias garbage; the
    # alias adjoint clears them again before SuperpositionTHC's adjoint runs.
    garbage_pool = (
        [w for i, w in enumerate(superposition_work) if i not in (0, 3, 6)] + garbage[n_sup:-2]
    )[:n_garbage]
    alias_work = (
        garbage_pool[:first_clean]  # contiguous address, QROM output, sigma sample
        + [clean[0]]  # alt_flag: restored by the adjoint comparator
        + garbage_pool[first_clean:]  # swap_flag, alt_edge_flag
        + clean[1:]  # comparator scratch, then extra wires for the QROM
    )

    return {
        "mu_wires": index[:n],
        "nu_wires": index[n:],
        # SuperpositionTHC runs first, while the shared pool is still untouched, and it
        # forwards anything past its fixed 3n+5 layout to its comparators and
        # multi-controlled gates. Given only the minimum it has *zero* scratch for those.
        "superposition_work": superposition_work + clean,
        "alias_work": alias_work,
        # SuperpositionTHC's flags: index 3 is true when nu = M (the one-body sentinel
        # column) and index 6 when the superposition was prepared correctly.
        "edge_flag": superposition_work[3],
        "success_flag": superposition_work[6],
        # AliasSamplingTHC holds its mu <-> nu symmetrization flag one wire above the
        # inequality-test flag.
        "swap_flag": alias_work[first_clean + 1],
        "spin_wires": spin_wires,
        "reflected": index + garbage,
    }


class QubitizationTHC(Operator2):
    r"""Apply the qubitization walk operator of a tensor hypercontracted (THC) Hamiltonian.

    This composes the three THC oracles into

    .. math::

        \mathcal{W} = \mathcal{R} \cdot \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT}
        \cdot \mathrm{PREPARE} ,
        \qquad \mathcal{R} = 2 \lvert \vec 0 \rangle \langle \vec 0 \rvert - I ,

    with :math:`\mathcal{R}` the reflection on ``index_wires + prep_garbage_wires``, following
    `Lee et al. (2021) <https://arxiv.org/abs/2011.03494>`_ (Figs. 3, 5 and 7).
    ``PREPARE`` is :class:`~.SuperpositionTHC` followed by :class:`~.AliasSamplingTHC` and a
    ``Hadamard`` on each of the two spin flags; ``SELECT`` is :class:`~.SelectTHC`.

    The :math:`\lvert \vec 0 \rangle` block of :math:`\mathcal{W}` is
    :math:`\hat{\mathcal{H}} / \lambda` with

    .. math::

        \hat{\mathcal{H}} = \sum_{\mu \nu = 0}^{M - 1} \zeta_{\mu \nu}
        \hat n_\mu \hat n_\nu
        - 2 \sum_{\mu = 0}^{M - 1} \Big( \sum_{\nu = 0}^{M - 1} \zeta_{\mu \nu} \Big)
        \hat n_\mu
        + 2 \sum_{\ell = 0}^{N/2 - 1} t_\ell \hat n_\ell
        + c\, \hat 1 ,

    .. math::

        \lambda = \sum_{\mu \nu = 0}^{M - 1} \lvert \zeta_{\mu \nu} \rvert
        + 2 \sum_{\ell = 0}^{N/2 - 1} \lvert t_\ell \rvert ,

    where :math:`\hat n_\mu = \sum_\sigma \hat c^\dagger_{\mu \sigma} \hat c_{\mu \sigma}`
    is the number operator of the THC leaf :math:`\chi_\mu` (the row ``chi[mu]``,
    normalized), :math:`\hat n_\ell` the number operator of the one-body eigenvector
    ``t_eigenvectors[:, ell]``, and :math:`c` an irrelevant identity shift that the block
    encoding is blind to.

    .. warning::

        The linear term :math:`- 2 \sum_\nu \zeta_{\mu \nu}` above is *not* optional: it
        is produced by rewriting :math:`\hat n = (\hat 1 - \hat V) / 2` in the two-body
        block and it lands on the leaves :math:`\chi_\mu`, not on the one-body
        eigenvectors. To block encode a target Hamiltonian
        :math:`\sum_{\mu\nu} \zeta_{\mu\nu} \hat n_\mu \hat n_\nu + \hat T`, the caller
        must fold that term into the one-body part *before* diagonalizing, i.e. pass the
        eigendecomposition of

        .. math::

            T' = T + \sum_\mu \Big( \sum_\nu \zeta_{\mu\nu} \Big)
            \frac{\chi_\mu \chi_\mu^T}{\lVert \chi_\mu \rVert^2}

        as ``(t_ell / 2, t_eigenvectors)``. This is the modified one-body matrix of
        Eq. (11) of `Lee et al. (2021) <https://arxiv.org/abs/2011.03494>`_.

    .. note::

        ``PREPARE`` succeeds with probability :math:`1` only when
        :math:`d = N/2 + M(M+1)/2` is at least :math:`2^{2n - 2}` with
        :math:`n = \lceil \log_2 (M + 1) \rceil`, the condition under which the single
        amplitude-amplification round of :class:`~.SuperpositionTHC` is exact. Otherwise the
        leftover garbage branch is *not* acted on by ``SELECT``, yet is mapped back onto
        :math:`\lvert \vec 0 \rangle` by ``PREPARE``:math:`^\dagger`, which contaminates
        the block. A ``ValueError`` is raised in that case rather than returning a
        silently wrong block encoding.

    .. note::

        ``gradient_wires`` must be prepared by the caller in the phase gradient state

        .. math::

            \lvert \phi \rangle = \frac{1}{\sqrt{2^{\mathrm{beth} + 1}}}
            \sum_{k = 0}^{2^{\mathrm{beth} + 1} - 1}
            e^{-2 \pi i k / 2^{\mathrm{beth} + 1}} \lvert k \rangle ,

        a product state that ``beth + 1`` ``Hadamard`` and ``beth + 1`` ``PhaseShift`` gates
        prepare. The walk leaves it unchanged, so a single register is prepared once and
        shared across every repetition of :math:`\mathcal{W}`.

    .. seealso:: :func:`~.qubitization_thc_wires`, :class:`~.SuperpositionTHC`,
        :class:`~.AliasSamplingTHC`, :class:`~.SelectTHC`.

    Args:
        zeta (tuple[tuple[float]]): the THC central tensor of shape ``(M, M)``, provided as
            a nested tuple (use ``tuple(map(tuple, arr))`` to convert an array). Must be
            symmetric
        t_ell (tuple[float]): the eigenvalues of the modified one-body matrix :math:`T'` as
            a tuple of length ``N // 2``. See the warning above
        chi (tuple[tuple[float]]): the THC leaf matrix of shape ``(M, N // 2)``, provided as
            a nested tuple
        t_eigenvectors (tuple[tuple[float]]): the eigenvectors of :math:`T'` as columns,
            shape ``(N // 2, N // 2)``, provided as a nested tuple
        aleph (int): the number of bits used to encode the ``PREPARE``
            keep-probabilities. The prepared coefficients match the target up to a
            discretization error that decreases as ``aleph`` grows
        beth (int): bits of precision per Givens angle in ``SELECT``
        system_wires (WiresLike): the ``N`` spin orbitals, spin-blocked: the ``N/2``
            spin-down spatial orbitals followed by the ``N/2`` spin-up ones
        index_wires (WiresLike): ``2 * ceil(log2(M + 1))`` wires, :math:`\mu` followed by
            :math:`\nu`
        prep_garbage_wires (WiresLike): the wires ``PREPARE`` leaves entangled, in the exact
            order and size reported by :func:`~.qubitization_thc_wires`
        gradient_wires (WiresLike): the ``beth + 1`` wires holding the phase gradient state
        work_wires (WiresLike): clean scratch, returned to :math:`\lvert 0 \rangle`
        num_batches (int): the number of batches the ``SELECT`` Givens angles are loaded
            in. The default of ``1`` loads all of them at once; larger values shrink
            ``work_wires`` at the cost of more ``QROM`` loads

    Raises:
        ValueError: if an array has the wrong shape, if a register has the wrong size, if
            two registers share a wire, or if ``PREPARE`` cannot reach unit success
            probability

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        M, N, aleph, beth = 2, 2, 1, 1
        rng = np.random.default_rng(0)
        zeta = rng.standard_normal((M, M))
        zeta = (zeta + zeta.T) / 2
        chi = rng.standard_normal((M, N // 2))
        t_ell = rng.standard_normal(N // 2)
        t_eigenvectors = np.eye(N // 2)

        sizes = qp.qubitization_thc_wires(M, N, aleph, beth)
        wires = qp.registers(sizes)
        gradient = wires["gradient_wires"]

        def gradient_state():
            for j, w in enumerate(gradient):
                qp.Hadamard(w)
                qp.PhaseShift(-2 * np.pi * 2 ** (len(gradient) - 1 - j) / 2 ** len(gradient), w)

        @qp.transforms.decompose(stopping_condition=lambda op: len(op.wires) <= 3)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            gradient_state()
            qp.QubitizationTHC(
                tuple(map(tuple, zeta)), tuple(t_ell), tuple(map(tuple, chi)),
                tuple(map(tuple, t_eigenvectors)), aleph, beth,
                wires["system_wires"], wires["index_wires"], wires["prep_garbage_wires"],
                gradient, wires["work_wires"],
            )
            qp.adjoint(gradient_state)()
            return qp.probs(wires=wires["index_wires"] + wires["prep_garbage_wires"])

    The first entry of the returned distribution is the probability that the reflected
    register returns to :math:`\lvert \vec 0 \rangle`, i.e. the squared norm of
    :math:`(\hat{\mathcal{H}} / \lambda) \lvert \psi \rangle`.

    """

    wire_argnames = (
        "system_wires",
        "index_wires",
        "prep_garbage_wires",
        "gradient_wires",
        "work_wires",
    )
    compilable_argnames = ("zeta", "t_ell", "chi", "t_eigenvectors", "aleph", "beth", "num_batches")
    arg_specs = {
        "system_wires": Wire[-1],
        "index_wires": Wire[-1],
        "prep_garbage_wires": Wire[-1],
        "gradient_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        zeta,
        t_ell,
        chi,
        t_eigenvectors,
        aleph,
        beth,
        system_wires: WiresLike,
        index_wires: WiresLike,
        prep_garbage_wires: WiresLike,
        gradient_wires: WiresLike,
        work_wires: WiresLike,
        num_batches=1,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        _validate_zeta(zeta)
        _validate_t_ell(t_ell)
        M, n_half = _validate_select_data(chi, t_eigenvectors)
        N = 2 * n_half
        # ``M`` and ``N`` come from ``chi``, so this also checks the two PREPARE arrays
        # against the two SELECT ones.
        _build_thc_pairs(M, N, zeta, t_ell)

        # PREPARE is exact only when the single amplitude-amplification round of
        # SuperpositionTHC reaches unit success probability. Otherwise the garbage branch is
        # skipped by SELECT but folded back onto |0> by PREPARE^dagger.
        n = _num_index_wires(M)
        d = n_half + M * (M + 1) // 2
        if d < 2 ** (2 * n - 2):
            raise ValueError(
                f"PREPARE cannot reach unit success probability for M={M}, N={N}: the valid "
                f"index set has size d = {d}, below the 2 ** (2 * ceil(log2(M + 1)) - 2) = "
                f"{2 ** (2 * n - 2)} needed by SuperpositionTHC's single amplification round. "
                f"Increase M towards 2 ** {n} - 1 = {2**n - 1}."
            )

        sizes = qubitization_thc_wires(M, N, aleph, beth, num_batches)
        registers = {
            "system_wires": Wires(system_wires),
            "index_wires": Wires(index_wires),
            "prep_garbage_wires": Wires(prep_garbage_wires),
            "gradient_wires": Wires(gradient_wires),
            "work_wires": Wires(work_wires),
        }
        for name in ("system_wires", "index_wires", "prep_garbage_wires", "gradient_wires"):
            if len(registers[name]) != sizes[name]:
                raise ValueError(
                    f"{name} must have exactly {sizes[name]} wires for M={M}, N={N}, "
                    f"aleph={aleph}, beth={beth}; got {len(registers[name])}."
                )
        if len(registers["work_wires"]) < sizes["work_wires"]:
            raise ValueError(
                f"work_wires must have at least {sizes['work_wires']} wires for M={M}, N={N}, "
                f"aleph={aleph}, beth={beth}; got {len(registers['work_wires'])}."
            )
        validate_no_wire_overlaps(registers)

        super().__init__(
            zeta, t_ell, chi, t_eigenvectors, aleph, beth, *registers.values(), num_batches
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return (
            self.system_wires
            + self.index_wires
            + self.prep_garbage_wires
            + self.gradient_wires
            + self.work_wires
        )


def _qubitization_thc_resources(
    zeta,
    t_ell,
    chi,
    t_eigenvectors,
    aleph,
    beth,
    system_wires,
    index_wires,
    prep_garbage_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Return the top-level resources of the QubitizationTHC decomposition."""
    M = len(zeta)
    N = 2 * len(chi[0])
    n_index, n_garbage = len(index_wires), len(prep_garbage_wires)
    n_work = len(work_wires)

    # The sub-register sizes only depend on how the input registers are split, so the
    # split is replayed on placeholder labels to keep it in sync with the decomposition.
    offsets = np.cumsum([0, n_index, n_garbage, n_work])
    registers = _prepare_registers(
        M,
        N,
        aleph,
        *(range(int(start), int(stop)) for start, stop in zip(offsets[:-1], offsets[1:])),
    )
    n = len(registers["mu_wires"])

    superposition = SuperpositionTHC(
        M, N, Wire[n], Wire[n], Wire[len(registers["superposition_work"])]
    )
    alias_args = (M, N, zeta, t_ell, Wire[n], Wire[n], Wire[1], Wire[len(registers["alias_work"])])
    alias = AliasSamplingTHC(*alias_args, aleph, apply_sign=True)
    alias_adjoint = AliasSamplingTHC(*alias_args, aleph, apply_sign=False)
    select = SelectTHC(
        chi,
        t_eigenvectors,
        beth,
        Wire[len(system_wires)],
        Wire[n_index],
        Wire[5],
        Wire[len(gradient_wires)],
        Wire[n_work],
        num_batches,
    )
    num_reflected = len(registers["reflected"])
    reflection = FlipSign([0] * num_reflected, Wire[num_reflected], work_wires=Wire[n_work])

    return {
        superposition: 1,
        adjoint(superposition): 1,
        alias: 1,
        adjoint(alias_adjoint): 1,
        Hadamard: 4,
        select: 1,
        reflection: 1,
        GlobalPhase: 1,
    }


@register_resources(_qubitization_thc_resources)
def _qubitization_thc_decomp(
    zeta,
    t_ell,
    chi,
    t_eigenvectors,
    aleph,
    beth,
    system_wires,
    index_wires,
    prep_garbage_wires,
    gradient_wires,
    work_wires,
    num_batches=1,
    **_,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    M = len(zeta)
    N = 2 * len(chi[0])

    registers = _prepare_registers(M, N, aleph, index_wires, prep_garbage_wires, work_wires)
    mu_wires, nu_wires = registers["mu_wires"], registers["nu_wires"]
    spin_wires = registers["spin_wires"]

    # PREPARE. The sign of each LCU coefficient must be applied an *odd* number of times
    # between PREPARE and PREPARE^dagger. AliasSamplingTHC applies it as a Z on the sign
    # qubit, so the adjoint below switches it off: keeping it on both sides would square
    # the sign away and block encode the coefficient *magnitudes* instead. The Z is
    # diagonal and SELECT never touches the sign qubit, so dropping it from the adjoint
    # still returns every PREPARE auxiliary wire to |0>.
    SuperpositionTHC(M, N, mu_wires, nu_wires, registers["superposition_work"])
    AliasSamplingTHC(
        M,
        N,
        zeta,
        t_ell,
        mu_wires,
        nu_wires,
        registers["edge_flag"],
        registers["alias_work"],
        aleph,
        apply_sign=True,
    )
    # The two spin flags are the |+> controls that route each V onto a spin sector.
    for wire in spin_wires:
        Hadamard(wire)

    SelectTHC(
        chi,
        t_eigenvectors,
        beth,
        system_wires,
        list(index_wires),
        [
            registers["success_flag"],
            registers["edge_flag"],
            registers["swap_flag"],
            spin_wires[0],
            spin_wires[1],
        ],
        gradient_wires,
        work_wires,
        num_batches=num_batches,
    )

    # PREPARE^dagger. Hadamard is self-inverse, so only the two templates are adjointed.
    for wire in spin_wires:
        Hadamard(wire)
    adjoint(
        AliasSamplingTHC(
            M,
            N,
            zeta,
            t_ell,
            mu_wires,
            nu_wires,
            registers["edge_flag"],
            registers["alias_work"],
            aleph,
            apply_sign=False,
        )
    )
    adjoint(SuperpositionTHC(M, N, mu_wires, nu_wires, registers["superposition_work"]))

    # R = 2|0><0| - I on the full PREPARE register. The global sign is fixed so that the
    # |0> block is + H / lambda: the sign flip of |0> that a bare I - 2|0><0| would give
    # is exactly what SELECT's rewriting of n = (1 - V) / 2 already supplies.
    reflected = registers["reflected"]

    # SELECT and PREPARE both restore work_wires, so they are zeroed auxiliary wires here
    # and make the multi-controlled Z much cheaper.
    FlipSign([0] * len(reflected), reflected, work_wires=work_wires)
    GlobalPhase(np.pi)


def _ctrl_qubitization_thc_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    """Return the top-level resources of the controlled QubitizationTHC decomposition."""
    # pylint: disable=unused-argument
    zeta, t_ell, chi, t_eigenvectors = base.zeta, base.t_ell, base.chi, base.t_eigenvectors
    aleph, beth = base.aleph, base.beth
    M = len(zeta)
    N = 2 * len(chi[0])
    n_index, n_garbage = len(base.index_wires), len(base.prep_garbage_wires)
    n_work = len(base.work_wires)

    # The sub-register sizes only depend on how the input registers are split, so the
    # split is replayed on placeholder labels to keep it in sync with the decomposition.
    offsets = np.cumsum([0, n_index, n_garbage, n_work])
    registers = _prepare_registers(
        M,
        N,
        aleph,
        *(range(int(start), int(stop)) for start, stop in zip(offsets[:-1], offsets[1:])),
    )
    n = len(registers["mu_wires"])

    superposition = SuperpositionTHC(
        M, N, Wire[n], Wire[n], Wire[len(registers["superposition_work"])]
    )
    alias_args = (M, N, zeta, t_ell, Wire[n], Wire[n], Wire[1], Wire[len(registers["alias_work"])])
    alias = AliasSamplingTHC(*alias_args, aleph, apply_sign=True)
    alias_adjoint = AliasSamplingTHC(*alias_args, aleph, apply_sign=False)
    ctrl_kwargs = {
        "control": Wire[len(control_wires)],
        "work_wires": Wire[len(work_wires)],
        "work_wire_type": work_wire_type,
    }
    ctrl_select = ctrl(
        SelectTHC(
            chi,
            t_eigenvectors,
            beth,
            Wire[len(base.system_wires)],
            Wire[n_index],
            Wire[5],
            Wire[len(base.gradient_wires)],
            Wire[n_work],
            base.num_batches,
        ),
        **ctrl_kwargs,
    )
    num_reflected = len(registers["reflected"])
    ctrl_reflection = ctrl(
        FlipSign([0] * num_reflected, Wire[num_reflected], work_wires=Wire[n_work]), **ctrl_kwargs
    )

    return {
        superposition: 1,
        adjoint(superposition): 1,
        alias: 1,
        ctrl(Z(Wire[1]), **ctrl_kwargs): 1,
        adjoint(alias_adjoint): 1,
        Hadamard: 4,
        ctrl_select: 1,
        ctrl_reflection: 1,
        ctrl(
            Z(Wire[1]),
            control=Wire[len(control_wires) - 1],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1,
        Z: 1,
    }


@register_resources(_ctrl_qubitization_thc_resource, exact=False)
def _ctrl_qubitization_thc_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    # pylint: disable=unused-argument
    zeta = base.zeta
    t_ell = base.t_ell
    chi = base.chi
    t_eigenvectors = base.t_eigenvectors
    aleph = base.aleph
    beth = base.beth
    M = len(zeta)
    N = 2 * len(chi[0])

    registers = _prepare_registers(
        M, N, aleph, base.index_wires, base.prep_garbage_wires, base.work_wires
    )
    mu_wires, nu_wires = registers["mu_wires"], registers["nu_wires"]
    spin_wires = registers["spin_wires"]

    # PREPARE. The sign of each LCU coefficient must be applied an *odd* number of times
    # between PREPARE and PREPARE^dagger. AliasSamplingTHC applies it as a Z on the sign
    # qubit, so the adjoint below switches it off: keeping it on both sides would square
    # the sign away and block encode the coefficient *magnitudes* instead. The Z is
    # diagonal and SELECT never touches the sign qubit, so dropping it from the adjoint
    # still returns every PREPARE auxiliary wire to |0>.
    SuperpositionTHC(M, N, mu_wires, nu_wires, registers["superposition_work"])
    AliasSamplingTHC(
        M,
        N,
        zeta,
        t_ell,
        mu_wires,
        nu_wires,
        registers["edge_flag"],
        registers["alias_work"],
        aleph,
        apply_sign=True,
    )
    # The two spin flags are the |+> controls that route each V onto a spin sector.
    for wire in spin_wires:
        Hadamard(wire)

    ctrl(
        SelectTHC(
            chi,
            t_eigenvectors,
            beth,
            base.system_wires,
            list(base.index_wires),
            [
                registers["success_flag"],
                registers["edge_flag"],
                registers["swap_flag"],
                spin_wires[0],
                spin_wires[1],
            ],
            base.gradient_wires,
            base.work_wires,
            num_batches=base.num_batches,
        ),
        control=control_wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )

    # PREPARE^dagger. Hadamard is self-inverse, so only the two templates are adjointed.
    for wire in spin_wires:
        Hadamard(wire)

    # If the control condition does _not_ trigger, we undo AliasSamplingTHC only up to the
    # difference between apply_sign=True and apply_sign=False. So we apply the corresponding
    # PauliZ here unconditionally, and then once more under the control condition:
    # - if control activates, the two inserted ops (Z and ctrl(Z)) cancel and everything is correct
    # - if control does not activate, only the Z triggers, undoing the Z applied by the difference
    # between AliasSamplingTHC and adjoint(AliasSamplingTHC).
    alias_sizes = alias_sampling_thc_wires(M, N, aleph)
    sign_wire = registers["alias_work"][alias_sizes["sign_wire"]]
    Z(sign_wire)
    ctrl(Z(sign_wire), control=control_wires, work_wires=work_wires, work_wire_type=work_wire_type)
    adjoint(
        AliasSamplingTHC(
            M,
            N,
            zeta,
            t_ell,
            mu_wires,
            nu_wires,
            registers["edge_flag"],
            registers["alias_work"],
            aleph,
            apply_sign=False,
        )
    )
    adjoint(SuperpositionTHC(M, N, mu_wires, nu_wires, registers["superposition_work"]))

    # R = 2|0><0| - I on the full PREPARE register. The global sign is fixed so that the
    # |0> block is + H / lambda: the sign flip of |0> that a bare I - 2|0><0| would give
    # is exactly what SELECT's rewriting of n = (1 - V) / 2 already supplies.
    reflected = registers["reflected"]

    # SELECT and PREPARE both restore work_wires, so they are zeroed auxiliary wires here
    # and make the multi-controlled Z much cheaper.
    ctrl(
        FlipSign([0] * len(reflected), reflected, work_wires=base.work_wires),
        control=control_wires,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
    if len(control_wires) > 1:
        ctrl(
            Z(control_wires[-1]),
            control=control_wires[:-1],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    else:
        Z(control_wires[-1])


add_decomps(QubitizationTHC, _qubitization_thc_decomp)
add_decomps("C(QubitizationTHC)", flip_zero_control2(_ctrl_qubitization_thc_decomp))
