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
"""Qubitization walk operator for the tensor hypercontracted (THC) Hamiltonian."""

from functools import partial

import numpy as np

import pennylane as qp
from pennylane.labs.templates.alias_sampling_thc import (
    _num_address_wires,
    _num_index_wires,
    alias_sampling_thc,
    alias_sampling_thc_wires,
)
from pennylane.labs.templates.select_thc import select_thc, select_thc_wires
from pennylane.labs.templates.superposition_thc import SuperpositionTHC
from pennylane.wires import Wires


def _alias_wire_split(M, N, aleph):
    r"""Split :func:`alias_sampling_thc`'s work register into garbage and clean wires.

    That function takes a single ``work_wires`` register, but only part of it is garbage:
    with ``f = n_d + 2 + 2 n + 2 aleph`` its layout ends in ``alt_flag`` at ``f``,
    ``swap_flag`` at ``f + 1``, ``alt_edge_flag`` at ``f + 2``, and the comparator/QROM
    scratch from ``f + 3``. The adjoint inequality test returns ``alt_flag`` and the
    comparator scratch to :math:`\lvert 0 \rangle`, so those ``aleph`` wires do *not* have
    to live inside the reflected register and can be drawn from the shared clean pool
    instead. Everything below ``f``, plus ``swap_flag`` and ``alt_edge_flag``, stays
    entangled and must be reflected.

    Returns:
        tuple[int, int]: ``(first_clean, num_garbage)``, the index ``f`` of the first clean
        wire and the number of garbage wires ``f + 2``
    """
    first_clean = _num_address_wires(M, N) + 2 + 2 * _num_index_wires(M) + 2 * aleph
    return first_clean, first_clean + 2


def qubitization_thc_wires(M, N, aleph, beth, num_batches=1):
    r"""Return the wire counts required by :func:`qubitization_thc`.

    Args:
        M (int): the THC rank
        N (int): the number of spin orbitals. Requires ``N // 2 <= M + 1``
        aleph (int): the number of bits used to encode the ``PREPARE``
            keep-probabilities
        beth (int): bits of precision per Givens angle in ``SELECT``
        num_batches (int): the number of batches the Givens angles are loaded in

    Returns:
        dict: ``{"system_wires": N, "index_wires": 2 * n, "prep_garbage_wires": n_prep,
        "gradient_wires": beth, "work_wires": n_work}``

        * ``system_wires`` (``N``): the spin orbitals, the ``N/2`` spin-down spatial
          orbitals followed by the ``N/2`` spin-up ones
        * ``index_wires`` (``2 * ceil(log2(M + 1))``): the LCU index :math:`\mu` followed by
          :math:`\nu`, i.e. the register ``PREPARE`` writes the coefficient amplitudes on and
          ``SELECT`` reads to pick which :math:`V` to apply
        * ``prep_garbage_wires``: *not* a state preparation register. These are the wires
          ``PREPARE`` leaves entangled with the index and that therefore have to be reflected
          along with it, laid out as the
          :class:`~pennylane.labs.templates.SuperpositionTHC` work register, then whatever
          :func:`~pennylane.labs.templates.alias_sampling_thc` garbage it cannot supply,
          then the two spin flags. The spin flags are the exception to the name: they are
          genuine LCU index wires, placed here because ``index_wires`` is sized exactly
          ``2 n`` by :func:`~pennylane.labs.templates.select_thc_wires`. Exact: the
          reflection acts on ``index_wires + prep_garbage_wires``, so a spare wire here
          changes the walk operator
        * ``gradient_wires`` (``beth``): the phase gradient register, which must be
          prepared by the caller and is left unchanged
        * ``work_wires``: the shared clean pool, returned to :math:`\lvert 0 \rangle`. This
          is a minimum; extra wires are forwarded to ``SELECT``, to ``PREPARE``'s ``QROM``
          and to the multi-controlled :math:`Z` of the reflection to lower their T counts

    .. note::

        Every zeroed auxiliary wire is shared rather than duplicated, which is what keeps
        ``prep_garbage_wires`` well below the sum of the sub-template registers:

        * ``SuperpositionTHC`` returns all but three of its ``3 n + 5`` work wires to
          :math:`\lvert 0 \rangle` (the exceptions are its indices ``0``, ``3`` and ``6``),
          so those ``3 n + 2`` wires are recycled as ``alias_sampling_thc`` garbage
        * of ``alias_sampling_thc``'s work register, only
          ``n_d + 2 n + 2 aleph + 4`` wires stay entangled; the remaining ``aleph``
          (its ``alt_flag`` and comparator scratch) are restored, so they are taken from
          ``work_wires`` instead of ``prep_garbage_wires``. This both saves ``aleph`` qubits and
          removes ``aleph`` controls from the reflection
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

    >>> from pennylane.labs.templates import qubitization_thc_wires
    >>> qubitization_thc_wires(M=2, N=2, aleph=1, beth=1)
    {'system_wires': 2, 'index_wires': 4, 'prep_garbage_wires': 18, 'gradient_wires': 1, 'work_wires': 1}
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
        "gradient_wires": beth,
        # One shared clean pool: SELECT's scratch, the alias comparator/QROM scratch that
        # ``alias_sampling_thc`` restores, and at least one zeroed auxiliary wire for the
        # reflection's multi-controlled Z.
        "work_wires": max(select_sizes["work_wires"], aleph, 1),
    }


def qubitization_thc(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
):
    r"""Apply the qubitization walk operator of a tensor hypercontracted (THC) Hamiltonian.

    This composes the three THC oracles into

    .. math::

        \mathcal{W} = \mathcal{R} \cdot \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT}
        \cdot \mathrm{PREPARE} ,
        \qquad \mathcal{R} = 2 \lvert \vec 0 \rangle \langle \vec 0 \rvert - I ,

    with :math:`\mathcal{R}` the reflection on ``index_wires + prep_garbage_wires``, following
    `Lee et al. (2021) <https://arxiv.org/abs/2011.03494>`_ (Figs. 3, 5 and 7).
    ``PREPARE`` is :class:`~pennylane.labs.templates.SuperpositionTHC` followed by
    :func:`~pennylane.labs.templates.alias_sampling_thc` and a ``Hadamard`` on each of the
    two spin flags; ``SELECT`` is :func:`~pennylane.labs.templates.select_thc`.

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
        amplitude-amplification round of
        :class:`~pennylane.labs.templates.SuperpositionTHC` is exact. Otherwise the
        leftover garbage branch is *not* acted on by ``SELECT``, yet is mapped back onto
        :math:`\lvert \vec 0 \rangle` by ``PREPARE``:math:`^\dagger`, which contaminates
        the block. A ``ValueError`` is raised in that case rather than returning a
        silently wrong block encoding.

    .. note::

        ``gradient_wires`` must be prepared by the caller in the phase gradient state

        .. math::

            \lvert \phi \rangle = \frac{1}{\sqrt{2^{\mathrm{beth}}}}
            \sum_{k = 0}^{2^{\mathrm{beth}} - 1}
            e^{-2 \pi i k / 2^{\mathrm{beth}}} \lvert k \rangle ,

        a product state that ``beth`` ``Hadamard`` and ``beth`` ``PhaseShift`` gates
        prepare. The walk leaves it unchanged, so a single register is prepared once and
        shared across every repetition of :math:`\mathcal{W}`.

    .. seealso:: :func:`~pennylane.labs.templates.qubitization_thc_wires`,
        :func:`~pennylane.labs.templates.alias_sampling_thc`,
        :func:`~pennylane.labs.templates.select_thc`.

    Args:
        zeta (tensor_like): the THC central tensor, shape ``(M, M)``. Must be symmetric
        t_ell (tensor_like): the eigenvalues of the modified one-body matrix :math:`T'`,
            shape ``(N // 2,)``. See the warning above
        chi (tensor_like): the THC leaf matrix, shape ``(M, N // 2)``
        t_eigenvectors (tensor_like): the eigenvectors of :math:`T'` as columns, shape
            ``(N // 2, N // 2)``
        aleph (int): the number of bits used to encode the ``PREPARE``
            keep-probabilities. The prepared coefficients match the target up to a
            discretization error that decreases as ``aleph`` grows
        beth (int): bits of precision per Givens angle in ``SELECT``
        system_wires (WiresLike): the ``N`` spin orbitals, spin-blocked: the ``N/2``
            spin-down spatial orbitals followed by the ``N/2`` spin-up ones
        index_wires (WiresLike): ``2 * ceil(log2(M + 1))`` wires, :math:`\mu` followed by
            :math:`\nu`
        prep_garbage_wires (WiresLike): the wires ``PREPARE`` leaves entangled, in the exact order and size
            reported by :func:`qubitization_thc_wires`
        gradient_wires (WiresLike): the ``beth`` wires holding the phase gradient state
        work_wires (WiresLike): clean scratch, returned to :math:`\lvert 0 \rangle`
        num_batches (int): the number of batches the ``SELECT`` Givens angles are loaded
            in. The default of ``1`` loads all of them at once; larger values shrink
            ``work_wires`` at the cost of more ``QROM`` loads

    Raises:
        ValueError: if an array has the wrong shape, if a register has the wrong size, or
            if ``PREPARE`` cannot reach unit success probability

    **Example**

    .. code-block:: python

        from functools import partial

        import numpy as np
        import pennylane as qp
        from pennylane.labs.templates import qubitization_thc, qubitization_thc_wires

        M, N, aleph, beth = 2, 2, 1, 1
        rng = np.random.default_rng(0)
        zeta = rng.standard_normal((M, M))
        zeta = (zeta + zeta.T) / 2
        chi = rng.standard_normal((M, N // 2))
        t_ell = rng.standard_normal(N // 2)
        t_eigenvectors = np.eye(N // 2)

        sizes = qubitization_thc_wires(M, N, aleph, beth)
        wires = qp.registers(sizes)

        def gradient_state():
            for j, w in enumerate(wires["gradient_wires"]):
                qp.Hadamard(w)
                qp.PhaseShift(-2 * np.pi * 2 ** (beth - 1 - j) / 2**beth, wires=w)

        @partial(qp.transforms.decompose, stopping_condition=lambda op: len(op.wires) <= 3)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            gradient_state()
            qubitization_thc(
                zeta, t_ell, chi, t_eigenvectors, aleph, beth,
                wires["system_wires"], wires["index_wires"], wires["prep_garbage_wires"],
                wires["gradient_wires"], wires["work_wires"],
            )
            qp.adjoint(gradient_state)()
            return qp.probs(wires=wires["index_wires"] + wires["prep_garbage_wires"])

    .. warning::

        Two things about that device line are load-bearing.

        First, ``wires`` is deliberately left unset. ``PREPARE``'s ``QROM`` requests dynamic
        work wires when it decomposes, and those come on top of the registers reported by
        :func:`qubitization_thc_wires`, so a device fixed at ``sum(sizes.values())`` raises
        ``AllocationError``. Measured peak of concurrent dynamic wires: ``0`` at
        ``M = 1, N = 2``, ``1`` at ``M = 2, N = 2``, and ``3`` at ``M = 7, N = 8``. Leaving
        ``wires`` unset lets the device size itself.

        On a fixed-width device — which is what ``lightning.qubit`` requires, since it
        cannot take a :class:`~pennylane.allocation.DynamicWire` at all — resolve the
        dynamic requests yourself against an explicit pool:

        .. code-block:: python

            total = sum(sizes.values())
            pool = [total]  # size it to the peak above; 1 is enough at M = 2, N = 2

            @partial(qp.transforms.resolve_dynamic_wires, zeroed=pool)
            @partial(qp.transforms.decompose, stopping_condition=lambda op: len(op.wires) <= 3)
            @qp.qnode(qp.device("default.qubit", wires=total + len(pool)))
            def circuit():
                ...

        A pool smaller than the peak raises ``AllocationError: no wires left to allocate``,
        so the failure is loud rather than silent.

        Second, the :func:`~pennylane.transforms.decompose` wrapper is not an optimization.
        Without it the simulator tries to build the dense matrix of the reflection's
        multi-controlled :math:`Z` and dies with a ``MemoryError`` asking for over 100 TiB
        already at ``M = 2, N = 2``.

    The first entry of the returned distribution is the probability that the reflected
    register returns to :math:`\lvert \vec 0 \rangle`, i.e. the squared norm of
    :math:`(\hat{\mathcal{H}} / \lambda) \lvert \psi \rangle`.
    """
    M = qp.math.shape(zeta)[0]
    n_half = qp.math.shape(chi)[1]
    N = 2 * n_half
    n = int(qp.math.ceil_log2(M + 1))

    if qp.math.shape(zeta) != (M, M):
        raise ValueError(f"zeta must be square; got {qp.math.shape(zeta)}.")
    if qp.math.shape(chi) != (M, n_half):
        raise ValueError(
            f"chi must have shape ({M}, {n_half}) to match zeta and t_eigenvectors; "
            f"got {qp.math.shape(chi)}."
        )
    if qp.math.shape(t_eigenvectors) != (n_half, n_half):
        raise ValueError(
            f"t_eigenvectors must have shape ({n_half}, {n_half}); "
            f"got {qp.math.shape(t_eigenvectors)}."
        )

    # ``PREPARE`` is exact only when the single amplitude-amplification round of
    # SuperpositionTHC reaches unit success probability. Otherwise the garbage branch is
    # skipped by ``SELECT`` but folded back onto |0> by ``PREPARE^dagger``.
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
    }
    for name, register in registers.items():
        if len(register) != sizes[name]:
            raise ValueError(
                f"{name} must have exactly {sizes[name]} wires for M={M}, N={N}, "
                f"aleph={aleph}, beth={beth}; got {len(register)}."
            )
    work_wires = Wires(work_wires)
    if len(work_wires) < sizes["work_wires"]:
        raise ValueError(
            f"work_wires must have at least {sizes['work_wires']} wires for M={M}, N={N}, "
            f"aleph={aleph}, beth={beth}; got {len(work_wires)}."
        )

    index = list(registers["index_wires"])
    mu_wires, nu_wires = index[:n], index[n:]

    # Split ``prep_garbage_wires`` into the two work registers and the two spin flags. The layout
    # is the one documented by ``qubitization_thc_wires``.
    garbage = list(registers["prep_garbage_wires"])
    n_sup = alias_sampling_thc_wires(M, N, aleph)["superposition_work_wires"]
    first_clean, n_garbage = _alias_wire_split(M, N, aleph)
    superposition_work = garbage[:n_sup]
    spin_wires = garbage[-2:]

    # ``alias_sampling_thc`` takes one work register that mixes garbage and restored wires.
    # Feed the garbage slots from the reflected register and the restored slots from the
    # shared clean pool, so ``prep_garbage_wires`` carries no wire that ends in |0>.
    #
    # ``SuperpositionTHC`` returns every work wire to |0> except its three flags (indices
    # 0, 3 and 6), so the rest are reflected wires that are free to carry alias garbage;
    # the alias adjoint clears them again before ``SuperpositionTHC``'s adjoint runs.
    garbage_pool = (
        [w for i, w in enumerate(superposition_work) if i not in (0, 3, 6)] + garbage[n_sup:-2]
    )[:n_garbage]
    clean_pool = list(work_wires)
    alias_work = (
        garbage_pool[:first_clean]  # contiguous address, QROM output, sigma sample
        + [clean_pool[0]]  # alt_flag: restored by the adjoint comparator
        + garbage_pool[first_clean:]  # swap_flag, alt_edge_flag
        + clean_pool[1:]  # comparator scratch, then extra wires for the QROM
    )

    # ``SuperpositionTHC`` flags: work_wires[3] is true when nu = M (the one-body
    # sentinel column) and work_wires[6] when the superposition was prepared correctly.
    edge_flag = superposition_work[3]
    success_flag = superposition_work[6]
    # ``alias_sampling_thc`` holds its mu <-> nu symmetrization flag one wire above the
    # inequality-test flag; see the wire map in that function.
    swap_flag = alias_work[first_clean + 1]

    # ``SuperpositionTHC`` runs first, while the shared pool is still untouched, and it
    # forwards anything past its fixed 3n+5 layout to its comparators and multi-controlled
    # gates. Given only the minimum it has *zero* scratch for those, so append the pool.
    superposition_all = superposition_work + clean_pool

    def prepare(apply_sign):
        SuperpositionTHC(M, N, mu_wires, nu_wires, superposition_all)
        alias_sampling_thc(
            M,
            N,
            zeta,
            t_ell,
            mu_wires,
            nu_wires,
            edge_flag,
            alias_work,
            aleph,
            apply_sign=apply_sign,
        )
        # The two spin flags are the |+> controls that route each V onto a spin sector.
        for wire in spin_wires:
            qp.Hadamard(wire)

    # The sign of each LCU coefficient must be applied an *odd* number of times between
    # PREPARE and PREPARE^dagger. ``alias_sampling_thc`` applies it as a Z on the sign
    # qubit, so the adjoint call must switch it off: keeping it on both sides would square
    # the sign away and block encode the coefficient *magnitudes* instead. The Z is
    # diagonal and SELECT never touches the sign qubit, so dropping it from the adjoint
    # still returns every PREPARE auxiliary wire to |0>.
    prepare(apply_sign=True)

    select_thc(
        chi,
        t_eigenvectors,
        beth,
        list(registers["system_wires"]),
        index,
        [success_flag, edge_flag, swap_flag, spin_wires[0], spin_wires[1]],
        list(registers["gradient_wires"]),
        list(work_wires),
        num_batches=num_batches,
    )

    # ``partial`` rather than ``qp.adjoint(prepare)(apply_sign=False)``: under ``qp.qjit``
    # the callable form of ``adjoint`` traces its arguments, which would turn the flag into
    # a JAX tracer and make ``if apply_sign`` fail. Closing over it keeps it static.
    qp.adjoint(partial(prepare, apply_sign=False))()

    # R = 2|0><0| - I on the full PREPARE register. The global sign is fixed so that the
    # |0> block is + H / lambda: the sign flip of |0> that a bare I - 2|0><0| would give
    # is exactly what ``SELECT``'s rewriting of n = (1 - V) / 2 already supplies.
    reflected = index + garbage
    for wire in reflected:
        qp.X(wire)
    # ``SELECT`` and ``PREPARE`` both restore ``work_wires``, so they are zeroed auxiliary wires
    # here and make the multi-controlled Z much cheaper.
    qp.ctrl(
        qp.Z(reflected[-1]),
        control=reflected[:-1],
        control_values=[1] * (len(reflected) - 1),
        work_wires=list(work_wires),
        work_wire_type="zeroed",
    )
    for wire in reflected:
        qp.X(wire)
    qp.GlobalPhase(np.pi)
