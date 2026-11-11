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

from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .alias_sampling_thc import (
    _build_thc_pairs,
    _num_address_wires,
    _num_index_wires,
    _validate_t_ell,
    _validate_zeta,
    alias_sampling_thc_wires,
)
from .select_thc import _validate_select_data, select_thc_wires


def _alias_wire_split(M, N, aleph):
    r"""Split :class:`~.AliasSamplingTHC`'s work register into garbage and clean wires.

    That template takes a single ``work_wires`` register, but only its lower part is
    garbage. With ``f = n_d + 1 + 2 n`` everything up to and including the sampling
    register at ``f + aleph + 2 : f + 2 aleph + 2`` stays entangled and must be reflected:
    the conditional swaps correlate the sampling register with the index registers, so the
    closing ``Hadamard`` layer rotates it rather than returning it to
    :math:`\lvert 0 \rangle`. The inequality-test flag at ``f + 2 aleph + 2`` and the
    comparator/QROM scratch above it *are* restored, so they do not have to live inside the
    reflected register and are drawn from the shared clean pool instead.

    Returns:
        tuple[int, int]: ``(swap_index, num_garbage)``, the index of
        :class:`~.AliasSamplingTHC`'s symmetrization flag within its work register and the
        number of leading garbage wires
    """
    f = _num_address_wires(M, N) + 1 + 2 * _num_index_wires(M)
    return f + aleph + 1, f + 2 * aleph + 2


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
          ``n_d + 2 n + 2 aleph + 3`` wires stay entangled; the rest (its inequality-test
          flag, the comparator scratch and the ``QROM`` scratch) are restored, so they are taken from
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
    {'system_wires': 2, 'index_wires': 4, 'prep_garbage_wires': 17, 'gradient_wires': 2, 'work_wires': 2}
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

        ``gradient_wires`` must be prepared by the caller in the phase gradient state

        .. math::

            \lvert \phi \rangle = \frac{1}{\sqrt{2^{\mathrm{beth} + 1}}}
            \sum_{k = 0}^{2^{\mathrm{beth} + 1} - 1}
            e^{-2 \pi i k / 2^{\mathrm{beth} + 1}} \lvert k \rangle ,

        which can be done with :class:`~.PhaseGradientStatePrep`.
        The walk leaves it unchanged, so a single register is prepared once and
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
        ValueError: if an array has the wrong shape, if a register has the wrong size, or if
            two registers share a wire.

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


        @qp.transforms._tape_decompose(stopping_condition=lambda op: len(op.wires) <= 3)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.PhaseGradientStatePrep(gradient)
            qp.QubitizationTHC(
                tuple(map(tuple, zeta)), tuple(t_ell), tuple(map(tuple, chi)),
                tuple(map(tuple, t_eigenvectors)), aleph, beth,
                wires["system_wires"], wires["index_wires"], wires["prep_garbage_wires"],
                gradient, wires["work_wires"],
            )

            return qp.probs(wires=wires["index_wires"] + wires["prep_garbage_wires"])

    The first entry of the returned distribution is the probability that the reflected
    register returns to :math:`\lvert \vec 0 \rangle`, i.e. the squared norm of
    :math:`(\hat{\mathcal{H}} / \lambda) \lvert \psi \rangle`.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.QubitizationTHC
            :stacked:

            .. decomposition-rule:: _qubitization_thc_decomp
                :image: /_static/decomposition_thumbnails/qubitization_thc_walk.svg

                :class:`~.SuperpositionTHC` | :math:`1`
                :class:`Adjoint(SuperpositionTHC) <.SuperpositionTHC>` | :math:`1`
                :class:`~.AliasSamplingTHC` | :math:`1`
                :class:`Adjoint(AliasSamplingTHC) <.AliasSamplingTHC>` | :math:`1`
                :class:`~.Hadamard` | :math:`4`
                :class:`~.Z` | :math:`1`
                :class:`~.SelectTHC` | :math:`1`
                :class:`~.FlipSign` | :math:`1`
                :class:`~.GlobalPhase` | :math:`1`

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
