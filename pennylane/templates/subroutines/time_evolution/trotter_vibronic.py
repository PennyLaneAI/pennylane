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
"""Contains the ``TrotterVibronic`` template for vibronic Hamiltonian time evolution."""

import warnings
from collections import defaultdict
from functools import reduce

import numpy as np

from pennylane import capture, compiler, math
from pennylane.allocation import allocate
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, Hadamard, X, adjoint, cond, ctrl
from pennylane.typing import AbstractWires, Float, Wire
from pennylane.wires import Wires, WiresLike

from ..aqft import AQFT
from ..arithmetic.incrementer import Incrementer
from ..arithmetic.out_multiplier import OutMultiplier
from ..arithmetic.semi_adder import SemiAdder
from ..arithmetic.signed_out_multiplier import SignedOutMultiplier, _twos_complement_helper
from ..arithmetic.signed_out_square import SignedOutSquare
from ..qrom import QROM

# Keys expected in the dense vibronic Hamiltonian dictionary.
HAMILTONIAN_KEYS = ("constant", "linear", "quadratic", "kinetic")


def _aqft(order, wires):
    """Construct an :class:`~.AQFT`, suppressing its "use QFT instead" advisory.

    ``TrotterVibronic`` uses the AQFT as its position <-> momentum transform and deliberately
    allows the full order (``aqft_order=None`` resolves to ``k - 1``), which always trips AQFT's
    advice to use ``QFT`` instead. That advisory is not actionable here, so it is silenced.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*QFT class is recommended.*")
        return AQFT(order=order, wires=wires)


class TrotterVibronic(Operator2):
    r"""Second-order Trotter circuit for vibronic Hamiltonian simulation using phase-gradient
    arithmetic.

    This template realizes :math:`U \approx e^{-iHt}` for a vibronic Hamiltonian, using the
    phase-gradient arithmetic construction of `Motlagh et al, arXiv:2411.13669
    <https://arxiv.org/abs/2411.13669>`__. The Hamiltonian acts on :math:`N` electronic states
    (represented on :math:`n = \lceil\log_2(N)\rceil` qubits) and :math:`M` vibrational modes,
    each discretized on a grid represented with :math:`k` qubits. The vibronic Hamiltonian is
    partitioned into position fragments (diagonal in a fragment-specific electronic basis and
    polynomial in the mode positions up to second order) and a single trailing kinetic fragment
    (quadratic in the mode momenta).

    Args:
        evolution_time (float): time :math:`t` for which to evolve under the vibronic Hamiltonian.
        num_trotter_steps (int): number of second-order Trotter steps to use.
        hamiltonian (dict): the vibronic Hamiltonian as a dictionary of dense coefficient tensors.
            The expected keys and shapes are

            * ``"constant"``: ``(F, N, N)`` -- the constant (mode-independent) coefficients of the
              ``F`` position fragments;
            * ``"linear"``: ``(F, N, N, M)`` -- the linear-in-position coefficients;
            * ``"quadratic"``: ``(F, N, N, M, M)`` -- the quadratic-in-position coefficients (see
              the Implementation Details section for the diagonal/off-diagonal convention);
            * ``"kinetic"``: ``(N, N, M, M)`` -- the quadratic-in-momentum coefficients of the
              single kinetic fragment.
        electronic_wires (WiresLike): the :math:`n` electronic-state wires.
        vib_wires (WiresLike): the :math:`M \cdot k` vibrational-mode wires, provided as a single
            flattened register. Internally these are reshaped into ``M`` registers of ``k`` wires,
            one per mode, via ``np.array(vib_wires).reshape(M, -1)``.
        phase_gradient_wires (WiresLike): at least :math:`b` wires holding the `phase-gradient
            resource state <https://pennylane.ai/compilation/phase-gradient>`__, assumed to be
            prepared upfront. Extra wires beyond :math:`b` add unused precision headroom.
        coefficient_wires (WiresLike): the :math:`b` wires of the Hamiltonian-coefficient
            register. If omitted (default), they are dynamically allocated in the decomposition,
            with :math:`b` equal to ``len(phase_gradient_wires)``.
        cache_wires (WiresLike): the :math:`2k` cache wires for squared/multiplied mode registers.
            If omitted (default), they are dynamically allocated in the decomposition.
        work_wires (WiresLike): the :math:`\max(n-1, 2k, 2b+2)` work wires for data loading and
            arithmetic. If omitted (default), they are dynamically allocated in the decomposition.
        aqft_order (int): approximation order of the :class:`~.AQFT` used to transform between
            position and momentum space. If ``None`` (default), no approximation is made
            (``aqft_order = k - 1``).

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        n_states, n_modes, k, b = 2, 1, 2, 3
        n = int(np.ceil(np.log2(n_states)))
        # A single diagonal position fragment with non-zero constant/linear coefficients, and a
        # diagonal kinetic fragment.
        constant = np.zeros((1, n_states, n_states))
        constant[0, 0, 0], constant[0, 1, 1] = 0.4, -0.4
        linear = np.zeros((1, n_states, n_states, n_modes))
        linear[0, 0, 0, 0], linear[0, 1, 1, 0] = 0.2, -0.2
        hamiltonian = {
            "constant": constant,
            "linear": linear,
            "quadratic": np.zeros((1, n_states, n_states, n_modes, n_modes)),
            "kinetic": np.einsum(
                "ab,cd->abcd", np.eye(n_states), np.diag(0.3 * np.ones(n_modes))
            ),
        }
        wires = qp.registers({
            "electronic": n, "vib_wires": n_modes * k, "cache": 2 * k,
            "coefficients": b, "phase_gradient": b, "work": max(n - 1, 2 * k, 2 * b + 2),
        })
        all_wires = qp.wires.Wires.all_wires(list(wires.values()))

        @qp.decompose(max_expansion=1)  # to see the top-level sub-templates
        @qp.qnode(qp.device("default.qubit", wires=all_wires))
        def circuit():
            qp.TrotterVibronic(
                evolution_time=1.0, num_trotter_steps=1, hamiltonian=hamiltonian,
                electronic_wires=wires["electronic"], vib_wires=wires["vib_wires"],
                cache_wires=wires["cache"], coefficient_wires=wires["coefficients"],
                phase_gradient_wires=wires["phase_gradient"], work_wires=wires["work"],
                aqft_order=1,
            )
            return qp.probs(wires=wires["electronic"])

    >>> qp.specs(circuit)().resources.quantum_operations
    {'QROM': 6, 'SemiAdder': 2, 'CNOT': 24, 'C(Incrementer)': 4, 'OutMultiplier': 3, 'AQFT': 1, 'SignedOutSquare': 1, 'Adjoint(SignedOutSquare)': 1, 'Adjoint(AQFT)': 1}

    .. details::
        :title: Register Sizes
        :href: register-sizes

        With :math:`N` electronic states (:math:`n = \lceil\log_2(N)\rceil` qubits), :math:`M`
        vibrational modes (:math:`k` qubits each), and :math:`b`-qubit coefficient and
        phase-gradient registers, the wire registers should have the following sizes:

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - argument
             - expected size
             - information content
           * - ``electronic_wires``
             - :math:`n`
             - electronic state
           * - ``vib_wires``
             - :math:`M \cdot k`
             - positions of all vibrational modes (signed), flattened
           * - ``cache_wires``
             - :math:`2k`
             - cached squares/products of modes (signed/unsigned)
           * - ``coefficient_wires``
             - :math:`b`
             - Hamiltonian coefficients (unsigned); optional, dynamically allocated if omitted
           * - ``phase_gradient_wires``
             - :math:`\ge b`
             - phase-gradient state (unsigned)
           * - ``work_wires``
             - :math:`\max(n-1, 2k, 2b+2)`
             - data-loading/arithmetic scratch

    .. details::
        :title: Implementation Details
        :href: implementation-details

        This section shows how the dense ``hamiltonian`` is turned into gates, making every phase
        prefactor explicit, following `Motlagh et al, arXiv:2411.13669
        <https://arxiv.org/abs/2411.13669>`__ (see also this `PennyLane demo
        <https://pennylane.ai/demos/simulating_vibronic_dynamics>`__ for a from-scratch,
        step-by-step implementation). The mode positions :math:`Q_r` and momenta
        :math:`P_r` are the signed (two's-complement) integers on the mode sub-registers of
        ``vib_wires``, and :math:`a` labels an ``electronic_wires`` basis state. Operator products
        below use the standard convention that the rightmost factor is applied first.

        The Hamiltonian is a sum of :math:`F` position fragments and one kinetic fragment,

        .. math::

            H = \sum_{i=0}^{F-1} H_i + H_{\text{kin}},
            \qquad
            H_i = V_i D_i(Q) V_i^\dagger,
            \qquad
            H_{\text{kin}} = \sum_r t_r P_r^2 .

        Here :math:`V_i` (step 3) rotates into a fragment-specific electronic eigenbasis, and
        :math:`D_i(Q)` is the diagonal generator on ``electronic_wires``,

        .. math::

            D_i(Q) = \sum_a \theta_{i,a}(Q) |a\rangle\langle a|,

        whose entries for electronic state :math:`a` are degree-2 polynomials in the mode positions,

        .. math::

            \theta_{i,a}(Q) = c_a + \sum_r l_{a,r} Q_r + \sum_r q_{a,r} Q_r^2
                              + \sum_{r<s} \beta_{a,rs} Q_r Q_s .

        The entries :math:`c, l, q, \beta` are the diagonalized ``constant``/``linear``/``quadratic``
        tensors and :math:`t_r` the diagonal of ``kinetic``. The ``quadratic`` tensor packs both
        :math:`q_{a,r}` (diagonal, ``[..., r, r]``) and :math:`\beta_{a,rs}` (read once from the
        entry with :math:`r < s`). The mirror entry :math:`(s, r)` is ignored, so put the full
        weight on :math:`r < s`. Splitting it evenly between the two would drop half.

        The Hamiltonian must follow the "XOR" fragmentation scheme of the reference above:
        position fragment :math:`i` may only populate the electronic matrix elements at index
        pairs :math:`(j, i \oplus j)` (bitwise XOR), for :math:`i = 0, \dots, F-1`. This is what
        lets :math:`V_i` (step 3) be the fixed single-Hadamard/CNOT circuit derived from the key
        :math:`(0, i)`. A Hamiltonian with a different fragment structure will silently produce an
        incorrect circuit. :math:`N` itself must be a power of 2, since :math:`V_i` is only
        guaranteed orthogonal when it spans the full :math:`2^n`-dimensional electronic space. See
        the demo's `fragmentation and diagonalization
        <https://pennylane.ai/demos/simulating_vibronic_dynamics#fragmentation-and-diagonalization>`__
        section for the general (non-XOR) construction of :math:`V_i`.

        **1. Second-order Trotter.**
        With :math:`n_T =` ``num_trotter_steps`` and :math:`\Delta t = t/n_T`, :math:`e^{-iHt}` is
        approximated by :math:`n_T` repetitions of the symmetric step

        .. math::

            S_2(\Delta t) = \Big(\prod_{i=0}^{F-1} e^{-i H_i \Delta t/2}\Big)
                            e^{-i H_{\text{kin}} \Delta t}
                            \Big(\prod_{i=F-1}^{0} e^{-i H_i \Delta t/2}\Big),

        so each position fragment is visited twice per step at :math:`\tau = \Delta t/2` and the
        kinetic fragment once at :math:`\Delta t`. See the demo's `assembling the Trotter step
        <https://pennylane.ai/demos/simulating_vibronic_dynamics#assembling-the-trotter-step>`__
        section for the fragment loop this unrolls into.

        **2. Phase-gradient arithmetic.**
        Each fragment is diagonal, so evolving it reduces to imprinting a phase per basis state
        with the `phase-gradient trick <https://pennylane.ai/compilation/phase-gradient>`__: the
        ``phase_gradient_wires`` are required to hold

        .. math::

            |\nabla n\rangle = \tfrac{1}{\sqrt{2^b}}\sum_{j=0}^{2^b-1} e^{-2\pi i j/2^b}|j\rangle,
            \qquad
            \text{ADD}(m)|\nabla n\rangle = e^{+2\pi i m/2^b}|\nabla n\rangle .

        A phase :math:`\tau\,\theta_{i,a}` is imprinted by loading its :math:`b`-bit fixed-point
        encoding :math:`m` into ``coefficient_wires`` and adding it into ``phase_gradient_wires``
        with a :class:`~.SemiAdder`:

        .. code-block::

            coefficient : ─|m⟩─╭SemiAdder─|m⟩─┤
            |∇n⟩        : ─────╰SemiAdder─────┤  e^{+2πi m/2^b}|∇n⟩

        The sign of the ADD identity is positive, so coefficients are negated before encoding,
        :math:`m = \text{encode}(-\tau\theta_{i,a})`, giving the target :math:`e^{-i\tau\theta}`.
        Rounding to :math:`b` bits (mod :math:`2\pi`) is a discretization error on top of the
        Trotter truncation.

        **3. Electronic diagonalization.**
        Each position fragment is diagonalized by a Clifford :math:`V_i` (Fig. 2 of the reference),

        .. math::

            e^{-i H_i \tau} = V_i e^{-i D_i(Q)\tau} V_i^\dagger,
            \qquad
            V_i = \Big(\prod_{j \ne c} \text{CNOT}_{c \to j}\Big) H_c,
            \quad V_0 = I .

        Here :math:`c` is the least-significant set bit of :math:`i`, and the product runs over its
        other set bits :math:`j`. For :math:`i = 3` on two ``electronic_wires``:

        .. code-block::

            electronic[0]: ──────╭X─┤
            electronic[1]: ──H───╰●─┤

        **4. Position fragment.**
        Each monomial of :math:`\theta_{i,a}(Q)` follows the same pattern: a :class:`~.QROM`
        controlled on ``electronic_wires`` loads its coefficient into ``coefficient_wires``
        (differentially, i.e. only the change from the previous term), an arithmetic block writes
        the monomial's phase into ``phase_gradient_wires`` as in step 2, and a final
        :class:`~.QROM` unloads the last coefficient back to :math:`|0\rangle`:

        .. code-block::

            electronic  : ─╭QROM─────────────╭QROM─┤
            coefficient : ─╰QROM─╭arithmetic─╰QROM─┤
            |∇n⟩        : ───────╰arithmetic───────┤  e^{-iτ θ}

        The arithmetic block per monomial (skipped when the coefficient vanishes):

        * :math:`c_a` -- :class:`~.SemiAdder`.
        * :math:`l_{a,r} Q_r` -- a half-signed multiply of the (unsigned) coefficient by the
          signed :math:`Q_r`.
        * :math:`q_{a,r} Q_r^2` -- :class:`~.SignedOutSquare` then a plain :class:`~.OutMultiplier`.
        * :math:`\beta_{a,rs} Q_r Q_s` -- :class:`~.SignedOutMultiplier` then the same
          half-signed multiply as the linear term.

        The half-signed multiply is a plain :class:`~.OutMultiplier` on the magnitudes, flanked by
        extra sign-bit CNOTs and two controlled :class:`~.Incrementer` calls that apply the
        two's-complement (visible as ``C(Incrementer)`` in the resource count above).
        :math:`Q_r^2` uses the low :math:`2k-1` ``cache_wires`` and :math:`Q_r Q_s` all
        :math:`2k`, and each square/product is uncomputed after use. See the
        demo's `potential step
        <https://pennylane.ai/demos/simulating_vibronic_dynamics#the-potential-step>`__ section for
        an explicit, ungraphed version of this pattern.

        **5. Kinetic fragment.**
        The kinetic fragment is mode-diagonal in momentum space,

        .. math::

            e^{-i H_{\text{kin}} \Delta t} = \prod_r \text{QFT}_r^\dagger e^{-i t_r P_r^2 \Delta t}
                                             \text{QFT}_r,

        where :math:`\text{QFT}_r` maps mode :math:`r` from position to momentum space
        (implemented as an :class:`~.AQFT`), :math:`t_r` is read from the :math:`(0,0)` electronic
        block of ``kinetic`` (any electronic dependence is ignored), and
        :math:`e^{-i t_r P_r^2 \Delta t}` reuses the
        square-and-multiply of the quadratic term (step 4). Since :math:`t_r` is state-independent,
        it is written directly to ``coefficient_wires`` without a :class:`~.QROM`. See the demo's
        `kinetic step <https://pennylane.ai/demos/simulating_vibronic_dynamics#the-kinetic-step>`__
        section for an explicit, ungraphed version of this basis change and arithmetic.
    """

    dynamic_argnames = ("evolution_time",)
    hybrid_argnames = ("hamiltonian",)
    wire_argnames = (
        "electronic_wires",
        "vib_wires",
        "coefficient_wires",
        "phase_gradient_wires",
        "cache_wires",
        "work_wires",
    )
    # Static rather than dynamic: sets the resource-function gate counts, and can't be
    # `compilable_argnames` alongside the hybrid `hamiltonian` argument.
    static_argnames = ("num_trotter_steps", "aqft_order")

    # `coefficient_wires`, `cache_wires` and `work_wires` are optional (dynamically allocated
    # when omitted), so they have no fixed wire count and are excluded here.
    arg_specs = {
        "evolution_time": Float,
        "electronic_wires": Wire[-1],
        "vib_wires": Wire[-1],
        "phase_gradient_wires": Wire[-1],
    }

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        evolution_time,
        num_trotter_steps,
        hamiltonian,
        electronic_wires: WiresLike,
        vib_wires: WiresLike,
        phase_gradient_wires: WiresLike,
        coefficient_wires: WiresLike = (),
        cache_wires: WiresLike = (),
        work_wires: WiresLike = (),
        aqft_order=None,
    ):
        hamiltonian = _validate_hamiltonian(hamiltonian)
        # Sort dict keys for stable pytree round-trip.
        hamiltonian = {key: hamiltonian[key] for key in sorted(hamiltonian)}
        # Reject ``bool`` (subclass of ``int``); accept numpy integers.
        if (
            isinstance(num_trotter_steps, bool)
            or not isinstance(num_trotter_steps, (int, np.integer))
            or num_trotter_steps <= 0
        ):
            raise ValueError(
                "The number of Trotter steps should be a positive integer, "
                f"but got {num_trotter_steps}."
            )

        # Check register sizes against the Hamiltonian shape eagerly, when possible (wires may
        # be abstract, e.g. under capture, in which case this is deferred to decomposition time).
        if _wires_are_concrete(vib_wires) and _wires_are_concrete(electronic_wires):
            n_states = hamiltonian["constant"].shape[1]
            n_modes = hamiltonian["linear"].shape[-1]
            # `Wires(...)` cast needed since a single wire label is valid `WiresLike` but not
            # itself `len()`-able.
            num_vib_wires = len(Wires(vib_wires))
            if num_vib_wires % n_modes != 0:
                raise ValueError(
                    f"The number of vibrational wires ({num_vib_wires}) must be divisible by the "
                    f"number of modes ({n_modes})."
                )
            num_electronic = math.ceil_log2(n_states)
            if len(Wires(electronic_wires)) != num_electronic:
                raise ValueError(
                    f"Expected {num_electronic} electronic qubits for {n_states} electronic "
                    f"states, but got {len(Wires(electronic_wires))}."
                )

        super().__init__(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            electronic_wires,
            vib_wires,
            phase_gradient_wires,
            coefficient_wires,
            cache_wires,
            work_wires,
            aqft_order,
        )


# pylint: disable-next=too-many-arguments
def _trotter_vibronic_resources(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic_wires,
    vib_wires,
    coefficient_wires,
    phase_gradient_wires,
    cache_wires,
    work_wires,
    aqft_order,
):
    """Coarse (upper-bound) gate counts for the vibronic Trotter circuit.

    This estimate is intentionally inexact (``exact=False``): terms whose coefficients happen to
    vanish are skipped at runtime, and the sub-operations are counted at their top level.
    """
    # ``evolution_time`` and ``cache_wires`` are part of the shared resource/decomposition
    # signature but do not affect the (structural) gate counts.
    # pylint: disable=unused-argument
    num_fragments = hamiltonian["constant"].shape[0]
    n_states = hamiltonian["constant"].shape[1]
    n_modes = hamiltonian["linear"].shape[-1]
    n_elec = len(electronic_wires)
    # ``coefficient_wires`` may be dynamically allocated (empty here); fall back to
    # ``len(phase_gradient_wires)`` so the sub-operations are counted consistently.
    b = _coefficient_register_size(coefficient_wires, phase_gradient_wires)
    n_pg = len(phase_gradient_wires)
    k = len(vib_wires) // n_modes
    # ``work_wires`` may be dynamically allocated (empty here), in which case fall back to the
    # required size so the sub-operations are counted consistently with the decomposition.
    n_work = len(work_wires) if len(work_wires) > 0 else max(n_elec - 1, 2 * k, 2 * b + 2)
    num_pairs = n_modes * (n_modes - 1) // 2

    # Each Trotter step visits every position fragment twice (forward + backward).
    position_visits = 2 * num_fragments * num_trotter_steps

    # Data loading (QROM) and the arithmetic primitives, sized to match ``_extract_registers``.
    # ``Wire[size]`` builds an abstract wire register of the given size; only the sizes matter
    # for resource counting, so registers of the same operator need not be kept disjoint.
    qrom = QROM(
        np.zeros((n_states, b), dtype=int),
        control_wires=Wire[n_elec],
        target_wires=Wire[b],
        work_wires=Wire[max(n_elec - 1, 0)],
    )
    semi_adder = SemiAdder(x_wires=Wire[b], y_wires=Wire[n_pg], work_wires=Wire[n_work])

    # ``OutMultiplier`` appears with three different ``y`` register sizes: inside the half-signed
    # multiplier for the linear terms (``y = k``, the mode register) and the bilinear terms
    # (``y = 2k``, the cache register), and directly for the quadratic/kinetic terms
    # (``y = 2k - 1``, the cache register minus its sign bit).
    out_mult_linear = OutMultiplier(
        x_wires=Wire[b],
        y_wires=Wire[k],
        output_wires=Wire[n_pg],
        work_wires=Wire[max(n_work - 1, 0)],
    )
    out_mult_bilinear = OutMultiplier(
        x_wires=Wire[b],
        y_wires=Wire[2 * k],
        output_wires=Wire[n_pg],
        work_wires=Wire[max(n_work - 1, 0)],
    )
    out_mult_quad = OutMultiplier(
        x_wires=Wire[b],
        y_wires=Wire[max(2 * k - 1, 1)],
        output_wires=Wire[n_pg],
        work_wires=Wire[n_work],
    )
    signed_out_mult = SignedOutMultiplier(
        x_wires=Wire[k],
        y_wires=Wire[k],
        output_wires=Wire[2 * k],
        work_wires=Wire[n_work],
        output_wires_zeroed=True,
    )
    signed_square = SignedOutSquare(
        x_wires=Wire[k],
        output_wires=Wire[max(2 * k - 1, 1)],
        work_wires=Wire[n_work],
        output_wires_zeroed=True,
    )
    # Mirror the decomposition's resolution of ``aqft_order`` (``k - 1`` when unset) so the
    # resource estimate and the emitted circuit agree.
    aqft = _aqft(order=(aqft_order if aqft_order is not None else k - 1), wires=Wire[k])

    # The compute/uncompute pairs are emitted as ``adjoint(...)`` wrappers in the decomposition, so
    # they must be declared as such (not as their bare base ops) for the graph to find a path.
    adj_signed_square = adjoint(signed_square)
    adj_signed_out_mult = adjoint(signed_out_mult)
    adj_aqft = adjoint(aqft)

    # The sign-bit-controlled two's complement (compute + uncompute) acts on the ``k``-wire mode
    # register (linear terms) and the ``2k``-wire cache register (bilinear terms).
    ctrl_incrementer_linear = ctrl(Incrementer(Wire[k], Wire[max(n_work - 1, 0)]), Wire[1])
    ctrl_incrementer_bilinear = ctrl(Incrementer(Wire[2 * k], Wire[max(n_work - 1, 0)]), Wire[1])

    resources = defaultdict(int)

    # -- Electronic diagonalization (forward + adjoint per visit) --
    resources[Hadamard(wires=Wire[1])] += 2 * position_visits

    # -- CNOTs: electronic diagonalization plus the half-signed multipliers, which cache the sign
    #    bit, invert the register for the two's complement, and apply the controlled output flips --
    resources[CNOT(wires=Wire[2])] += (
        2 * position_visits * max(n_elec - 1, 0)
        + position_visits * n_modes * (2 + 2 * k + 2 * n_pg)
        + position_visits * num_pairs * (2 + 4 * k + 2 * n_pg)
    )

    # -- Data loading: constant + each linear/quadratic/bilinear term + final unload --
    resources[qrom] += position_visits * (2 + 2 * n_modes + num_pairs)
    resources[semi_adder] += position_visits

    # -- Linear terms: one half-signed multiplier each --
    resources[out_mult_linear] += position_visits * n_modes
    resources[ctrl_incrementer_linear] += position_visits * n_modes * 2

    # -- Quadratic terms: two SignedOutSquares (compute + uncompute) and one OutMultiplier --
    resources[signed_square] += position_visits * n_modes
    resources[adj_signed_square] += position_visits * n_modes
    resources[out_mult_quad] += position_visits * n_modes

    # -- Bilinear terms: two SignedOutMultipliers (compute + uncompute) and one half-signed
    #    multiplier --
    resources[signed_out_mult] += position_visits * num_pairs
    resources[adj_signed_out_mult] += position_visits * num_pairs
    resources[out_mult_bilinear] += position_visits * num_pairs
    resources[ctrl_incrementer_bilinear] += position_visits * num_pairs * 2

    # -- Kinetic fragment (once per Trotter step): per mode, load the momentum coefficients as a
    #    basis state (conditional PauliX per coefficient wire), then AQFT + SignedOutSquare +
    #    OutMultiplier and their uncomputation --
    kinetic_visits = num_trotter_steps * n_modes
    resources[X(wires=Wire[1])] += kinetic_visits * 2 * b
    resources[aqft] += kinetic_visits
    resources[adj_aqft] += kinetic_visits
    resources[signed_square] += kinetic_visits
    resources[adj_signed_square] += kinetic_visits
    resources[out_mult_quad] += kinetic_visits

    return dict(resources)


def _coefficient_register_size(coefficient_wires, phase_gradient_wires):
    """Size ``b`` of the coefficient register: its own length, or ``len(phase_gradient_wires)``
    when dynamically allocated (both hold ``b``-bit precision data, see the class docstring)."""
    return len(coefficient_wires) if len(coefficient_wires) > 0 else len(phase_gradient_wires)


def _required_work_wire_sizes(hamiltonian, vib_wires, b):
    """Required ``(cache, work)`` register sizes for the vibronic Trotter circuit."""
    n_states = hamiltonian["constant"].shape[1]
    n_modes = hamiltonian["linear"].shape[-1]
    k = len(vib_wires) // n_modes
    n = math.ceil_log2(n_states)
    return 2 * k, max(n - 1, 2 * k, 2 * b + 2)


# pylint: disable-next=too-many-arguments
def _trotter_vibronic_work_wires(
    hamiltonian, vib_wires, coefficient_wires, phase_gradient_wires, cache_wires, work_wires, **_
):
    """Number of zeroed work wires allocated when ``coefficient_wires``/``cache_wires``/
    ``work_wires`` are omitted."""
    b = _coefficient_register_size(coefficient_wires, phase_gradient_wires)
    cache_size, work_size = _required_work_wire_sizes(hamiltonian, vib_wires, b)
    num_alloc = 0
    if len(coefficient_wires) == 0:
        num_alloc += b
    if len(cache_wires) == 0:
        num_alloc += cache_size
    if len(work_wires) == 0:
        num_alloc += work_size
    return {"zeroed": num_alloc}


@register_resources(
    _trotter_vibronic_resources, exact=False, work_wires=_trotter_vibronic_work_wires
)
# pylint: disable-next=too-many-arguments
def _trotter_vibronic_decomposition(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic_wires,
    vib_wires,
    coefficient_wires,
    phase_gradient_wires,
    cache_wires,
    work_wires,
    aqft_order,
):
    n_states = hamiltonian["constant"].shape[1]
    n_modes = hamiltonian["linear"].shape[-1]
    n_elec = len(electronic_wires)

    vib = list(vib_wires)
    # Reshape flat ``vib_wires`` into per-mode registers for dynamic-index access.
    if compiler.active() or capture.enabled():
        mode_registers = math.array(vib, like="jax").reshape(n_modes, -1)
    else:
        mode_registers = np.array(vib).reshape(n_modes, -1)

    resolved_aqft_order = aqft_order if aqft_order is not None else mode_registers.shape[1] - 1

    def _run(coefficients, cache, work):
        registers = {
            "electronic": electronic_wires,
            "cache": cache,
            "coefficients": coefficients,
            "phase_gradient": phase_gradient_wires,
            "work": work,
        }
        _validate_registers(registers, mode_registers, n_modes, n_states)
        _run_trotter_vibronic(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            registers,
            mode_registers,
            resolved_aqft_order,
            n_states,
            n_modes,
            n_elec,
        )

    # Dynamically allocate ``coefficient_wires``/``cache_wires``/``work_wires`` when omitted.
    b = _coefficient_register_size(coefficient_wires, phase_gradient_wires)
    cache_size, work_size = _required_work_wire_sizes(hamiltonian, vib_wires, b)
    coefficients = coefficient_wires
    cache = cache_wires
    work = work_wires
    need_coefficients = len(coefficients) == 0
    need_cache = len(cache) == 0
    need_work = len(work) == 0
    num_alloc = (
        (b if need_coefficients else 0)
        + (cache_size if need_cache else 0)
        + (work_size if need_work else 0)
    )

    if num_alloc == 0:
        _run(coefficients, cache, work)
        return

    with allocate(num_alloc, state="zero", restored=True) as allocated:
        start = 0
        if need_coefficients:
            coefficients = allocated[start : (start := start + b)]
        if need_cache:
            cache = allocated[start : (start := start + cache_size)]
        if need_work:
            work = allocated[start : (start := start + work_size)]
        _run(coefficients, cache, work)


add_decomps(TrotterVibronic, _trotter_vibronic_decomposition)


# ---------------------------------------------------------------------------
# ---------------------------- Electronic diagonalization -------------------
# ---------------------------------------------------------------------------


def _diagonalization_support(key, n_wires):
    """Return the control wire index and the CNOT target indices for a fragment key.

    ``key`` is the ``(row, column)`` index of the (first) non-zero off-diagonal
    electronic matrix element of a fragment. Following Fig. 2 of
    `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__, the
    fragment is diagonalized on the electronic register with a single Hadamard and a
    number of CNOTs, all acting on the wires where the bit strings of ``key[0]`` and
    ``key[1]`` differ.
    """
    bitstrings = [math.int_to_binary(int(k), n_wires) for k in key]
    diagonalization_key = (np.array(bitstrings[0]) + np.array(bitstrings[1])) % 2
    support = np.where(diagonalization_key)[0][::-1]
    return int(support[0]), [int(k) for k in support[1:]]


def _diagonalize_vibronic_circuit(fragment_idx, wires):
    r"""Diagonalize position fragment ``fragment_idx`` by applying Clifford operations on the
    electronic register, following the ``key = (0, fragment_idx)`` convention (see
    :func:`_diagonalization_matrix`).

    Based on Fig. 2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.
    Requires one :class:`~.Hadamard` and at most :math:`\lceil\log_2(N)\rceil - 1`
    :class:`~.CNOT` gates, where :math:`N` is the number of electronic states.

    ``fragment_idx`` may be a traced integer, so the circuit structure (whether any gate is
    applied, which wire is the control, which wires receive a CNOT) is expressed with
    ``cond``/``for_loop`` rather than Python-level branching.

    Args:
        fragment_idx (int): index of the position fragment. The circuit is the identity if
            ``fragment_idx == 0``.
        wires (WiresLike): electronic wires on which the fragment acts.
    """
    n = len(wires)
    # ``key = (0, fragment_idx)``, so the bits that differ between ``key[0]`` and ``key[1]``
    # (see ``_diagonalization_support``) are exactly the bits of ``fragment_idx`` itself.
    shifts = np.arange(n - 1, -1, -1)
    if compiler.active() or capture.enabled():
        wires_arr = math.array(list(wires), like="jax")
        shifts = math.array(shifts, like="jax")
    else:
        wires_arr = list(wires)
    diff = (fragment_idx >> shifts) & 1
    any_diff = math.sum(diff) > 0
    # Largest wire index with a set bit == least-significant differing bit, since wire 0 holds
    # the most significant bit (matching ``qp.math.int_to_binary``).
    control_pos = math.max(math.where(diff > 0, math.arange(n), -1))
    control_wire = wires_arr[control_pos]

    @cond(any_diff)
    def _apply():
        Hadamard(wires=[control_wire])

        @for_loop(n - 1, -1, -1)
        def _cnots(w):
            @cond((diff[w] > 0) & (w != control_pos))
            def _maybe_cnot():
                CNOT(wires=[control_wire, wires_arr[w]])

            _maybe_cnot()

        _cnots()  # pylint: disable=no-value-for-parameter

    _apply()


def _diagonalization_matrix(key, n_wires):
    """Dense matrix implemented by :func:`_diagonalize_vibronic_circuit` for ``key``.

    The matrix acts on ``2 ** n_wires`` basis states with wire ``0`` being the most
    significant qubit (matching :func:`pennylane.math.int_to_binary`).
    """
    dim = 2**n_wires
    if int(key[0]) == int(key[1]):
        return np.eye(dim)

    identity = np.eye(2)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    proj_0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    proj_1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])

    def _embed(gate, wire):
        return reduce(np.kron, [gate if w == wire else identity for w in range(n_wires)])

    def _embed_cnot(control, target):
        term0 = reduce(np.kron, [proj_0 if w == control else identity for w in range(n_wires)])
        term1 = reduce(
            np.kron,
            [
                proj_1 if w == control else (pauli_x if w == target else identity)
                for w in range(n_wires)
            ],
        )
        return term0 + term1

    control, targets = _diagonalization_support(key, n_wires)
    matrix = _embed(hadamard, control)
    for target in targets:
        matrix = _embed_cnot(control, target) @ matrix
    return matrix


# ---------------------------------------------------------------------------
# ------------------------- Half-signed multiplier --------------------------
# ---------------------------------------------------------------------------


def _half_signed_out_multiplier(x_wires, y_wires, output_wires, work_wires):
    r"""Out-of-place multiplier of an unsigned register ``x`` and a signed register ``y``.

    Computes :math:`|z\rangle \mapsto |(z + xy) \bmod 2^k\rangle` on ``output_wires``, where
    ``k = len(output_wires)``, ``x`` is an unsigned integer and ``y`` is a two's-complement
    signed integer. This is a specialized primitive used by :class:`~.TrotterVibronic`; it is
    a plain quantum function that queues its constituent operations.

    The sign bit of ``y`` is cached on the first work wire, the two's complement of ``y`` is
    computed controlled on that cached bit, the (unsigned) magnitudes are multiplied into the
    output register with a :class:`~.OutMultiplier` (with the output wires flipped before and
    after the multiplication if ``y`` is negative), and finally the two's complement and the
    cached sign bit are uncomputed.
    """
    y_aux, work_wires = work_wires[0], work_wires[1:]

    if compiler.active() or capture.enabled():
        output_arr = math.array(list(output_wires), like="jax")
    else:
        output_arr = list(output_wires)

    def _flip_outputs():
        # CNOT from the cached sign bit onto every output wire.
        @for_loop(len(output_wires))
        def _flip(w):
            CNOT([y_aux, output_arr[w]])

        # pylint: disable-next=no-value-for-parameter
        _flip()

    # Cache the sign bit of ``y`` on the auxiliary wire.
    CNOT([y_wires[0], y_aux])

    # Take the two's complement of ``y`` controlled on the cached sign bit.
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # Multiply the magnitudes into the output register. If ``y`` was negative, flip all output
    # qubits before and after the (unsigned) multiplication, effectively subtracting the product.
    _flip_outputs()
    OutMultiplier(x_wires, y_wires, output_wires, work_wires=work_wires, output_wires_zeroed=False)
    _flip_outputs()

    # Uncompute the two's complement and the cached sign bit.
    _twos_complement_helper(y_wires, y_aux, work_wires)
    CNOT([y_wires[0], y_aux])


def _load_basis(bitstring, wires):
    r"""Prepare the computational basis state ``bitstring`` on ``wires`` (assumed to start in
    :math:`|0\rangle`).

    Applies a conditional :class:`~.PauliX` on each wire whose bit is set. Self-inverse, so the
    same call unloads the state.
    """
    if compiler.active() or capture.enabled():
        bitstring = math.array(bitstring, like="jax")
        wires_arr = math.array(list(wires), like="jax")
    else:
        wires_arr = list(wires)

    @for_loop(len(wires))
    def _prep(i):
        cond(bitstring[i] != 0, X)(wires_arr[i])

    # pylint: disable-next=no-value-for-parameter
    _prep()


# ---------------------------------------------------------------------------
# ------------------------- Coefficient preprocessing -----------------------
# ---------------------------------------------------------------------------


# pylint: disable-next=too-many-arguments
def _position_coefficients(matrix, constant, linear, quadratic, n_states, n_modes):
    """Diagonalize a single position fragment and extract its phase coefficients.

    Args:
        matrix (np.ndarray): dense ``(n_states, n_states)`` diagonalization matrix of the fragment.
        constant (TensorLike): dense ``(n_states, n_states)`` constant coefficients.
        linear (TensorLike): dense ``(n_states, n_states, n_modes)`` linear coefficients.
        quadratic (TensorLike): dense ``(n_states, n_states, n_modes, n_modes)`` quadratic
            coefficients.
        n_states (int): number of electronic states.
        n_modes (int): number of vibrational modes.

    Returns:
        tuple: ``(constant, linear, quadratic, bilinear)`` diagonalized coefficient tensors of
        shapes ``(n_states,)``, ``(n_modes, n_states)``, ``(n_modes, n_states)`` and
        ``(n_pairs, n_states)``, where ``n_pairs = n_modes * (n_modes - 1) / 2``.
    """
    matrix_t = matrix.T

    rotated_constant = matrix_t @ constant @ matrix
    constant_diag = math.stack([rotated_constant[a, a] for a in range(n_states)])

    linear_full = math.einsum("ba,bcz,cd->zad", matrix, linear, matrix)
    linear_diag = math.stack([linear_full[:, a, a] for a in range(n_states)], axis=1)

    sec_order = math.einsum("ba,bcyz,cd->yzad", matrix, quadratic, matrix)
    sec_order_diag = math.stack([sec_order[:, :, a, a] for a in range(n_states)], axis=2)
    quadratic_diag = math.stack([sec_order_diag[k, k] for k in range(n_modes)])

    rows, cols = np.triu_indices(n_modes, 1)
    if len(rows) > 0:
        bilinear = math.stack([sec_order_diag[int(r), int(c)] for r, c in zip(rows, cols)])
    else:
        bilinear = np.zeros((0, n_states))

    return constant_diag, linear_diag, quadratic_diag, bilinear


def _momentum_coefficients(kinetic):
    """Extract per-mode momentum coefficients from a dense kinetic fragment.

    Args:
        kinetic (TensorLike): dense ``(n_states, n_states, n_modes, n_modes)`` kinetic
            coefficients.

    Returns:
        TensorLike: momentum coefficients of shape ``(n_modes,)``.
    """
    # The kinetic fragment is electronic-diagonal and mode-diagonal, so the per-mode momentum
    # coefficients are the diagonal of its ``(n_modes, n_modes)`` block for electronic state 0.
    return math.diag(kinetic[0, 0])


def _preprocess_data(time, hamiltonian, n_elec, n_states, n_modes):
    """Diagonalize all position fragments and scale their coefficients by the time step.

    Returns the tuple ``((constant, linear, quadratic, bilinear), bilinear_indices)`` where the
    coefficient tensors are stacked over the position fragments and scaled by ``time / 2`` (the
    first-order time step of the symmetric second-order Trotter step). Fragment ``i`` is
    diagonalized with the ``(0, i)`` key.
    """
    first_order_time_step = time / 2
    constant_dense = hamiltonian["constant"]
    linear_dense = hamiltonian["linear"]
    quadratic_dense = hamiltonian["quadratic"]
    num_fragments = constant_dense.shape[0]

    all_constant, all_linear, all_quadratic, all_bilinear = [], [], [], []
    for i in range(num_fragments):
        matrix = _diagonalization_matrix((0, i), n_elec)[:n_states, :n_states]
        constant, linear, quadratic, bilinear = _position_coefficients(
            matrix,
            constant_dense[i],
            linear_dense[i],
            quadratic_dense[i],
            n_states,
            n_modes,
        )
        all_constant.append(constant * first_order_time_step)
        all_linear.append(linear * first_order_time_step)
        all_quadratic.append(quadratic * first_order_time_step)
        all_bilinear.append(bilinear * first_order_time_step)

    all_constant = math.stack(all_constant)
    all_linear = math.stack(all_linear)
    all_quadratic = math.stack(all_quadratic)
    all_bilinear = math.stack(all_bilinear)
    # Reshape the bilinear data into a flattened structure with respect to mode pairs.
    # ``k < ell`` matches bilinear_coeffs, populated only in the upper triangle.
    bilinear_indices = np.array(np.triu_indices(n_modes, 1))

    if compiler.active() or capture.enabled():
        all_constant = math.array(all_constant, like="jax")
        all_linear = math.array(all_linear, like="jax")
        all_quadratic = math.array(all_quadratic, like="jax")
        all_bilinear = math.array(all_bilinear, like="jax")
        bilinear_indices = math.array(bilinear_indices, like="jax")

    return (all_constant, all_linear, all_quadratic, all_bilinear), bilinear_indices


# ---------------------------------------------------------------------------
# ------------------------------ Data loading -------------------------------
# ---------------------------------------------------------------------------


def _load_coefficients(coefficients, precision, prev_bitstrings, qrom_wires):
    """Load ``coefficients`` into the data-loading register via a differential :class:`~.QROM`.

    The coefficients are converted into ``precision``-bit strings, XOR-ed with the currently
    loaded ``prev_bitstrings`` reference point, and loaded with a :class:`~.QROM`. The newly
    loaded bit strings are returned so that they can be used as the reference point for the next
    data-loading step.

    The coefficients are negated before encoding: adding an integer into the standard
    phase-gradient state :math:`|\\nabla n\\rangle` imprints :math:`e^{+2\\pi i m/2^b}`, so the
    negation is what turns the accumulated phase into :math:`e^{-i\\theta}` (i.e. :math:`e^{-iHt}`).
    """
    new_bitstrings = math.binary_decimals(-coefficients, precision, unit=2 * np.pi)
    change_bitstrings = (prev_bitstrings + new_bitstrings) % 2
    QROM(change_bitstrings, **qrom_wires)
    return new_bitstrings


def _extract_registers(registers, mode_registers, term, *mode_ids):
    r"""Extract the wire registers required for a specific term of the Trotter step.

    See :class:`~.TrotterVibronic` for the meaning of the individual registers. ``term`` is one
    of ``"constant"``, ``"linear"``, ``"quadratic"``, ``"bilinear"`` or ``"QROM"``, and
    ``mode_ids`` are the indices of the mode(s) involved in the term.
    """
    if term == "quadratic":
        (k,) = mode_ids
        square_wires = {"output_wires": "cache", "work_wires": "work"}
        square_wires = {new: registers[old] for new, old in square_wires.items()}
        square_wires["x_wires"] = mode_registers[k]
        # The cache contains 2k wires, we just need 2k-1 here
        square_wires["output_wires"] = square_wires["output_wires"][1:]
        mult_wires = {
            "x_wires": "coefficients",
            "y_wires": "cache",
            "output_wires": "phase_gradient",
            "work_wires": "work",
        }
        mult_wires = {new: registers[old] for new, old in mult_wires.items()}
        # The cache contains 2k wires, we just need 2k-1 here, see above
        mult_wires["y_wires"] = mult_wires["y_wires"][1:]
        return square_wires, mult_wires

    if term == "bilinear":
        k, ell = mode_ids
        mode_mult_wires = {
            "output_wires": "cache",
            "work_wires": "work",
        }
        mode_mult_wires = {new: registers[old] for new, old in mode_mult_wires.items()}
        mode_mult_wires["x_wires"] = mode_registers[k]
        mode_mult_wires["y_wires"] = mode_registers[ell]
        # The signed register for _half_signed_out_multiplier must be the _second_ input
        coeff_mult_wires = {
            "x_wires": "coefficients",
            "y_wires": "cache",
            "output_wires": "phase_gradient",
            "work_wires": "work",
        }
        coeff_mult_wires = {new: registers[old] for new, old in coeff_mult_wires.items()}
        return mode_mult_wires, coeff_mult_wires

    if term == "QROM":
        reg = {"control_wires": "electronic", "target_wires": "coefficients", "work_wires": "work"}
        qrom_wires = {new: registers[old] for new, old in reg.items()}
        # Fix lambda=1
        qrom_wires["work_wires"] = qrom_wires["work_wires"][: len(qrom_wires["control_wires"]) - 1]
        return qrom_wires

    if term == "linear":
        (k,) = mode_ids
        # The signed register for _half_signed_out_multiplier must be the _second_ input
        reg = {
            "x_wires": "coefficients",
            "output_wires": "phase_gradient",
            "work_wires": "work",
        }
        mult_wires = {new: registers[old] for new, old in reg.items()}
        mult_wires["y_wires"] = mode_registers[k]
        return mult_wires

    if term == "constant":
        reg = {"x_wires": "coefficients", "y_wires": "phase_gradient", "work_wires": "work"}
    return {new: registers[old] for new, old in reg.items()}


# ---------------------------------------------------------------------------
# ------------------------------ Trotter step -------------------------------
# ---------------------------------------------------------------------------


# pylint: disable-next=too-many-arguments,too-many-statements
def _trotter_step_second_order(
    time, hamiltonian, registers, mode_registers, aqft_order, n_states, n_modes, n_elec
):
    r"""Emit a single symmetric second-order Trotter step.

    The (first-Trotter-order) kinetic evolution is placed in the middle, wrapped by the
    second-Trotter-order position-fragment evolutions (forward, then backward). Within each
    position fragment we iterate over the linear, quadratic and bilinear terms; within each of
    these we iterate over the modes (or mode pairs) using ``for_loop``\ s. The fragment loop
    itself is also a ``for_loop``, so ``fragment_idx`` may be a traced value.
    """
    precision = len(registers["phase_gradient"])
    all_coeffs, bilinear_indices = _preprocess_data(time, hamiltonian, n_elec, n_states, n_modes)
    all_constant, all_linear, all_quadratic, all_bilinear = all_coeffs
    qrom_wires = _extract_registers(registers, mode_registers, "QROM")
    num_position_fragments = hamiltonian["constant"].shape[0]

    def position_fragments(i):
        const_coeffs = all_constant[i]
        linear_coeffs = all_linear[i]
        quadratic_coeffs = all_quadratic[i]
        bilinear_coeffs = all_bilinear[i]

        adjoint(_diagonalize_vibronic_circuit, lazy=False)(
            fragment_idx=i, wires=registers["electronic"]
        )

        def constant_term(prev_bitstrings):
            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                new_bitstrings = _load_coefficients(
                    const_coeffs, precision, prev_bitstrings, qrom_wires
                )
                SemiAdder(**_extract_registers(registers, mode_registers, "constant"))
                return new_bitstrings

            return cond(math.allclose(const_coeffs, 0.0), skip_fn, actual_fn)()

        @for_loop(n_modes)
        def linear_terms(k, prev_bitstrings):
            _coeffs = linear_coeffs[k]

            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                new_bitstrings = _load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                _half_signed_out_multiplier(
                    **_extract_registers(registers, mode_registers, "linear", k)
                )
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        @for_loop(n_modes)
        def quadratic_terms(k, prev_bitstrings):
            _coeffs = quadratic_coeffs[k]

            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                new_bitstrings = _load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                SignedOutSquare(**square_wires, output_wires_zeroed=True)
                OutMultiplier(**mult_wires)
                adjoint(SignedOutSquare(**square_wires, output_wires_zeroed=True))
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        @for_loop(bilinear_indices.shape[1])
        def bilinear_terms(k, prev_bitstrings):
            _coeffs = bilinear_coeffs[k]
            ids = bilinear_indices[:, k]

            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                mode_mult_wires, coeff_mult_wires = _extract_registers(
                    registers, mode_registers, "bilinear", *ids
                )
                new_bitstrings = _load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                _half_signed_out_multiplier(**coeff_mult_wires)
                adjoint(SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True))
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        prev_bitstrings = np.zeros((n_states, precision), dtype=int)
        prev_bitstrings = constant_term(prev_bitstrings)
        # pylint: disable-next=no-value-for-parameter
        prev_bitstrings = linear_terms(prev_bitstrings)
        # pylint: disable-next=no-value-for-parameter
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        # Skip empty bilinear loop: zero-iteration ``for_loop`` still traces under capture.
        if bilinear_indices.shape[1] > 0:
            # pylint: disable-next=no-value-for-parameter
            prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        QROM(prev_bitstrings, **qrom_wires)
        _diagonalize_vibronic_circuit(fragment_idx=i, wires=registers["electronic"])

    def kinetic_fragment():
        # use ``time``, not ``first_order_time_step`` because the kinetic fragment is the
        # middle one in second-order Trotter, so the two neighbouring first-order steps merge.
        kinetic_coeffs = _momentum_coefficients(hamiltonian["kinetic"]) * time
        if compiler.active() or capture.enabled():
            kinetic_coeffs = math.array(kinetic_coeffs, like="jax")

        @for_loop(n_modes)
        def kinetic_terms(k):
            _coeffs = kinetic_coeffs[k]

            def skip_fn():
                """Do nothing."""

            def actual_fn():
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                # Negated for the same phase-gradient sign convention as ``_load_coefficients``.
                bitstring = math.binary_decimals(-_coeffs, precision, unit=2 * np.pi)
                _load_basis(bitstring, registers["coefficients"])
                _aqft(order=aqft_order, wires=mode_registers[k])
                SignedOutSquare(**square_wires, output_wires_zeroed=True)
                OutMultiplier(**mult_wires)
                adjoint(SignedOutSquare(**square_wires, output_wires_zeroed=True))
                adjoint(_aqft(order=aqft_order, wires=mode_registers[k]))
                # ``_load_basis`` is self-inverse, so the unload is the same call.
                _load_basis(bitstring, registers["coefficients"])

            cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        # pylint: disable-next=no-value-for-parameter
        kinetic_terms()

    # pylint: disable-next=no-value-for-parameter
    for_loop(num_position_fragments)(position_fragments)()
    kinetic_fragment()
    # pylint: disable-next=no-value-for-parameter
    for_loop(num_position_fragments - 1, -1, -1)(position_fragments)()


# pylint: disable-next=too-many-arguments
def _run_trotter_vibronic(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    registers,
    mode_registers,
    aqft_order,
    n_states,
    n_modes,
    n_elec,
):
    """Emit ``num_trotter_steps`` symmetric second-order Trotter steps."""
    trotter_time_step = evolution_time / num_trotter_steps

    def _step(_step_idx, hamiltonian):
        # Carry ``hamiltonian`` through the loop for valid traced loop-body inputs.
        _trotter_step_second_order(
            trotter_time_step,
            hamiltonian,
            registers,
            mode_registers,
            aqft_order,
            n_states,
            n_modes,
            n_elec,
        )
        return hamiltonian

    for_loop(num_trotter_steps)(_step)(hamiltonian)


# ---------------------------------------------------------------------------
# ------------------------------ Validation ---------------------------------
# ---------------------------------------------------------------------------


def _wires_are_concrete(wires):
    """Whether a wire register is concrete (i.e. its labels are known, not abstract/traced)."""
    if isinstance(wires, AbstractWires):
        return False
    return not any(math.is_abstract(w) for w in Wires(wires))


def _validate_hamiltonian(hamiltonian):
    """Validate the vibronic Hamiltonian dict; coerce list/tuple leaves to arrays."""
    if not isinstance(hamiltonian, dict):
        raise ValueError(
            f"Expected `hamiltonian` to be a dictionary, got {type(hamiltonian).__name__}."
        )
    if set(hamiltonian) != set(HAMILTONIAN_KEYS):
        raise ValueError(
            f"Expected the keys in `hamiltonian` to be {set(HAMILTONIAN_KEYS)}, "
            f"but got {set(hamiltonian)}."
        )
    # Materialize list/tuple leaves on the host with ``np.asarray`` (not ``math.asarray``).
    hamiltonian = {
        key: np.asarray(value) if isinstance(value, (list, tuple)) else value
        for key, value in hamiltonian.items()
    }
    expected_ndim = {"constant": 3, "linear": 4, "quadratic": 5, "kinetic": 4}
    for key, ndim in expected_ndim.items():
        if math.ndim(hamiltonian[key]) != ndim:
            raise ValueError(
                f"Expected `hamiltonian['{key}']` to be a {ndim}-dimensional array, "
                f"but got {math.ndim(hamiltonian[key])} dimensions."
            )
    # The electronic diagonalization (see ``_diagonalization_matrix``) embeds each fragment's
    # 2x2 Clifford circuit into a ``2 ** n``-dimensional matrix and then slices it down to
    # ``n_states``. That slice is only guaranteed to stay orthogonal (i.e. a valid change of
    # basis) when it spans the full space, i.e. when ``n_states`` is itself a power of 2.
    n_states = math.shape(hamiltonian["constant"])[1]
    if n_states & (n_states - 1) != 0:
        raise ValueError(
            f"`hamiltonian` implies {n_states} electronic states, but TrotterVibronic currently "
            "only supports a number of electronic states that is a power of 2."
        )
    return hamiltonian


def _validate_registers(registers, mode_registers, n_modes, n_states):
    """Light validation of the wire register sizes. See :class:`~.TrotterVibronic`."""
    b = len(registers["coefficients"])
    k = len(mode_registers[0])
    n = math.ceil_log2(n_states)
    needed_work_wires = max(n - 1, 2 * k, 2 * b + 2)

    if len(registers["electronic"]) != n:
        raise ValueError(
            f"Expected {n} qubits for {n_states} electronic states, but got "
            f"{len(registers['electronic'])}."
        )
    if len(registers["cache"]) != 2 * k:
        raise ValueError(
            f"Expected exactly {2 * k} cache qubits for {k} qubits per vibrational mode, "
            f"but got {len(registers['cache'])}."
        )
    if len(registers["phase_gradient"]) < b:
        raise ValueError(
            "Expected the phase-gradient register to have at least as many qubits as the "
            f"coefficients register ({b} qubits), but got {len(registers['phase_gradient'])}."
        )
    if len(registers["work"]) < needed_work_wires:
        raise ValueError(
            f"Expected at least {needed_work_wires} work qubits, but got {len(registers['work'])}."
        )
    vibr_sizes = [len(mode_registers[i]) for i in range(n_modes)]
    if any(size != vibr_sizes[0] for size in vibr_sizes[1:]):
        raise ValueError(
            f"Expected all vibrational mode registers to have the same size, but got {vibr_sizes}."
        )
