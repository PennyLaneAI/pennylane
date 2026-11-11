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

from functools import reduce

import numpy as np

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.numeric_hamiltonians import VibronicHamiltonian
from pennylane.typing import AbstractArray, AbstractWires, Float, Wire
from pennylane.wires import Wires, WiresLike

_BRANCH_TERMS = ("constant", "linear", "quadratic", "bilinear", "kinetic")


class TrotterVibronic(Operator2):
    r"""Second-order Trotter circuit for vibronic Hamiltonian simulation using phase-gradient
    arithmetic.

    This template realizes :math:`U \approx e^{-iHt}` for a vibronic Hamiltonian, using the
    phase-gradient arithmetic construction of `Motlagh et al, arXiv:2411.13669
    <https://arxiv.org/abs/2411.13669>`__,

    .. math::

        H = \sum_{i=0}^{F-1} H_i + H_{\text{kin}},
        \qquad
        H_i = V_i D_i(Q) V_i^\dagger,
        \qquad
        H_{\text{kin}} = \sum_r t_r P_r^2 .

    .. seealso:: :class:`pennylane.VibronicHamiltonian`

    Args:
        evolution_time (float): time :math:`t` for which to evolve under the vibronic Hamiltonian.
        num_trotter_steps (int): number of second-order Trotter steps to use.
        hamiltonian (:class:`pennylane.VibronicHamiltonian`): A
            :class:`pennylane.VibronicHamiltonian` instance whose arguments are ``constant``,
            ``linear``, ``quadratic`` and ``kinetic``. The expected shapes are
            ``constant: (F, N, N)``, ``linear: (F, N, N, M)``, ``quadratic: (F, N, N, M, M)`` and
            ``kinetic: (N, N, M, M)``, where ``F`` is the number of position fragments, ``N`` the
            number of electronic states and ``M`` the number of vibrational modes. ``N`` must be a
            power of 2, and the fragments must follow the "XOR" scheme described in the
            Implementation Details section. See the documentation for
            :class:`pennylane.VibronicHamiltonian` for more information.
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
        branch_probs (dict[str, float]): probabilities for the fragment terms to be applied rather
            than skipped for vanishing coefficients, used as resource hints.
            See note below for details.

    **Example**

    Let's first create a :class:`pennylane.VibronicHamiltonian` with dummy data.

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

        hamiltonian = qp.VibronicHamiltonian(
            constant=constant,
            linear=linear,
            quadratic=np.zeros((1, n_states, n_states, n_modes, n_modes)),
            kinetic=np.einsum(
                "ab,cd->abcd", np.eye(n_states), np.diag(0.3 * np.ones(n_modes))
            ),
        )

    With this, we can setup a simple circuit that includes ``TrotterVibronic``.

    .. code-block:: python

        wires = qp.registers({
            "electronic": n, "vib_wires": n_modes * k, "cache": 2 * k,
            "coefficients": b, "phase_gradient": b, "work": max(n - 1, 2 * k, 2 * b + 2),
        })
        all_wires = qp.wires.Wires.all_wires(list(wires.values()))

        # decompose by one step to see the top-level sub-templates
        @qp.transforms._tape_decompose(max_expansion=1, gate_set=qp.gate_sets.ALL_OPS | {qp.PhaseGradientStatePrep})
        @qp.qnode(qp.device("default.qubit", wires=all_wires))
        def circuit():
            qp.PhaseGradientStatePrep(wires["phase_gradient"])
            qp.TrotterVibronic(
                evolution_time=1.0, num_trotter_steps=1, hamiltonian=hamiltonian,
                electronic_wires=wires["electronic"], vib_wires=wires["vib_wires"],
                cache_wires=wires["cache"], coefficient_wires=wires["coefficients"],
                phase_gradient_wires=wires["phase_gradient"], work_wires=wires["work"],
                aqft_order=1,
            )
            return qp.probs(wires=wires["electronic"])

    >>> qp.specs(circuit)().resources.quantum_operations  # doctest: +SKIP
    {'PhaseGradientStatePrep': 1, 'QROM': 6, 'SemiAdder': 2, 'CNOT': 24, 'C(Incrementer)': 4, 'OutMultiplier': 3, 'MultiX': 2, 'AQFT': 1, 'SignedOutSquare': 1, 'Adjoint(SignedOutSquare)': 1, 'Adjoint(AQFT)': 1}

    .. note::

        The keys of ``branch_probs`` are ``"constant"``, ``"linear"``, ``"quadratic"``,
        ``"bilinear"`` and ``"kinetic"``. The probabilities are attached as ``"branch-prob"``
        hints via :func:`~.hint` to the conditionals that skip the respective terms, and thus
        only expect resource accounting but not the circuit itself. In addition,
        they weight the gate counts of the terms in the resources of the decomposition rule, and
        thus affect the weight of this rule in the graph-based decomposition system.

        If ``branch_probs=None`` (default), the probablities are computed from ``hamiltonian``
        and ``evolution_time`` if both are concrete, and omitted otherwise.

        Providing ``branch_probs`` replaces the probabilities computed from ``hamiltonian`` and
        ``evolution_time`` entirely, rather than updating them. Terms missing from
        ``branch_probs`` are left unhinted and counted at every visit.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.TrotterVibronic
            :stacked:
            :symbols: :math:`F` position fragments, :math:`M` modes with :math:`k` wires each,
                :math:`n` electronic wires, :math:`b` coefficient and phase gradient wires, and
                :math:`r` = ``num_trotter_steps``
            :example: :math:`F=1`, :math:`M=1`, :math:`k=1`, :math:`n=1`,
                :math:`b=1`, :math:`r=1`

            .. decomposition-rule:: _trotter_vibronic_decomposition
                :image: /_static/decomposition_thumbnails/trotter_vibronic_phase_gradient.svg
                :where: :math:`P = 2Fr` position-fragment visits and :math:`K = M(M-1)/2` mode
                    pairs. All counts are upper bounds: terms with zero coefficients are
                    skipped, and the :class:`~.Hadamard` and :class:`~.CNOT` counts also depend
                    on each fragment's electronic diagonalization

                :class:`~.Hadamard` | :math:`2P` | 0
                :class:`~.CNOT` | :math:`2P(n-1) + PM(2+2k+2b) + PK(2+4k+2b)` | 12
                :class:`~.QROM` | :math:`P(2 + 2M + K)` | 6
                :class:`~.SemiAdder` | :math:`P` | 2
                :class:`~.OutMultiplier` | :math:`P(2M + K) + rM` | 3
                :class:`~.Incrementer` with 1 control | :math:`2P(M + K)` | 4
                :class:`~.SignedOutSquare` | :math:`PM + rM` | 1
                :class:`Adjoint(SignedOutSquare) <.SignedOutSquare>` | :math:`PM + rM` | 1
                :class:`~.SignedOutMultiplier` | :math:`PK` | 0
                :class:`Adjoint(SignedOutMultiplier) <.SignedOutMultiplier>` | :math:`PK` | 0
                :class:`~.MultiX` | :math:`2rM` | 2
                :class:`~.AQFT` | :math:`rM` | 1
                :class:`Adjoint(AQFT) <.AQFT>` | :math:`rM` | 1

    .. details::
        :title: Register Sizes
        :href: register-sizes

        With :math:`N` electronic states (:math:`n = \lceil\log_2(N)\rceil` qubits), :math:`M`
        vibrational modes (:math:`k` qubits each), and :math:`b`-qubit coefficient and
        phase-gradient registers, the wire registers should have the following sizes:

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - Argument
             - Expected size
             - Physical meaning / information content
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

        This section shows how the dense :class:`pennylane.VibronicHamiltonian` is turned into gates,
        making every phase prefactor explicit, following `Motlagh et al, arXiv:2411.13669
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
        tensors of the :class:`~.VibronicHamiltonian` and :math:`t_r` the diagonal of its
        ``kinetic`` tensor. The ``quadratic`` tensor packs both
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
        "cache_wires",
        "coefficient_wires",
        "electronic_wires",
        "phase_gradient_wires",
        "vib_wires",
        "work_wires",
    )
    # Static rather than dynamic: sets the resource-function gate counts or compiler hints, and
    # can't be `compilable_argnames` alongside the hybrid `hamiltonian` argument.
    static_argnames = ("num_trotter_steps", "aqft_order", "branch_probs")

    arg_specs = {
        "evolution_time": Float,
        "electronic_wires": Wire[-1],
        "vib_wires": Wire[-1],
        "phase_gradient_wires": Wire[-1],
        "coefficient_wires": Wire[-1],
        "cache_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        evolution_time,
        num_trotter_steps,
        hamiltonian: VibronicHamiltonian,
        *,
        cache_wires: WiresLike = (),
        coefficient_wires: WiresLike = (),
        electronic_wires: WiresLike,
        phase_gradient_wires: WiresLike,
        vib_wires: WiresLike,
        work_wires: WiresLike = (),
        aqft_order=None,
        branch_probs=None,
    ):
        _validate_hamiltonian(hamiltonian)
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
            n_states = hamiltonian.num_states
            n_modes = hamiltonian.num_modes
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

        if branch_probs is None:
            branch_probs = _branch_probs(hamiltonian, evolution_time, num_trotter_steps)
        if branch_probs is not None:
            # Static arguments must be hashable, and a fixed order lets equal values hash equally
            branch_probs = dict(branch_probs)
            branch_probs = tuple((t, branch_probs[t]) for t in _BRANCH_TERMS if t in branch_probs)

        super().__init__(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            cache_wires=cache_wires,
            coefficient_wires=coefficient_wires,
            electronic_wires=electronic_wires,
            phase_gradient_wires=phase_gradient_wires,
            vib_wires=vib_wires,
            work_wires=work_wires,
            aqft_order=aqft_order,
            branch_probs=branch_probs,
        )


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
    constant_dense = hamiltonian.constant
    linear_dense = hamiltonian.linear
    quadratic_dense = hamiltonian.quadratic
    num_fragments = hamiltonian.num_fragments

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

    return (all_constant, all_linear, all_quadratic, all_bilinear), bilinear_indices


def _applied_fraction(coeffs):
    """Fraction of the terms, each given by a coefficient vector along the last axis of
    ``coeffs``, that are applied rather than skipped for vanishing coefficients.

    The tolerance must match the ``math.allclose(..., 0.0)`` skip conditions of the Trotter step.
    """
    applied = ~np.isclose(coeffs, 0.0).all(axis=-1)
    return float(np.mean(applied)) if applied.size else None


def _branch_probs(hamiltonian, evolution_time, num_trotter_steps):
    """Probabilities for the terms of a Trotter step to be applied rather than skipped for
    vanishing coefficients, see the ``branch_probs`` argument of :class:`~.TrotterVibronic`.

    The coefficients are scaled by the time steps as in :func:`_trotter_step_second_order`,
    because the skip conditions compare the scaled coefficients against a fixed tolerance.

    Returns ``None`` if the Hamiltonian data or the evolution time are not concrete, or if the
    Hamiltonian has more position fragments than the XOR scheme on the electronic register allows.
    """
    data = (evolution_time, *hamiltonian.tensors)
    if any(math.is_tracer(t) or isinstance(t, AbstractArray) for t in data):
        return None
    n_states, n_modes = hamiltonian.num_states, hamiltonian.num_modes
    n_elec = math.ceil_log2(n_states)
    if hamiltonian.num_fragments > 2**n_elec:
        return None
    # Compute on NumPy copies: within an active JAX trace, e.g., while capturing a circuit, JAX
    # operations return tracers even for concrete inputs.
    hamiltonian = VibronicHamiltonian(*(math.toarray(t) for t in hamiltonian.tensors))
    time_step = float(evolution_time) / num_trotter_steps
    coeffs, _ = _preprocess_data(time_step, hamiltonian, n_elec, n_states, n_modes)
    kinetic = (_momentum_coefficients(hamiltonian.kinetic) * time_step)[:, None]
    return {term: _applied_fraction(c) for term, c in zip(_BRANCH_TERMS, (*coeffs, kinetic))}


# ---------------------------------------------------------------------------
# ------------------------------ Data loading -------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ------------------------------ Trotter step -------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ------------------------------ Validation ---------------------------------
# ---------------------------------------------------------------------------


def _wires_are_concrete(wires):
    """Whether a wire register is concrete (i.e. its labels are known, not abstract/traced)."""
    if isinstance(wires, AbstractWires):
        return False
    return not any(math.is_tracer(w) for w in Wires(wires))


def _validate_hamiltonian(hamiltonian):
    """Validate the vibronic Hamiltonian against ``TrotterVibronic``'s own requirements.

    The tensor ranks and the consistency of the shared ``F``/``N``/``M`` dimensions are already
    enforced by :class:`pennylane.VibronicHamiltonian`; only the extra constraint this template imposes on
    top of that shape family is checked here.
    """
    if not isinstance(hamiltonian, VibronicHamiltonian):
        raise ValueError(
            "TrotterVibronic expects a VibronicHamiltonian for the hamiltonian argument. Got "
            f"{type(hamiltonian)}."
        )
    # The electronic diagonalization (see ``_diagonalization_matrix``) embeds each fragment's
    # 2x2 Clifford circuit into a ``2 ** n``-dimensional matrix and then slices it down to
    # ``n_states``. That slice is only guaranteed to stay orthogonal (i.e. a valid change of
    # basis) when it spans the full space, i.e. when ``n_states`` is itself a power of 2.
    # ``num_states`` is ``None`` when no tensor pins that axis (an abstract Hamiltonian declared
    # with a ``-1`` electronic axis), in which case there is nothing to check.
    n_states = hamiltonian.num_states
    if n_states is not None and n_states & (n_states - 1) != 0:
        raise ValueError(
            f"`hamiltonian` implies {n_states} electronic states, but TrotterVibronic currently "
            "only supports a number of electronic states that is a power of 2."
        )
