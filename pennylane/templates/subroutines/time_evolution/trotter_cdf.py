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
"""Contains the ``TrotterCDF`` template for Compressed Double Factorization Hamiltonians."""

from pennylane.core.operator import Operator2
from pennylane.numeric_hamiltonians import CDFHamiltonian
from pennylane.typing import AbstractArray, AbstractWires
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument


class TrotterCDF(Operator2):
    r"""Second-order Trotter time evolution for a Compressed Double Factorization (CDF) Hamiltonian.

    This template realizes :math:`U \approx e^{-iHt}` for an electronic-structure Hamiltonian in
    the Compressed Double Factorization (CDF) form (see `arXiv:2506.15784, Sec. III A
    <https://arxiv.org/abs/2506.15784>`__),

    .. math::

        H = C + \sum_{p} \epsilon_{p}\, \tilde{n}^{(0)}_{p}
            + \frac{1}{2}\sum_{l=1}^{L} \sum_{p,q} \lambda^{(l)}_{pq}\,
                \tilde{n}^{(l)}_{p} \tilde{n}^{(l)}_{q} .

    .. seealso:: :class:`pennylane.CDFHamiltonian`

    Args:
        evolution_time (float): Total evolution time ``t``.
        num_trotter_steps (int): Number of second-order Trotter steps.
        hamiltonian (:class:`pennylane.CDFHamiltonian`): A :class:`pennylane.CDFHamiltonian`
            instance whose arguments are ``nuc_constant``, ``core_tensors``, and ``leaf_tensors``.
            The expected shapes are ``core_tensors: (L+1, N, N)`` and ``leaf_tensors: (L+1, N, N)``,
            where ``N`` is the number of orbitals and ``L`` is the number of two-body fragments. See
            the documentation for :class:`pennylane.CDFHamiltonian` for more information.
        wires (Wires): The system wires. CDF expects ``2N`` wires (alpha / beta interleaved), i.e.
            wire ``2i + s`` holds spin-orbital ``(i, s)`` under the *blocked* Jordan-Wigner mode
            ordering ``m = i + s * N`` (all alpha modes, then all beta modes). The per-channel
            :class:`~.BasisRotation` factors on ``wires[::2]`` and ``wires[1::2]`` realize a fermionic
            Gaussian rotation only under this layout; mapping integrals with an interleaved mode
            order yields a different operator.
        double_phase (bool): Only affects a single-control decomposition. If ``False`` (default),
            :func:`~pennylane.ctrl` produces a genuine controlled unitary
            :math:`\text{diag}(1, U)` where :math:`U = e^{-iHt}` is the Trotter evolution.
            If ``True``, it produces a cheaper :math:`\text{diag}(U, U^\dagger)` instead, leading to the
            double phase trick for Hadamard test circuits (see `Fig. 6 <https://arxiv.org/abs/2506.15784>`__ and Usage Details below).
        work_wires (Wires): Optional auxiliary wires that the caller must supply in
            :math:`|0\rangle` and that the decomposition restores to :math:`|0\rangle`.
            The alternative two-body decomposition needs one
            work wire without control or with ``double_phase=True``, and two for standard
            single-qubit control (``double_phase=False``). With fewer work wires the
            original :class:`~.IsingZZ` decomposition is used.

    **Example**

    Let us create mock :class:`pennylane.CDFHamiltonian` data with the correct tensor shapes and real
    orthogonal leaves, and evolve with a few Trotter steps.

    .. code-block:: python

        import pennylane as qp
        import numpy as np

        rng = np.random.default_rng(42)
        N = 2  # orbitals
        L = 1  # two-body fragments

        def random_orthogonal(dim):
            q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
            return q * np.sign(np.diag(r))

        hamiltonian = qp.CDFHamiltonian(
            core_tensors = rng.standard_normal((L + 1, N, N)),
            leaf_tensors = np.stack([random_orthogonal(N) for _ in range(L + 1)]),
            nuc_constant = 0.5,
        )

        gate_set = {"BasisRotation", "RZ", "IsingZZ", "GlobalPhase"}

        @qp.qjit
        @qp.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit", wires=2 * N))
        def trotter_circuit():
            qp.TrotterCDF(
                evolution_time=1.0, num_trotter_steps=10, hamiltonian=hamiltonian,
                wires=range(2 * N),
            )
            return qp.state()

    We can inspect the quantum resources required. Note that the order of the keys in
    the ``quantum_operations`` dictionary is not guaranteed, so we sort it before printing:

    >>> specs = qp.specs(trotter_circuit)()["resources"].quantum_operations
    >>> dict(sorted(specs.items()))
    {'GlobalPhase': 1, 'IsingZZ': 66, 'RZ': 40, 'SingleExcitation': 44}

    The :class:`~.SingleExcitation` gates are due to :class:`~.BasisRotation` decomposing
    further on ``lightning.qubit``.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.TrotterCDF
            :stacked:
            :symbols: :math:`N` orbitals, :math:`L` two-body fragments and :math:`n` =
                ``num_trotter_steps``
            :example: the Hamiltonian above, :math:`N=2`, :math:`L=1`, :math:`n=1`. Note that all expressions below assume :math:`n \gt 0`

            .. decomposition-rule:: _trotter_cdf_decomposition
                :image: /_static/decomposition_thumbnails/trotter_cdf_second_order.svg
                :where: :math:`B = (2L-1)n + 1` two-body blocks and :math:`w` work wires. The
                    counts are upper bounds, as basis rotations that are the identity are
                    skipped. The circuit shows :math:`w = 0`

                :class:`~.BasisRotation` on :math:`N` wires | :math:`4Ln + 4` | 8
                :class:`~.IsingZZ` | :math:`BN(2N-1)` if :math:`w = 0` | 12
                :class:`~.RZ` | :math:`2Nn` if :math:`w = 0`, :math:`2Nn + BN` if :math:`w \geq 1` | 4
                :class:`~.CRZ` | :math:`BN(N-1)/2` if :math:`w \geq 1` | 0
                :class:`~.CNOT` | :math:`BN(N+1)` if :math:`w \geq 1` | 0
                :class:`~.TemporaryAND` | :math:`BN(N-1)/2` if :math:`w \geq 1` | 0
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`BN(N-1)/2` if :math:`w \geq 1` | 0
                :class:`~.GlobalPhase` | :math:`1` | 1

    .. details ::
        :title: Usage Details

        Controlling this operator with :func:`~pennylane.ctrl` produces by default a genuine
        controlled evolution :math:`\text{diag}(1, U)` (assuming a single control wire),
        where :math:`U = e^{-iHt}` is the Trotter evolution.

        With :math:`m>1` standard controls, supply :math:`m-1` clean ``work_wires`` to
        :func:`~pennylane.ctrl` with ``work_wire_type="zeroed"``. The registered
        multi-control decomposition combines the controls with :class:`~.TemporaryAND`,
        reuses the single-control ``TrotterCDF`` decomposition, and uncomputes the AND.
        These auxiliary wires must be distinct from the ``work_wires`` supplied to
        ``TrotterCDF``, which are used for the two-body layers.

        With a single control and ``double_phase=True``, :func:`~pennylane.ctrl` instead produces the double-phase Hadamard-test circuit
        of `Fig. 6 of arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`__, which realizes
        :math:`\text{diag}(U, U^\dagger)` (control-0 branch :math:`U = e^{-iHt}`, control-1 branch :math:`U^\dagger = e^{+iHt}`).

        In this decomposition, each diagonal rotation block is sandwiched by ``CNOT`` gates from the control wire (an
        ancilla-system ``ZZ`` coupling).

        .. code-block::

            double_phase=True:

                c: ─╭●──────┤ ->        c: ─╭●────────╭●─┤
            wires: ─╰RZ(2ϕ)─┤       wires: ─╰X──RZ(ϕ)─╰X─┤

        This is not a true controlled operation, but can be used to reduce the cost in Hadamard test circuits instead of the controlled evolution.
        The two circuits above are not equal as unitaries; only the relative phase between the
        control-0 and control-1 branches -- which is what the Hadamard test reads out -- agrees.

    .. details::
        :title: Combining four IsingZZ gates into one CRZ

        Consider only the four interactions between distinct orbitals :math:`p<q`, with
        :math:`a=p\alpha`, :math:`b=p\beta`, :math:`c=q\alpha`, and :math:`d=q\beta`:

        .. code-block:: text

            a: ─╭ZZ(θ)─╭ZZ(θ)───────────────┤
            b: ─│──────│──────╭ZZ(θ)─╭ZZ(θ)─┤
            c: ─╰ZZ(θ)─│──────╰ZZ(θ)─│──────┤
            d: ────────╰ZZ(θ)────────╰ZZ(θ)─┤

        Here ``ZZ(θ)`` denotes :class:`~.IsingZZ` with
        :math:`\theta=\lambda_{pq}\tau/2`, where :math:`\lambda_{pq}=`
        ``core_tensors[l][p, q]`` and :math:`\tau` is the duration of the diagonal block.
        In computational basis order :math:`|abcd\rangle`, this circuit has the matrix

        .. math::

            U_{pq}(\theta)=\operatorname{diag}\!\left(
            e^{-2i\theta},1,1,e^{2i\theta},
            \underbrace{1,\ldots,1}_{8\text{ entries}},
            e^{2i\theta},1,1,e^{-2i\theta}\right).

        Only states with :math:`a=b` and :math:`c=d` acquire a phase. These are precisely
        the cases selected by the clean work wire :math:`e` in the following circuit:

        .. code-block:: text

            a: ─╭●────╭●─────────╭●────╭●─┤
            b: ─╰X─╭○─│──────────│──○╮─╰X─┤
            c: ─╭●─│──╰X─╭RZ(4θ)─╰X──│─╭●─┤
            d: ─╰X─├○────│──────────○┤─╰X─┤
            e: ────╰⊕────╰●─────────⊕╯────┤

        The symbols ``○`` indicate controls on zero. The left AND gate on :math:`(b,d,e)`
        is :class:`~.TemporaryAND` with ``control_values=(0, 0)``; the mirrored gate is its
        adjoint, which undoes the AND.
        The work wire :math:`e` starts and ends in :math:`|0\rangle`.

        The first CNOTs store :math:`a\oplus b` and :math:`c\oplus d` on :math:`b,d`, so
        the AND sets :math:`e=1` exactly when the two input bits of each orbital agree.
        The next CNOT stores :math:`a\oplus c` on :math:`c`: this is zero for inputs
        :math:`0000,1111` and one for :math:`0011,1100`. The required phases therefore match
        those of a controlled rotation, in basis order :math:`|ec\rangle`:

        .. math::

            \mathrm{CRZ}_{e\to c}(4\theta)
            =\operatorname{diag}(1,1,e^{-2i\theta},e^{2i\theta}).

        Undoing the CNOT and AND operations restores the original system basis and the
        clean work wire, reproducing :math:`U_{pq}(\theta)` for any system input state.

    .. details ::
        :title: Implementation Details

        This section shows how the :class:`pennylane.CDFHamiltonian` is turned into the concrete gate angles, making
        every numerical prefactor explicit. Each orbital :math:`p` corresponds to two spin-orbital
        wires :math:`p\alpha, p\beta` that share the same leaf, and the number operator is
        spin-summed, :math:`\tilde{n}_p = n_{p\alpha} + n_{p\beta}`. Throughout we use
        :math:`n_w = (I - Z_w)/2` for a single wire :math:`w` and the PennyLane conventions
        :math:`RZ(\theta) = e^{-i\theta Z/2}` and
        :math:`IsingZZ(\theta) = e^{-i\theta\, Z\otimes Z/2}`.

        Recall that we start from the Hamiltonian

        .. math::

            H = C + \sum_{p} \epsilon_{p}\, \tilde{n}^{(0)}_{p}
                + \frac{1}{2}\sum_{l=1}^{L} \sum_{p,q} \lambda^{(l)}_{pq}\,
                  \tilde{n}^{(l)}_{p} \tilde{n}^{(l)}_{q} ,

        with :math:`\tilde{n}^{(l)}_{p} = \mathcal{U}^{(l)\dagger} \tilde{n}_p \mathcal{U}^{(l)}`.

        **1. Second-order Trotter**
        The Hamiltonian splits into :math:`L+1` fragments, :math:`H = C + H_0 + \sum_{l=1}^{L} H_l`, the
        one-body fragment :math:`H_0` and :math:`L` two-body fragments :math:`H_l`,

        .. math::

            H_0 = \sum_{p} \epsilon_{p}\, \tilde{n}^{(0)}_{p} ,
            \qquad
            H_l = \frac{1}{2}\sum_{p,q} \lambda^{(l)}_{pq}\, \tilde{n}^{(l)}_{p} \tilde{n}^{(l)}_{q}
            \quad (l \ge 1) ,

        with the scalar :math:`C =` ``nuc_constant`` (handled in step 5). With
        :math:`n =` ``num_trotter_steps`` and step duration :math:`\Delta t = t/n`, :math:`e^{-iHt}`
        is approximated by :math:`n` repetitions of the second-order (Strang) step

        .. math::

            S_2(\Delta t) = \Big(\prod_{l=1}^{L} e^{-i H_l \Delta t/2}\Big)\,
                            e^{-i H_0 \Delta t}\,
                            \Big(\prod_{l=L}^{1} e^{-i H_l \Delta t/2}\Big) ,

        which visits each two-body fragment *twice* per step (at the half-step duration
        ``first_order_time_step`` :math:`= \Delta t/2`) and the central one-body fragment *once* (at the
        full :math:`\Delta t`). The next steps derive :math:`e^{-i H_l \tau}` for a single fragment and
        duration :math:`\tau`. In the circuit, adjacent :math:`H_1` half evolutions at
        internal step boundaries are merged into one full-duration evolution.

        **2. Evolving a fragment**
        The fragments are optimized and derived via :doc:`compressed double factorization <demo:demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians>`.
        Each fragment is diagonal in its own orbital basis,
        :math:`H_l = \mathcal{U}^{(l)\dagger} D_l\, \mathcal{U}^{(l)}`, where
        :math:`\mathcal{U}^{(l)} =` :class:`~.BasisRotation` ``(leaf_tensors[l])`` rotates from the
        bare basis into the fragment's diagonal basis and :math:`D_l` is diagonal, so

        .. math::

            e^{-i H_l \tau} = \mathcal{U}^{(l)\dagger}\, e^{-i D_l \tau}\, \mathcal{U}^{(l)} .

        The Givens-rotation network used to synthesize :math:`\mathcal{U}^{(l)}` is not hand-rolled
        here; :class:`~.BasisRotation` decomposes any orthogonal/unitary single-particle rotation
        into a sequence of :class:`~.SingleExcitation`/:class:`~.PhaseShift` gates on its own (see
        its ``compute_decomposition``), so this template only supplies the leaf and lets
        :class:`~.BasisRotation` handle the gate-level realization.

        This is implemented by applying :math:`\mathcal{U}^{(l)}` (to the :math:`\alpha` and
        :math:`\beta` spin channels separately), the diagonal rotations for :math:`e^{-i D_l \tau}`
        (see steps 3-4 below), and the inverse rotation :math:`\mathcal{U}^{(l)\dagger}`. Consecutive
        fragment rotations are merged as :math:`\mathcal{U}^{(l-1)\dagger}\mathcal{U}^{(l)}`, so only
        one basis rotation is emitted per fragment boundary, plus a single trailing
        :math:`\mathcal{U}^{(1)\dagger}` after the last step. Each leaf is first normalized to
        determinant :math:`+1` (negating one orbital column, a no-op on :math:`\tilde{n}_p`) so that
        no :class:`~.BasisRotation` needs a determinant-correcting phase flip.

        **3. One-body diagonal**
        In the fragment basis the one-body generator is :math:`D_0 = \sum_w \epsilon_p\, n_w`, summed
        over the :math:`2N` wires, with :math:`\epsilon_p =` ``core_tensors[0][p, p]`` for the orbital
        :math:`p` of wire :math:`w`. Using :math:`n_w=(I-Z_w)/2`,

        .. math::

            \epsilon_p\, n_w = \frac{\epsilon_p}{2}\, I - \frac{\epsilon_p}{2}\, Z_w ,

        so the :math:`Z_w` piece over duration :math:`\tau` is
        :math:`e^{+i(\epsilon_p/2) Z_w \tau} = RZ(-\epsilon_p \tau)`. The one-body fragment is the
        central full :math:`\Delta t` term (visited once per step), so the emitted angle is
        :math:`-\epsilon_p \Delta t = -2 \epsilon_p (\Delta t/2)`, accumulating :math:`RZ(-\epsilon_p t)`
        over the :math:`n` steps. The constant :math:`\epsilon_p/2` is deferred to the global phase (step 5).

        **4. Two-body diagonal**
        In the fragment basis the two-body generator is a sum over distinct wire pairs
        :math:`(w, w')`, whose coupling :math:`\lambda_{pq} =` ``core_tensors[l][p, q]`` is indexed by
        their orbitals :math:`p, q`. Using :math:`n=(I-Z)/2`, each term expands as

        .. math::

            \lambda_{pq}\, n_w n_{w'} =
                \frac{\lambda_{pq}}{4}\left(I - Z_w - Z_{w'} + Z_w Z_{w'}\right) ,

        a constant :math:`\lambda_{pq}/4`, single-site terms
        :math:`-\tfrac{\lambda_{pq}}{4}(Z_w + Z_{w'})`, and the two-site term
        :math:`\tfrac{\lambda_{pq}}{4} Z_w Z_{w'}`. In the *regrouped* input the single-site terms are
        already absorbed into ``core_tensors[0]`` and the constants into the global phase, so each
        two-body layer implements only the two-site term

        .. math::

            e^{-i(\lambda_{pq}/4) Z_w Z_{w'}\, \tau} = \text{IsingZZ} \left(\lambda_{pq} \tfrac{\tau}{2}\right).

        Summing each unordered wire pair once turns the orbital-level :math:`\tfrac{1}{2}` of
        :math:`H_l` into a single :class:`~.IsingZZ` per pair. Each two-body fragment is visited twice
        per step at :math:`\tau = \Delta t/2`, so the emitted angle is
        :math:`\tfrac{1}{2}\,\lambda_{pq}\,(\Delta t/2)` (the ``0.5`` prefactor in the code),
        accumulating :math:`IsingZZ(\lambda_{pq}\, t/2)` over the :math:`n` steps.

        With sufficient ``work_wires``, the four rotations for each :math:`p<q` pair are
        combined as shown in "Combining four IsingZZ gates into one CRZ" above. Without
        control, the same-orbital terms :math:`p=q` become :math:`RZ(\lambda_{pp}\tau/2)` on
        :math:`p\beta`. At the start of each layer, CNOTs store each orbital's alpha/beta
        parity on its beta wire. All pairs use these parities, and the same CNOTs restore
        the system wires at the end of the layer.

        An uncontrolled layer therefore uses :math:`N(N-1)/2` CRZ, :math:`N` RZ,
        :math:`N(N+1)` explicit CNOTs, and :math:`N(N-1)/2` TemporaryAND/adjoint pairs.
        With ``double_phase=True``, the :math:`p=q` rotations become control-system IsingZZ
        gates, and two control CNOTs enclosing all pairs with the same first orbital
        add :math:`2(N-1)` explicit CNOTs. Standard control first computes the AND of
        the control and the zero parity of orbital :math:`p`, sharing it across all
        :math:`q>p`. A second work wire combines this result with the zero parity of
        orbital :math:`q`. This uses :math:`N(N-1)/2 + N-1` TemporaryAND/adjoint pairs
        per layer; its :math:`p=q` rotations are CRZ.

        **5. Constant terms (global phase / energy shift)**
        The scalar :math:`C =` ``nuc_constant`` together with all identity contributions produced by
        the :math:`n=(I-Z)/2` substitutions are collected into a single scalar :math:`s` and applied
        as one :class:`~.GlobalPhase` with angle :math:`s\,t`, i.e. :math:`e^{-ist}`, matching the
        :math:`I`-part of :math:`e^{-iHt}`. Following Eq. (A29) of
        `arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`__,

        .. math::

            s = C + \sum_p \epsilon_p
                  - \frac{1}{2}\sum_{l\ge 1}\sum_{p,q} \lambda^{(l)}_{pq}
                  + \frac{1}{4}\sum_{l\ge 1}\sum_{p} \lambda^{(l)}_{pp} ,

        which the code accumulates as the energy shift (``trace(core_tensors[0])`` for the
        one-body part and ``-sum(core_tensors[1:])/2 + trace(core_tensors[1:])/4`` for the two-body
        part).

    """

    dynamic_argnames = ("evolution_time",)
    hybrid_argnames = ("hamiltonian",)
    wire_argnames = ("wires", "work_wires")
    # `hybrid_argnames` and `compilable_argnames` cannot both be non-empty on the same
    # operator, so `num_trotter_steps` (a plain Python int that drives Python-level
    # control flow) is treated as `static_argnames` instead.
    static_argnames = (
        "num_trotter_steps",
        "double_phase",
    )

    def __init__(
        self,
        evolution_time: int | AbstractArray,
        num_trotter_steps: float | AbstractArray,
        hamiltonian: CDFHamiltonian,
        wires: WiresLike | AbstractWires,
        double_phase=False,
        work_wires: WiresLike | AbstractWires = (),
    ):
        # ``hamiltonian`` is a hybrid argument: its array-like leaves must be arrays (or scalars)
        # to be captured and lowered correctly. Cast only list/tuple inputs, leaving array inputs
        # as-is so the hybrid argument survives the capture round-trip unchanged. Abstract capture
        # placeholders (which are neither list/tuple nor real arrays) are left untouched and skip
        # the ``ndim`` validation below, so the operator can be reconstructed during capture.

        if not isinstance(hamiltonian, CDFHamiltonian):
            raise ValueError(
                "TrotterCDF expects a CDFHamiltonian for the hamiltonian argument. Got "
                f"{type(hamiltonian)}."
            )

        Z = hamiltonian.core_tensors
        U = hamiltonian.leaf_tensors
        if hasattr(Z, "ndim") and hasattr(U, "ndim") and not (Z.ndim == 3 and U.ndim == 3):
            raise ValueError(
                "TrotterCDF expects a CDFHamiltonian with core_tensors.ndim == 3 and "
                f"leaf_tensors.ndim == 3. Got core_tensors.ndim={Z.ndim}, "
                f"leaf_tensors.ndim={U.ndim}. For vibrational (CGF) Hamiltonians, use TrotterCGF."
            )
        wires = Wires(wires)
        work_wires = Wires(work_wires)
        validate_no_wire_overlaps({"wires": wires, "work_wires": work_wires})
        super().__init__(
            evolution_time, num_trotter_steps, hamiltonian, wires, double_phase, work_wires
        )
