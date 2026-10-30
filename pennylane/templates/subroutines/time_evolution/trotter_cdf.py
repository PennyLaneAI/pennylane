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

from collections import defaultdict
from functools import partial

import numpy as np

from pennylane import math
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.numeric_hamiltonians import CDFHamiltonian
from pennylane.ops import CNOT, CRZ, RZ, GlobalPhase, IsingZZ, PhaseShift
from pennylane.ops.op_math import adjoint, ctrl
from pennylane.ops.op_math.controlled2 import ctrl_many_zeroed_work_wires
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.typing import AbstractArray, AbstractWires, Float, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from ._trotter_utils import _emit_one_body_rz, _run_trotter_steps

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

        gate_set = {"BasisRotation", "RZ", "IsingZZ", "GlobalPhase", "ForLoop"}

        @qp.qjit
        @qp.transforms._tape_decompose(gate_set=gate_set)
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
            :example: the Hamiltonian above, :math:`N=2`, :math:`L=1`, :math:`n=1`

            .. decomposition-rule:: _trotter_cdf_decomposition
                :image: /_static/decomposition_thumbnails/trotter_cdf_second_order.svg
                :where: the counts are upper bounds, as basis rotations that are the identity are
                    skipped

                :class:`~.BasisRotation` on :math:`N` wires | :math:`4Ln + 4` | 8
                :class:`~.IsingZZ` | :math:`((2L-1)n + 1)\, N(2N-1)` | 12
                :class:`~.RZ` | :math:`2Nn` | 4
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


def _apply_system_basis_rotation(U, wires):
    """Apply a fragment's basis rotation on the alpha and beta spin channels."""
    if math.is_tracer(U) or not np.allclose(U, np.eye(len(U))):
        BasisRotation(unitary_matrix=U, wires=wires[::2])
        BasisRotation(unitary_matrix=U, wires=wires[1::2])


def _merge_leaves(U_prev, U_curr):
    """Fragment-rotation merge rule: conjugate-transpose of ``U_prev`` times ``U_curr``."""
    return _transpose_leaf(U_prev) @ U_curr


def _transpose_leaf(U):
    """Conjugate transpose (adjoint) of a leaf rotation."""
    return U.conj().T


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, double_phase):
    r"""Apply the two-body ``IsingZZ`` layer (base / double-phase / genuine controlled).

    Genuine control and double-phase are mutually exclusive constructions for a controlled
    ``IsingZZ``, chosen once here via ``double_phase``: genuine control uses
    ``ctrl(IsingZZ(...))`` on each term; double-phase instead shares *one* ``CNOT`` sandwich
    across every term touching a given ``wire_idx0``, which is cheaper since ``IsingZZ`` itself
    stays uncontrolled either way.
    """
    num_cas = Z.shape[0]

    def _angle(wire_idx0, wire_idx1):
        # Two-body prefactor. In the fragment basis the generator is a sum over distinct wire
        # pairs (w, w') with coupling lambda = core_tensors[l][p, q] for their orbitals p, q.
        # With n = (I - Z)/2, the Z_w Z_w' piece of lambda n_w n_w' is (lambda/4) Z_w Z_w', and
        # exp(-i (lambda/4) Z_w Z_w' tau) = IsingZZ(lambda tau / 2) since
        # IsingZZ(theta) = exp(-i theta Z Z / 2). Each fragment is visited at
        # tau = first_order_time_step = dt_trotter / 2, so the angle per visit is
        #   IsingZZ(lambda tau / 2) = IsingZZ(0.5 * lambda * first_order_time_step),
        # i.e. beta_twoB = 0.5. The two visits per Trotter step accumulate
        # IsingZZ(lambda * dt_trotter / 2), i.e. IsingZZ(lambda t / 2) over the full evolution.
        return 0.5 * Z[wire_idx0 // 2, wire_idx1 // 2] * first_order_time_step

    def zz_rotations(wire_idx0):
        if double_phase:

            @for_loop(wire_idx0 + 1, 2 * num_cas)
            def _zz_rotations(wire_idx1):
                IsingZZ(_angle(wire_idx0, wire_idx1), [wires[wire_idx0], wires[wire_idx1]])

            CNOT([control_wires[0], wires[wire_idx0]])
            _zz_rotations()
            CNOT([control_wires[0], wires[wire_idx0]])
        else:

            @for_loop(wire_idx0 + 1, 2 * num_cas)
            def _zz_rotations(wire_idx1):
                angle = _angle(wire_idx0, wire_idx1)
                zz_wires = [wires[wire_idx0], wires[wire_idx1]]
                if len(control_wires) == 0:
                    IsingZZ(angle, zz_wires)
                else:
                    ctrl(IsingZZ(angle, zz_wires), control=control_wires)

            _zz_rotations()

    for wire_idx0 in range(2 * num_cas - 1):
        zz_rotations(wire_idx0)


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires, double_phase):
    """Apply the one-body ``RZ`` layer (base / double-phase / genuine controlled)."""
    num_cas = Z_one_body.shape[0]

    @for_loop(2 * num_cas)
    def z_rotations(wire_idx):
        # One-body prefactor derivation (identical to CGF).
        #   n_p = (I - Z_p) / 2  ->  the Z-piece of eps n_p has coefficient -eps/2.
        #   target operator exp(-i eps n_p t) has Z-piece exp(+i eps t / 2 Z) = RZ(-eps t).
        #   The one-body fragment is visited ONCE per Trotter step, at duration
        #   first_order_time_step = dt_trotter / 2, so to accumulate a total RZ angle of
        #   -eps * dt_trotter we need angle-per-step = -2 * eps * first_order_time_step,
        #   i.e. alpha_oneB = -2.
        angle = -2.0 * Z_one_body[wire_idx // 2, wire_idx // 2] * first_order_time_step
        _emit_one_body_rz(angle, wires[wire_idx], control_wires, double_phase)

    z_rotations()


def _apply_two_body_diagonal_with_work_wires(
    Z, wires, first_order_time_step, control_wires, double_phase, work_wires
):
    """Combine each orbital pair's four ZZ rotations using clean work wires.

    CNOTs store each orbital's alpha/beta parity on its beta wire. For p<q, an AND
    selects the two zero parities and a CRZ supplies the phase. Standard control
    shares the AND of the external control and the first orbital's zero parity
    across the inner loop. The work wires and original system basis are restored
    at the end of the layer.
    """
    assert len(control_wires) <= 1, "The work-wire decomposition supports at most one control wire."
    num_orbitals = Z.shape[0]
    standard_control = len(control_wires) > 0 and not double_phase

    @for_loop(num_orbitals)
    def orbital_parity_cnots(orbital):
        CNOT([wires[2 * orbital], wires[2 * orbital + 1]])

    @for_loop(num_orbitals)
    def same_orbital_rotations(orbital):
        _emit_one_body_rz(
            0.5 * Z[orbital, orbital] * first_order_time_step,
            wires[2 * orbital + 1],
            control_wires,
            double_phase,
        )

    @for_loop(num_orbitals - 1)
    def orbital_pairs(p):
        # The inner loop visits N-1, ..., 1 pairs, averaging N/2 iterations.
        @hint({"num-iters": num_orbitals / 2})
        @for_loop(p + 1, num_orbitals)
        def pair(q):
            if standard_control:
                pair_and_wires = [work_wires[0], wires[2 * q + 1], work_wires[1]]
                control_values = (1, 0)
                rotation_control = work_wires[1]
            else:
                pair_and_wires = [wires[2 * p + 1], wires[2 * q + 1], work_wires[0]]
                control_values = (0, 0)
                rotation_control = work_wires[0]
            TemporaryAND(pair_and_wires, control_values=control_values)
            rotation_target = wires[2 * q]
            CNOT([wires[2 * p], rotation_target])
            CRZ(2 * Z[p, q] * first_order_time_step, [rotation_control, rotation_target])
            CNOT([wires[2 * p], rotation_target])
            adjoint(TemporaryAND)(pair_and_wires, control_values=control_values)

        if standard_control:
            control_and_wires = [wires[2 * p + 1], control_wires[0], work_wires[0]]
            TemporaryAND(control_and_wires, control_values=(0, 1))
        # Flipping p-alpha reverses all (p, q) rotations, so one CNOT pair serves this loop.
        if double_phase:
            CNOT([control_wires[0], wires[2 * p]])
        pair()
        if double_phase:
            CNOT([control_wires[0], wires[2 * p]])
        if standard_control:
            adjoint(TemporaryAND)(control_and_wires, control_values=(0, 1))

    orbital_parity_cnots()
    same_orbital_rotations()
    orbital_pairs()
    orbital_parity_cnots()


def _energy_shift(hamiltonian):
    r"""Zero-of-energy shift applied as a ``GlobalPhase`` (or control ``RZ``).

    Collects ``nuc_constant`` (:math:`C`) and every identity contribution produced by the
    :math:`n = (I - Z)/2` substitutions into a single scalar :math:`s` (see "Implementation
    Details", step 5, in :class:`~.TrotterCDF`'s docstring for the full derivation):

    .. math::

        s = C + \sum_p \epsilon_p - \frac{1}{2}\sum_{l\ge 1}\sum_{p,q} \lambda^{(l)}_{pq}
              + \frac{1}{4}\sum_{l\ge 1}\sum_{p} \lambda^{(l)}_{pp} ,

    where :math:`\epsilon_p =` ``core_tensors[0][p, p]`` and :math:`\lambda^{(l)}_{pq} =`
    ``core_tensors[l][p, q]``. Below, ``phase_from_mod_one_body`` is :math:`\sum_p \epsilon_p`
    and ``phase_from_two_body`` is the two-body sum/trace pair.
    """
    nuc_constant = hamiltonian.nuc_constant
    Z_tensor = hamiltonian.core_tensors
    phase_from_mod_one_body = math.trace(Z_tensor[0])
    phase_from_two_body = (
        -math.sum(Z_tensor[1:]) / 2 + math.sum(math.trace(Z_tensor[1:], axis1=1, axis2=2)) / 4
    )
    return nuc_constant + phase_from_mod_one_body + phase_from_two_body


_CDF_HELPERS = {
    "apply_system_basis_rotation": _apply_system_basis_rotation,
    "apply_two_body_diagonal": _apply_two_body_diagonal,
    "apply_one_body_diagonal": _apply_one_body_diagonal,
    "merge_leaves": _merge_leaves,
    "transpose_leaf": _transpose_leaf,
}


def _use_work_wire_decomposition(num_work_wires, has_control, double_phase):
    """Whether enough clean work wires are supplied to combine the four ZZ rotations."""
    return num_work_wires >= 1 + int(has_control and not double_phase)


def _cdf_decomposition_helpers(work_wires, has_control=False, double_phase=False):
    """Select the two-body decomposition and bind its work wires when available."""
    if _use_work_wire_decomposition(len(work_wires), has_control, double_phase):
        return {
            **_CDF_HELPERS,
            "apply_two_body_diagonal": partial(
                _apply_two_body_diagonal_with_work_wires, work_wires=work_wires
            ),
        }
    return _CDF_HELPERS


def _cdf_resource_counts(
    num_trotter_steps, hamiltonian, has_control, double_phase=False, num_work_wires=0
):
    """Shared (upper-bound) gate counts for the base and controlled CDF circuits.

    The exact gate count can be lower at runtime because fragments whose basis rotation
    happens to be the identity are skipped when the Hamiltonian data is concrete (not
    traced); this estimate assumes no such fragment is skipped.
    """
    if num_trotter_steps <= 0:
        return {}

    leaf_tensors = hamiltonian.leaf_tensors
    num_two_body_fragments = leaf_tensors.shape[0] - 1
    num_cas = leaf_tensors.shape[-1]

    resources = defaultdict(int)
    num_sysrot_calls = 2 * num_two_body_fragments * num_trotter_steps + 2
    num_twobody_blocks = (2 * num_two_body_fragments - 1) * num_trotter_steps + 1
    num_onebody_blocks = num_trotter_steps
    num_twobody_rotations = num_twobody_blocks * num_cas * (2 * num_cas - 1)
    num_onebody_rotations = num_onebody_blocks * 2 * num_cas

    # NOTE: 'BasisRotation' decomposes differently depending on the dtype of the unitary matrix.
    dtype_leaf, _ = math.get_abstract_and_cast_dtype(leaf_tensors)
    sysrot_key = BasisRotation(dtype_leaf[num_cas, num_cas], wires=Wire[num_cas])
    resources[sysrot_key] += 2 * num_sysrot_calls

    if not has_control:
        one_body_rotation = RZ
        two_body_rotation = IsingZZ
        resources[GlobalPhase] += 1
    elif double_phase:
        one_body_rotation = two_body_rotation = IsingZZ
        resources[RZ] += 1
    else:
        one_body_rotation = ctrl(RZ(Float, wires=Wire[1]), control=Wire[1])
        two_body_rotation = ctrl(IsingZZ(Float, wires=Wire[2]), control=Wire[1])
        resources[PhaseShift] += 1
    resources[one_body_rotation] += num_onebody_rotations

    if _use_work_wire_decomposition(num_work_wires, has_control, double_phase):
        num_pairs = num_twobody_blocks * num_cas * (num_cas - 1) // 2
        num_same_orbital_rotations = num_twobody_blocks * num_cas
        standard_control = has_control and not double_phase
        if has_control and double_phase:
            resources[CNOT] += 2 * num_twobody_blocks * (num_cas - 1)
        resources[one_body_rotation] += num_same_orbital_rotations
        resources[one_body_rotation if standard_control else CRZ] += num_pairs
        resources[CNOT] += 2 * num_twobody_blocks * num_cas + 2 * num_pairs
        num_ands = num_pairs
        if standard_control:
            num_ands += num_twobody_blocks * (num_cas - 1)
        resources[TemporaryAND] += num_ands
        resources[adjoint(TemporaryAND(Wire[3]))] += num_ands
    else:
        resources[two_body_rotation] += num_twobody_rotations
        if has_control and double_phase:
            resources[CNOT] += num_twobody_blocks * 2 * (2 * num_cas - 1)

    return {op: count for op, count in resources.items() if count}


def _trotter_cdf_resources(
    evolution_time, num_trotter_steps, hamiltonian, wires, double_phase, work_wires=()
):
    return _cdf_resource_counts(
        num_trotter_steps, hamiltonian, has_control=False, num_work_wires=len(work_wires)
    )


@register_resources(_trotter_cdf_resources, exact=False)
def _trotter_cdf_decomposition(
    evolution_time, num_trotter_steps, hamiltonian, wires, double_phase, work_wires=()
):
    # ``double_phase`` only affects the controlled decomposition; the base operator is
    # always the plain (uncontrolled) e^{-iHt} circuit.
    if num_trotter_steps > 0:
        _run_trotter_steps(
            evolution_time,
            num_trotter_steps,
            hamiltonian.normalize_leaf_determinant(),
            wires,
            (),
            **_cdf_decomposition_helpers(work_wires),
        )
        phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
        GlobalPhase(phi)


add_decomps(TrotterCDF, _trotter_cdf_decomposition)


def _controlled_trotter_cdf_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return _cdf_resource_counts(
        base.arguments["num_trotter_steps"],
        base.arguments["hamiltonian"],
        has_control=True,
        double_phase=base.arguments["double_phase"],
        num_work_wires=len(base.arguments["work_wires"]),
    )


@register_condition(lambda base, control_wires, *_, **__: len(control_wires) == 1)
@register_resources(_controlled_trotter_cdf_resource, exact=False)
def _controlled_trotter_cdf_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    evolution_time = base.arguments["evolution_time"]
    num_trotter_steps = base.arguments["num_trotter_steps"]
    hamiltonian = base.arguments["hamiltonian"]
    wires = base.arguments["wires"]
    double_phase = base.arguments["double_phase"]
    cdf_work_wires = base.arguments["work_wires"]
    validate_no_wire_overlaps({"control_wires": control_wires, "work_wires": cdf_work_wires})

    if num_trotter_steps == 0:
        return

    helpers = _cdf_decomposition_helpers(
        cdf_work_wires, has_control=True, double_phase=double_phase
    )
    phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
    hamiltonian = hamiltonian.normalize_leaf_determinant()

    if double_phase:
        # Double-phase (Fig. 6 in https://arxiv.org/abs/2506.15784) circuit: each full-time diagonal block is CNOT-sandwiched by
        # the control wire, so the bare rotations give control-0 / control-1 branches
        # e^{-i(H - s)t} / e^{+i(H - s)t}, where s = _energy_shift is the identity part of H.
        # We apply the energy shift explicitly and symmetrically as RZ(2*phi) on the control
        # wire (= diag(e^{-i s t}, e^{+i s t})), making the branches exactly the full-time
        # e^{-iHt} / e^{+iHt}.
        _run_trotter_steps(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            wires,
            control_wires,
            True,
            **helpers,
        )
        RZ(2 * phi, control_wires)
        return

    # Genuine controlled unitary: control each diagonal rotation at the full angle (basis
    # rotations stay uncontrolled and telescope to the identity in the control-0 branch),
    # and apply the global phase as controlled-GlobalPhase(phi) = PhaseShift(-phi).
    _run_trotter_steps(
        evolution_time,
        num_trotter_steps,
        hamiltonian,
        wires,
        control_wires,
        False,
        **helpers,
    )
    PhaseShift(-phi, control_wires)


@register_condition(ctrl_many_zeroed_work_wires.is_applicable)
@register_condition(lambda base, **_: not base.arguments["double_phase"])
@register_resources(
    lambda *args, **kwargs: ctrl_many_zeroed_work_wires.compute_resources(
        *args, **kwargs
    ).gate_counts,
    exact=False,
)
def _multi_controlled_trotter_cdf_decomp(
    base, control_wires, control_values, work_wires, work_wire_type
):
    """Reuse the single-control CDF decomposition after combining multiple controls."""
    validate_no_wire_overlaps(
        {
            "control_wires": control_wires,
            "control_work_wires": work_wires,
            "work_wires": base.arguments["work_wires"],
        }
    )
    ctrl_many_zeroed_work_wires(base, control_wires, control_values, work_wires, work_wire_type)


add_decomps(
    "C(TrotterCDF)",
    flip_zero_control2(_controlled_trotter_cdf_decomp),
    _multi_controlled_trotter_cdf_decomp,
)
