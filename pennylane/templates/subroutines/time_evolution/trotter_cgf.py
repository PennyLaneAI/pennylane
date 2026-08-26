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
"""Contains the ``TrotterCGF`` template for Christiansen Greedy Fragmentation Hamiltonians."""

from collections import defaultdict

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.numeric_hamiltonians import CGFHamiltonian
from pennylane.ops import CNOT, RZ, GlobalPhase, IsingZZ, PhaseShift
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.templates.subroutines.qchem.basis_rotation import BasisRotation
from pennylane.typing import Complex, Wire, AbstractArray, AbstractWires
from pennylane.wires import WiresLike

from ._trotter_utils import _emit_one_body_rz, _emit_two_body_isingzz, _run_trotter_steps

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument


class TrotterCGF(Operator2):
    r"""Second-order Trotter time evolution for a Christiansen Greedy Fragmentation (CGF) Hamiltonian.

    This template realizes :math:`U \approx e^{-iHt}` for a vibrational Hamiltonian in the
    Christiansen Greedy Fragmentation (CGF) form (see `arXiv:2508.11865, Sec. III C
    <https://arxiv.org/abs/2508.11865>`__).

    .. seealso:: :class:`pennylane.CGFHamiltonian`

    Args:
        evolution_time (float): Total evolution time ``t``.
        num_trotter_steps (int): Number of second-order Trotter steps.
        hamiltonian (:class:`pennylane.CGFHamiltonian`): A :class:`pennylane.CGFHamiltonian`
            instance whose arguments are ``nuc_constant``,
            ``core_tensors``, and ``leaf_tensors``. The expected shapes are
            ``core_tensors: (L+1, M, M, N, N)`` and ``leaf_tensors: (L+1, M, N, N)``,
            where ``M`` is the number of modes, ``N`` is the number of modals per mode,
            and ``L`` is the number of two-body fragments. See the documentation for
            :class:`pennylane.CGFHamiltonian` for more information.
        wires (Wires): The system wires. CGF expects ``M*N`` wires arranged mode-major:
            wire ``l*N + p`` corresponds to modal ``p`` of mode ``l`` (unary/SBE layout).
        double_phase (bool): Only affects a single-control decomposition. If ``False`` (default),
            :func:`~pennylane.ctrl` produces a genuine controlled unitary
            :math:`\text{diag}(1, U)` where :math:`U = e^{-iHt}` is the Trotter evolution.
            If ``True``, it produces :math:`\text{diag}(U, U^\dagger)` instead, leading to the
            double phase trick for Hadamard test circuits (see `Fig. 6 <https://arxiv.org/abs/2506.15784>`__ and Usage Details below).

    **Example**

    Let us create mock :class:`pennylane.CGFHamiltonian` data with the correct tensor shapes and real
    orthogonal leaves.

    .. code-block:: python

        import pennylane as qp

        rng = np.random.default_rng(42)
        L, M, N = 1, 2, 3  # two-body fragments, modes, modals

        def random_orthogonal(dim):
            q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
            return q * np.sign(np.diag(r))

        hamiltonian = qp.CGFHamiltonian(
            core_tensors = rng.standard_normal((L + 1, M, M, N, N)),
            leaf_tensors = np.stack(
                [np.stack([random_orthogonal(N) for _ in range(M)]) for _ in range(L + 1)]
            ),
            nuc_constant = 0.5,
        )

    With this, we can set up a Hadamard test circuit, consisting of a :class:`~.Hadamard`
    gate and a controlled Trotter evolution obtained via :func:`~pennylane.ctrl`.

    .. code-block:: python

        registers = qp.registers({"hadamard": 1, "system": M * N})
        gate_set = {"Hadamard", "BasisRotation", "RZ", "CNOT", "PhaseShift", "ForLoop"}

        @qp.qjit
        @qp.transforms.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])

            qp.ctrl(
                qp.TrotterCGF(
                    evolution_time=1.0, num_trotter_steps=10, hamiltonian=hamiltonian,
                    wires=registers["system"],
                ),
                control=registers["hadamard"],
            )

            return qp.expval(qp.X(registers["hadamard"]))

    We can now run this circuit consisting of just ``10`` Trotter steps.

    >>> trotter_circuit()
    Array(0.99599768, dtype=float64)

    Or check the quantum resources required for this task. Because the (default) controlled
    decomposition is a genuine controlled unitary, each diagonal rotation is individually
    controlled, so it decomposes into :class:`~.CNOT` and :class:`~.RZ` gates rather than
    :class:`~.IsingZZ`. Note that the order of the keys in the ``quantum_operations`` dictionary
    is not guaranteed, so we sort it before printing:

    >>> specs = qp.specs(trotter_circuit)()["resources"].quantum_operations
    >>> dict(sorted(specs.items()))
    {'CNOT': 840, 'Hadamard': 1, 'PhaseShift': 1, 'RZ': 480, 'SingleExcitation': 186}

    The :class:`~.SingleExcitation` gates are due to :class:`~.BasisRotation` decomposing into
    :class:`~.PhaseShift` and :class:`~.SingleExcitation` on ``lightning.qubit``.

    .. details ::
        :title: Usage Details

        Controlling this operator with :func:`~pennylane.ctrl` produces by default a genuine
        controlled evolution :math:`\text{diag}(1, U)` (assuming a single control wire),
        where :math:`U = e^{-iHt}` is the Trotter evolution.

        With ``double_phase=True``, :func:`~pennylane.ctrl` instead produces the double-phase
        Hadamard-test circuit of `Fig. 6 of arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`__,
        which realizes :math:`\text{diag}(U, U^\dagger)` (control-0 branch :math:`U = e^{-iHt}`,
        control-1 branch :math:`U^\dagger = e^{+iHt}`).

        In this decomposition, each diagonal rotation block is sandwiched by ``CNOT`` gates from the
        control wire (an ancilla-system ``ZZ`` coupling).

        .. code-block::

            double_phase=True:

                c: ─╭●──────┤ ->        c: ─╭●────────╭●─┤
            wires: ─╰RZ(2ϕ)─┤       wires: ─╰X──RZ(ϕ)─╰X─┤

        This is not a true controlled operation, but can be used to reduce the cost in Hadamard test
        circuits instead of the controlled evolution.

    .. details ::
        :title: Implementation Details

        This section shows how the :class:`pennylane.CGFHamiltonian` is turned into the concrete gate angles, making
        every numerical prefactor explicit. Throughout we use :math:`n_{lp} = (I - Z_{lp})/2`
        and the PennyLane conventions
        :math:`RZ(\theta) = e^{-i\theta Z/2}` and
        :math:`IsingZZ(\theta) = e^{-i\theta\, Z\otimes Z/2}`.

        Recall that we start from the Hamiltonian

        .. math::

            H = C + \sum_{l,p} \epsilon_{lp}\, \tilde{n}^{(0)}_{lp}
                + \sum_{\nu=1}^{L} \sum_{l>m} \sum_{p,q} \lambda^{(\nu)}_{lmpq}\,
                  \tilde{n}^{(\nu)}_{lp} \tilde{n}^{(\nu)}_{mq} ,

        **1. Fragment splitting and second-order Trotter**
        The Hamiltonian splits into :math:`L+1` fragments,
        :math:`H = C + H_0 + \sum_{\nu=1}^{L} H_\nu`, the one-body fragment :math:`H_0` and
        :math:`L` two-body fragments :math:`H_\nu`,

        .. math::

            H_0 = \sum_{l,p} \epsilon_{lp}\, \tilde{n}^{(0)}_{lp} ,
            \qquad
            H_\nu = \sum_{l>m} \sum_{p,q} \lambda^{(\nu)}_{lmpq}\,
                    \tilde{n}^{(\nu)}_{lp} \tilde{n}^{(\nu)}_{mq}
            \quad (\nu \ge 1) ,

        with :math:`\tilde{n}^{(\nu)}_{lp} = \mathcal{U}^{(\nu,l)} n_{lp} \mathcal{U}^{(\nu,l)\dagger}`
        and the scalar :math:`C =` ``nuc_constant`` (handled in step 5). With
        :math:`n =` ``num_trotter_steps`` and step duration :math:`\Delta t = t/n`, :math:`e^{-iHt}` is
        approximated by :math:`n` repetitions of the second-order (Strang) step

        .. math::

            S_2(\Delta t) = \Big(\prod_{\nu=1}^{L} e^{-i H_\nu \Delta t/2}\Big)\,
                            e^{-i H_0 \Delta t}\,
                            \Big(\prod_{\nu=L}^{1} e^{-i H_\nu \Delta t/2}\Big) ,

        which visits each two-body fragment *twice* per step (at the half-step duration
        ``first_order_time_step`` :math:`= \Delta t/2`) and the central one-body fragment *once* (at the
        full :math:`\Delta t`). The next steps derive :math:`e^{-i H_\nu \tau}` for a single fragment and
        duration :math:`\tau`.

        **2. Evolving a fragment**
        The fragments are obtained via Christiansen greedy fragmentation (see `arXiv:2508.11865
        <https://arxiv.org/abs/2508.11865>`__). Each fragment is diagonal in its own per-mode basis,
        :math:`H_\nu = \mathcal{U}^{(\nu)\dagger} D_\nu\, \mathcal{U}^{(\nu)}`, where
        :math:`\mathcal{U}^{(\nu)} = \prod_l \mathcal{U}^{(\nu,l)}` is the product of the per-mode
        rotations and :math:`D_\nu` is diagonal, so

        .. math::

            e^{-i H_\nu \tau} = \mathcal{U}^{(\nu)\dagger}\, e^{-i D_\nu \tau}\, \mathcal{U}^{(\nu)} .

        This is implemented as a per-mode :class:`~.BasisRotation` (one on each mode's modal register),
        the diagonal rotations for :math:`e^{-i D_\nu \tau}` (see steps 3-4 below), and the inverse
        rotations. The one-body and two-body leaves store their per-mode rotation with opposite
        conventions, so the emitted rotation is :class:`~.BasisRotation` ``(leaf_tensors[0][l])`` for
        the one-body fragment (eigenvectors on the columns) and :class:`~.BasisRotation`
        ``(leaf_tensors[nu][l].T)`` for a two-body fragment (modal index on the rows). Consecutive
        fragment rotations telescope: at each fragment boundary the previous fragment's inverse
        rotation and the next fragment's rotation combine into a single per-mode
        :class:`~.BasisRotation`, so only one basis rotation is emitted per mode per boundary. The
        leaves are assumed real orthogonal (they are merged by transpose, not adjoint). Each leaf is
        first normalized to determinant :math:`+1` (negating one orbital line, a no-op on
        :math:`\tilde{n}_{lp}`) so that :class:`~.BasisRotation`'s real-orthogonal sign gauge is
        consistent across fragments; otherwise leaves with mixed determinants realize a different
        Hamiltonian.

        **3. One-body diagonal**
        The one-body generator is :math:`D_0 = \sum_{l,p} \epsilon_{lp}\, n_{lp}`, with
        :math:`\epsilon_{lp} =` ``core_tensors[0][l, l, p, p]``. Using :math:`n_{lp}=(I-Z_{lp})/2`,

        .. math::

            \epsilon_{lp}\, n_{lp} = \frac{\epsilon_{lp}}{2}\, I - \frac{\epsilon_{lp}}{2}\, Z_{lp} ,

        so the :math:`Z_{lp}` piece over duration :math:`\tau` is
        :math:`e^{+i(\epsilon_{lp}/2) Z_{lp} \tau} = RZ(-\epsilon_{lp} \tau)`. The one-body fragment is
        the central full :math:`\Delta t` term (visited once per step), so the emitted angle is
        :math:`-\epsilon_{lp} \Delta t = -2 \epsilon_{lp} \Delta t/2`, accumulating
        :math:`RZ(-\epsilon_{lp} t)` over the :math:`n` steps. The constant :math:`\epsilon_{lp}/2` is
        deferred to the global phase (step 5).

        **4. Two-body diagonal**
        A two-body generator is :math:`D_\nu = \sum_{l>m}\sum_{p,q}\lambda_{lmpq}\, n_{lp} n_{mq}`, with
        :math:`\lambda_{lmpq} =` ``core_tensors[nu][l, m, p, q]``. Using :math:`n=(I-Z)/2`,

        .. math::

            \lambda_{lmpq}\, n_{lp} n_{mq} =
                \frac{\lambda_{lmpq}}{4}\left(I - Z_{lp} - Z_{mq} + Z_{lp} Z_{mq}\right) ,

        a constant :math:`\lambda_{lmpq}/4`, single-site terms
        :math:`-\tfrac{\lambda_{lmpq}}{4}(Z_{lp} + Z_{mq})`, and the two-site term
        :math:`\tfrac{\lambda_{lmpq}}{4} Z_{lp} Z_{mq}`. In the *regrouped* input the single-site terms
        are already absorbed into ``core_tensors[0]`` and the constants into the global phase, so each
        two-body layer implements only the two-site term

        .. math::

            e^{-i(\lambda_{lmpq}/4) Z_{lp} Z_{mq}\, \tau}
                = \text{IsingZZ} \left(\lambda_{lmpq} \tfrac{\tau}{2}\right).

        Each two-body fragment is visited twice per step at :math:`\tau = \Delta t/2`, so the emitted
        angle is :math:`\tfrac{1}{2}\,\lambda_{lmpq}\,(\Delta t/2)` (the ``0.5`` prefactor in the code),
        accumulating :math:`IsingZZ(\lambda_{lmpq}\, t/2)` over the :math:`n` steps.

        **5. Constant terms (global phase / energy shift)**
        The code applies a single :class:`~.GlobalPhase` with angle :math:`s\,t` (i.e. :math:`e^{-ist}`),
        with

        .. math::

            s = C + \frac{1}{2}\sum_{l,p} \epsilon_{lp} ,

        i.e. ``nuc_constant`` plus ``trace(trace(core_tensors[0]))/2`` for the one-body
        :math:`n = (I - Z)/2` identity part (step 3). Restricted to the one-excitation-per-mode
        (unary/SBE) subspace the circuit acts on, each two-body :class:`~.IsingZZ` also carries an
        :math:`I`-component, so the two-body layers add a further
        :math:`\tfrac{1}{4}\sum_{\nu}\sum_{l>m}\sum_{p,q}\lambda^{(\nu)}_{lmpq}` identity phase that
        :math:`s` does *not* include. Consequently :math:`C =` ``nuc_constant`` must already have this
        balance folded in,

        .. math::

            C = C_\text{phys} - \frac{1}{4} \sum_{\nu}\sum_{l>m}\sum_{p,q} \lambda^{(\nu)}_{lmpq} ,

        where :math:`C_\text{phys}` is the bare scalar constant of :math:`H`. This is the convention the
        CGF construction pipeline stores; note it differs from :class:`~.TrotterCDF`, whose energy shift
        computes the analogous two-body contribution internally so that its ``nuc_constant`` is the bare
        constant.
    """

    dynamic_argnames = ("evolution_time",)
    hybrid_argnames = ("hamiltonian",)
    # `hybrid_argnames` and `compilable_argnames` cannot both be non-empty on the same
    # operator, so `num_trotter_steps` (a plain Python int that drives Python-level
    # control flow) is treated as `static_argnames` instead.
    static_argnames = ("num_trotter_steps", "double_phase")

    def __init__(
        self,
        evolution_time: int | AbstractArray,
        num_trotter_steps: float | AbstractArray,
        hamiltonian: CGFHamiltonian,
        wires: WiresLike | AbstractWires,
        double_phase=False,
    ):
        # ``hamiltonian`` is a hybrid argument: its array-like leaves must be arrays (or scalars)
        # to be captured and lowered correctly. Cast only list/tuple inputs, leaving array inputs
        # as-is so the hybrid argument survives the capture round-trip unchanged. Abstract capture
        # placeholders (which are neither list/tuple nor real arrays) are left untouched and skip
        # the ``ndim`` validation below, so the operator can be reconstructed during capture.

        if not isinstance(hamiltonian, CGFHamiltonian):
            raise ValueError(
                "TrotterCGF expects a CGFHamiltonian for the hamiltonian argument. Got "
                f"{type(hamiltonian)}."
            )

        Z = hamiltonian.core_tensors
        U = hamiltonian.leaf_tensors
        if hasattr(Z, "ndim") and hasattr(U, "ndim") and not (Z.ndim == 5 and U.ndim == 4):
            raise ValueError(
                "TrotterCGF expects a CGF Hamiltonian with core_tensors.ndim == 5 and "
                f"leaf_tensors.ndim == 4. Got core_tensors.ndim={Z.ndim}, "
                f"leaf_tensors.ndim={U.ndim}. For electronic (CDF) Hamiltonians, use TrotterCDF."
            )
        super().__init__(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase)


def _apply_system_basis_rotation(U, wires):
    """Apply a fragment's per-mode basis rotation on the whole system."""
    num_modes, n_states, _ = U.shape
    # The fragment's leaf O stores the "rotate-from-bare-to-diagonal-basis" direction.
    # qml.BasisRotation(U) with U = O^T implements the single-particle map O^T (moving
    # from bare modal states to the diagonal basis), so we pass the transpose.
    for l in range(num_modes):
        U_l = math.swapaxes(U[l], -2, -1)
        mode_wires = wires[l * n_states : (l + 1) * n_states]
        if math.is_abstract(U_l) or not np.allclose(U_l, np.eye(n_states)):
            BasisRotation(unitary_matrix=U_l, wires=mode_wires)


def _merge_leaves(U_prev, U_curr):
    """Per-mode fragment-rotation merge rule ``U_curr @ U_prev^T`` (vectorized via einsum).

    ``_apply_system_basis_rotation`` emits ``BasisRotation(leaf^T)``, so the single
    :class:`~.BasisRotation` that replaces the ``leaf_prev`` un-rotation followed by the
    ``leaf_curr`` rotation must be ``BasisRotation((U_curr @ U_prev^T)^T)`` for the frames
    to telescope. (This differs from :class:`~.TrotterCDF`, whose ``apply`` does not
    transpose and whose merge is therefore ``U_prev^dagger @ U_curr``.)
    """
    return math.einsum("lij,lkj->lik", U_curr, U_prev)


def _transpose_leaf(U):
    """Batch-transpose a leaf over the leading mode axis."""
    return math.swapaxes(U, -2, -1)


def _align_one_body_leaf(hamiltonian):
    """Transpose the one-body leaf so both sectors share the scaffolding's row convention.

    The scaffolding assumes each ``leaf_tensors[nu][l]`` stores its per-mode diagonalizing
    rotation with the modal index on the *rows* (the two-body :math:`U^{(l)}_{pa}` convention
    of `arXiv:2508.11865 <https://arxiv.org/abs/2508.11865>`__). The one-body leaf, however,
    is the eigenvector matrix of the effective one-body integrals and stores its eigenvectors
    as *columns*, so it is transposed here to match. Leaves are (special) orthogonal, so this
    is the inverse rotation.
    """
    leaves = hamiltonian.leaf_tensors
    leaves = math.concatenate([math.swapaxes(leaves[:1], -2, -1), leaves[1:]], axis=0)
    return CGFHamiltonian(
        core_tensors=hamiltonian.core_tensors,
        leaf_tensors=leaves,
        nuc_constant=hamiltonian.nuc_constant,
    )


def _normalize_leaf_determinant(hamiltonian):
    r"""Force every per-mode leaf to determinant ``+1`` so :class:`~.BasisRotation`'s real-orthogonal
    sign gauge is identical across fragments.

    :class:`~.BasisRotation` realizes a real orthogonal leaf only up to a determinant-dependent
    :math:`\pm 1` gauge, so leaves with *mixed* determinants (e.g. an ``eigh`` one-body leaf with
    ``det = -1`` next to ``expm`` two-body leaves with ``det = +1``) would be rotated into
    inconsistent bases and realize a different Hamiltonian. Negating one orbital line leaves the
    projector :math:`|v\rangle\langle v|`, and hence the fragment, unchanged, so this is a physical
    no-op. The orbital is stored on the *columns* of the one-body leaf and on the *rows* of the
    two-body leaves, so the two sectors negate a column and a row respectively.
    """
    leaves = hamiltonian.leaf_tensors
    signs = math.sign(math.linalg.det(leaves))  # (L+1, M)
    line = math.concatenate(
        [signs[..., None], math.ones_like(leaves[..., 0, 1:])], axis=-1
    )  # (L+1, M, N): +/-1 in the first slot, 1 elsewhere
    one_body = leaves[:1] * line[:1][..., None, :]  # eigenvectors on columns -> scale column 0
    two_body = leaves[1:] * line[1:][..., :, None]  # modal index on rows -> scale row 0
    return CGFHamiltonian(
        core_tensors=hamiltonian.core_tensors,
        leaf_tensors=math.concatenate([one_body, two_body], axis=0),
        nuc_constant=hamiltonian.nuc_constant,
    )


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, double_phase):
    """Apply the two-body ``IsingZZ`` layer (base / double-phase / genuine controlled).

    Genuine control and double-phase are mutually exclusive constructions for a controlled
    ``IsingZZ``, chosen once here via ``is_double_phase``: genuine control sandwiches each
    ``IsingZZ`` individually (inside :func:`~._emit_two_body_isingzz`); double-phase instead
    shares *one* ``CNOT`` sandwich across every term touching a given ``wire_lp``, which is
    cheaper since ``IsingZZ`` itself stays uncontrolled either way.
    """
    num_modes = Z.shape[0]
    n_states = Z.shape[2]
    # Double-phase assumes a single control wire; ``register_condition`` below enforces this.
    is_double_phase = len(control_wires) == 1 and double_phase

    for l in range(1, num_modes):
        for m in range(l):  # strict lower triangle: l > m
            Z_lm = Z[l, m]

            @for_loop(n_states)
            def _p_loop(p, Z_lm=Z_lm, l=l, m=m):
                wire_lp = wires[l * n_states + p]

                # Symmetrization is already taken into account here.
                def _angle(q):
                    return 0.5 * Z_lm[p, q] * first_order_time_step

                if is_double_phase:

                    @for_loop(n_states)
                    def _q_loop(q, wire_lp=wire_lp, m=m):
                        wire_mq = wires[m * n_states + q]
                        IsingZZ(_angle(q), [wire_lp, wire_mq])

                    CNOT([control_wires[0], wire_lp])
                    _q_loop()
                    CNOT([control_wires[0], wire_lp])
                else:

                    @for_loop(n_states)
                    def _q_loop(q, wire_lp=wire_lp, m=m):
                        wire_mq = wires[m * n_states + q]
                        _emit_two_body_isingzz(_angle(q), wire_lp, wire_mq, control_wires)

                    _q_loop()

            _p_loop()


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires, double_phase):
    """Apply the one-body ``RZ`` layer (base / double-phase / genuine controlled)."""
    num_modes = Z_one_body.shape[0]
    n_states = Z_one_body.shape[2]

    @for_loop(num_modes)
    def mode_loop(l):

        @for_loop(n_states)
        def modal_loop(p):
            wire_lp = wires[l * n_states + p]
            # One-body prefactor derivation:
            #   n^l_p = (I - Z_{l,p}) / 2  ->  Z-piece has coefficient -eps/2
            #   target operator: exp(-i eps n t) has Z-piece exp(+i eps t / 2 Z)
            #     which equals RZ(-eps t).
            #   This fragment is visited ONCE per Trotter step, at duration
            #     first_order_time_step = dt_trotter / 2, so to accumulate a total
            #     RZ angle of -eps * dt_trotter we need angle-per-step
            #     = -eps * dt_trotter = -2 * eps * first_order_time_step. So alpha_oneB = -2.
            angle = -2.0 * Z_one_body[l, l, p, p] * first_order_time_step
            _emit_one_body_rz(angle, wire_lp, control_wires, double_phase)

        modal_loop()

    mode_loop()


def _energy_shift(hamiltonian):
    """Zero-of-energy shift applied as a ``GlobalPhase`` (or control ``RZ``)."""
    nuc_constant = hamiltonian.nuc_constant
    Z_tensor = hamiltonian.core_tensors
    # One-body diagonal: nested traces over axes (-2, -1) twice pick out the
    # (l == m, p == q) entries and sum eps^l_p.
    one_body_diag = math.trace(
        math.trace(Z_tensor[0], axis1=-2, axis2=-1),
        axis1=-2,
        axis2=-1,
    )  # scalar: sum_{l, p} eps^l_p
    return nuc_constant + one_body_diag / 2


_CGF_HELPERS = {
    "apply_system_basis_rotation": _apply_system_basis_rotation,
    "apply_two_body_diagonal": _apply_two_body_diagonal,
    "apply_one_body_diagonal": _apply_one_body_diagonal,
    "merge_leaves": _merge_leaves,
    "transpose_leaf": _transpose_leaf,
}


def _cgf_resource_counts(num_trotter_steps, hamiltonian, has_control, double_phase=False):
    """Shared (upper-bound) gate counts for the base and controlled CGF circuits.

    The exact gate count can be lower at runtime because fragments whose basis rotation
    happens to be the identity are skipped when the Hamiltonian data is concrete (not
    traced); this estimate assumes no such fragment is skipped.
    """
    if num_trotter_steps <= 0:
        return {}

    leaf_tensors = hamiltonian.leaf_tensors
    num_two_body_fragments = leaf_tensors.shape[0] - 1
    num_modes = leaf_tensors.shape[1]
    n_states = leaf_tensors.shape[2]

    resources = defaultdict(int)
    num_sysrot_calls = num_trotter_steps * (2 * num_two_body_fragments + 1) + 1
    num_twobody_blocks = num_trotter_steps * 2 * num_two_body_fragments
    num_onebody_blocks = num_trotter_steps
    num_pairs = num_modes * (num_modes - 1) // 2
    num_twobody_rotations = num_twobody_blocks * num_pairs * n_states**2
    num_onebody_rotations = num_onebody_blocks * num_modes * n_states

    sysrot_key = BasisRotation(Complex[n_states, n_states], wires=Wire[n_states])
    resources[sysrot_key] += num_modes * num_sysrot_calls

    if not has_control:
        resources[IsingZZ] += num_twobody_rotations
        resources[RZ] += num_onebody_rotations
        resources[GlobalPhase] += 1
    elif double_phase:
        # Double-phase (Fig. 6 https://arxiv.org/abs/2506.15784): bare IsingZZ / RZ rotations, plus one CNOT pair around
        # each diagonal block, plus an RZ on the control wire for the global phase.
        resources[IsingZZ] += num_twobody_rotations
        resources[RZ] += num_onebody_rotations
        resources[CNOT] += num_twobody_blocks * num_pairs * 2 * n_states
        resources[CNOT] += num_onebody_blocks * 2 * num_modes * n_states
        resources[RZ] += 1
    else:
        # Genuine controlled: each IsingZZ -> controlled-IsingZZ (4 CNOT + 2 RZ) and each
        # RZ -> controlled-RZ (2 CNOT + 2 RZ); the global phase becomes a PhaseShift on
        # the control wire. There are no bare IsingZZ gates.
        resources[RZ] += 2 * num_twobody_rotations + 2 * num_onebody_rotations
        resources[CNOT] += 4 * num_twobody_rotations + 2 * num_onebody_rotations
        resources[PhaseShift] += 1

    return dict(resources)


def _trotter_cgf_resources(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase):
    return _cgf_resource_counts(num_trotter_steps, hamiltonian, has_control=False)


@register_resources(_trotter_cgf_resources, exact=False)
def _trotter_cgf_decomposition(evolution_time, num_trotter_steps, hamiltonian, wires, double_phase):
    # ``double_phase`` only affects the controlled decomposition; the base operator is
    # always the plain (uncontrolled) e^{-iHt} circuit.
    if num_trotter_steps > 0:
        _run_trotter_steps(
            evolution_time,
            num_trotter_steps,
            _align_one_body_leaf(_normalize_leaf_determinant(hamiltonian)),
            wires,
            (),
            **_CGF_HELPERS,
        )
        phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
        GlobalPhase(phi)


add_decomps(TrotterCGF, _trotter_cgf_decomposition)


def _controlled_trotter_cgf_resource(
    base, control_wires, control_values, work_wires, work_wire_type
):
    return _cgf_resource_counts(
        base.arguments["num_trotter_steps"],
        base.arguments["hamiltonian"],
        has_control=True,
        double_phase=base.arguments["double_phase"],
    )


@register_condition(lambda control_wires, **__: len(control_wires) == 1)
@register_resources(_controlled_trotter_cgf_resource, exact=False)
def _controlled_trotter_cgf_decomp(base, control_wires, control_values, work_wires, work_wire_type):
    evolution_time = base.arguments["evolution_time"]
    num_trotter_steps = base.arguments["num_trotter_steps"]
    hamiltonian = base.arguments["hamiltonian"]
    wires = base.arguments["wires"]
    double_phase = base.arguments["double_phase"]

    if num_trotter_steps == 0:
        return

    phi = (_energy_shift(hamiltonian) * evolution_time) % (4 * np.pi)
    hamiltonian = _align_one_body_leaf(_normalize_leaf_determinant(hamiltonian))

    if double_phase:
        # Double-phase (Fig. 6 https://arxiv.org/abs/2506.15784) circuit: each full-time diagonal block is CNOT-sandwiched by
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
            **_CGF_HELPERS,
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
        **_CGF_HELPERS,
    )
    PhaseShift(-phi, control_wires)


add_decomps("C(TrotterCGF)", flip_zero_control2(_controlled_trotter_cgf_decomp))
