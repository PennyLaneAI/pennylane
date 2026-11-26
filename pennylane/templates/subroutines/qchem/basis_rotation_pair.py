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
"""Contains the BasisRotationPair template."""

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import GlobalPhase, PauliZ, PhaseShift, cond
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from ..rz_layer import RZLayer
from .basis_rotation import (
    BasisRotation,
    _adjust_determinant,
    _is_jax_jit,
    _prepare_args,
    _qjit_or_capture,
)
from .givens_pair import GivensPair


class BasisRotationPair(Operator2):
    r"""Applies the same :class:`~.BasisRotation` to the spin-up and to the spin-down wires.

    For an :math:`N\times N` unitary matrix :math:`u`, this is

    .. math::

        \operatorname{BasisRotationPair}(u) = U(u)_{\uparrow} U(u)_{\downarrow},

    where :math:`U(u)` is the basis rotation of :class:`~.BasisRotation`, applied to the
    :math:`N` spin-up and to the :math:`N` spin-down wires, respectively. Both rotations consist of
    the same Givens rotations and phase shifts, so that decompositions can apply each Givens
    rotation as one :class:`~.GivensPair` and each phase shift as two equal rotations in an
    :class:`~.RZLayer`, whose decompositions can exploit the equal angles.

    Args:
        unitary_matrix (TensorLike): :math:`N\times N` matrix specifying the basis transformation
        wires_up (WiresLike): the :math:`N` spin-up wires
        wires_down (WiresLike): the :math:`N` spin-down wires
        work_wires (WiresLike): zeroed auxiliary wires that decompositions may use.
        check (bool): test unitarity of the provided ``unitary_matrix``

    Raises:
        ValueError: if the matrix is not square, if ``wires_up`` and ``wires_down`` differ in
            length or contain fewer than two wires each, or if the wire arguments overlap
        ValueError: if ``check=True`` and the matrix is not unitary

    **Example**

    By default, the operator decomposes into one :class:`~.BasisRotation` per spin:

    >>> U = np.array([[0.6, -0.8], [0.8, 0.6]])
    >>> op = qp.BasisRotationPair(U, wires_up=[0, 2], wires_down=[1, 3])
    >>> [(o.name, o.wires) for o in op.decomposition()]
    [('BasisRotation', Wires([0, 2])), ('BasisRotation', Wires([1, 3]))]

    Alternatively, the ``real`` and ``complex`` rules apply each Givens rotation to both spins as
    one :class:`~.GivensPair`. For the real-valued matrix above:

    >>> rule = qp.list_decomps(qp.BasisRotationPair)["real"]
    >>> [(o.name, o.wires) for o in qp.tape.make_qscript(rule)(**op.arguments).operations]
    [('GivensPair', Wires([0, 2, 1, 3]))]

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.BasisRotationPair
            :stacked:
            :symbols: :math:`N \times N` matrix :math:`u` (``unitary_matrix``) on :math:`N`
                spin-up and :math:`N` spin-down wires
            :example: :math:`N=3`, with :math:`\det u = -1` for the real-valued rule

            .. decomposition-rule:: to_basis_rotations
                :image: /_static/decomposition_thumbnails/basis_rotation_pair_basis_rotations.svg

                :class:`~.BasisRotation` on :math:`N` wires | :math:`2` | 2

            .. decomposition-rule:: real
                :image: /_static/decomposition_thumbnails/basis_rotation_pair_real_givens.svg
                :requires: real-valued :math:`u`
                :where: :math:`[\cdot]` is 1 if the condition holds and 0 otherwise, and the
                    :class:`~.GivensPair` gates receive the ``work_wires``

                :class:`~.PauliZ` | :math:`2\,[\det u = -1]` | 2
                :class:`~.GivensPair` | :math:`N(N-1)/2` | 3

            .. decomposition-rule:: complex
                :image: /_static/decomposition_thumbnails/basis_rotation_pair_complex_givens.svg
                :requires: complex-valued :math:`u`
                :where: the :class:`~.GivensPair` and :class:`~.RZLayer` gates receive the
                    ``work_wires``

                :class:`~.RZLayer` on two wires | :math:`\leq N(N+1)/2` | 6
                :class:`~.GivensPair` | :math:`N(N-1)/2` | 3
                :class:`~.GlobalPhase` | :math:`\leq N(N-1)/2 + 1` | 4
    """

    dynamic_argnames = ("unitary_matrix",)
    wire_argnames = ("wires_up", "wires_down", "work_wires")
    compilable_argnames = ("check",)

    # As for BasisRotation, 'unitary_matrix' is deliberately left out of the 'arg_specs'
    # so that real-valued and complex-valued matrices decompose differently.
    arg_specs = {"wires_up": Wire[-1], "wires_down": Wire[-1], "work_wires": Wire[-1]}

    grad_method = None

    def __init__(
        self,
        unitary_matrix: TensorLike,
        wires_up: WiresLike,
        wires_down: WiresLike,
        work_wires: WiresLike = (),
        check: bool = False,
    ):
        M, N = math.shape(unitary_matrix)

        if M != N:
            raise ValueError(f"The unitary matrix should be of shape NxN, got {(M, N)}")

        if check:
            if not math.is_tracer(unitary_matrix) and not math.allclose(
                unitary_matrix @ math.conj(unitary_matrix).T,
                math.eye(M, dtype=complex),
                atol=1e-4,
            ):
                raise ValueError("The provided transformation matrix should be unitary.")

        wires_up, wires_down, work_wires = Wires(wires_up), Wires(wires_down), Wires(work_wires)
        if len(wires_up) != len(wires_down):
            raise ValueError(
                "wires_up and wires_down must have the same number of wires, got "
                f"{len(wires_up)} and {len(wires_down)}."
            )
        if len(wires_up) < 2:
            raise ValueError(
                f"This template requires at least two wires per spin, got {len(wires_up)}"
            )
        validate_no_wire_overlaps(
            {"wires_up": wires_up, "wires_down": wires_down, "work_wires": work_wires}
        )

        super().__init__(
            unitary_matrix, wires_up=wires_up, wires_down=wires_down, work_wires=work_wires
        )

    @property
    def wires(self) -> Wires:
        """All wires involved in the operation, with the work wires last."""
        return self.wires_up + self.wires_down + self.work_wires


def _basis_rotation_pair_two_basis_rotations_resources(unitary_matrix, wires_up, **_):
    return {BasisRotation(unitary_matrix, wires=Wire[len(wires_up)]): 2}


@register_resources(_basis_rotation_pair_two_basis_rotations_resources, name="to_basis_rotations")
def _basis_rotation_pair_two_basis_rotations(unitary_matrix, wires_up, wires_down, **_):
    BasisRotation(unitary_matrix, wires=wires_up)
    BasisRotation(unitary_matrix, wires=wires_down)


def _real_basis_rotation_pair_givens_pairs_resources(unitary_matrix, wires_up, work_wires, **_):
    """Upper bound on the gates emitted by '_real_basis_rotation_pair_givens_pairs'.

    - GivensPair: Exact count of N(N-1)/2 for the Givens rotations.
    - PauliZ: *At most* two for the determinant correction, one per spin; skipped if det = +1.
      Under ``jax.jit`` without qjit or program capture, these are PhaseShift gates instead.

    """
    num_wires = len(wires_up)
    correction = PhaseShift if _is_jax_jit(unitary_matrix) else PauliZ
    givens_pair = GivensPair(Float, Wire[4], Wire[len(work_wires)])

    return {correction: 2, givens_pair: num_wires * (num_wires - 1) // 2}


@register_condition(lambda unitary_matrix, **_: not math.is_complex_dtype(unitary_matrix))
@register_resources(_real_basis_rotation_pair_givens_pairs_resources, exact=False, name="real")
def _real_basis_rotation_pair_givens_pairs(unitary_matrix, wires_up, wires_down, work_wires, **_):
    unitary, wires_up = _prepare_args(unitary_matrix, wires_up)
    _, wires_down = _prepare_args(unitary_matrix, wires_down)

    was_negative, unitary = _adjust_determinant(unitary)

    if _is_jax_jit(was_negative):
        # Plain jax.jit cannot branch on the traced determinant, so apply PauliZ as PhaseShift(pi).
        phase = math.where(was_negative, np.pi, 0.0)
        PhaseShift(phase, wires_up[0])
        PhaseShift(phase, wires_down[0])
    else:

        def _flip_signs():
            PauliZ(wires_up[0])
            PauliZ(wires_down[0])

        cond(was_negative, _flip_signs)()

    _, givens_list = math.decomposition.givens_decomposition(unitary)
    givens_matrices, givens_ids = zip(*givens_list, strict=True)

    if _qjit_or_capture():
        givens_ids = math.array(givens_ids, like="jax")
        givens_matrices = math.array(givens_matrices, like="jax")

    def _givens_pair(idx):
        grot_mat = givens_matrices[idx]
        i, j = givens_ids[idx]
        theta = math.arctan2(math.real(grot_mat[1, 0]), math.real(grot_mat[0, 0]))
        GivensPair(2 * theta, [wires_up[i], wires_up[j], wires_down[i], wires_down[j]], work_wires)

    for_loop(len(givens_list))(_givens_pair)()


def _complex_basis_rotation_pair_givens_pairs_resources(wires_up, work_wires, **_):
    """Upper bound on the gates emitted by '_complex_basis_rotation_pair_givens_pairs'.

    - GivensPair: Exact count of N(N-1)/2 for the Givens rotations.
    - RZLayer on a pair of wires: Exact count of N for the diagonal phases, and *at most*
      N(N-1)/2 for the phases of the Givens rotations; skipped if the phase is zero.
    - GlobalPhase: Exact count of one for the diagonal phases, and *at most* N(N-1)/2 for the
      phases of the Givens rotations.

    """
    num_wires = len(wires_up)
    num_givens = num_wires * (num_wires - 1) // 2
    num_work_wires = len(work_wires)

    return {
        RZLayer(Float, Wire[2], Wire[num_work_wires]): num_wires + num_givens,
        GlobalPhase: 1 + num_givens,
        GivensPair(Float, Wire[4], Wire[num_work_wires]): num_givens,
    }


@register_condition(lambda unitary_matrix, **_: math.is_complex_dtype(unitary_matrix))
@register_resources(
    _complex_basis_rotation_pair_givens_pairs_resources, exact=False, name="complex"
)
def _complex_basis_rotation_pair_givens_pairs(
    unitary_matrix, wires_up, wires_down, work_wires, **_
):
    unitary, wires_up = _prepare_args(unitary_matrix, wires_up)
    _, wires_down = _prepare_args(unitary_matrix, wires_down)

    phase_list, givens_list = math.decomposition.givens_decomposition(unitary)
    givens_matrices, givens_ids = zip(*givens_list, strict=True)

    if _qjit_or_capture():
        givens_ids = math.array(givens_ids, like="jax")
        givens_matrices = math.array(givens_matrices, like="jax")

    # PhaseShift(a) on both spins equals GlobalPhase(-a) times RZ(a) on both spins.
    phases = math.angle(phase_list)

    def _diagonal_phase(i):
        RZLayer(phases[i], [wires_up[i], wires_down[i]], work_wires)

    for_loop(len(wires_up))(_diagonal_phase)()
    GlobalPhase(-math.sum(phases))

    def _phase_pair(phi, i):
        GlobalPhase(-phi)
        RZLayer(phi, [wires_up[i], wires_down[i]], work_wires)

    def _givens_pair(idx):
        grot_mat = givens_matrices[idx]
        i, j = givens_ids[idx]
        theta = math.arctan2(-math.real(grot_mat[0, 1]), math.real(grot_mat[1, 1]))
        # The first column of grot_mat is e^{i phi} (cos(theta), sin(theta)) with both entries
        # non-negative before the phase, so their sum always carries the phase.
        phi = math.angle(grot_mat[0, 0] + grot_mat[1, 0])
        GivensPair(2 * theta, [wires_up[i], wires_up[j], wires_down[i], wires_down[j]], work_wires)
        if _is_jax_jit(phi):
            _phase_pair(phi, i)
        else:
            cond(math.logical_not(math.allclose(phi, 0.0)), _phase_pair)(phi, i)

    for_loop(len(givens_list))(_givens_pair)()


add_decomps(
    BasisRotationPair,
    _basis_rotation_pair_two_basis_rotations,
    _real_basis_rotation_pair_givens_pairs,
    _complex_basis_rotation_pair_givens_pairs,
)
