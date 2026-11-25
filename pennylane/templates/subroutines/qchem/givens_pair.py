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
"""Contains the GivensPair template."""

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import PPR_4, PauliX, SingleExcitation, adjoint, change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle as adjoint_rotation2
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.ops.subcircuit import subcircuit
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from ..rz_layer import RZLayer


class GivensPair(Operator2):
    r"""Applies the same :class:`~.SingleExcitation` to a spin-up and a spin-down pair of wires.

    For the wires ``[p_up, q_up, p_down, q_down]``, this is

    .. math::

        \operatorname{GivensPair}(\phi) = G(\phi)_{p_\uparrow q_\uparrow}
        G(\phi)_{p_\downarrow q_\downarrow},

    where :math:`G(\phi)` is a :class:`~.SingleExcitation`. Basis rotations that act identically
    on both spin channels consist of such pairs.

    Args:
        phi (TensorLike): rotation angle :math:`\phi`
        wires (WiresLike): the four wires ``[p_up, q_up, p_down, q_down]``
        work_wires (WiresLike): auxiliary wires in the zero state that decompositions may use and
            return to the zero state. They are included in ``op.wires``. Defaults to ``()``.

    Raises:
        ValueError: if the number of wires is not four, or if ``wires`` and ``work_wires`` overlap

    **Example**

    >>> op = qp.GivensPair(0.4, wires=[0, 1, 2, 3])
    >>> op.decomposition()
    [SingleExcitation(0.4, wires=[0, 1]), SingleExcitation(0.4, wires=[2, 3])]

    In a Clifford frame, both excitations together become a single :class:`~.RZLayer` that applies
    the angle :math:`\phi/2` to all four wires, so that decompositions of ``RZLayer`` can exploit
    the four equal angles.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.GivensPair

            .. decomposition-rule:: to_single_excitations
                :image: /_static/decomposition_thumbnails/givens_pair_single_excitations.svg

                :class:`~.SingleExcitation` | :math:`2`

            .. decomposition-rule:: to_rz_layer
                :image: /_static/decomposition_thumbnails/givens_pair_rz_layer.svg
                :where: the :class:`~.RZLayer` applies :math:`\phi/2` to all four wires and
                    receives the ``work_wires``

                :class:`PPR_4(1, "XX") <.PPR_4>` | :math:`2`
                :class:`PPR_4(-1, "XX") <.PPR_4>` | :math:`2`
                :class:`~.X` | :math:`4`
                :class:`~.RZLayer` on four wires | :math:`1`

            .. decomposition-rule:: negate_angle
                :image: /_static/decomposition_thumbnails/givens_pair_adjoint.svg
                :region: adjoint
                :where: ``phi`` changes sign

                :class:`~.GivensPair` | :math:`1`
    """

    dynamic_argnames = ("phi",)
    wire_argnames = ("wires", "work_wires")
    arg_specs = {"phi": Float, "wires": Wire[4], "work_wires": Wire[-1]}

    grad_method = None

    def __init__(self, phi: TensorLike, wires: WiresLike, work_wires: WiresLike = ()):
        wires = Wires(wires)
        work_wires = Wires(work_wires)
        validate_no_wire_overlaps({"wires": wires, "work_wires": work_wires})

        super().__init__(phi, wires=wires, work_wires=work_wires)

    @property
    def wires(self) -> Wires:
        """All wires involved in the operation, with the work wires last."""
        return self.arguments["wires"] + self.work_wires

    def matrix(self, wire_order: WiresLike | None = None) -> TensorLike:
        single_excitation = SingleExcitation.compute_matrix(self.phi)
        canonical_matrix = math.kron(
            math.kron(single_excitation, single_excitation), math.eye(2 ** len(self.work_wires))
        )
        return self._expand_canonical_matrix(canonical_matrix, wire_order)

    def adjoint(self) -> "GivensPair":
        return GivensPair(-self.phi, **self.wire_args)


@register_resources({SingleExcitation: 2}, name="to_single_excitations")
def _givens_pair_two_single_excitations(phi, wires, **_):
    SingleExcitation(phi, wires=wires[:2])
    SingleExcitation(phi, wires=wires[2:])


@subcircuit(arg_specs={"wires": Wire[4]})
@register_resources({PPR_4(1, "XX", wires=Wire[2]): 2, PauliX: 2})
def _GivensPairFrame(wires: WiresLike):
    for pair in (wires[:2], wires[2:]):
        PPR_4(1, "XX", wires=pair)
        PauliX(pair[1])


def _givens_pair_rzlayer_resources(work_wires, **_):
    frame = _GivensPairFrame(Wire[4])
    rotations = RZLayer(Float, Wire[4], Wire[len(work_wires)])
    return {_change_op_basis_abstract(frame, rotations, adjoint(frame)): 1}


@register_resources(_givens_pair_rzlayer_resources, name="to_rz_layer")
def _givens_pair_rzlayer(phi, wires, work_wires):
    r"""In the frame of ``_single_excitation_ppr_rz``, a SingleExcitation on ``(p, q)`` is
    :math:`RZ(\phi/2)_p RZ(-\phi/2)_q`. Conjugating ``q`` with ``PauliX`` flips the second angle,
    so that both pairs become four rotations by the same angle :math:`\phi/2`."""
    change_op_basis(_GivensPairFrame(wires), RZLayer(phi / 2, wires, work_wires))


add_decomps(GivensPair, _givens_pair_two_single_excitations, _givens_pair_rzlayer)
add_decomps("Adjoint(GivensPair)", adjoint_rotation2)
