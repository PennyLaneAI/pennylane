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

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
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
    r"""Applies the same :class:`~.SingleExcitation` to two pairs of wires, for example to two spin channels.

    For the wires :math:`\{p_\uparrow , q_\uparrow , p_\downarrow, q_\downarrow\}`, this is

    .. math::

        \operatorname{GivensPair}(\phi) = G(\phi)_{p_\uparrow q_\uparrow}
        G(\phi)_{p_\downarrow q_\downarrow},

    where :math:`G(\phi)` is a :class:`~.SingleExcitation`. Basis rotations that act identically
    on both spin channels consist of such pairs.

    Args:
        phi (TensorLike): rotation angle :math:`\phi`
        wires (WiresLike): the four wires :math:`\{p_\uparrow , q_\uparrow , p_\downarrow, q_\downarrow\}`
        work_wires (WiresLike): zeroed auxiliary wires that decompositions may use.

    Raises:
        ValueError: if the number of wires is not four, or if ``wires`` and ``work_wires`` overlap

    It decomposes into two :class:`~.SingleExcitation` gates or, in a Clifford frame, into a single
    :class:`~.RZLayer` that applies :math:`\phi/2` to all four wires, see the table below.

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

    def adjoint(self) -> "GivensPair":
        return GivensPair(-self.phi, **self.wire_args)


@register_resources({SingleExcitation: 2}, name="to_single_excitations")
def _givens_pair_two_single_excitations(phi, wires, **_):
    if capture.enabled() or compiler.active():
        wires = math.array(wires, like="jax")

    def _pair(i):
        SingleExcitation(phi, wires=[wires[2 * i], wires[2 * i + 1]])

    for_loop(2)(_pair)()


@subcircuit(arg_specs={"wires": Wire[4]})
@register_resources({PPR_4(1, "XX", wires=Wire[2]): 2, PauliX: 2})
def _GivensPairFrame(wires: WiresLike):
    if capture.enabled() or compiler.active():
        wires = math.array(wires, like="jax")

    def _pair(i):
        PPR_4(1, "XX", wires=[wires[2 * i], wires[2 * i + 1]])
        PauliX(wires[2 * i + 1])

    for_loop(2)(_pair)()


def _givens_pair_rzlayer_resources(work_wires, **_):
    frame = _GivensPairFrame(Wire[4])
    rotations = RZLayer(Float, Wire[4], Wire[len(work_wires)])
    return {_change_op_basis_abstract(frame, rotations, adjoint(frame)): 1}


@register_resources(_givens_pair_rzlayer_resources, name="to_rz_layer")
def _givens_pair_rzlayer(phi, wires, work_wires):
    r"""In the frame of ``_single_excitation_ppr_rz``, a SingleExcitation on ``(p, q)`` is
    :math:`RZ(\phi/2)_p RZ(-\phi/2)_q`. Conjugating ``q`` with ``PauliX`` flips the second angle,
    so that both pairs together become four rotations by the same angle :math:`\phi/2`."""
    change_op_basis(_GivensPairFrame(wires), RZLayer(phi / 2, wires, work_wires))


add_decomps(GivensPair, _givens_pair_two_single_excitations, _givens_pair_rzlayer)
add_decomps("Adjoint(GivensPair)", adjoint_rotation2)
