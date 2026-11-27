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
"""Contains the RZLayer template."""

from pennylane2 import capture, compiler, math
from pennylane2.control_flow import for_loop
from pennylane2.core.operator import Operator2
from pennylane2.decomposition import add_decomps, register_resources
from pennylane2.ops import RZ
from pennylane2.ops.op_math.adjoint2 import _adjoint_negate_angle as adjoint_rotation2
from pennylane2.typing import Float, TensorLike, Wire
from pennylane2.wires import Wires, WiresLike, validate_no_wire_overlaps


class RZLayer(Operator2):
    r"""Applies :class:`~.RZ` rotations by the same angle to each of its wires.

    For :math:`m` wires, this is

    .. math::

        \operatorname{RZLayer}(\theta) = \prod_{j=0}^{m-1} R_Z(\theta)_{\texttt{wires}[j]}.

    Args:
        angle (TensorLike): the rotation angle :math:`\theta`
        wires (WiresLike): the :math:`m` wires the rotations act on
        work_wires (WiresLike): auxiliary wires in the zero state that decompositions may use and
            return to the zero state. Defaults to ``()``.

    Raises:
        ValueError: if ``wires`` and ``work_wires`` overlap

    **Example**

    >>> qp.RZLayer(0.5, wires=[0, 1, 2]).decomposition()
    [RZ(0.5, wires=[0]), RZ(0.5, wires=[1]), RZ(0.5, wires=[2])]

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.RZLayer
            :symbols: :math:`m` wires and :math:`w` ``work_wires``
            :example: :math:`m=3`, :math:`w=1`

            .. decomposition-rule:: to_rz
                :image: /_static/decomposition_thumbnails/rz_layer_individual.svg

                :class:`~.RZ` | :math:`m` | 3

            .. decomposition-rule:: negate_angle
                :image: /_static/decomposition_thumbnails/rz_layer_adjoint.svg
                :region: adjoint
                :where: ``angle`` changes sign

                :class:`~.RZLayer` | :math:`1`
    """

    dynamic_argnames = ("angle",)
    wire_argnames = ("wires", "work_wires")
    arg_specs = {"angle": Float, "wires": Wire[-1], "work_wires": Wire[-1]}

    grad_method = None

    def __init__(self, angle: TensorLike, wires: WiresLike, work_wires: WiresLike = ()):
        wires = Wires(wires)
        work_wires = Wires(work_wires)
        validate_no_wire_overlaps({"wires": wires, "work_wires": work_wires})

        super().__init__(angle, wires=wires, work_wires=work_wires)

    @property
    def wires(self) -> Wires:
        """All wires involved in the operation, with the work wires last."""
        return self.arguments["wires"] + self.work_wires

    def adjoint(self) -> "RZLayer":
        return RZLayer(-self.angle, **self.wire_args)


@register_resources(lambda wires, **_: {RZ: len(wires)}, name="to_rz")
def _rzlayer_individual(angle, wires, **_):
    if capture.enabled() or compiler.active():
        wires = math.array(wires, like="jax")

    def _rz(i):
        RZ(angle, wires=wires[i])

    for_loop(len(wires))(_rz)()


add_decomps(RZLayer, _rzlayer_individual)
add_decomps("Adjoint(RZLayer)", adjoint_rotation2)
