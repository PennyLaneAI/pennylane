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

from pennylane import capture, compiler, math
from pennylane.allocation import allocate
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import RZ, GlobalPhase, PhaseShift, subcircuit
from pennylane.ops.op_math import ChangeOpBasis2, change_op_basis
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle as adjoint_rotation2
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .arithmetic.hamming_weight import _HammingWeight


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

            .. decomposition-rule:: hamming_weight_phasing
                :image: /_static/decomposition_thumbnails/rz_layer_hwp_synthesis.svg
                :lightbox:
                :requires: :math:`m \geq 3` and :math:`a` work wires
                :where: :math:`n = m - 1` if :math:`m` is a power of two and :math:`n = m`
                    otherwise, :math:`a = n - \operatorname{popcount}(n)`, and
                    :math:`\ell(x) = \lfloor \log_2 x \rfloor + 1` is the number of bits of
                    :math:`x`

                :class:`~.TemporaryAND` | :math:`a` | 1
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`a` | 1
                :class:`~.CNOT` | :math:`2(5n - 4\ell(n) - \operatorname{popcount}(n))` | 10
                :class:`~.PhaseShift` | :math:`\ell(m)` | 2
                :class:`~.GlobalPhase` | :math:`1` | 1

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


def _hwp_num_hw_wires(num_wires):
    """Helper method for Hamming-weight-phasing (HWP).

    Number of wires for which the synthesis rule computes the Hamming weight: either all of them or
    one less, which is then rotated separately. Prefers fewer rotations, then fewer ANDs."""

    def _cost(num_hw_wires):
        num_rotations = num_hw_wires.bit_length() + num_wires - num_hw_wires
        return num_rotations, num_hw_wires - num_hw_wires.bit_count()

    return min(num_wires, num_wires - 1, key=_cost)


@subcircuit(dynamic_argnames=("angle",), wire_argnames=("wires", "extra_wires"))
@register_resources(lambda wires, extra_wires, **_: {PhaseShift: len(wires) + len(extra_wires)})
def _HWPPhaseShifts(angle, wires, extra_wires):
    """Rotate the wire of bit ``j`` of a Hamming weight, ``wires[j]``, by ``2**j * angle`` and each
    of the ``extra_wires`` by ``angle``."""
    for j, wire in enumerate(wires):
        PhaseShift(2**j * angle, wires=wire)
    for wire in extra_wires:
        PhaseShift(angle, wires=wire)


def _rzlayer_hwp_synthesis_resources(wires, **_):
    """Resources of the Hamming-weight-phasing (HWP) decomposition."""
    num_hw_wires = _hwp_num_hw_wires(len(wires))
    num_work = num_hw_wires - num_hw_wires.bit_count()
    compute_op = _HammingWeight(Wire[num_hw_wires], Wire[num_work])
    target_op = _HWPPhaseShifts(
        Float, Wire[num_hw_wires.bit_length()], Wire[len(wires) - num_hw_wires]
    )
    return {ChangeOpBasis2(compute_op, target_op): 1, GlobalPhase: 1}


def _rzlayer_hwp_synthesis_work_wires(wires, work_wires, **_):
    num_hw_wires = _hwp_num_hw_wires(len(wires))
    num_work = num_hw_wires - num_hw_wires.bit_count()
    return {"zeroed": max(num_work - len(work_wires), 0)}


@register_condition(lambda wires, **_: len(wires) >= 3)
@register_resources(
    _rzlayer_hwp_synthesis_resources,
    work_wires=_rzlayer_hwp_synthesis_work_wires,
    name="hamming_weight_phasing",
)
def _rzlayer_hwp_synthesis(angle, wires, work_wires):
    """Hamming-weight phasing (HWP) decomposition: :math:`m` rotations ``RZ(theta)`` equal
    ``PhaseShift(2**j * theta)`` on bit :math:`j` of the Hamming weight of their wires and
    a ``GlobalPhase(m * theta / 2)``."""
    num_hw_wires = _hwp_num_hw_wires(len(wires))
    num_work = num_hw_wires - num_hw_wires.bit_count()
    work_wires = work_wires[:num_work]
    hw_wires, extra_wires = wires[:num_hw_wires], wires[num_hw_wires:]

    with allocate(num_work - len(work_wires), restored=True) as extra_work_wires:
        work_wires = [*work_wires, *extra_work_wires]
        out_wires = [hw_wires[-1], *work_wires[: num_hw_wires.bit_length() - 1]]
        phase_shifts = _HWPPhaseShifts(angle, out_wires, extra_wires)
        change_op_basis(_HammingWeight(hw_wires, work_wires), phase_shifts)

    GlobalPhase(len(wires) * angle / 2)


add_decomps(RZLayer, _rzlayer_individual, _rzlayer_hwp_synthesis)
add_decomps("Adjoint(RZLayer)", adjoint_rotation2)
