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
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import RZ
from pennylane.ops.op_math.adjoint2 import adjoint_rotation as adjoint_rotation2
from pennylane.typing import Float, TensorLike, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


class RZLayer(Operator2):
    r"""Applies :class:`~.RZ` rotations to blocks of wires, where all wires of a block share
    one angle.

    For :math:`k` angles and :math:`km` wires, the angle :math:`\theta_i` is applied to every
    wire of the :math:`i`-th block ``wires[i * m : (i + 1) * m]``:

    .. math::

        \operatorname{RZLayer}(\boldsymbol{\theta}) = \prod_{i=0}^{k-1} \prod_{j=0}^{m-1}
        R_Z(\theta_i)_{\texttt{wires}[im+j]}.

    The block size :math:`m` is fixed by the shapes of ``angles`` and ``wires``, so that
    decompositions can exploit equal angles without inspecting their values.

    Args:
        angles (TensorLike): one-dimensional array of :math:`k\geq 1` rotation angles
        wires (WiresLike): the :math:`km` wires the rotations act on
        work_wires (WiresLike): auxiliary wires in the zero state that decompositions may use and
            return to the zero state. They are included in ``op.wires``. Defaults to ``()``.

    Raises:
        ValueError: if ``angles`` is not one-dimensional or empty, if the number of wires is not
            a multiple of the number of angles, or if ``wires`` and ``work_wires`` overlap

    **Example**

    Two angles on four wires apply the first angle to the wires ``0, 1`` and the second angle
    to the wires ``2, 3``:

    >>> op = qp.RZLayer([0.1, 0.2], wires=[0, 1, 2, 3])
    >>> op.decomposition()
    [RZ(0.1, wires=[0]), RZ(0.1, wires=[1]), RZ(0.2, wires=[2]), RZ(0.2, wires=[3])]

    With one angle per wire, every wire gets its own angle:

    >>> qp.RZLayer([0.1, 0.2, 0.3], wires=[0, 1, 2]).decomposition()
    [RZ(0.1, wires=[0]), RZ(0.2, wires=[1]), RZ(0.3, wires=[2])]

    With a single angle, all wires share it:

    >>> qp.RZLayer([0.5], wires=[0, 1, 2]).decomposition()
    [RZ(0.5, wires=[0]), RZ(0.5, wires=[1]), RZ(0.5, wires=[2])]

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.RZLayer
            :symbols: :math:`k` angles on :math:`km` wires and :math:`w` ``work_wires``
            :example: :math:`k=2`, :math:`m=3`, :math:`w=1`

            .. decomposition-rule:: _rzlayer_individual
                :image: /_static/decomposition_thumbnails/rz_layer_individual.svg

                :class:`~.RZ` | :math:`km` | 6

            .. decomposition-rule:: adjoint_rotation
                :image: /_static/decomposition_thumbnails/rz_layer_adjoint.svg
                :region: adjoint
                :where: ``angles`` change sign

                :class:`~.RZLayer` | :math:`1`
    """

    dynamic_argnames = ("angles",)
    wire_argnames = ("wires", "work_wires")
    arg_specs = {"angles": Float[-1], "wires": Wire[-1], "work_wires": Wire[-1]}

    grad_method = None

    def __init__(self, angles: TensorLike, wires: WiresLike, work_wires: WiresLike = ()):
        if isinstance(angles, (list, tuple)):
            angles = math.array(angles, like=math.get_deep_interface(angles))
        wires = Wires(wires)
        work_wires = Wires(work_wires)

        if math.ndim(angles) != 1:
            raise ValueError("The angles argument must be a one-dimensional array.")
        if len(angles) == 0:
            raise ValueError("RZLayer requires at least one angle.")
        if len(wires) % len(angles) != 0:
            raise ValueError(
                f"The number of wires ({len(wires)}) must be a multiple of the number of angles "
                f"({len(angles)})."
            )
        validate_no_wire_overlaps({"wires": wires, "work_wires": work_wires})

        super().__init__(angles, wires=wires, work_wires=work_wires)

    @property
    def wires(self) -> Wires:
        """All wires involved in the operation, with the work wires last."""
        return self.arguments["wires"] + self.work_wires

    def adjoint(self) -> "RZLayer":
        return RZLayer(-self.angles, **self.wire_args)


def _rzlayer_individual_resources(wires, **_):
    return {RZ: len(wires)}


@register_resources(_rzlayer_individual_resources)
def _rzlayer_individual(angles, wires, **_):
    block_size = len(wires) // len(angles)
    if capture.enabled() or compiler.active():
        angles = math.array(angles, like="jax")
        wires = math.array(wires, like="jax")

    def _rz(i):
        RZ(angles[i // block_size], wires=wires[i])

    for_loop(len(wires))(_rz)()


add_decomps(RZLayer, _rzlayer_individual)
add_decomps("Adjoint(RZLayer)", adjoint_rotation2)
