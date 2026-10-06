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
"""Internal building blocks of the ripple-carry adder from
`arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_ (figures 2 and 4).

The left half adder block is :class:`~.TemporaryAND` itself, so it has no dedicated operator.

These operators are not meant to be used directly, but to compose adder decompositions such as
those of :class:`~.SemiAdder`. Each block assumes the state produced by the matching blocks of
an adder ladder, as stated in its docstring.
"""

from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, adjoint, ctrl
from pennylane.ops.op_math.controlled2 import _validate_work_wire_type
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike

from .temporary_and import TemporaryAND


class _AdderBlock(Operator2, is_baseclass=True):
    """Shared ``Operator2`` boilerplate for the uncontrolled adder blocks."""

    grad_method = None

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires)


class LeftFullAdder(_AdderBlock):
    r"""Left full adder block, computing the carry of a single bit position.

    The wires are ``[c, x, y, aux]`` for the input carry :math:`c`, the bits :math:`x` and
    :math:`y` to be added, and the zeroed output carry wire. The block acts as

    .. math::

        |c, x, y, 0\rangle \mapsto |c, x\oplus c, y\oplus c, \operatorname{maj}(c, x, y)\rangle.
    """

    arg_specs = {"wires": Wire[4]}


@register_resources({CNOT: 3, TemporaryAND: 1})
def _left_full_adder(wires):
    ck, ik, tk, aux = wires
    CNOT([ck, ik])
    CNOT([ck, tk])
    TemporaryAND([ik, tk, aux])
    CNOT([ck, aux])


add_decomps(LeftFullAdder, _left_full_adder)


class RightFullAdder(_AdderBlock):
    r"""Right full adder block, uncomputing the carry of a single bit position and writing the
    sum bit.

    The wires are ``[c, x, y, aux]`` as for :class:`~.LeftFullAdder`, whose output state
    this block expects:

    .. math::

        |c, x\oplus c, y\oplus c, \operatorname{maj}(c, x, y)\rangle
        \mapsto |c, x, x\oplus y\oplus c, 0\rangle.
    """

    arg_specs = {"wires": Wire[4]}


@register_resources({CNOT: 3, adjoint(TemporaryAND(Wire[3])): 1})
def _right_full_adder(wires):
    ck, ik, tk, aux = wires
    CNOT([ck, aux])
    adjoint(TemporaryAND([ik, tk, aux]))
    CNOT([ck, ik])
    CNOT([ik, tk])


add_decomps(RightFullAdder, _right_full_adder)


class RightHalfAdder(_AdderBlock):
    r"""Right half adder block, uncomputing the carry of a bit position without input bit and
    writing the sum bit.

    The wires are ``[c, y, aux]`` for the input carry :math:`c`, the bit :math:`y` and the
    output carry computed by :class:`~.TemporaryAND`:

    .. math::

        |c, y, c\cdot y\rangle \mapsto |c, y\oplus c, 0\rangle.
    """

    arg_specs = {"wires": Wire[3]}


@register_resources({CNOT: 1, adjoint(TemporaryAND(Wire[3])): 1})
def _right_half_adder(wires):
    adjoint(TemporaryAND(wires))
    CNOT(wires[:2])


add_decomps(RightHalfAdder, _right_half_adder)


class _CtrlRightAdderBlock(Operator2, is_baseclass=True):
    """Shared ``Operator2`` boilerplate for the controlled right adder blocks."""

    grad_method = None
    wire_argnames = ("control_wires", "wires", "work_wires")
    static_argnames = ("work_wire_type",)

    def __init__(
        self,
        control_wires: WiresLike,
        wires: WiresLike,
        work_wires: WiresLike | None = None,
        work_wire_type: str = "borrowed",
    ):
        work_wires = Wires(()) if work_wires is None else work_wires
        _validate_work_wire_type(work_wire_type)
        super().__init__(control_wires, wires, work_wires, work_wire_type)


def _ctrl_cnot_rep(control_wires, work_wires, work_wire_type):
    return ctrl(
        CNOT(Wire[2]),
        Wire[len(control_wires)],
        work_wires=Wire[len(work_wires)],
        work_wire_type=work_wire_type,
    )


class CtrlRightFullAdder(_CtrlRightAdderBlock):
    r"""Right full adder block in which only the sum bit is written controlled on
    ``control_wires`` (figure 4 in `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_).

    The ``wires`` are ``[c, x, y, aux]`` and the input state is as for
    :class:`~.RightFullAdder`. If all control wires are in the state :math:`|1\rangle`,
    this block acts like :class:`~.RightFullAdder`, otherwise it acts like the inverse of
    :class:`~.LeftFullAdder`. In particular, this is *not* a controlled
    :class:`~.RightFullAdder`.

    The ``work_wires`` and ``work_wire_type`` are passed on to the controlled ``CNOT``.
    """

    arg_specs = {"control_wires": Wire[-1], "wires": Wire[4], "work_wires": Wire[-1]}


def _ctrl_right_full_adder_resources(control_wires, wires, work_wires, work_wire_type):
    # pylint: disable=unused-argument
    return {
        CNOT: 3,
        adjoint(TemporaryAND(Wire[3])): 1,
        _ctrl_cnot_rep(control_wires, work_wires, work_wire_type): 1,
    }


@register_resources(_ctrl_right_full_adder_resources)
def _ctrl_right_full_adder(control_wires, wires, work_wires, work_wire_type):
    ck, ik, tk, aux = wires
    CNOT([ck, aux])
    adjoint(TemporaryAND([ik, tk, aux]))
    ctrl(CNOT([ik, tk]), control_wires, work_wires=work_wires, work_wire_type=work_wire_type)
    CNOT([ck, tk])
    CNOT([ck, ik])


add_decomps(CtrlRightFullAdder, _ctrl_right_full_adder)


class CtrlRightHalfAdder(_CtrlRightAdderBlock):
    r"""Right half adder block in which only the sum bit is written controlled on
    ``control_wires`` (figure 4 in `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_).

    The ``wires`` are ``[c, y, aux]`` and the input state is as for
    :class:`~.RightHalfAdder`. If all control wires are in the state :math:`|1\rangle`,
    this block acts like :class:`~.RightHalfAdder`, otherwise it acts like the inverse of
    :class:`~.TemporaryAND`.

    The ``work_wires`` and ``work_wire_type`` are passed on to the controlled ``CNOT``.
    """

    arg_specs = {"control_wires": Wire[-1], "wires": Wire[3], "work_wires": Wire[-1]}


def _ctrl_right_half_adder_resources(control_wires, wires, work_wires, work_wire_type):
    # pylint: disable=unused-argument
    return {
        adjoint(TemporaryAND(Wire[3])): 1,
        _ctrl_cnot_rep(control_wires, work_wires, work_wire_type): 1,
    }


@register_resources(_ctrl_right_half_adder_resources)
def _ctrl_right_half_adder(control_wires, wires, work_wires, work_wire_type):
    adjoint(TemporaryAND(wires))
    ctrl(CNOT(wires[:2]), control_wires, work_wires=work_wires, work_wire_type=work_wire_type)


add_decomps(CtrlRightHalfAdder, _ctrl_right_half_adder)
