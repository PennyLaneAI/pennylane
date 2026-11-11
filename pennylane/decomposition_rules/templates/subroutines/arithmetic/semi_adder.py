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

"""Built-in decomposition rules for templates.subroutines.arithmetic.semi_adder."""

from pennylane.allocation import allocate
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition.resources import resolve_work_wire_type
from pennylane.decomposition_rules.op_math.controlled2 import (
    flip_zero_control as flip_zero_control2,
)
from pennylane.ops import CNOT, adjoint, ctrl
from pennylane.templates.subroutines.arithmetic.semi_adder import (
    SemiAdder,
    _ctrl_right_block,
    _ctrl_right_block_zeroed,
    _effective_skip_input_pos,
    _left_block,
    _left_block_zeroed,
)
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.typing import Wire
from pennylane.wires import Wires


def _right_block(wires: list):
    """Right full adder unit. Wires are input carry, input bit, target bit and output carry."""
    ck, ik, tk, aux = wires
    CNOT([ck, aux])
    adjoint(TemporaryAND([ik, tk, aux]))
    CNOT([ck, ik])
    CNOT([ik, tk])


def _right_block_zeroed(wires: list):
    """Right half adder unit. Wires are input carry, target bit and output carry."""
    adjoint(TemporaryAND(wires))
    CNOT(wires[:2])


def _left_ladder(x_wires, y_wires, work_wires, carry_flip=None, skip_input_pos=None):
    """Implement a ladder formed from the left block in figure 2, https://arxiv.org/pdf/1709.06648.

    Args:
        x_wires(WiresLike): Wires encoding the integer :math:`x` to be added onto :math:`y`.
            Must be in non-PennyLane ordering, i.e., little endian.
        y_wires(WiresLike): Wires encoding the integer :math:`y` onto which :math:`x` is added.
            Must be in non-PennyLane ordering, i.e., little endian.
        work_wires(WiresLike): Work wires for the addition.
        carry_flip(Callable[[Wire], None], optional): if given, called with ``work_wires[0]``
            right after it is computed, to simulate a ``1`` input carry (see
            ``_adder_flipped_first_work_wire`` and ``_c_subtract_then_add_one``).
    """
    num_y_wires = len(y_wires)

    TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])
    if carry_flip is not None:
        carry_flip(work_wires[0])

    x_pos = 1
    for i in range(1, num_y_wires - 1):
        if i in skip_input_pos:
            # For a skipped input position, we don't have an input bit in x, so we just
            # need to propagate the carry over y
            _left_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            # Add the bit of x as well as the previous carry to the bit of y, and compute
            # the next carry
            _left_block([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos += 1

    return x_pos


def _right_ladder(x_wires, y_wires, work_wires, carry_flip=None, skip_input_pos=None):
    """Implement a ladder formed from the right block in figure 2, https://arxiv.org/pdf/1709.06648.

    Args:
        x_wires(WiresLike): Wires encoding the integer :math:`x` to be added onto :math:`y`.
            Must be in non-PennyLane ordering, i.e., little endian.
        y_wires(WiresLike): Wires encoding the integer :math:`y` onto which :math:`x` is added.
            Must be in non-PennyLane ordering, i.e., little endian.
        work_wires(WiresLike): Work wires for the addition.
        carry_flip(Callable[[Wire], None], optional): if given, called with ``work_wires[0]``
            right before it is uncomputed, undoing the flip applied by ``_left_ladder``'s own
            ``carry_flip`` (see ``_adder_flipped_first_work_wire`` and ``_c_subtract_then_add_one``).
        skip_input_pos (set[int]): Set of input qubit positions at which no qubit from ``x_wires``
            is used. Instead, a fixed zeroed input is assumed, and all subsequent input qubits
            from ``x_wires`` are shifted to the next (not skipped) position.
    """
    num_y_wires = len(y_wires)
    # This is x_pos as computed by _left_ladder, minus one.
    x_pos = sum(i not in skip_input_pos for i in range(1, num_y_wires - 1))

    for i in range(num_y_wires - 2, 0, -1):
        if i in skip_input_pos:
            # For these bits, we don't have any bits in x, we only need to uncompute the
            # carry propagation
            _right_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            # Uncompute the carry and the addition of the bit of x and the next
            # less-significant carry into the bit of y.
            _right_block([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos -= 1

    if carry_flip is not None:
        carry_flip(work_wires[0])
    adjoint(TemporaryAND([x_wires[0], y_wires[0], work_wires[0]]))
    CNOT([x_wires[0], y_wires[0]])


def _controlled_right_ladder(x_wires, y_wires, non_ctrl_work_wires, carry_flip=None, **ctrl_kwargs):
    """Implement a ladder formed from the right block in figure 4, https://arxiv.org/pdf/1709.06648.

    Args:
        x_wires(WiresLike): Wires encoding the integer :math:`x` to be added onto :math:`y`.
            Must be in non-PennyLane ordering, i.e., little endian.
        y_wires(WiresLike): Wires encoding the integer :math:`y` onto which :math:`x` is added.
            Must be in non-PennyLane ordering, i.e., little endian.
        work_wires(WiresLike): Work wires for the addition.
        carry_flip(Callable[[Wire], None], optional): see ``_right_ladder``.
    """
    # We need to use a different name for this variable in the function signature because
    # work_wires is a key in ctrl_kwargs. This allows us to keep passing ctrl_kwargs around as
    # a convenient variable. Here we rename the variable passed to the function to work_wires,
    # in order to be more consistent with `_left_ladder` and `_right_ladder`.
    work_wires = non_ctrl_work_wires
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    crossover = min(num_y_wires - 1, num_x_wires)

    for i in range(len(y_wires) - 2, crossover - 1, -1):
        _ctrl_right_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]], **ctrl_kwargs)
    for i in range(crossover - 1, 0, -1):
        _ctrl_right_block([work_wires[i - 1], x_wires[i], y_wires[i], work_wires[i]], **ctrl_kwargs)

    if carry_flip is not None:
        carry_flip(work_wires[0])
    adjoint(TemporaryAND([x_wires[0], y_wires[0], work_wires[0]]))
    ctrl(CNOT([x_wires[0], y_wires[0]]), **ctrl_kwargs)


# pylint: disable-next=unused-argument
def _semi_adder_resources(x_wires, y_wires, work_wires=None, skip_input_pos=None):
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    if num_y_wires == 1:
        return {CNOT: 1}

    # Process skip_input_pos into standard format, taking
    # size of x_wires into account
    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos)
    num_elbows = num_y_wires - 1
    # Determine whether the second CNOT in the middle of the decomposition is present
    second_middle_cnot = int((num_y_wires - 1) not in skip_input_pos)
    # The number of zeroed blocks is given by skip_input_pos, except for the value
    # `num_y_wires-1`, which does not modify a block but the second middle CNOT
    num_zeroed_blocks = max(len(skip_input_pos) - (1 - second_middle_cnot), 0)
    # There are num_y_wires-2 blocks in total, not counting the initial half adder
    # and the middle CNOTs
    num_nonzeroed_blocks = max((num_y_wires - 2) - num_zeroed_blocks, 0)
    # each _left_block uses 3 CNOTs, each _left_block_zeroed none
    # each _right_block uses 3 CNOTs, each _right_block_zeroed just 1 CNOT
    # the remaining construction uses 2 + int(num_y_wires-1 not in skip_input_pos) CNOTs
    return {
        TemporaryAND: num_elbows,
        adjoint(TemporaryAND(Wire[3])): num_elbows,
        CNOT: num_zeroed_blocks + 6 * num_nonzeroed_blocks + 2 + second_middle_cnot,
    }


# pylint: disable-next=unused-argument
def _semi_adder_work_wires(x_wires, y_wires, work_wires):
    """The work wires that the ladders need, minus the ones that were already provided.

    Symbolic rules like ``C(SemiAdder)`` reuse this spec but are called with the symbolic
    operator's arguments, so ``base`` is set instead of ``y_wires``. The requirement is the one
    of the wrapped ``SemiAdder``, whose own ``work_wires`` are the relevant ones.
    """
    num_work_wires_needed = len(y_wires) - 1
    num_work_wires_provided = len(work_wires)
    return {"zeroed": max(num_work_wires_needed - num_work_wires_provided, 0)}


@register_resources(_semi_adder_resources, work_wires=_semi_adder_work_wires)
def _semi_adder(x_wires, y_wires, work_wires=None, carry_flip=None, skip_input_pos=None):
    num_y_wires = len(y_wires)
    num_x_wires = len(x_wires)

    if num_y_wires == 1:
        CNOT([x_wires[-1], y_wires[0]])
        return

    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos)
    work_wires = [] if work_wires is None else list(work_wires)
    # The right ladder restores the work wires to zero, so they can be borrowed and returned.
    # ``allocate(0)`` records nothing when every work wire was already provided.
    with allocate(max(num_y_wires - 1 - len(work_wires), 0), restored=True) as extra_work_wires:
        work_wires += list(extra_work_wires)

        # Turn wires from big endian to little endian
        # Truncate x_wires, as values larger than 2**num_y_wires-1 can anyways not be stored. If
        # there are skipped input positions in skip_input_pos, we could truncate even further,
        # which happens anyways in the ladder functions.
        x_wires = x_wires[::-1][:num_y_wires]
        y_wires = y_wires[::-1]
        work_wires = work_wires[: num_y_wires - 1][::-1]

        x_pos = _left_ladder(
            x_wires, y_wires, work_wires, carry_flip=carry_flip, skip_input_pos=skip_input_pos
        )

        CNOT([work_wires[-1], y_wires[-1]])

        if num_y_wires - 1 not in skip_input_pos:
            CNOT([x_wires[x_pos], y_wires[-1]])

        _right_ladder(
            x_wires,
            y_wires,
            work_wires,
            carry_flip=carry_flip,
            skip_input_pos=skip_input_pos,
        )


add_decomps(SemiAdder, _semi_adder)


# pylint: disable-next=too-many-arguments,unused-argument
def _ctrl_semi_adder_resource(base, control_wires, control_values, work_wires, work_wire_type):
    r"""
    Resources calculated from `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.

    ``control_values`` is unused: this resource function is only ever registered wrapped in
    ``flip_zero_control``, which normalizes control values to all-ones (accounting for any
    zero-valued controls itself via extra ``X`` gates) before this function ever runs.
    """
    x_wires = base.x_wires
    y_wires = base.y_wires
    base_work_wires = base.work_wires
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)

    num_control_wires = len(control_wires)
    # Note: don't re-wrap `work_wires` in `Wires(...)` here -- it may already be an
    # `AbstractWires` instance (when this resource function runs on abstractified
    # arguments), and `Wires(some_abstract_wires)` would wrap it as a single opaque
    # element instead of preserving its length. `len()` alone works on both.
    num_extra_work_wires = 0 if work_wires is None else len(work_wires)
    # The base's own work_wires beyond the (num_y_wires - 1) consumed by the ladders
    # are available, in addition to any extra work_wires passed to `ctrl`, to the ctrl-CNOTs.
    # Clamped at 0: if the base has too few, the ladders allocate and none are left over here.
    num_extra_work_wires_from_base = max(len(base_work_wires) - (num_y_wires - 1), 0)
    num_work_wires = num_extra_work_wires + num_extra_work_wires_from_base
    work_wire_type = resolve_work_wire_type(
        Wire[num_extra_work_wires_from_base], "zeroed", work_wires, work_wire_type
    )

    if num_y_wires == 1:
        return {
            ctrl(
                CNOT(Wire[2]),
                Wire[num_control_wires],
                work_wires=Wire[num_work_wires],
                work_wire_type=work_wire_type,
            ): 1
        }

    crossover = min(num_y_wires - 1, num_x_wires)

    # _left_ladder uses (num_y_wires - 1) TemporaryANDs
    # and 3 * (crossover - 1) CNOTs
    # _controlled_right_ladder uses (num_y_wires - 1) TemporaryANDs, (num_y_wires - 1) controlled
    # CNOTs, and 3 * (crossover - 1) CNOTs.
    # There are 1 + int(num_x_wires>=num_y_wires) additional ctrl-CNOTs in the main function
    num_cnots = 6 * (crossover - 1)
    num_ctrl_cnots = num_y_wires + int(num_x_wires >= num_y_wires)
    return {
        TemporaryAND: num_y_wires - 1,
        adjoint(TemporaryAND(Wire[3])): num_y_wires - 1,
        CNOT: num_cnots,
        ctrl(
            CNOT(Wire[2]),
            Wire[num_control_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): num_ctrl_cnots,
    }


@register_resources(
    _ctrl_semi_adder_resource,
    work_wires=lambda base, *_, **__: _semi_adder_work_wires(**base.arguments),
)
def _controlled_semi_adder(
    base,
    control_wires,
    control_values=None,
    work_wires=None,
    work_wire_type="borrowed",
    carry_flip=None,
):  # pylint: disable=too-many-arguments
    r"""
    Decomposition extracted from `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_
    using building block described in Figure 4.
    """
    y_wires = base.y_wires
    x_wires = base.x_wires
    base_work_wires = base.work_wires
    # Slice out the needed work wires for the left and right ladders, the extra work wires
    # will be used as work wires for `ctrl`
    extra_work_wires_from_base = base_work_wires[len(y_wires) - 1 :]
    base_work_wires = list(base_work_wires[: len(y_wires) - 1])
    # The right ladder restores the work wires to zero, so they can be borrowed and returned.
    # ``allocate(0)`` records nothing when every work wire was already provided.
    num_to_allocate = max(len(y_wires) - 1 - len(base_work_wires), 0)
    with allocate(num_to_allocate, restored=True) as alloc_work_wires:
        base_work_wires += list(alloc_work_wires)
        work_wires = [] if work_wires is None else work_wires
        ctrl_work_wires = Wires.all_wires([work_wires, extra_work_wires_from_base])
        ctrl_work_wire_type = resolve_work_wire_type(
            extra_work_wires_from_base, "zeroed", work_wires, work_wire_type
        )
        ctrl_kwargs = {
            "control": control_wires,
            "control_values": control_values,
            "work_wires": ctrl_work_wires,
            "work_wire_type": ctrl_work_wire_type,
        }

        num_y_wires = len(y_wires)
        num_x_wires = len(x_wires)
        if num_y_wires == 1:
            ctrl(CNOT([x_wires[-1], y_wires[0]]), **ctrl_kwargs)
            return

        # Turn wires from big endian to little endian
        # Truncate x_wires, as values larger than 2**num_y_wires-1 can anyways not be stored
        x_wires = x_wires[::-1][:num_y_wires]
        y_wires = y_wires[::-1]
        work_wires = base_work_wires[::-1]

        skip_input_pos = _effective_skip_input_pos(len(x_wires), len(y_wires), [])
        _left_ladder(
            x_wires, y_wires, work_wires, carry_flip=carry_flip, skip_input_pos=skip_input_pos
        )

        ctrl(CNOT([work_wires[-1], y_wires[-1]]), **ctrl_kwargs)
        if num_x_wires >= num_y_wires:
            ctrl(CNOT([x_wires[-1], y_wires[-1]]), **ctrl_kwargs)

        _controlled_right_ladder(x_wires, y_wires, work_wires, carry_flip=carry_flip, **ctrl_kwargs)


add_decomps("C(SemiAdder)", flip_zero_control2(_controlled_semi_adder))
