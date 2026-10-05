# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the SemiAdder template for performing the semi-out-place addition."""

from pennylane.allocation import allocate
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, ctrl
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from .adder_blocks import (
    CtrlRightFullAdder,
    CtrlRightHalfAdder,
    LeftFullAdder,
    RightFullAdder,
    RightHalfAdder,
)
from .temporary_and import TemporaryAND


def _left_ladder(x_wires, y_wires, work_wires, skip_input_pos):
    """Implement the bit positions ``1`` to ``len(y_wires) - 2`` of the ladder formed from the
    left blocks in figure 2, https://arxiv.org/pdf/1709.06648.

    Args:
        x_wires(WiresLike): Wires encoding the integer :math:`x` to be added onto :math:`y`.
            Must be in non-PennyLane ordering, i.e., little endian.
        y_wires(WiresLike): Wires encoding the integer :math:`y` onto which :math:`x` is added.
            Must be in non-PennyLane ordering, i.e., little endian.
        work_wires(WiresLike): Work wires for the addition.
        skip_input_pos (set[int]): Set of input qubit positions at which no qubit from ``x_wires``
            is used. Instead, a fixed zeroed input is assumed, and all subsequent input qubits
            from ``x_wires`` are shifted to the next (not skipped) position.

    Returns:
        int: The position in ``x_wires`` of the input bit for the most significant bit.
    """
    num_y_wires = len(y_wires)

    x_pos = 1
    for i in range(1, num_y_wires - 1):
        if i in skip_input_pos:
            # For a skipped input position, we don't have an input bit in x, so we just
            # need to propagate the carry over y
            TemporaryAND([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            # Add the bit of x as well as the previous carry to the bit of y, and compute
            # the next carry
            LeftFullAdder([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos += 1

    return x_pos


def _right_ladder(x_wires, y_wires, work_wires, skip_input_pos, ctrl_kwargs=None):
    """Implement the bit positions ``len(y_wires) - 2`` to ``1`` of the ladder formed from the
    right blocks in figure 2 (or figure 4 if ``ctrl_kwargs`` are given),
    https://arxiv.org/pdf/1709.06648.

    Args:
        x_wires(WiresLike): Wires encoding the integer :math:`x` to be added onto :math:`y`.
            Must be in non-PennyLane ordering, i.e., little endian.
        y_wires(WiresLike): Wires encoding the integer :math:`y` onto which :math:`x` is added.
            Must be in non-PennyLane ordering, i.e., little endian.
        work_wires(WiresLike): Work wires for the addition.
        skip_input_pos (set[int]): See ``_left_ladder``.
        ctrl_kwargs (dict | None): If given, the sum bits are written controlled, using the
            ``control_wires``, ``work_wires`` and ``work_wire_type`` from this dictionary.
    """
    num_y_wires = len(y_wires)
    # This is x_pos as computed by _left_ladder, minus one.
    x_pos = sum(i not in skip_input_pos for i in range(1, num_y_wires - 1))

    for i in range(num_y_wires - 2, 0, -1):
        if i in skip_input_pos:
            # For these bits, we don't have any bits in x, we only need to uncompute the
            # carry propagation
            _right_half_block([work_wires[i - 1], y_wires[i], work_wires[i]], ctrl_kwargs)
        else:
            # Uncompute the carry and the addition of the bit of x and the next
            # less-significant carry into the bit of y.
            wires = [work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]]
            _right_full_block(wires, ctrl_kwargs)
            x_pos -= 1


def _right_full_block(wires, ctrl_kwargs=None):
    if ctrl_kwargs is None:
        RightFullAdder(wires)
    else:
        CtrlRightFullAdder(wires=wires, **ctrl_kwargs)


def _right_half_block(wires, ctrl_kwargs=None):
    if ctrl_kwargs is None:
        RightHalfAdder(wires)
    else:
        CtrlRightHalfAdder(wires=wires, **ctrl_kwargs)


class SemiAdder(Operator2):
    r"""This operator performs the plain addition of two integers :math:`x` and :math:`y` in the computational basis:

    .. math::

        \text{SemiAdder} |x \rangle | y \rangle = |x \rangle | x + y  \rangle,

    This operation is also referred to as semi-out-place addition or quantum-quantum in-place addition in the literature.

    The implementation is based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.

    Args:
        x_wires (Sequence[int]): The wires that store the integer :math:`x`. The number of wires must be sufficient to
            represent :math:`x` in binary.
        y_wires (Sequence[int]): The wires that store the integer :math:`y`. The number of wires must be sufficient to
            represent :math:`y` in binary. These wires are also used
            to encode the integer :math:`x+y` which is computed modulo :math:`2^{\text{len(y_wires)}}` in the computational basis.
        work_wires (Optional(Sequence[int])): The auxiliary wires to use for the addition. The
            addition uses ``len(y_wires) - 1`` work wires; any of them that are not provided are
            dynamically allocated by the decomposition.

    **Example**

    This example computes the sum of two integers :math:`x=3` and :math:`y=4`.

    .. code-block:: python

        x = 3
        y = 4

        wires = qp.registers({"x":3, "y":6, "work":5})

        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(wires["x"]))
            y_bin = qp.math.int_to_binary(y, len(wires["y"]))
            qp.BasisEmbedding(x_bin, wires=wires["x"])
            qp.BasisEmbedding(y_bin, wires=wires["y"])
            qp.SemiAdder(wires["x"], wires["y"], wires["work"])
            return qp.sample(wires=wires["y"])

    .. code-block:: pycon

        >>> print(circuit())
        [[0 0 0 1 1 1]]

    The result :math:`[[0 0 0 1 1 1]]`, is the binary representation of :math:`3 + 4 = 7`.

    Note that the result is computed modulo :math:`2^{\text{len(y_wires)}}` which makes the computed value dependent on the size of the ``y_wires`` register. This behavior is demonstrated in the following example.

    .. code-block:: python

        x = 3
        y = 1

        wires = qp.registers({"x":3, "y":2, "work":1})

        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(wires["x"]))
            y_bin = qp.math.int_to_binary(y, len(wires["y"]))
            qp.BasisEmbedding(x_bin, wires=wires["x"])
            qp.BasisEmbedding(y_bin, wires=wires["y"])
            qp.SemiAdder(wires["x"], wires["y"], wires["work"])
            return qp.sample(wires=wires["y"])

    >>> print(circuit())
    [[0 0]]

    The result :math:`[0\ 0]` is the binary representation of :math:`3 + 1 = 4` where :math:`4 \mod 2^2 = 0`.
    """

    grad_method = None

    wire_argnames = ("x_wires", "y_wires", "work_wires")
    arg_specs = {"x_wires": Wire[-1], "y_wires": Wire[-1], "work_wires": Wire[-1]}

    def __init__(self, x_wires: WiresLike, y_wires: WiresLike, work_wires: WiresLike | None = None):

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        work_wires = Wires(work_wires if work_wires is not None else [])

        wire_args = {"x_wires": x_wires, "y_wires": y_wires, "work_wires": work_wires}
        validate_no_wire_overlaps(wire_args)

        super().__init__(x_wires=x_wires, y_wires=y_wires, work_wires=work_wires)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.x_wires + self.y_wires + self.work_wires


def _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos):
    if skip_input_pos is None:
        skip_input_pos = []
    assert 0 not in skip_input_pos
    used_x = 0
    new_skip_input_pos = []
    for i in range(num_y_wires):
        if i in skip_input_pos or used_x >= num_x_wires:
            new_skip_input_pos.append(i)
        else:
            used_x += 1
    return set(new_skip_input_pos)


# pylint: disable-next=unused-argument
def _semi_adder_resources(x_wires, y_wires, work_wires=None, skip_input_pos=None):
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    if num_y_wires == 1:
        return {CNOT: 1}

    # Process skip_input_pos into standard format, taking
    # size of x_wires into account
    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos)
    num_half_blocks, num_full_blocks = _num_ladder_blocks(num_y_wires, skip_input_pos)
    # Determine whether the second CNOT in the middle of the decomposition is present
    second_middle_cnot = int((num_y_wires - 1) not in skip_input_pos)
    # The least significant bit position is always a half adder, with x_wires[0] as carry
    return {
        TemporaryAND: num_half_blocks + 1,
        LeftFullAdder: num_full_blocks,
        RightFullAdder: num_full_blocks,
        RightHalfAdder: num_half_blocks + 1,
        CNOT: 1 + second_middle_cnot,
    }


def _num_ladder_blocks(num_y_wires, skip_input_pos):
    """The number of half and full adder blocks in ``_left_ladder`` (or ``_right_ladder``)."""
    num_half_blocks = sum(i in skip_input_pos for i in range(1, num_y_wires - 1))
    return num_half_blocks, num_y_wires - 2 - num_half_blocks


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

        TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])
        if carry_flip is not None:
            carry_flip(work_wires[0])

        x_pos = _left_ladder(x_wires, y_wires, work_wires, skip_input_pos)

        CNOT([work_wires[-1], y_wires[-1]])

        if num_y_wires - 1 not in skip_input_pos:
            CNOT([x_wires[x_pos], y_wires[-1]])

        _right_ladder(x_wires, y_wires, work_wires, skip_input_pos)

        if carry_flip is not None:
            carry_flip(work_wires[0])
        RightHalfAdder([x_wires[0], y_wires[0], work_wires[0]])


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
    num_work_wires = num_extra_work_wires + max(len(base_work_wires) - (num_y_wires - 1), 0)
    ctrl_cnot = ctrl(
        CNOT(Wire[2]),
        Wire[num_control_wires],
        work_wires=Wire[num_work_wires],
        work_wire_type=work_wire_type,
    )

    if num_y_wires == 1:
        return {ctrl_cnot: 1}

    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, [])
    num_half_blocks, num_full_blocks = _num_ladder_blocks(num_y_wires, skip_input_pos)
    second_middle_cnot = int((num_y_wires - 1) not in skip_input_pos)
    ctrl_block_kwargs = {
        "control_wires": Wire[num_control_wires],
        "work_wires": Wire[num_work_wires],
        "work_wire_type": work_wire_type,
    }
    return {
        TemporaryAND: num_half_blocks + 1,
        LeftFullAdder: num_full_blocks,
        CtrlRightFullAdder(wires=Wire[4], **ctrl_block_kwargs): num_full_blocks,
        CtrlRightHalfAdder(wires=Wire[3], **ctrl_block_kwargs): num_half_blocks + 1,
        ctrl_cnot: 1 + second_middle_cnot,
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
):  # pylint: disable=too-many-arguments,unused-argument
    r"""
    Decomposition extracted from `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_
    using building block described in Figure 4.

    ``control_values`` are ignored, i.e., all control values are assumed to be ``1``. This rule
    is registered wrapped in ``flip_zero_control``, which takes care of zero-valued controls.
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
        ctrl_kwargs = {
            "control_wires": control_wires,
            "work_wires": Wires.all_wires([work_wires, extra_work_wires_from_base]),
            "work_wire_type": work_wire_type,
        }

        num_y_wires = len(y_wires)
        if num_y_wires == 1:
            _ctrl_cnot([x_wires[-1], y_wires[0]], ctrl_kwargs)
            return

        # Turn wires from big endian to little endian
        # Truncate x_wires, as values larger than 2**num_y_wires-1 can anyways not be stored
        x_wires = x_wires[::-1][:num_y_wires]
        y_wires = y_wires[::-1]
        work_wires = base_work_wires[::-1]

        skip_input_pos = _effective_skip_input_pos(len(x_wires), num_y_wires, [])
        TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])
        if carry_flip is not None:
            carry_flip(work_wires[0])

        x_pos = _left_ladder(x_wires, y_wires, work_wires, skip_input_pos)

        _ctrl_cnot([work_wires[-1], y_wires[-1]], ctrl_kwargs)
        if num_y_wires - 1 not in skip_input_pos:
            _ctrl_cnot([x_wires[x_pos], y_wires[-1]], ctrl_kwargs)

        _right_ladder(x_wires, y_wires, work_wires, skip_input_pos, ctrl_kwargs)

        if carry_flip is not None:
            carry_flip(work_wires[0])
        _right_half_block([x_wires[0], y_wires[0], work_wires[0]], ctrl_kwargs)


def _ctrl_cnot(wires, ctrl_kwargs):
    ctrl(
        CNOT(wires),
        ctrl_kwargs["control_wires"],
        work_wires=ctrl_kwargs["work_wires"],
        work_wire_type=ctrl_kwargs["work_wire_type"],
    )


add_decomps("C(SemiAdder)", flip_zero_control2(_controlled_semi_adder))


def _self_ctrl_one_sparse_add_resources(
    num_x_wires, num_y_wires, num_work_wires, first_and_is_output_copy=False
):
    """Resources for _self_ctrl_one_sparse_add below."""
    if num_y_wires == 1:
        return {CNOT: 1}

    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos=[1])
    num_half_blocks, num_full_blocks = _num_ladder_blocks(num_y_wires, skip_input_pos)
    second_middle_cnot = int(num_y_wires - 1 not in skip_input_pos)
    ctrl_kwargs = {
        "control_wires": Wire[1],
        "work_wires": Wire[num_work_wires - (num_y_wires - 1)],
        "work_wire_type": "zeroed",
    }
    ccnot_rep = ctrl(
        CNOT(Wire[2]),
        Wire[1],
        work_wires=ctrl_kwargs["work_wires"],
        work_wire_type="zeroed",
    )
    # The least significant bit position either copies y_0 into the work wire twice, or uses
    # a half adder block pair with x_0 as carry, which is uncontrolled because x_0 is the control.
    first_is_copy = int(first_and_is_output_copy)
    return {
        TemporaryAND: num_half_blocks + 1 - first_is_copy,
        RightHalfAdder: 1 - first_is_copy,
        LeftFullAdder: num_full_blocks,
        CtrlRightFullAdder(wires=Wire[4], **ctrl_kwargs): num_full_blocks,
        CtrlRightHalfAdder(wires=Wire[3], **ctrl_kwargs): num_half_blocks,
        CNOT: 3 * first_is_copy,
        ccnot_rep: 1 + second_middle_cnot,
    }


def _self_ctrl_one_sparse_add(x_wires, y_wires, work_wires, first_and_is_output_copy=False):
    """Specialized arithmetic unit: Addition controlled on the first qubit of the first addend,
    with a classically fixed bit in state |0> injected into first addend register at the position
    of the 2's bit. All other input bits are shifted in position.

    Effectively, we are adding :math:`x_0 * (x_{n-1} x_{n-2} ... x_1 0 x_0)_2` to ``y_wires``.
    """
    num_y_wires = len(y_wires)
    if num_y_wires == 1:
        CNOT([x_wires[0], y_wires[0]])
        return

    # Set up control structure for controlled ops within decomposition
    ctrl_kwargs = {
        "control_wires": x_wires[:1],
        "work_wires": work_wires[num_y_wires - 1 :],  # Pass only additional work qubits
        "work_wire_type": "zeroed",
    }

    num_x_wires = len(x_wires)
    # We use static zeroed=[1] for this subroutine
    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos=[1])
    work_wires = work_wires[: num_y_wires - 1]

    if first_and_is_output_copy:
        CNOT([y_wires[0], work_wires[0]])
    else:
        TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])

    x_pos = _left_ladder(x_wires, y_wires, work_wires, skip_input_pos)

    _ctrl_cnot([work_wires[-1], y_wires[-1]], ctrl_kwargs)
    if num_y_wires - 1 not in skip_input_pos:
        _ctrl_cnot([x_wires[x_pos], y_wires[-1]], ctrl_kwargs)

    _right_ladder(x_wires, y_wires, work_wires, skip_input_pos, ctrl_kwargs)

    # The sum bit write at the least significant position is controlled on x_0 itself, and thus
    # uncontrolled.
    if first_and_is_output_copy:
        CNOT([y_wires[0], work_wires[0]])
        CNOT([x_wires[0], y_wires[0]])
    else:
        RightHalfAdder([x_wires[0], y_wires[0], work_wires[0]])
