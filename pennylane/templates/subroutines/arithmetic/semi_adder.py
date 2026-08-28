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

from pennylane import math
from pennylane.allocation import allocate
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, adjoint, ctrl
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike

from .temporary_and import TemporaryAND


def _left_block(wires: list):
    """Left full adder unit. Wires are input carry, input bit, target bit and output carry."""
    ck, ik, tk, aux = wires
    CNOT([ck, ik])
    CNOT([ck, tk])
    TemporaryAND([ik, tk, aux])
    CNOT([ck, aux])


_left_block_zeroed = TemporaryAND
"""Left half adder unit. Wires are input carry, target bit and output carry."""


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


def _left_ladder(x_wires, y_wires, work_wires, carry_flip=None):
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
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)

    TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])
    if carry_flip is not None:
        carry_flip(work_wires[0])
    crossover = min(num_y_wires - 1, num_x_wires)

    for i in range(1, crossover):
        # Add the bit of x as well as the previous carry to the bit of y, and compute the next carry
        _left_block([work_wires[i - 1], x_wires[i], y_wires[i], work_wires[i]])

    # From here on, we don't have any bits in x left, so we just need to propagate the carry over y
    for i in range(crossover, num_y_wires - 1):
        _left_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])


def _right_ladder(x_wires, y_wires, work_wires, carry_flip=None):
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
    """
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    crossover = min(num_y_wires - 1, num_x_wires)
    # For these bits, we don't have any bits in x, we only need to uncompute the carry propagation
    for i in range(num_y_wires - 2, crossover - 1, -1):
        _right_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])

    for i in range(crossover - 1, 0, -1):
        # Uncompute the carry and the addition of the bit of x and the next less-significant carry
        # into the bit of y.
        _right_block([work_wires[i - 1], x_wires[i], y_wires[i], work_wires[i]])

    if carry_flip is not None:
        carry_flip(work_wires[0])
    adjoint(TemporaryAND([x_wires[0], y_wires[0], work_wires[0]]))
    CNOT([x_wires[0], y_wires[0]])


def _ctrl_right_block_zeroed(wires, **ctrl_kwargs):
    ck, tk, aux = wires
    adjoint(TemporaryAND([ck, tk, aux]))
    ctrl(CNOT(wires=[ck, tk]), **ctrl_kwargs)


def _ctrl_right_block(wires, **ctrl_kwargs):
    ck, ik, tk, aux = wires
    CNOT([ck, aux])
    adjoint(TemporaryAND([ik, tk, aux]))
    ctrl(CNOT(wires=[ik, tk]), **ctrl_kwargs)
    CNOT([ck, tk])
    CNOT([ck, ik])


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

        _wires_are_traced = any(
            math.is_abstract(w) for ws in (x_wires, y_wires, work_wires) for w in ws
        )

        # Wire overlap/length validation must be skipped when wires are JAX tracers,
        # as their concrete values are not available during tracing.
        if not _wires_are_traced:
            if work_wires:
                if work_wires.intersection(x_wires):
                    raise ValueError(
                        "None of the wires in work_wires should be included in x_wires."
                    )
                if work_wires.intersection(y_wires):
                    raise ValueError(
                        "None of the wires in work_wires should be included in y_wires."
                    )
            if x_wires.intersection(y_wires):
                raise ValueError("None of the wires in y_wires should be included in x_wires.")

        super().__init__(x_wires=x_wires, y_wires=y_wires, work_wires=work_wires)

    # pylint: disable=arguments-differ
    def __abstract_init__(self, x_wires, y_wires, work_wires=None):
        work_wires = work_wires if work_wires is not None else []
        super().__abstract_init__(
            x_wires=Wire[len(x_wires)],
            y_wires=Wire[len(y_wires)],
            work_wires=Wire[len(work_wires)],
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.x_wires + self.y_wires + self.work_wires


def _semi_adder_resources(x_wires, y_wires, **_):
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    if num_y_wires == 1:
        return {CNOT: 1}
    # Resources extracted from `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.
    # _left_ladder uses (num_y_wires - 1) TemporaryANDs
    # and 3 * (crossover - 1) CNOTs
    # _right_ladder uses (num_y_wires - 1) Adjoint(TemporaryAND)s
    # and 3 * (crossover - 1) + (num_y_wires - 1 - crossover) + 1 CNOTs
    # There are 1 + int(num_x_wires>=num_y_wires) additional CNOTs in the main decomp. function
    crossover = min(num_y_wires - 1, num_x_wires)
    return {
        TemporaryAND: num_y_wires - 1,
        adjoint(TemporaryAND(Wire[3])): num_y_wires - 1,
        CNOT: 5 * crossover + num_y_wires - 5 + int(num_x_wires >= num_y_wires),
    }


def _semi_adder_work_wires(y_wires=None, work_wires=(), base=None, **_):
    """The work wires that the ladders need, minus the ones that were already provided.

    Symbolic rules like ``C(SemiAdder)`` reuse this spec but are called with the symbolic
    operator's arguments, so ``base`` is set instead of ``y_wires``. The requirement is the one
    of the wrapped ``SemiAdder``, whose own ``work_wires`` are the relevant ones.
    """
    if base is not None:
        return _semi_adder_work_wires(**base.arguments)
    num_work_wires_needed = len(y_wires) - 1
    num_work_wires_provided = len(work_wires)
    return {"zeroed": max(num_work_wires_needed - num_work_wires_provided, 0)}


@register_resources(_semi_adder_resources, work_wires=_semi_adder_work_wires)
def _semi_adder(x_wires, y_wires, work_wires=None, carry_flip=None):
    num_y_wires = len(y_wires)
    num_x_wires = len(x_wires)

    if num_y_wires == 1:
        CNOT([x_wires[-1], y_wires[0]])
        return

    work_wires = [] if work_wires is None else list(work_wires)
    if len(work_wires) < num_y_wires - 1:
        # The right ladder restores the work wires to zero, so they can be borrowed and returned.
        work_wires += list(allocate(num_y_wires - 1 - len(work_wires), restored=True))

    # Turn wires from big endian to little endian
    # Truncate x_wires, as values larger than 2**num_y_wires-1 can anyways not be stored
    x_wires = x_wires[::-1][:num_y_wires]
    y_wires = y_wires[::-1]
    work_wires = work_wires[: num_y_wires - 1][::-1]

    _left_ladder(x_wires, y_wires, work_wires, carry_flip=carry_flip)

    CNOT([work_wires[-1], y_wires[-1]])

    if num_x_wires >= num_y_wires:
        CNOT([x_wires[-1], y_wires[-1]])

    _right_ladder(x_wires, y_wires, work_wires, carry_flip=carry_flip)


add_decomps(SemiAdder, _semi_adder)


def _controlled_semi_adder_resource(
    base, control_wires, control_values, work_wires=None, work_wire_type="borrowed"
):  # pylint: disable=too-many-arguments,unused-argument
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


@register_resources(_controlled_semi_adder_resource, work_wires=_semi_adder_work_wires)
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
    if len(base_work_wires) < len(y_wires) - 1:
        # The right ladder restores the work wires to zero, so they can be borrowed and returned.
        base_work_wires += list(allocate(len(y_wires) - 1 - len(base_work_wires), restored=True))
    work_wires = [] if work_wires is None else work_wires
    ctrl_kwargs = {
        "control": control_wires,
        "control_values": control_values,
        "work_wires": Wires.all_wires([work_wires, extra_work_wires_from_base]),
        "work_wire_type": work_wire_type,
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

    _left_ladder(x_wires, y_wires, work_wires, carry_flip=carry_flip)

    ctrl(CNOT([work_wires[-1], y_wires[-1]]), **ctrl_kwargs)
    if num_x_wires >= num_y_wires:
        ctrl(CNOT([x_wires[-1], y_wires[-1]]), **ctrl_kwargs)

    _controlled_right_ladder(x_wires, y_wires, work_wires, carry_flip=carry_flip, **ctrl_kwargs)


add_decomps("C(SemiAdder)", flip_zero_control2(_controlled_semi_adder))
