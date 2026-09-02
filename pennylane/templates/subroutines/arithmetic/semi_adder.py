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

import numpy as np

from pennylane import compiler
from pennylane.allocation import allocate
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import (
    CNOT,
    GlobalPhase,
    PauliRot,
    X,
    Z,
    adjoint,
    cond,
    ctrl,
    pauli_measure,
)
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.ops.op_math.controlled2 import flip_zero_control as flip_zero_control2
from pennylane.typing import Float, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

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

        wire_args = {"x_wires": x_wires, "y_wires": y_wires, "work_wires": work_wires}
        validate_no_wire_overlaps(wire_args)

        super().__init__(x_wires=x_wires, y_wires=y_wires, work_wires=work_wires)

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


#######################################################################################
# Measurement-based (Pauli-product) decomposition
#######################################################################################
#
# The same ladder written with Pauli product rotations and Pauli product measurements.
# Only the four pi/8 rotations of each temporary AND stay unitary; every CNOT becomes a
# Pauli-controlled-Pauli
#
#     Lambda(Pc | Pt) = Proj(+Pc) (x) I  +  Proj(-Pc) (x) Pt ,
#
# which takes three PPMs, an ancilla and a few classically conditioned Pauli corrections.
# The gadget flips the ancilla between |0> and |+>, so each block spends two of them and
# hands the ancilla back in |0>.


def _minus():
    GlobalPhase(np.pi)


def _ppm_left_block(wires):
    """Wires are input carry, input bit, target bit, output carry, shared ancilla."""
    ck, ik, tk, out, aux = wires

    PauliRot(+np.pi / 4, "Y", wires=[out])
    PauliRot(-np.pi / 4, "ZZY", wires=[ck, tk, out])
    PauliRot(-np.pi / 4, "ZZY", wires=[ck, ik, out])
    PauliRot(+np.pi / 4, "ZZY", wires=[ik, tk, out])

    def flip_tk_out():
        X(tk)
        X(out)

    # Lambda(Z_ck | X_tk X_out), ancilla |0> -> |+>
    m1 = pauli_measure("ZX", wires=[ck, aux])
    m2 = pauli_measure("ZXX", wires=[aux, tk, out])
    m3 = pauli_measure("X", wires=[aux])
    cond(m3 == 1, Z)(aux)
    cond(m2 == 1, Z)(ck)
    cond((m1 ^ m3) == 1, flip_tk_out)()
    cond((m2 & (m1 ^ m3)) == 1, _minus)()

    # Lambda(Z_ik | X_tk), ancilla |+> -> |0>
    m4 = pauli_measure("ZZ", wires=[ik, aux])
    m5 = pauli_measure("XX", wires=[aux, tk])
    m6 = pauli_measure("Z", wires=[aux])
    cond(m6 == 1, X)(aux)
    cond(m5 == 1, Z)(ik)
    cond((m4 ^ m6) == 1, X)(tk)
    cond((m5 & (m4 ^ m6)) == 1, _minus)()


def _ppm_last_block(wires):
    """Wires are input carry, input bit, target bit, shared ancilla. No output carry."""
    ck, ik, tk, aux = wires

    # Lambda(Z_ck | X_tk), ancilla |0> -> |+>
    m1 = pauli_measure("ZX", wires=[ck, aux])
    m2 = pauli_measure("ZX", wires=[aux, tk])
    m3 = pauli_measure("X", wires=[aux])
    cond(m3 == 1, Z)(aux)
    cond(m2 == 1, Z)(ck)
    cond((m1 ^ m3) == 1, X)(tk)
    cond((m2 & (m1 ^ m3)) == 1, _minus)()

    # Lambda(Z_ik | X_tk), ancilla |+> -> |0>
    m4 = pauli_measure("ZZ", wires=[ik, aux])
    m5 = pauli_measure("XX", wires=[aux, tk])
    m6 = pauli_measure("Z", wires=[aux])
    cond(m6 == 1, X)(aux)
    cond(m5 == 1, Z)(ik)
    cond((m4 ^ m6) == 1, X)(tk)
    cond((m5 & (m4 ^ m6)) == 1, _minus)()


def _ppm_right_block(wires):
    """Wires are input carry, input bit, target bit, output carry."""
    ck, ik, tk, out = wires

    def correct():
        Z(out)  # |-> -> |+>
        Z(ck)

        def flip_ctrl():
            Z(ck)
            Z(ik)

        def flip_target():
            Z(ik)
            Z(tk)

        # Lambda(Z_ck Z_ik | Z_ik Z_tk), with the freed carry as the |+> ancilla
        m1 = pauli_measure("ZZZ", wires=[ck, ik, out])
        m2 = pauli_measure("XZZ", wires=[out, ik, tk])
        m3 = pauli_measure("Z", wires=[out])
        cond(m3 == 1, X)(out)
        cond(m2 == 1, flip_ctrl)()
        cond((m1 ^ m3) == 1, flip_target)()
        cond((m2 & (m1 ^ m3)) == 1, _minus)()

    def clean():
        outcome = pauli_measure("Z", wires=[out])
        cond(outcome == 1, X)(out)

    readout = pauli_measure("X", wires=[out])
    cond(readout == 1, correct, clean)()


def _semi_adder_ppm_work_wires(y_wires=None, work_wires=(), **_):
    """The carries, the shared ancilla and one wire held at |0>."""
    return {"zeroed": max(len(y_wires) + 1 - len(work_wires), 0)}


def _semi_adder_ppm_condition(x_wires=None, y_wires=None, **_):
    """``pauli_measure`` and the conditional corrections need an active compiler."""
    if not compiler.active():
        return False
    return len(x_wires) > 0 and len(y_wires) > 0


def _semi_adder_ppm_resources(y_wires=None, **_):
    """Counts on the branch where every right block needs its correction."""
    num_blocks = len(y_wires) - 1
    return {
        PauliRot(Float, "Y", Wire[1]): num_blocks,
        PauliRot(Float, "ZZY", Wire[3]): 3 * num_blocks,
        PauliMeasure("Z", wires=Wire[1]): 4 * num_blocks + 2,
        PauliMeasure("ZZ", wires=Wire[2]): 3 * num_blocks + 4,
        PauliMeasure("ZZZ", wires=Wire[3]): 3 * num_blocks,
        X: 5 * num_blocks + 3,
        Z: 9 * num_blocks + 3,
        GlobalPhase(Float): 3 * num_blocks + 2,
    }


@register_condition(_semi_adder_ppm_condition)
@register_resources(_semi_adder_ppm_resources, work_wires=_semi_adder_ppm_work_wires, exact=False)
def _semi_adder_ppm(x_wires, y_wires, work_wires=None):
    """The ladder of `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_ written with
    Pauli product rotations and Pauli product measurements.

    Requires ``len(y_wires) + 1`` work wires, all of them returned to ``|0>``: the
    ``len(y_wires) - 1`` carries, the shared gadget ancilla and one wire held at ``|0>``.
    That last wire stands in for the missing input carry of the first block and for the
    missing bits of ``x``, so that every block is the same circuit.
    """
    num_y_wires = len(y_wires)

    work_wires = [] if work_wires is None else list(work_wires)
    if len(work_wires) < num_y_wires + 1:
        # The right ladder restores the work wires to zero, so they can be borrowed.
        work_wires += list(allocate(num_y_wires + 1 - len(work_wires), restored=True))

    aux = work_wires[num_y_wires - 1]
    zero = work_wires[num_y_wires]

    # Turn wires from big endian to little endian.
    # Truncate x_wires, as values larger than 2**num_y_wires-1 can anyways not be stored.
    x_wires = list(x_wires[::-1][:num_y_wires])
    x_wires += [zero] * (num_y_wires - len(x_wires))
    y_wires = list(y_wires[::-1])
    carries = [zero] + list(work_wires[: num_y_wires - 1])

    for i in range(num_y_wires - 1):
        _ppm_left_block([carries[i], x_wires[i], y_wires[i], carries[i + 1], aux])

    _ppm_last_block([carries[-1], x_wires[-1], y_wires[-1], aux])

    for i in reversed(range(num_y_wires - 1)):
        _ppm_right_block([carries[i], x_wires[i], y_wires[i], carries[i + 1]])


add_decomps(SemiAdder, _semi_adder, _semi_adder_ppm)


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
