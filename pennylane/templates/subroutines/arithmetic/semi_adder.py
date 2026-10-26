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

from math import pi

from pennylane import capture, compiler
from pennylane.allocation import allocate
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT, PPR_8, GlobalPhase, X, Z, adjoint, cond, ctrl, pauli_measure
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


# Native decomposition signatures group wire arguments alphabetically. Operator2 sorts the
# class declaration into constructor order, so set the compilation order after class creation.
# The wires property above preserves the public x/y/work order.
SemiAdder.wire_argnames = ("work_wires", "x_wires", "y_wires")


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


#######################################################################################
# Measurement-based (Pauli-product) decomposition
#######################################################################################


def _minus():
    """Correct the outcome-dependent sign of a PPM gadget."""
    GlobalPhase(pi)


def _ppm_left_block(wires):
    """Wires are input carry, input bit, target bit, output carry, shared auxiliary qubit."""
    ck, ik, tk, out, aux = wires

    PPR_8(1, "Y", wires=[out])
    PPR_8(-1, "ZZY", wires=[ck, tk, out])
    PPR_8(-1, "ZZY", wires=[ck, ik, out])
    PPR_8(1, "ZZY", wires=[ik, tk, out])

    # Lambda(Z_ck | X_tk X_out), auxiliary qubit |0> -> |+>
    m1 = pauli_measure("ZX", wires=[ck, aux])
    m2 = pauli_measure("ZXX", wires=[aux, tk, out])
    m3 = pauli_measure("X", wires=[aux])
    cond(m3 == 1, Z)(aux)
    cond(m2 == 1, Z)(ck)
    cond(m1 != m3, X)(out)
    cond((m2 == 1) & (m1 != m3), _minus)()

    # Lambda(Z_ik | X_tk), auxiliary qubit |+> -> |0>
    m4 = pauli_measure("ZZ", wires=[ik, aux])
    m5 = pauli_measure("XX", wires=[aux, tk])
    m6 = pauli_measure("Z", wires=[aux])
    cond(m6 == 1, X)(aux)
    cond(m5 == 1, Z)(ik)
    # X_tk of the first gadget commutes with all three measurements of the second one, so
    # it is deferred and grouped with the X_tk correction of the second gadget.
    cond((m1 != m3) != (m4 != m6), X)(tk)
    cond((m5 == 1) & (m4 != m6), _minus)()


def _ppm_last_block(wires):
    """Wires are input carry, input bit, target bit, shared auxiliary qubit. No output carry.

    Both CNOTs of the last block share the same target, and

    .. math:: \\Lambda(Z_{c_k}, X_{t_k}) \\Lambda(Z_{i_k}, X_{t_k})
        = \\Lambda(Z_{i_k} Z_{c_k}, X_{t_k}),

    so a single gadget is enough here.
    """
    ck, ik, tk, aux = wires

    def flip_ctrl():
        Z(ck)
        Z(ik)

    # Lambda(Z_ck Z_ik | X_tk), auxiliary qubit |0> -> |+>
    m1 = pauli_measure("ZZX", wires=[ck, ik, aux])
    m2 = pauli_measure("ZX", wires=[aux, tk])
    m3 = pauli_measure("X", wires=[aux])
    cond(m3 == 1, Z)(aux)
    cond(m2 == 1, flip_ctrl)()
    cond(m1 != m3, X)(tk)
    cond((m2 == 1) & (m1 != m3), _minus)()

    # Only one gadget ran, so the auxiliary qubit is left in |+>: reset it to |0>.
    m4 = pauli_measure("Z", wires=[aux])
    cond(m4 == 1, X)(aux)


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

        # Lambda(Z_ck Z_ik | Z_ik Z_tk), with the freed carry as the |+> auxiliary qubit
        m1 = pauli_measure("ZZZ", wires=[ck, ik, out])
        m2 = pauli_measure("XZZ", wires=[out, ik, tk])
        m3 = pauli_measure("Z", wires=[out])
        cond(m3 == 1, X)(out)
        cond(m2 == 1, flip_ctrl)()
        cond(m1 != m3, flip_target)()
        cond((m2 == 1) & (m1 != m3), _minus)()

    def clean():
        outcome = pauli_measure("Z", wires=[out])
        cond(outcome == 1, X)(out)

    readout = pauli_measure("X", wires=[out])
    cond(readout == 1, correct, clean)()


def _semi_adder_ppm_work_wires(y_wires=None, work_wires=(), **_):
    """The carries, the shared auxiliary qubit and one wire held at |0>."""
    return {"zeroed": max(len(y_wires) + 1 - len(work_wires), 0)}


def _semi_adder_ppm_condition(x_wires=None, y_wires=None, **_):
    """``pauli_measure`` and the corrections require capture or an active compiler."""
    if not (capture.enabled() or compiler.active()):
        return False
    return len(x_wires) > 0 and len(y_wires) > 0


def _semi_adder_ppm_resources(y_wires=None, **_):
    """Counts on the branch where every right block needs its correction.

    Each of the ``num_blocks`` left blocks contributes one ``Y`` and three ``ZZY``
    rotations and the six measurements of its two gadgets; the single last block
    contributes one gadget plus the reset of the auxiliary qubit; each right block
    contributes its ``X`` readout and the three measurements of its correction.
    """
    num_blocks = len(y_wires) - 1
    return {
        PPR_8(1, "Y", Wire[1]): num_blocks,
        PPR_8(-1, "ZZY", Wire[3]): 2 * num_blocks,
        PPR_8(1, "ZZY", Wire[3]): num_blocks,
        # left blocks + last block + right block readouts
        PauliMeasure("X", wires=Wire[1]): 2 * num_blocks + 1,
        # left blocks + auxiliary reset of the last block + right block corrections
        PauliMeasure("Z", wires=Wire[1]): 2 * num_blocks + 1,
        PauliMeasure("ZX", wires=Wire[2]): num_blocks + 1,
        PauliMeasure("ZZ", wires=Wire[2]): num_blocks,
        PauliMeasure("XX", wires=Wire[2]): num_blocks,
        PauliMeasure("ZXX", wires=Wire[3]): num_blocks,
        PauliMeasure("ZZX", wires=Wire[3]): 1,
        PauliMeasure("ZZZ", wires=Wire[3]): num_blocks,
        PauliMeasure("XZZ", wires=Wire[3]): num_blocks,
        X: 4 * num_blocks + 2,
        Z: 9 * num_blocks + 3,
        GlobalPhase(Float): 3 * num_blocks + 1,
    }


@register_condition(_semi_adder_ppm_condition)
@register_resources(_semi_adder_ppm_resources, work_wires=_semi_adder_ppm_work_wires, exact=False)
def _semi_adder_ppm(x_wires, y_wires, work_wires=None):
    """The ladder of `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_ written with
    Pauli product rotations and Pauli product measurements.

    Only the four pi/8 rotations of each temporary AND stay unitary; every CNOT becomes a
    Pauli-controlled-Pauli

    .. math:: \\Lambda(P_c | P_t) = \\Pi_{+P_c} \\otimes I + \\Pi_{-P_c} \\otimes P_t,

    which takes three Pauli product measurements, an auxiliary qubit and a few classically
    conditioned Pauli corrections. The gadget flips the auxiliary qubit between ``|0>`` and
    ``|+>``, so each left block spends two of them and hands the auxiliary qubit back in
    ``|0>``, while the last block runs a single gadget and resets the auxiliary qubit.

    Requires ``len(y_wires) + 1`` work wires, all of them returned to ``|0>``: the
    ``len(y_wires) - 1`` carries, the shared gadget auxiliary qubit and one wire held at
    ``|0>``. That last wire stands in for the missing input carry of the first block and for
    the missing bits of ``x``, so that every block is the same circuit.
    """
    # TODO: the extra |0> wire only exists to keep the first block and the blocks past the end
    # of ``x`` identical to the rest. Specialising those blocks would remove it, bringing the
    # requirement down to the len(y_wires) - 1 carries plus the shared auxiliary qubit.
    work_wires = [] if work_wires is None else list(work_wires)
    num_missing = max(len(y_wires) + 1 - len(work_wires), 0)
    # The right ladder restores the work wires to zero, so they can be borrowed and
    # returned; the context manager deallocates them once the ladder is done with them.
    with allocate(num_missing, restored=True) as new_wires:
        _ppm_ladder(x_wires, y_wires, work_wires + list(new_wires))


def _ppm_ladder(x_wires, y_wires, work_wires):
    """The left/last/right block sequence of the PPM decomposition of ``SemiAdder``."""
    num_y_wires = len(y_wires)

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
        ctrl_kwargs = {
            "control": control_wires,
            "control_values": control_values,
            "work_wires": ctrl_work_wires,
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

        skip_input_pos = _effective_skip_input_pos(len(x_wires), len(y_wires), [])
        _left_ladder(
            x_wires, y_wires, work_wires, carry_flip=carry_flip, skip_input_pos=skip_input_pos
        )

        ctrl(CNOT([work_wires[-1], y_wires[-1]]), **ctrl_kwargs)
        if num_x_wires >= num_y_wires:
            ctrl(CNOT([x_wires[-1], y_wires[-1]]), **ctrl_kwargs)

        _controlled_right_ladder(x_wires, y_wires, work_wires, carry_flip=carry_flip, **ctrl_kwargs)


def _ppm_controlled_pauli(control_wires, target_wires, aux, target_basis="Z", zeroed=True):
    """Implement a Z-parity-controlled Pauli, toggling the auxiliary between zero and plus."""

    def flip_control():
        for wire in control_wires:
            Z(wire)

    def flip_target():
        for wire in target_wires:
            (X if target_basis == "X" else Z)(wire)

    first_basis, second_basis = ("X", "Z") if zeroed else ("Z", "X")
    m1 = pauli_measure("Z" * len(control_wires) + first_basis, [*control_wires, aux])
    m2 = pauli_measure(second_basis + target_basis * len(target_wires), [aux, *target_wires])
    m3 = pauli_measure(first_basis, [aux])
    cond(m3 == 1, Z if zeroed else X)(aux)
    cond(m2 == 1, flip_control)()
    cond(m1 != m3, flip_target)()
    cond((m2 == 1) & (m1 != m3), _minus)()


def _ppm_controlled_left_block(wires, zeroed=True):
    """Compute the carry while leaving the input and target bits unchanged."""
    ck, ik, tk, out, aux = wires
    PPR_8(1, "Y", [out])
    PPR_8(-1, "ZZY", [ck, tk, out])
    PPR_8(-1, "ZZY", [ck, ik, out])
    PPR_8(1, "ZZY", [ik, tk, out])
    _ppm_controlled_pauli([ck], [out], aux, target_basis="X", zeroed=zeroed)


def _ppm_controlled_erase_carry(wires):
    """Erase the carry before applying the controlled sum to the target bit."""
    ck, ik, tk, out = wires

    def correct():
        Z(out)
        Z(ck)
        _ppm_controlled_pauli([ck, ik], [ck, tk], out, zeroed=False)

    def clean():
        outcome = pauli_measure("Z", [out])
        cond(outcome == 1, X)(out)

    readout = pauli_measure("X", [out])
    cond(readout == 1, correct, clean)()


def _ppm_controlled_sum(control, wires, zeroed=True):
    """Add ``control & (carry ^ input)`` using a temporary AND of the two parities."""
    ck, ik, tk, out, aux = wires
    PPR_8(1, "Y", [out])
    PPR_8(-1, "ZY", [control, out])
    PPR_8(-1, "ZZY", [ck, ik, out])
    PPR_8(1, "ZZZY", [control, ck, ik, out])
    _ppm_controlled_pauli([out], [tk], aux, target_basis="X", zeroed=zeroed)

    def correct():
        Z(out)
        _ppm_controlled_pauli([control], [ck, ik], out, zeroed=False)

    def clean():
        outcome = pauli_measure("Z", [out])
        cond(outcome == 1, X)(out)

    readout = pauli_measure("X", [out])
    cond(readout == 1, correct, clean)()


def _controlled_semi_adder_ppm_condition(base, control_wires, **_):
    """The PPM rule requires capture or a compiler, and one quantum control."""
    return (
        (capture.enabled() or compiler.active())
        and len(control_wires) == 1
        and len(base.x_wires) > 0
        and len(base.y_wires) > 0
    )


def _controlled_semi_adder_ppm_work_wires(
    base, *_, work_wires=None, work_wire_type="borrowed", **__
):
    """Count the carries, gadget qubit, zero padding and top-bit temporary AND."""
    num_available = len(base.work_wires)
    if work_wire_type == "zeroed" and work_wires is not None:
        num_available += len(work_wires)
    return {"zeroed": max(len(base.y_wires) + 2 - num_available, 0)}


def _controlled_semi_adder_ppm_resources(base, *_, **__):
    """Upper bounds with every measurement-based uncomputation taking its correction."""
    num_bits = len(base.y_wires)
    num_carries = num_bits - 1
    return {
        PPR_8(1, "Y", Wire[1]): num_carries + num_bits,
        PPR_8(-1, "ZY", Wire[2]): num_bits,
        PPR_8(-1, "ZZY", Wire[3]): 2 * num_carries + num_bits,
        PPR_8(1, "ZZY", Wire[3]): num_carries,
        PPR_8(1, "ZZZY", Wire[4]): num_bits,
        PauliMeasure("X", wires=Wire[1]): 3 * num_bits - 1,
        PauliMeasure("Z", wires=Wire[1]): 3 * num_bits - 1,
        PauliMeasure("ZX", wires=Wire[2]): 2 * num_bits,
        PauliMeasure("ZZ", wires=Wire[2]): num_carries + num_bits,
        PauliMeasure("XX", wires=Wire[2]): num_carries,
        PauliMeasure("ZZZ", wires=Wire[3]): num_carries,
        PauliMeasure("XZZ", wires=Wire[3]): num_carries + num_bits,
        X: 5 * num_bits - 2,
        Z: 13 * num_bits - 7,
        GlobalPhase(Float): 2 * (num_carries + num_bits),
    }


@register_condition(_controlled_semi_adder_ppm_condition)
@register_resources(
    _controlled_semi_adder_ppm_resources,
    work_wires=_controlled_semi_adder_ppm_work_wires,
    exact=False,
)
def _controlled_semi_adder_ppm(
    base, control_wires, work_wires=None, work_wire_type="borrowed", **_
):
    r"""The single-control version of Gidney's Fig. 4 in PPRs and PPMs.

    Commuting the carry fan-out through the temporary AND leaves the four rotated
    axes of ``_ppm_controlled_left_block``. Its data bits remain unchanged, so the
    carry readout is corrected by
    :math:`Z_c\Lambda(Z_c Z_i \mid Z_c Z_t)` before the target is modified.
    The two controlled sum updates combine into a temporary AND of the external
    control with the parity :math:`c \oplus i`, costing four more pi/8 PPRs.

    The shared gadget auxiliary alternates between ``|0>`` and ``|+>`` and only
    needs one final reset. Each erased carry or sum flag supplies the auxiliary
    for its own correction. For ``n = len(base.y_wires)``, this gives ``8n - 4``
    non-Clifford PPRs and at most ``14n - 6`` direct PPMs. The expected direct PPM
    count is ``12n - 5``, including the resets when no correction is needed.

    The ``flip_zero_control`` wrapper normalizes the control value to one. This rule uses
    ``len(base.y_wires) + 2`` zeroed work wires and restores them all. Borrowed
    control work wires are untouched; zeroed control work wires can supplement
    those supplied to the base operator.
    """
    zeroed_wires = list(base.work_wires)
    if work_wire_type == "zeroed" and work_wires is not None:
        zeroed_wires += list(work_wires)
    num_missing = max(len(base.y_wires) + 2 - len(zeroed_wires), 0)
    with allocate(num_missing, restored=True) as new_wires:
        _ppm_controlled_ladder(base, control_wires[0], zeroed_wires + list(new_wires))


def _ppm_controlled_ladder(base, control, work_wires):
    """Compute carries, update the top bit and erase carries before each lower update."""
    num_bits = len(base.y_wires)
    aux, zero, top = work_wires[num_bits - 1 : num_bits + 2]
    x_wires = list(base.x_wires[::-1][:num_bits])
    x_wires += [zero] * (num_bits - len(x_wires))
    y_wires = list(base.y_wires[::-1])
    carries = [zero] + list(work_wires[: num_bits - 1])

    for i in range(num_bits - 1):
        _ppm_controlled_left_block(
            [carries[i], x_wires[i], y_wires[i], carries[i + 1], aux], zeroed=i % 2 == 0
        )
    _ppm_controlled_sum(
        control,
        [carries[-1], x_wires[-1], y_wires[-1], top, aux],
        zeroed=(num_bits - 1) % 2 == 0,
    )
    for i in reversed(range(num_bits - 1)):
        wires = [carries[i], x_wires[i], y_wires[i], carries[i + 1]]
        _ppm_controlled_erase_carry(wires)
        _ppm_controlled_sum(control, [*wires, aux], zeroed=i % 2 == 0)
    outcome = pauli_measure("Z", [aux])
    cond(outcome == 1, X)(aux)


add_decomps(
    "C(SemiAdder)",
    flip_zero_control2(_controlled_semi_adder),
    flip_zero_control2(_controlled_semi_adder_ppm),
)


def _self_ctrl_one_sparse_add_resources(
    num_x_wires, num_y_wires, num_work_wires, first_and_is_output_copy=False
):
    """Resources for _self_ctrl_one_sparse_add below."""
    if num_y_wires == 1:
        return {CNOT: 1}

    skip_input_pos = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos=[1])
    num_elbows = num_y_wires - 1 - int(first_and_is_output_copy)
    num_blocks = num_y_wires - 2
    second_middle_cnot = int(num_y_wires - 1 not in skip_input_pos)
    num_zeroed_blocks = len(skip_input_pos) - (1 - second_middle_cnot)
    # each _left_block uses 3 CNOTs, each _left_block_zeroed none
    # each _ctrl_right_block uses 3 CNOTs, each _right_block_zeroed none
    # each _ctrl_right_block or _ctrl_right_block_zeroed uses 1 ctrl(CNOT)
    # the remaining construction uses 1 CNOT and 1 + second_middle_cnot ctrl(CNOT)s
    ccnot_rep = ctrl(
        CNOT(Wire[2]),
        Wire[1],
        work_wires=Wire[num_work_wires - (num_y_wires - 1)],
        work_wire_type="zeroed",
    )
    return {
        TemporaryAND: num_elbows,
        adjoint(TemporaryAND(Wire[3])): num_elbows,
        CNOT: 6 * (num_blocks - num_zeroed_blocks) + 1 + 2 * int(first_and_is_output_copy),
        ccnot_rep: num_blocks + (1 + second_middle_cnot),
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
        "control": x_wires[:1],
        "control_values": [1],
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
    x_pos = 1
    for i in range(1, num_y_wires - 1):
        if i in skip_input_pos:
            _left_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            _left_block([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos += 1

    ctrl(CNOT([work_wires[-1], y_wires[-1]]), **ctrl_kwargs)
    if num_y_wires - 1 not in skip_input_pos:
        ctrl(CNOT([x_wires[x_pos], y_wires[-1]]), **ctrl_kwargs)

    x_pos -= 1
    for i in range(num_y_wires - 2, 0, -1):
        if i in skip_input_pos:
            _ctrl_right_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]], **ctrl_kwargs)
        else:
            _wires = [work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]]
            _ctrl_right_block(_wires, **ctrl_kwargs)
            x_pos -= 1

    if first_and_is_output_copy:
        CNOT([y_wires[0], work_wires[0]])
    else:
        adjoint(TemporaryAND([x_wires[0], y_wires[0], work_wires[0]]))
    CNOT([x_wires[0], y_wires[0]])
