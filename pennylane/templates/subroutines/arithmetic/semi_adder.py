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

from pennylane.core.operator import Operator2
from pennylane.ops import CNOT, adjoint, ctrl
from pennylane.typing import Wire
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
