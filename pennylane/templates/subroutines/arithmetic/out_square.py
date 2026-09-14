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
"""
Contains the OutSquare template.
"""

from collections import defaultdict

from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.ops import CNOT
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps

from ..multix import MultiX
from .out_multiplier import _c_add_sub, _c_add_sub_resources
from .semi_adder import (
    SemiAdder,
    _self_ctrl_one_sparse_add,
    _self_ctrl_one_sparse_add_resources,
    _semi_adder,
    _semi_adder_resources,
)
from .temporary_and import TemporaryAND


class _SquareArithmeticOp(Operator2, is_baseclass=True):
    """Shared ``Operator2`` boilerplate for :class:`~.OutSquare` and :class:`~.SignedOutSquare`."""

    wire_argnames = ("x_wires", "output_wires", "work_wires")
    compilable_argnames = ("output_wires_zeroed",)

    arg_specs = {
        "x_wires": Wire[-1],
        "output_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    @staticmethod
    def _min_work_wires(n, m, output_wires_zeroed):
        """The minimum number of work wires required for the given register sizes. Must be
        overridden by subclasses."""
        raise NotImplementedError

    def _check_work_wires(self, n, m, work_wires, output_wires_zeroed):
        num_required_work_wires = self._min_work_wires(n, m, output_wires_zeroed)
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"{type(self).__name__} requires at least {num_required_work_wires} work wires "
                f"for {n} input wires, {m} output wires and {output_wires_zeroed=}. "
                f"Got {len(work_wires)} work wires instead."
            )

    def __init__(
        self,
        x_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        output_wires_zeroed: bool = False,
    ):
        x_wires = Wires(x_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)

        self._check_work_wires(len(x_wires), len(output_wires), work_wires, output_wires_zeroed)

        wire_args = {"x_wires": x_wires, "output_wires": output_wires, "work_wires": work_wires}
        validate_no_wire_overlaps(wire_args)

        super().__init__(
            x_wires,
            output_wires,
            work_wires,
            output_wires_zeroed=output_wires_zeroed,
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.x_wires + self.output_wires + self.work_wires


class OutSquare(_SquareArithmeticOp):
    r"""Performs out-of-place squaring.

    This operator performs the squaring of an :math:`n`-qubit integer :math:`x` modulo
    :math:`2^m` into an :math:`m`-qubit output register:

    .. math::
        \text{OutSquare} |x \rangle |y \rangle = |x \rangle |(y + x^2) \; \text{mod} \; 2^m \rangle.

    There are two implementations available, differing in their :class:`~.Toffoli` and auxiliary
    qubit counts. The first is based on Schoolbook multiplication, using controlled addition.
    The second uses controlled add-subtract blocks that also are used by
    Litinski in `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__ to reduce the
    cost of multiplication.

    .. seealso:: :class:`~.OutMultiplier`, :class:`~.SemiAdder` , and :class:`~.Multiplier`.

    Args:
        x_wires (WiresLike): wires that store the integer :math:`x`.
        output_wires (WiresLike): wires that store the squaring result. If the register initially
            encodes a non-zero value :math:`y`, the solution will be added to this value.
            If the register is guaranteed to be in the zero state, it is recommended to set
            ``output_wires_zeroed=True``.
        work_wires (WiresLike): the auxiliary wires to use for the squaring.
            :math:`m-1` work wires are required if ``output_wires_zeroed=False``,
            otherwise :math:`\min(\max(m-4, 0), n-1)` work wires are required.
        output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    **Example**

    Let's compute the square of :math:`x=3` and :math:`x=7` in superposition, added to a
    :math:`m=6`-qubit register that holds the value :math:`y=5` initially.
    The computation will be modulo :math:`2^m=2^6=64`.

    .. code-block:: python

        import pennylane as qp

        n = 3
        m = 6
        wires = qp.registers({"x": n, "out": m, "work": m})

        dev = qp.device("lightning.qubit", wires=n + 2 * m, seed=295)

        @qp.qnode(dev, shots=1_000)
        def circuit(output_wires):
            # Create a uniform superposition between integers 3 and 7
            qp.H(wires["x"][0]) # Superposition between 0 and 4
            # Add 3, by preparing lower-precision wires
            qp.BasisState(qp.math.int_to_binary(3, len(wires["x"][1:])), wires=wires["x"][1:])
            # Prepare initial state on output wires
            qp.BasisState(qp.math.int_to_binary(5, len(output_wires)), wires=output_wires)
            # Square
            qp.OutSquare(wires["x"], output_wires, wires["work"])
            return qp.counts(wires=output_wires)

    >>> counts = circuit(wires["out"])
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {14: np.int64(498), 54: np.int64(502)}

    We correctly obtain the squared numbers added to :math:`y=5`, namely
    :math:`5+3^2=14` and :math:`5+7^2=54`.

    Note that reducing the size of the output register (here from ``m=6`` to ``m=4``)
    changes the computed numbers via the reduced modulus:

    >>> counts = circuit(wires["out"][:4])
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {6: np.int64(501), 14: np.int64(499)}

    The new results are consistent with the previous ones: the smaller output :math:`14` remains
    unchanged because :math:`14 < 16=2^4`, and :math:`54` is changed to :math:`54\!\mod\!2^4=6`.

    .. details::
        :title: Usage Details

        **Cheaper decomposition for zeroed output state**

        If we know that the qubits in ``output_wires`` are in the state
        :math:`|0\rangle^{\otimes m}` before ``OutSquare`` is applied, we can pass this information
        to the template via ``output_wires_zeroed``, leading to a cheaper decomposition.
        Consider the following example, where we control this information with the ``QNode``
        argument ``zeroed``:

        .. code-block:: python

            n = 4
            m = 8
            x_wires = list(range(n))
            output_wires = list(range(n, n + m))
            work_wires = list(range(n + m, n + 2 * m))

            dev = qp.device("lightning.qubit", wires=20, seed=295)

            @qp.decompose(max_expansion=1) # To see resources easily
            @qp.qnode(dev, shots=1_000)
            def circuit(zeroed):
                qp.BasisState(qp.math.int_to_binary(13, len(x_wires)), wires=x_wires)
                qp.OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
                return qp.counts(wires=output_wires)

        We can compute the required resources with ``zeroed=False``, i.e., when not passing
        the information to the template:

        >>> specs_false = qp.specs(circuit)(False).resources.quantum_operations
        >>> print(specs_false)
        {'BasisState': 1, 'C(BasisState)': 4, 'MultiControlledX': 12, 'TemporaryAND': 19, 'CNOT': 49, 'Adjoint(TemporaryAND)': 19, 'MultiX': 6, 'SemiAdder': 2}

        When we do pass the information, we reduce the required resources by a lot:

        >>> specs_true = qp.specs(circuit)(True).resources.quantum_operations
        >>> print(specs_true)
        {'BasisState': 1, 'TemporaryAND': 11, 'CNOT': 22, 'MultiControlledX': 8, 'Adjoint(TemporaryAND)': 8}

        Of course, both decompositions are correctly implementing the squaring operation:

        >>> print(circuit(False))
        {np.str_('10101001'): np.int64(1000)}
        >>> print(circuit(True))
        {np.str_('10101001'): np.int64(1000)}

        Here, :math:`(10101001)_2=128 + 32 + 8 + 1=169` is the expected result of :math:`13^2`.
        To conclude, we draw the more efficient circuit variant:

        >>> print(qp.draw(circuit)(True))
         0: ─╭|Ψ⟩───────╭●───────────────────╭X────╭●───────────●╮─╭●────╭X─────────────────────────── ···
         1: ─├|Ψ⟩────╭●─│────────╭X────╭●────│─────│─────────────│─│─────│──────●╮─╭●────╭X─────────── ···
         2: ─├|Ψ⟩─╭●─│──│──╭●────│─────│─────│─────│─────╭●──────│─├●────│───────│─├●────│──────╭●──●╮ ···
         3: ─╰|Ψ⟩─├●─├●─├●─│─────│─────│─────│─────│─────│───────│─│─────│───────│─│─────│──────│────│ ···
         4: ──────│──│──│──│─────│─────│─────│─────│─────│───────│─│─────│───────│─│─────│──────│────│ ···
         5: ──────│──│──│──│─────│─────│─────│─────│─────├X──────│─│─────│───────│─│─────│──────│────│ ···
         6: ──────│──│──│──│─────│─────│─────│──╭X─├●────│──────●┤─╰X─╭X─│───────│─│─────│──────│────│ ···
         7: ──────│──│──╰⊕─│─────│──╭X─├●────│──│──│─────│───────│────│──│──────●┤─╰X─╭X─│──────│────│ ···
         8: ──────│──╰⊕────│──╭●─│──│──│─────│──│──│─────│───────│────│──│───────│────│──│───●╮─├X───│ ···
         9: ──────╰⊕───────├●─│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│───●┤ ···
        10: ───────────────│──│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│────│ ···
        11: ───────────────│──│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│────│ ···
        12: ───────────────╰⊕─├●─│──│──│─────│──│──│─────│───────│────│──│───────│────│──│───●┤─╰●──⊕╯ ···
        13: ──────────────────╰⊕─╰●─╰●─│──╭●─│──│──│─────│───────│────│──│──╭●───│────╰●─╰●──⊕╯─────── ···
        14: ───────────────────────────╰⊕─╰X─╰●─╰●─│──╭●─│──╭●───│────╰●─╰●─╰X──⊕╯──────────────────── ···
        15: ───────────────────────────────────────╰⊕─╰X─╰●─╰X──⊕╯──────────────────────────────────── ···
        <BLANKLINE>
         0: ··· ──────────╭X────╭●───────────●╮─╭●────╭X───────────────╭●─╭●──●╮─╭●────┤
         1: ··· ────╭●────│─────│─────╭●──────│─├●────│──────╭●──●╮─╭●─│──│────│─│─────┤
         2: ··· ─╭●─│─────│─────│─────│───────│─│─────│──────│────│─│──│──│────│─│─────┤
         3: ··· ─│──│─────│─────│─────│───────│─│─────│──────│────│─│──│──│────│─│──╭●─┤
         4: ··· ─│──│─────│─────│─────├X──────│─│─────│──────│────│─│──│──├X───│─│──│──┤ ╭Counts
         5: ··· ─│──│─────│──╭X─├●────│──────●┤─╰X─╭X─│──────│────│─│──├●─│───●┤─╰X─│──┤ ├Counts
         6: ··· ─│──│──╭●─│──│──│─────│───────│────│──│───●╮─├X───│─│──│──│────│────│──┤ ├Counts
         7: ··· ─│──├●─│──│──│──│─────│───────│────│──│────│─│───●┤─╰X─│──│────│────│──┤ ├Counts
         8: ··· ─│──│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
         9: ··· ─╰X─│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
        10: ··· ────│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
        11: ··· ────│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────╰X─┤ ╰Counts
        12: ··· ────╰⊕─├●─│──│──│─────│───────│────│──│───●┤─╰●──⊕╯────╰⊕─╰●──⊕╯───────┤
        13: ··· ───────╰⊕─╰●─╰●─│──╭●─│──╭●───│────╰●─╰●──⊕╯───────────────────────────┤
        14: ··· ────────────────╰⊕─╰X─╰●─╰X──⊕╯────────────────────────────────────────┤
        15: ··· ───────────────────────────────────────────────────────────────────────┤

    """

    @staticmethod
    def _min_work_wires(n, m, output_wires_zeroed):
        return min(n - 1, max(m - 4, 0)) if output_wires_zeroed else m - 1


def _out_square_with_adder_zeroed_condition(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> bool:
    if not output_wires_zeroed:
        return False
    n = len(x_wires)
    m = len(output_wires)
    # pylint: disable-next=protected-access
    return len(work_wires) >= OutSquare._min_work_wires(n, m, output_wires_zeroed)


def _out_square_with_adder_zeroed_resources(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> dict:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    num_work_wires = len(work_wires)
    resources = defaultdict(int)
    # Copying of first bit is a CNOT, all other bits require a TemporaryAND
    resources[CNOT] += 1
    resources[TemporaryAND] = min(n - 1, m - 2)
    num_work_wires += 2  # Using the 1s and 2s output bits as work wires
    p = min(n, m // 2 + m % 2) - 1
    for i in range(1, p + 1):
        x_size = n - i
        y_size = min(m - 2 * i, n + 2 - i)
        # First self-controlled 1-sparse adder can copy the first input bit instead of recomputing
        # a temporary AND (Improvement #4 from Sec IIIA).
        for k, val in _self_ctrl_one_sparse_add_resources(
            x_size, y_size, num_work_wires, i == 1
        ).items():
            resources[k] += val

    return dict(resources)


@register_condition(_out_square_with_adder_zeroed_condition)
@register_resources(_out_square_with_adder_zeroed_resources, name="out_square_with_adder")
def _out_square_with_adder_zeroed(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool = False,
):
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    x_wires = x_wires[::-1]
    output_wires = output_wires[::-1]
    work_wires = Wires.all_wires([work_wires, output_wires[:2]])

    # Copy x, controlled on the least significant bit (LSB) of x, to the output register,
    # which is in |0>. This can be reduced to a CNOT for the LSB and TemporaryANDs for
    # the other bits. The CNOT is performed at the very end, to use the 1s bit of the output as
    # work wire until then.
    num_elbows = min(n - 1, m - 2)
    copy_input = x_wires[1 : num_elbows + 1]
    copy_output = output_wires[2 : num_elbows + 2]
    for x_wire, out_wire in zip(copy_input, copy_output, strict=True):
        TemporaryAND([x_wires[0], x_wire, out_wire])

    p = min(n, m // 2 + (m % 2)) - 1
    for i in range(1, p + 1):
        # Perform specialized "self"-controlled addition with zeroed 2s input bit, using
        # sliced x_wires and output_wires.
        # First self-controlled 1-sparse adder can copy the first input bit instead of recomputing
        # a temporary AND (Improvement #4 from Sec IIIA).
        _self_ctrl_one_sparse_add(
            x_wires[i:], output_wires[2 * i : min(m, n + 2 + i)], work_wires, i == 1
        )

    CNOT([x_wires[0], output_wires[0]])  # First control-copy, delayed until end of decomp.


def _out_square_with_caddsub_condition(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> bool:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    if output_wires_zeroed and (n == 1 or m == 1):
        # Just a single CNOT in the decomposition.
        return True
    return len(work_wires) >= m - 1


def _out_square_with_caddsub_resources(
    x_wires, output_wires, work_wires, output_wires_zeroed=False
) -> dict:
    # pylint: disable=unused-argument
    n = len(x_wires)
    m = len(output_wires)
    num_work_wires = len(work_wires)
    p = min(n - 1, m // 2)

    resources = defaultdict(int)

    # Controlled add-subtract loop
    for i in range(p):
        size = min(n - i, m - 2 * i - 1) if output_wires_zeroed else m - 2 * i - 1
        for key, value in _c_add_sub_resources(n - i - 1, size).items():
            resources[key] += value

    if output_wires_zeroed and p == 0:
        resources[CNOT] += 1
    else:
        skips = [1] + [2 * j for j in range(1, n)]
        sparse_adder_res = _semi_adder_resources(x_wires, output_wires, skip_input_pos=skips)
        for key, value in sparse_adder_res.items():
            resources[key] += value

    if n > 1 and m > 1:
        # Subtract 2 x_{[1:]}
        resources[MultiX(Bool[m - 1], Wire[m - 1])] += 2
        resources[SemiAdder(Wire[n - 1], Wire[m - 1], Wire[num_work_wires])] += 1

        if m > n:
            # Shifted addition
            resources[MultiX(Bool[m - n], Wire[m - n])] += 2
            resources[MultiX(Bool[n - 1], Wire[n - 1])] += 2
            resources[SemiAdder(Wire[n - 1], Wire[m - n], Wire[num_work_wires])] += 1

    return dict(resources)


def _shifted_adder(x_wires, output_wires, work_wires):
    """Perform shifted addition y -> y + x - 2^n + 1."""
    x_ones = [True] * len(x_wires)
    output_ones = [True] * len(output_wires)
    MultiX(x_ones, x_wires)
    MultiX(output_ones, output_wires)
    SemiAdder(x_wires[::-1], output_wires[::-1], work_wires)
    MultiX(output_ones, output_wires)
    MultiX(x_ones, x_wires)


@register_condition(_out_square_with_caddsub_condition)
@register_resources(_out_square_with_caddsub_resources)
def _out_square_with_caddsub(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool = False,
):
    r"""This decomposition uses controlled add-subtract blocks, and three correction
    steps. See Sec. II for details."""
    x_wires = x_wires[::-1]
    output_wires = output_wires[::-1]
    n = len(x_wires)
    m = len(output_wires)
    p = min(n - 1, m // 2)

    for i, x_wire in enumerate(x_wires[:p]):
        if output_wires_zeroed:
            _out_reg = output_wires[2 * i + 1 : n + 1 + i]
        else:
            _out_reg = output_wires[2 * i + 1 :]
        _c_add_sub(x_wire, x_wires[i + 1 :][::-1], _out_reg[::-1], work_wires)

    if output_wires_zeroed and p == 0:
        # output register is still zeroed, no need for a full adder. p=0 holds for n=1 or m=1
        # in both cases we just need a CNOT to copy the LSB of the input into the zeroed output.
        CNOT([x_wires[0], output_wires[0]])
    else:
        _semi_adder(
            x_wires, output_wires, work_wires, skip_input_pos=[1] + [2 * j for j in range(1, n)]
        )

    if n > 1 and m > 1:
        _output = output_wires[1:]
        output_ones = [True] * len(_output)

        MultiX(output_ones, _output)
        SemiAdder(x_wires[1:][::-1], _output[::-1], work_wires)
        MultiX(output_ones, _output)

        # shifted addition
        if m > n:
            _shifted_adder(x_wires[:-1], output_wires[n:], work_wires)


add_decomps(
    OutSquare,
    _out_square_with_adder_zeroed,
    _out_square_with_caddsub,
)
