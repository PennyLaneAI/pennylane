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
Contains the SignedOutSquare template.
"""

from .out_square import _SquareArithmeticOp


class SignedOutSquare(_SquareArithmeticOp):
    r"""Performs out-of-place squaring of a signed integer.

    This operator performs the squaring of a signed integer :math:`x` in 2s complement
    convention into an unsigned register. The computation is modulo :math:`2^m`,
    where ``m=len(output_wires)``:

    .. math::
        \text{SignedOutSquare} |x \rangle |b \rangle = |x \rangle |(b + x^2) \; \text{mod} \; 2^m \rangle,

    .. seealso:: :class:`~.OutSquare` and :class:`~.OutMultiplier`.

    The input is given in `2s complement <https://en.wikipedia.org/wiki/Two%27s_complement>`__.
    Wire :math:`0` stores the sign bit and wire :math:`i` stores the bit with weight :math:`2^{n-1-i}`
    for a register of length :math:`n`. For example, the value :math:`x=-5` encoded by a bitstring
    of length six is given by

    .. math::

        -5 = -32 + 16 + 8 + 2 + 1 = -2^{n-1} + 2^{n-2} + 2^{n-3} + 2^{n-5} + 2^{n-6} = (111011)_{\text{2s compl}}.

    The first bit of each encoded bitstring gives the sign of the encoded number.
    :math:`1 \mapsto -`, :math:`0 \mapsto +`. This is however not a sign-magnitude encoding. Iff
    the encoded number is negative, the rest of the bits do not give the magnitude.

    Args:
        x_wires (WiresLike): wires that store the integer :math:`x`.
        output_wires (WiresLike): wires that store the squaring result. If the
            register is in a non-zero state :math:`b`, the solution will be added to this value.
            If the register is guaranteed to be in the zero state, it is recommended to set
            ``output_wires_zeroed=True`` to reduce the cost of the operation.
        work_wires (WiresLike): the auxiliary wires to use for the squaring.
            ``len(output_wires)`` work wires are required if ``output_wires_zeroed=False``,
            otherwise ``min(len(output_wires), len(x_wires))`` work wires are required.
        output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    **Example**

    Let's compute the square of :math:`x=-5` and :math:`x=7` in superposition, added to :math:`b=5`
    modulo :math:`2^n=2^6=64`.

    .. code-block:: python

        import pennylane as qp

        x_wires = list(range(4))
        output_wires = list(range(4, 10))
        work_wires = list(range(10, 16))

        dev = qp.device("lightning.qubit", wires=16, seed=295)

        @qp.qnode(dev, shots=1_000)
        def circuit(output_wires):
            # Create a uniform superposition between integers -5 and 7:
            # - Create superposition between -8=(1000)_{2s compl} and 0=(0000)_{2s compl}
            qp.H(x_wires[0])
            # - Flip 4's bit if sign bit=0 (=> superposition of -8 and 4)
            qp.ctrl(qp.X(x_wires[1]), x_wires[0], control_values=[0])
            # - Add 3 (x_wires[2:] are zeroed, can just do bit flips) => superposition of -5 and 7
            qp.BasisEmbedding(qp.math.int_to_binary(3, len(x_wires[2:])), wires=x_wires[2:])
            # Prepare initial state on output wires
            qp.BasisEmbedding(qp.math.int_to_binary(5, len(output_wires)), wires=output_wires)
            # Signed square
            qp.SignedOutSquare(x_wires, output_wires, work_wires)
            return qp.counts(wires=output_wires)

    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {30: np.int64(498), 54: np.int64(502)}

    We correctly obtain the squared numbers added to :math:`b=5`, namely
    :math:`5+(-5)^2=30` and :math:`5+7^2=54`.

    Note that reducing the size of the output register (here from ``m=6`` to ``m=4``)
    changes the computed numbers via the reduced modulus:

    >>> output_wires = list(range(4, 8))
    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {6: np.int64(501), 14: np.int64(499)}

    The new results are consistent with the previous ones: the smaller output :math:`30` is
    changed to :math:`30\!\mod\!2^4=14` and :math:`54` is changed to :math:`54\!\mod\!2^4=6`.

    Note that the keyword argument ``output_wires_zeroed`` is passed on to the :class:`~.OutSquare`
    used in the decomposition, leading to smaller cost of this main component. See the usage
    details of ``OutSquare`` for details.

    .. details::
        :title: Theoretical background
        :href: theory

        We compute the square of the signed input register in three steps. First, note that
        :math:`x=x_u-2^{n-1}x_s`, where :math:`x_u` is the unsigned part of :math:`x`
        and :math:`x_s` is the sign bit. As an example, consider the bit string
        :math:`(1101)_{\text{2s compl}}=(101)_2 - 2^3 = 5-8=-3`. We have :math:`x_u=5`
        and :math:`x_s=1`.

        Consider the binomial expansion of :math:`x^2`:

        .. math::

            x^2 = (x_u-2^{n-1}x_s)^2=x_u^2 - x_s (2^{n}x_u) + 2^{2n-2}x_s^2.
            =x_u^2 + x_s 2^{n} (2^{n-2}-x_u)

        In order to arrive at steps that we can easily compute, we rewrite this as

        .. math::

            x^2 = x_u^2 + x_s 2^n (2^{n-1}-x_u) - x_s 2^{2n-2}

        The three steps in our calculation directly correspond to these three terms. That is,
        we first compute the square of :math:`x_u` into the output register, shown here for
        three input and 7 output wires.

        .. code-block:: pycon

                ╭ 0: ────────────
              x ┤ 1: ─╭OutSquare─
                ╰ 2: ─├OutSquare─
                ╭ 3: ─├OutSquare─
                │ 4: ─├OutSquare─
                │ 5: ─├OutSquare─
              y ┤ 6: ─├OutSquare─
                │ 7: ─├OutSquare─
                │ 8: ─├OutSquare─
                │ 9: ─├OutSquare─
                ╰10: ─├OutSquare─
                ╭11: ─├OutSquare─
                │12: ─├OutSquare─
            aux ┤13: ─├OutSquare─
                │14: ─├OutSquare─
                ╰15: ─╰OutSquare─

        Then we add :math:`2^n(2^{n-1}-x_u)`, controlled on the sign bit :math:`x_s`. For this,
        we simply combine some simple bit flips with a controlled :class:`~.SemiAdder`. In our
        example from above we obtain:

        .. code-block:: pycon

                ╭ 0: ───────────────────────╭●────────╭●──────────────╭●──────
              x ┤ 1: ─|Ψ⟩───────╭X────╭●────│──────●╮─├●────╭X────────│───|Ψ⟩─
                ╰ 2: ─────╭●────│─────│─────│───────│─│─────│──────●╮─├●──────
                ╭ 3: ─────│─────│─────│─────│───────│─│─────│───────│─│───────
                │ 4: ─────│─────│─────│─────├X──────│─│─────│───────│─│───X───
                │ 5: ─────│─────│──╭X─├●────│──────●┤─╰X─╭X─│───────│─│───X───
              y ┤ 6: ─X───├●────│──│──│─────│───────│────│──│──────●┤─╰X──X───
                │ 7: ─────│─────│──│──│─────│───────│────│──│───────│─────────
                │ 8: ─────│─────│──│──│─────│───────│────│──│───────│─────────
                │ 9: ─────│─────│──│──│─────│───────│────│──│───────│─────────
                ╰10: ─────│─────│──│──╰⊕─╭X─╰●─╭X──⊕╯────│──│───────│─────────
            aux  11: ─────╰⊕──X─╰●─╰●────╰●────╰●────────╰●─╰●──X──⊕╯─────────

        Finally, we subtract :math:`x_s 2^{2n-2}` from the output, by wrapping an
        adder onto the most significant bits in bit flips of the same output subregister.

        Overall, we arrive at the following decomposition:

        >>> op = qp.SignedOutSquare(range(3), range(3, 10), range(10, 16), True)
        >>> qp.inspect_decomps(op, "signed_square_from_unsigned_square")
        Decomposition 0 (name: signed_square_from_unsigned_square)
         0: ──────────────────────────────────────────╭●────────╭●──────────────╭●──────────── ···
         1: ─╭OutSquare──MultiX(M0)───────╭X────╭●────│──────●╮─├●────╭X────────│───MultiX(M0) ···
         2: ─├OutSquare─────────────╭●────│─────│─────│───────│─│─────│──────●╮─├●──────────── ···
         3: ─├OutSquare─────────────│─────│─────│─────│───────│─│─────│───────│─│───────────── ···
         4: ─├OutSquare─────────────│─────│─────│─────├X──────│─│─────│───────│─│───X───────── ···
         5: ─├OutSquare─────────────│─────│──╭X─├●────│──────●┤─╰X─╭X─│───────│─│───X───────── ···
         6: ─├OutSquare──X──────────├●────│──│──│─────│───────│────│──│──────●┤─╰X──X───────── ···
         7: ─├OutSquare─────────────│─────│──│──│─────│───────│────│──│───────│─────────────── ···
         8: ─├OutSquare─────────────│─────│──│──│─────│───────│────│──│───────│─────────────── ···
         9: ─├OutSquare─────────────│─────│──│──│─────│───────│────│──│───────│─────────────── ···
        10: ─├OutSquare─────────────│─────│──│──╰⊕─╭X─╰●─╭X──⊕╯────│──│───────│─────────────── ···
        11: ─├OutSquare─────────────╰⊕──X─╰●─╰●────╰●────╰●────────╰●─╰●──X──⊕╯─────────────── ···
        12: ─├OutSquare─────────────────────────────────────────────────────────────────────── ···
        13: ─├OutSquare─────────────────────────────────────────────────────────────────────── ···
        14: ─├OutSquare─────────────────────────────────────────────────────────────────────── ···
        15: ─╰OutSquare─────────────────────────────────────────────────────────────────────── ···
        <BLANKLINE>
         0: ··· ─╭SemiAdder────┤
         1: ··· ─│─────────────┤
         2: ··· ─│─────────────┤
         3: ··· ─│─────────────┤
         4: ··· ─├SemiAdder──X─┤
         5: ··· ─├SemiAdder──X─┤
         6: ··· ─│─────────────┤
         7: ··· ─│─────────────┤
         8: ··· ─│─────────────┤
         9: ··· ─│─────────────┤
        10: ··· ─├SemiAdder────┤
        11: ··· ─├SemiAdder────┤
        12: ··· ─├SemiAdder────┤
        13: ··· ─├SemiAdder────┤
        14: ··· ─├SemiAdder────┤
        15: ··· ─╰SemiAdder────┤
        M0 =
        [ True]
        Gate Count: {Adjoint(TemporaryAND): 2, CNOT: 6, Controlled(PauliX, control_wires=AbstractWires(2), work_wires=AbstractWires(4), control_values=AbstractArray((2,), bool)): 3, MultiX(AbstractArray((1,), bool, weak_type=True), wires=AbstractWires(1)): 2, OutSquare(x_wires=AbstractWires(2), output_wires=AbstractWires(7), work_wires=AbstractWires(6), output_wires_zeroed=True): 1, PauliX: 8, SemiAdder(x_wires=AbstractWires(1), y_wires=AbstractWires(2), work_wires=AbstractWires(6)): 1, TemporaryAND: 2}

    """

    @staticmethod
    def _min_work_wires(n, m, output_wires_zeroed):
        # Work wires required for the unsigned square (its input is reduced by one)
        # Work wires required for the first correction adder are `min(m-n, n)-1`, which is smaller
        # than `m` and smaller than `n`, so that the unsigned square always needs more work wires.
        return min(n, m) if output_wires_zeroed else m
