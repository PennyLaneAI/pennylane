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

from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


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
            qp.BasisState(qp.math.int_to_binary(3, n-1), wires=wires["x"][1:])
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

            @qp.transforms._tape_decompose(max_expansion=1) # To see resources easily
            @qp.qnode(dev, shots=1_000)
            def circuit(zeroed):
                qp.BasisState(qp.math.int_to_binary(13, len(x_wires)), wires=x_wires)
                qp.OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
                return qp.counts(wires=output_wires)

        We can compute the required resources with ``zeroed=False``, i.e., when not passing
        the information to the template:

        >>> specs_false = qp.specs(circuit)(False).resources.quantum_operations  # doctest: +SKIP
        >>> print(specs_false)  # doctest: +SKIP
        {'BasisState': 1, 'C(MultiX)': 4, 'MultiControlledX': 12, 'TemporaryAND': 19, 'CNOT': 49, 'Adjoint(TemporaryAND)': 19, 'MultiX': 4, 'SemiAdder': 2}

        When we do pass the information, we reduce the required resources by a lot:

        >>> specs_true = qp.specs(circuit)(True).resources.quantum_operations  # doctest: +SKIP
        >>> print(specs_true)  # doctest: +SKIP
        {'BasisState': 1, 'TemporaryAND': 10, 'CNOT': 24, 'MultiControlledX': 8, 'Adjoint(TemporaryAND)': 7}

        Of course, both decompositions are correctly implementing the squaring operation:

        >>> print(circuit(False))
        {np.str_('10101001'): np.int64(1000)}
        >>> print(circuit(True))
        {np.str_('10101001'): np.int64(1000)}

        Here, :math:`(10101001)_2=128 + 32 + 8 + 1=169` is the expected result of :math:`13^2`.
        To conclude, we draw the more efficient circuit variant:

        >>> print(qp.draw(circuit)(True))
         0: ─╭|Ψ⟩───────╭●─────────────╭X────╭●───────────●╮─╭●────╭X─────────────────────────────────── ···
         1: ─├|Ψ⟩────╭●─│──╭X────╭●────│─────│─────────────│─│─────│──────●╮─╭●────╭X──────────────╭●─── ···
         2: ─├|Ψ⟩─╭●─│──│──│─────│─────│─────│─────╭●──────│─├●────│───────│─├●────│──────╭●────╭●─│──── ···
         3: ─╰|Ψ⟩─├●─├●─├●─│─────│─────│─────│─────│───────│─│─────│───────│─│─────│──────│─────│──│──── ···
         4: ──────│──│──│──│─────│─────│─────│─────│───────│─│─────│───────│─│─────│──────│─────│──│──── ···
         5: ──────│──│──│──│─────│─────│─────│─────├X──────│─│─────│───────│─│─────│──────│─────│──│──── ···
         6: ──────│──│──│──│─────│─────│──╭X─├●────│──────●┤─╰X─╭X─│───────│─│─────│──────│─────│──│──╭● ···
         7: ──────│──│──╰⊕─│──╭X─├●────│──│──│─────│───────│────│──│──────●┤─╰X─╭X─│──────│─────│──├●─│─ ···
         8: ──────│──╰⊕─╭●─│──│──│─────│──│──│─────│───────│────│──│───────│────│──│───●╮─├X────│──│──│─ ···
         9: ──────╰⊕─╭●─│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│──╭●─╰X─│──│─ ···
        10: ─────────│──│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│──│─────│──│─ ···
        11: ─────────│──│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│──│─────│──│─ ···
        12: ─────────╰X─├●─│──│──│─────│──│──│─────│───────│────│──│───────│────│──│───●┤─╰●─╰X────╰⊕─├● ···
        13: ────────────╰⊕─╰●─╰●─│──╭●─│──│──│─────│───────│────│──│──╭●───│────╰●─╰●──⊕╯─────────────╰⊕ ···
        14: ─────────────────────╰⊕─╰X─╰●─╰●─│──╭●─│──╭●───│────╰●─╰●─╰X──⊕╯──────────────────────────── ···
        15: ─────────────────────────────────╰⊕─╰X─╰●─╰X──⊕╯──────────────────────────────────────────── ···
        <BLANKLINE>
         0: ··· ─╭X────╭●───────────●╮─╭●────╭X───────────────╭●─╭●──●╮─╭●────┤
         1: ··· ─│─────│─────╭●──────│─├●────│──────╭●──●╮─╭●─│──│────│─│─────┤
         2: ··· ─│─────│─────│───────│─│─────│──────│────│─│──│──│────│─│─────┤
         3: ··· ─│─────│─────│───────│─│─────│──────│────│─│──│──│────│─│──╭●─┤
         4: ··· ─│─────│─────├X──────│─│─────│──────│────│─│──│──├X───│─│──│──┤ ╭Counts
         5: ··· ─│──╭X─├●────│──────●┤─╰X─╭X─│──────│────│─│──├●─│───●┤─╰X─│──┤ ├Counts
         6: ··· ─│──│──│─────│───────│────│──│───●╮─├X───│─│──│──│────│────│──┤ ├Counts
         7: ··· ─│──│──│─────│───────│────│──│────│─│───●┤─╰X─│──│────│────│──┤ ├Counts
         8: ··· ─│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
         9: ··· ─│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
        10: ··· ─│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
        11: ··· ─│──│──│─────│───────│────│──│────│─│────│────│──│────│────╰X─┤ ╰Counts
        12: ··· ─│──│──│─────│───────│────│──│───●┤─╰●──⊕╯────╰⊕─╰●──⊕╯───────┤
        13: ··· ─╰●─╰●─│──╭●─│──╭●───│────╰●─╰●──⊕╯───────────────────────────┤
        14: ··· ───────╰⊕─╰X─╰●─╰X──⊕╯────────────────────────────────────────┤
        15: ··· ──────────────────────────────────────────────────────────────┤

    """

    @staticmethod
    def _min_work_wires(n, m, output_wires_zeroed):
        return min(n - 1, max(m - 4, 0)) if output_wires_zeroed else m - 1
