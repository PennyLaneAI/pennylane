# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Contains the Select template.
"""

from pennylane import math
from pennylane.core.operator import Operator, Operator2
from pennylane.ops import CNOT, adjoint, ctrl
from pennylane.typing import Wire
from pennylane.wires import Wires, validate_no_wire_overlaps

from .arithmetic.temporary_and import TemporaryAND


class Select(Operator2):
    r"""The ``Select`` operator, also available as :class:`~.Multiplexer` and :class:`~.Multiplexor`,
    applies different operations depending on the state of designated control wires.

    .. math:: Select|i\rangle \otimes |\psi\rangle = |i\rangle \otimes U_i |\psi\rangle

    .. figure:: ../../../doc/_static/templates/subroutines/select.png
                    :align: center
                    :width: 70%
                    :target: javascript:void(0);

    If the applied operations :math:`\{U_i\}` are all single-qubit Pauli rotations about the
    same axis, with the angle determined by the control wires, this is also called a
    **uniformly controlled rotation** gate.

    .. seealso:: :class:`~.SelectPauliRot`

    Args:
        ops (list[Operator]): operations to apply
        control (Sequence[int]): the wires controlling which operation is applied.
            At least :math:`\lceil \log_2 K\rceil` wires are required for :math:`K` operations.
        work_wires (Union[Wires, Sequence[int], or int]): auxiliary wire(s) that may be
            utilized during the decomposition of the operator into native operations.
            For details, see the section on the unary iterator decomposition below.
        partial (bool): Whether the state on the wires provided in ``control`` are compatible with
            a `partial Select <https://pennylane.ai/compilation/partial-select>`__ decomposition.
            See the note below for details.

    .. note::
        The position of the operation in the list determines which qubit state implements that
        operation. For example, when the qubit register is in the state :math:`|00\rangle`,
        we will apply ``ops[0]``. When the qubit register is in the state :math:`|10\rangle`,
        we will apply ``ops[2]``. To obtain the list position ``index`` for a given binary
        bitstring representing the control state we can use the following relationship:
        ``index = int(state_string, 2)``. For example, ``2 = int('10', 2)``.

    .. note::
        Using ``partial=True`` assumes that the quantum state :math:`|\psi\rangle` on the
        ``control`` wires satisfies :math:`\langle j|\psi\rangle=0` for all :math:`j\in [K, 2^c)`,
        where :math:`K` is the number of operators (``len(ops)``) and :math:`c` is the number of
        control wires (``len(control)``).
        If you are unsure whether this condition is satisfied, set ``partial=False`` to guarantee
        a correct, even though more expensive, decomposition.
        For more details on the partial Select decomposition, see
        `its compilation page <https://pennylane.ai/compilation/partial-select>`__.

    **Example**

    >>> dev = qp.device('default.qubit', wires=4)
    >>> ops = [qp.X(2), qp.X(3), qp.Y(2), qp.SWAP([2, 3])]
    >>> @qp.qnode(dev)
    ... def circuit():
    ...     qp.Select(ops, control=[0,1])
    ...     return qp.state()
    ...
    >>> print(qp.draw(circuit, level='device')())
    0: ─╭○─╭○─╭●─╭●────┤ ╭State
    1: ─├○─├●─├○─├●────┤ ├State
    2: ─╰X─│──╰Y─├SWAP─┤ ├State
    3: ────╰X────╰SWAP─┤ ╰State

    If there are fewer operators to be applied than possible for the given number of control
    wires, we call the ``Select`` operator a `partial Select <https://pennylane.ai/compilation/partial-select>`__.
    In this case, the control structure can be simplified if the state on the control wires
    does not have overlap with the unused computational basis states (:math:`|j\rangle` with
    :math:`j>K-1`). Passing ``partial=True`` tells ``Select`` that this criterion is
    satisfied, and allows the decomposition to make use of the simplification:

    >>> ops = [qp.X(2), qp.X(3), qp.SWAP([2, 3])]
    >>> @qp.qnode(dev)
    ... def circuit():
    ...     qp.Select(ops, control=[0, 1], partial=True)
    ...     return qp.state()
    ...
    >>> print(qp.draw(circuit, level='device')())
    0: ─╭○────╭●────┤ ╭State
    1: ─├○─╭●─│─────┤ ├State
    2: ─╰X─│──├SWAP─┤ ├State
    3: ────╰X─╰SWAP─┤ ╰State

    Note how the first (second) control node of the second (third) operator was skipped.

    .. details::
        :title: Unary iterator decomposition

        Generically, ``Select`` is decomposed into one multi-controlled operator for each target
        operator. However, if auxiliary wires are available, a decomposition using a
        "unary iterator" can be applied. It was introduced by
        `Babbush et al. (2018) <https://arxiv.org/abs/1805.03662>`__.

        **Principle**

        Unary iteration leverages auxiliary wires to store intermediate values for reuse between
        the different multi-controlled operators, avoiding unnecessary recomputation.
        In addition to this caching functionality, unary iteration reduces the cost of the
        computation directly, because the involved reversible AND (or Toffoli) gates can be
        implemented at lower cost if the target is known to be in the :math:`|0\rangle` state
        (see :class:`~TemporaryAND`).

        For :math:`K` operators to be Select-applied, :math:`c=\lceil\log_2 K\rceil` control
        wires are required. Unary iteration demands an additional :math:`c-1` auxiliary wires.
        Below we first show an example for :math:`K` being a power of two, i.e., :math:`K=2^c`.
        Then we elaborate on implementation details for the case :math:`K<2^c`, which we call
        a *partial Select* operator.

        **Example**

        Assume that we want to Select-apply :math:`K=8=2^3` operators to two target wires,
        which requires :math:`c=\lceil \log_2 K\rceil=3` control wires. The generic
        decomposition for this takes the form

        .. code-block::

            0: ─╭○─────╭○─────╭○─────╭○─────╭●─────╭●─────╭●─────╭●─────┤
            1: ─├○─────├○─────├●─────├●─────├○─────├○─────├●─────├●─────┤
            2: ─├○─────├●─────├○─────├●─────├○─────├●─────├○─────├●─────┤
            3: ─├U(M0)─├U(M1)─├U(M2)─├U(M3)─├U(M4)─├U(M5)─├U(M6)─├U(M7)─┤
            4: ─╰U(M0)─╰U(M1)─╰U(M2)─╰U(M3)─╰U(M4)─╰U(M5)─╰U(M6)─╰U(M7)─┤.

        Unary iteration then uses :math:`c-1=2` auxiliary wires, denoted ``aux0`` and ``aux1``
        below, to first rewrite the control structure:

        .. code-block::

            0:    ─╭○───────○╮─╭○───────○╮─╭○───────○╮─╭○───────○╮─╭●───────●╮─╭●───────●╮─╭●───────●╮─╭●───────●╮─┤
            1:    ─├○───────○┤─├○───────○┤─├●───────●┤─├●───────●┤─├○───────○┤─├○───────○┤─├●───────●┤─├●───────●┤─┤
            aux0:  ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ ╰─╭●───●╮─╯ │
            2:    ───├○───○┤─────├●───●┤─────├○───○┤─────├●───●┤─────├○───○┤─────├●───●┤─────├○───○┤─────├●───●┤───┤
            aux1:    ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯     ╰─╭●──╯   │
            3:    ─────├U(M0)──────├U(M1)──────├U(M2)──────├U(M3)──────├U(M4)──────├U(M5)──────├U(M6)──────├U(M7)──┤
            4:    ─────╰U(M0)──────╰U(M1)──────╰U(M2)──────╰U(M3)──────╰U(M4)──────╰U(M5)──────╰U(M6)──────╰U(M7)──┤

        Here, we used the symbols

        .. code-block::

            0: ─╭●──       ─●─╮─
            1: ─├●──  and  ─●─┤─
            2:  ╰───       ───╯

        for :class:`~.TemporaryAND` and its adjoint, respectively, and skipped drawing the
        auxiliary wires in areas where they are guaranteed to be in the state :math:`|0\rangle`.
        We will need three simplification rules for pairs of ``TemporaryAND`` gates:

        .. code-block::

            ─○─╮─╭○──   ──     ─○─╮─╭○──   ─╭○─       ─○─╮─╭●──   ─╭●────
            ─○─┤─├○── = ──,    ─○─┤─├●── = ─│──, and  ─●─┤─├○── = ─│──╭●─.
            ───╯ ╰───   ──     ───╯ ╰───   ─╰X─       ───╯ ╰───   ─╰X─╰X─

        Applying these simplifications reduces the computational cost of the ``Select``
        template:

        .. code-block::

            0:    ─╭○────────────────╭○──────────────────╭●─────────────────────╭●─────────────────●╮─┤
            1:    ─├○────────────────│───────────────────│──╭●──────────────────│──────────────────●┤─┤
            aux0:  ╰─╭●─────╭●────●╮─╰X─╭●─────╭●─────●╮─╰X─╰X─╭●─────╭●─────●╮─╰X─╭●─────╭●─────●╮─╯ │
            2:    ───├○─────│─────●┤────├○─────│──────●┤───────├○─────│──────●┤────├○─────│──────●┤───┤
            aux1:    ╰─╭●───╰X─╭●──╯    ╰─╭●───╰X──╭●──╯       ╰─╭●───╰X──╭●──╯    ╰─╭●───╰X──╭●──╯   │
            3:    ─────├U(M0)──├U(M1)─────├U(M2)───├U(M3)────────├U(M4)───├U(M5)─────├U(M6)───├U(M7)──┤
            4:    ─────╰U(M0)──╰U(M1)─────╰U(M2)───╰U(M3)────────╰U(M4)───╰U(M5)─────╰U(M6)───╰U(M7)──┤

        An additional cost reduction then results from the fact that the ``TemporaryAND``
        gate and its adjoint require four and zero :class:`~T` gates, respectively,
        in contrast to the seven ``T`` gates required by a decomposition of :class:`~Toffoli`.

        For general :math:`c` and :math:`K=2^c`, the decomposition takes a similar form, with
        alternating control and auxiliary wires.

        An implementation of the unary iterator is achieved in the following steps:
        We first define a recursive sub-circuit ``R``;
        given :math:`L` operators and :math:`2 \lceil\log_2(L)\rceil + 1` control and
        auxiliary wires, there are three cases that ``R`` distinguishes. First, if ``L>1``,
        it applies the circuit

        .. code-block::

            aux_j:   ╭R   ─╭●────╭●────●─╮─
            j+1:     ├R = ─├○────│─────●─┤─
            aux_j+1: ╰R    ╰──R──╰X─R────╯ ,

        where each label ``R`` symbolizes a call to ``R`` itself, on the next recursion level.
        These next-level calls use
        :math:`L' = 2^{\lceil\log_2(L)\rceil-1}` (i.e. half of :math:`L`, rounded up to the next
        power of two) and :math:`L-L'` (i.e. the rest) operators, respectively.

        Second, if ``L=1``, the single operator is applied, controlled on the first control wire.
        Finally, if ``L=0``, ``R`` does not apply any operators.

        With ``R`` defined, we are ready to outline the main circuit structure:

        #. Apply the left-most ``TemporaryAND`` controlled on qubits ``0`` and ``1``.
        #. Split the target operators into four "quarters" (often with varying sizes)
           and apply the first quarter using ``R``.
        #. Apply ``[X(0), CNOT([0, "aux0"]), X(0)]``.
        #. Apply the second quarter using ``R``.
        #. Apply ``[CNOT([0, "aux0"]), CNOT([1, "aux0"])]``.
        #. Apply the third quarter using ``R``.
        #. Apply ``[CNOT([0, "aux0"])]``.
        #. Apply the last quarter using ``R``.
        #. Apply the right-most ``adjoint(TemporaryAND)`` controlled on qubits ``0`` and ``1``.

        **Partial Select decomposition**

        The unary iterator decomposition of the ``Select`` template can be
        simplified further if both of the following criteria are met:

        #. There are fewer target operators than would maximally be possible for the given
           number of control wires, i.e. :math:`K<2^c`.

        #. The state :math:`|\psi\rangle` of the control wires satisfies
           :math:`\langle j | \psi\rangle=0` for all computational basis states with :math:`j\geq K`.

        We do not derive this reduction here but discuss the modifications to the implementation
        above that result from it.

        Given :math:`K=2^c-b` operators, where :math:`c` is defined as above and we
        have :math:`0\leq b<2^{c-1}`, the nine steps above are modified into one of three variants.
        In each variant, the first :math:`2^{c-1}` operators are applied in two equal portions,
        containing :math:`2^{c-2}` operators each.
        After this, :math:`\ell=2^{c-1} -b` operators remain and the three circuit variants are
        distinguished, based on :math:`\ell`:

        - if :math:`\ell \geq 2^{c-2}`, the following, rather generic, circuit is applied:

          .. code-block::

              0:    ─╭○─────╭○─────╭●────────╭●─────●─╮─
              1:    ─├○─────│──────│──╭●─────│──────●─┤─
              aux0:  ╰──╭R──╰X─╭R──╰X─╰X─╭R──╰X─╭R────╯
              2:    ────├R─────├R────────├R─────├R──────
              aux1:     ╰R     ╰R        ╰R     ╰R      .

          Here, each operator with three ``R`` labels symbolizes a call to ``R``. The first
          call in the second half applies :math:`2^{\lceil\log_2(\ell)\rceil-1}` operators.
          Note that this case is triggered if :math:`K` is larger than or equal to
          :math:`\tfrac{3}{4}` of the maximal capacity for :math:`c` control wires.
          Also note how the two middle ``TemporaryAND`` gates were merged into two CNOTs,
          like for the non-partial Select operator.

        - if :math:`1<\ell < 2^{c-2}`, the following circuit is applied:

          .. code-block::

              0:    ─╭○─────╭○─────○─╮╭●─────╭●─────●─╮─
              1:    ─├○─────│──────●─┤│──────│────────│─
              aux0:  ╰──╭R──╰X─╭R────╯│      │        │
              2:    ────├R─────├R─────├○─────│──────●─┤─
              aux1:     ╰R     ╰R     ╰───R──╰X──R────╯

          where the second half may skip more than one control and auxiliary wire each.
          In this diagram, both the operators with three and one ``R`` labels represent calls to
          ``R``, with single-label instances applying fewer operators.
          The first call to ``R`` in the second half applies :math:`2^{\lceil\log_2(\ell)\rceil-1}`
          operators. The middle elbows act on distinct wire triples and can not be merged as
          above.

        - if :math:`\ell=1`, the following circuit is applied:

          .. code-block::

              0:    ─╭○─────╭○─────○─╮╭●──
              1:    ─├○─────│──────●─┤│───
              aux0:  ╰──╭R──╰X─╭R────╯│───
              2:    ────├R─────├R─────│───
              aux1:     ╰R     ╰R     ╰U  .

          Here, the three connected ``R`` labels symbolize a call to ``R`` and
          apply :math:`2^{c-2}` operators each.
          The controlled gate on the right applies the single remaining operator.

    """

    hybrid_argnames = ("ops",)
    wire_argnames = ("control", "work_wires")
    static_argnames = ("partial",)

    arg_specs = {"control": Wire[-1], "work_wires": Wire[-1]}

    def __init__(self, ops, control, work_wires=None, partial=False):
        work_wires = () if work_wires is None else work_wires
        super().__init__(ops=ops, control=control, work_wires=work_wires, partial=partial)

        if 2 ** len(self.control) < len(self.ops):
            raise ValueError(
                f"Not enough control wires ({len(self.control)}) for the desired number of "
                + f"operations ({len(ops)}). At least {math.ceil_log2(len(ops))} control "
                + "wires are required."
            )

        target_wire_args = tuple(op.wires for op in self.ops if isinstance(op, Operator))
        self._target_wires = Wires.all_wires(target_wire_args)
        all_wire_args = (self.control, self._target_wires)
        self._wires = Wires.all_wires(all_wire_args)

        wire_args = {
            "target_wires": self._target_wires,
            "control": self.control,
            "work_wires": self.work_wires,
        }
        validate_no_wire_overlaps(wire_args)

    @property
    def target_wires(self):
        """The wires of the target operators."""
        return self._target_wires

    @property
    def data(self) -> tuple:
        """The trainable parameters of the target operators."""
        return tuple(d for op in self.ops for d in op.data)

    @property
    def num_params(self) -> int:
        """The number of trainable parameters of the target operators."""
        return sum(op.num_params for op in self.ops)

    def __repr__(self):
        # Overriding so that work_wires aren't included
        return f"Select(ops={self.ops}, control={self.control}, partial={self.partial})"


# Decomposition of Select using multi-control strategy


# Decomposition of Select using unary iterator


def _add_k_units(ops, controls, work_wires, k):
    """Add k controlled-applied operators within the unary iterator scheme, in a recursive
    manner.

    In the documentation, this subroutine is called ``R``.
    This is _not_ used for the outer-most recursion level, see _add_first_k_units instead.

    We are given ``K=len(ops)`` operators and ``2 * ⌈log_2(K)⌉ + 1`` control and auxiliary wires.
    If ``K=0``, nothing is applied.
    If ``K=1``, the single operator is applied, controlled on the first control wire.

    In all other cases, this function applies the circuit

    .. code-block::

        ─╭●────╭●────●─╮─
        ─├○────│─────●─┤─
         ╰──■──╰X─■────╯

    where each box symbolizes calls to itself on the next recursion level.
    The next-level calls to ``_add_k_units` use

    ``k_first = 2 ** (⌈log_2(k)⌉-1)`` (i.e. half of ``k``, rounded up to the next power of two)
    and
    ``k_second = k-k_first`` (i.e. the rest)

    operators, respectively. Accordingly, two fewer control wires are used.

    """
    assert k == len(ops) > 0
    num_bits = math.ceil_log2(k)
    needed_controls = 2 * num_bits + 1
    assert len(controls) >= needed_controls, f"{len(controls)=}, {needed_controls=}"

    if k == 1:
        assert num_bits == 0
        return [ctrl(ops[0], control=controls[0], control_values=[1], work_wires=work_wires)]

    assert num_bits != 0
    controls = controls[:1] + controls[-(needed_controls - 1) :]

    and_wires = controls[:3]
    new_controls = controls[2:]
    k_first = 2 ** (num_bits - 1)
    return (
        [TemporaryAND(and_wires, control_values=(1, 0))]
        + _add_k_units(ops[:k_first], new_controls, work_wires, k_first)
        + [CNOT(and_wires[::2])]
        + _add_k_units(ops[k_first:], new_controls, work_wires, k - k_first)
        + [adjoint(TemporaryAND)(and_wires)]
    )


# Decomposition of Select using one work wire to control the target operations


Multiplexer = Select
r"""Multiplexer(ops, control, work_wires=None, partial=False)

Alias for :class:`~.Select`.
"""

Multiplexor = Select
r"""Multiplexor(ops, control, work_wires=None, partial=False)

Alias for :class:`~.Select`.
"""
