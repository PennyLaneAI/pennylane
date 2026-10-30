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
"""Contains the LeftClassicalComparator template for performing an inequality
test of a quantum register and a classical integer."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, X, cond
from pennylane.typing import AbstractArray, AbstractWires, Int, Wire
from pennylane.wires import Wires, WiresLike

from .temporary_and import Elbow


class LeftClassicalComparator(Operator2):
    r"""This operator performs an inequality test between a quantum register :math:`\lvert x\rangle` and a
    classical integer :math:`\mathrm{value}`, storing the result in a target qubit.

    The operator evaluates the following relation, given by the ``comparator="<"`` argument

    .. math::

        \text{LeftClassicalComparator}_{<} \lvert x\rangle \lvert 0\rangle = \lvert x\rangle \lvert x < \mathrm{value}\rangle \text{ if comparator} = \text{'<'}

    The decomposition is based on the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_, adapted for a classical constant.

    .. warning::

        Note that the decomposition uses auxiliary wires and in order to clean them,
        one must apply the adjoint of this operator after using the target qubit.

    Args:
        x_wires (WiresLike): The wires that store the quantum integer :math:`x`.
        value (int): The classical integer to compare against. It must be non-negative and at most
            :math:`2^{n}`, where :math:`n` is the number of ``x_wires`` (the upper value
            :math:`2^{n}` is only allowed for the ``'<'`` and ``'>='`` comparators, for which the
            test is statically satisfied for every representable ``x``).
        target_wire (WiresLike): The wire that stores the value of the inequality test.
        work_wires (WiresLike): The auxiliary wires to use for the comparison.
            At least ``len(x_wires) - 1`` zeroed work wires should be provided.
            They are not returned in the zero state.
        comparator (str): The operator used in the inequality. Possible values are:
            ``'<'``, ``'<='``, ``'>='`` and ``'>'``.

    **Example**

    .. code-block:: python

        import pennylane as qp

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.BasisState(qp.math.int_to_binary(3, 3), wires=[0, 1, 2])
            qp.LeftClassicalComparator(
                x_wires=[0, 1, 2],
                value=2,
                target_wire=3,
                work_wires=[4, 5],
                comparator=">=",
            )
            return qp.probs(wires=3)

    >>> print(circuit())
    [0. 1.]

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.LeftClassicalComparator
            :symbols: :math:`n` ``x_wires``, the constant :math:`L` and ``comparator`` :math:`c`
            :example: :math:`n=3`, :math:`L=6`, :math:`c={<}`

            .. decomposition-rule:: _left_classical_comparator
                :image: /_static/decomposition_thumbnails/left_classical_comparator_elbow_ladder.svg
                :where: :math:`L' = L + [c \in \{\leq, >\}]`, :math:`b_0 = L' \bmod 2`,
                    :math:`N = \mathrm{wt}(\lfloor L'/2 \rfloor)`, with :math:`\mathrm{wt}` the
                    number of ones in the binary representation, and
                    :math:`[\cdot]` equal to 1 if the condition holds and 0 otherwise. The counts
                    hold for :math:`L' < 2^n`; for :math:`L' = 2^n`, the rule only applies
                    :math:`1 + [c \in \{>, \geq\}]` :class:`~.X` gates

                :class:`~.TemporaryAND` | :math:`n - 1` | 2
                :class:`~.CNOT` | :math:`n - 1 + b_0` | 2
                :class:`~.X` | :math:`2b_0 + 4N + [c \in \{>, \geq\}]` | 8
    """

    wire_argnames = ("x_wires", "target_wire", "work_wires")
    dynamic_argnames = ("value",)
    compilable_argnames = ("comparator",)
    ndim_params = (0,)
    num_params = 0  # The integer threshold is dynamic but is not differentiable.
    arg_specs = {
        "value": Int,
        "x_wires": Wire[-1],
        "target_wire": Wire[1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        x_wires: WiresLike,
        value: int,
        target_wire: WiresLike,
        work_wires: WiresLike,
        comparator: str,
    ):  # pylint: disable=too-many-arguments

        if math.shape(value) != ():
            raise ValueError("value must be a scalar integer.")

        if comparator not in ["<", "<=", ">=", ">"]:
            raise ValueError("Allowed values for 'comparator' are: '<', '<=', '>=' and '>'.")

        if isinstance(x_wires, AbstractWires):
            super().__init__(x_wires, value, target_wire, work_wires, comparator)
            return

        x_wires = Wires(x_wires)
        target_wire = Wires(target_wire)
        work_wires = Wires(work_wires)

        if len(work_wires) < len(x_wires) - 1:
            raise ValueError(
                f"At least {len(x_wires) - 1} work_wires are required, but only "
                f"{len(work_wires)} were provided. (x_wires={list(x_wires)}, "
                f"work_wires={list(work_wires)})"
            )
        overlap = work_wires.intersection(target_wire)
        if overlap:
            raise ValueError(
                f"work_wires and target_wire must be disjoint, but share: {list(overlap)}. "
                f"(work_wires={list(work_wires)}, target_wire={list(target_wire)})"
            )
        overlap = work_wires.intersection(x_wires)
        if overlap:
            raise ValueError(
                f"work_wires and x_wires must be disjoint, but share: {list(overlap)}. "
                f"(work_wires={list(work_wires)}, x_wires={list(x_wires)})"
            )
        overlap = x_wires.intersection(target_wire)
        if overlap:
            raise ValueError(
                f"x_wires and target_wire must be disjoint, but share: {list(overlap)}. "
                f"(x_wires={list(x_wires)}, target_wire={list(target_wire)})"
            )
        # The comparators ``"<"`` and ``">="`` accept the degenerate bound ``value == 2 ** n``
        # (``x < 2 ** n`` is statically True for every representable ``x``); ``"<="`` / ``">"``
        # add one internally, so their largest allowed input stays ``2 ** n - 1``.
        max_value = 2 ** len(x_wires) if comparator in ["<", ">="] else 2 ** len(x_wires) - 1
        if (
            not isinstance(value, AbstractArray)
            and not math.is_tracer(value)
            and not 0 <= value <= max_value
        ):
            raise ValueError(
                f"value must be less than or equal to {max_value} and non-negative for comparator "
                f"'{comparator}'. Got {value=} and {2 ** len(x_wires)=}."
            )

        super().__init__(x_wires, value, target_wire, work_wires, comparator)


def _get_specific_bit(value, i):
    # returns the i-th bit of the binary representation of value in little endian convention.
    return (value >> i) & 1


def _left_classical_comparator_resources(x_wires, value, target_wire, work_wires, comparator, **_):
    """Return conservative worst-case gate counts without knowing the runtime threshold."""
    # pylint: disable=unused-argument
    n = len(x_wires)
    # Count every conditional gate: two Xs for bit 0, four per remaining low bit,
    # and one for bit n. The final negation depends only on the comparator.
    return {
        Elbow: n - 1,
        CNOT: n,
        X: 4 * n - 1 + int(comparator in [">", ">="]),
    }


@register_resources(_left_classical_comparator_resources, exact=False)
def _left_classical_comparator(x_wires, value, target_wire, work_wires, comparator, **_):
    x_wires = x_wires[::-1]

    def _negate_output():
        X(wires=target_wire)

    if comparator in ["<=", ">"]:
        value += 1

    used_work_wires = Wires.all_wires([work_wires[: len(x_wires) - 1], target_wire])

    bit = _get_specific_bit(value, 0)
    cond(bit, X)(wires=[x_wires[0]])
    cond(bit, CNOT)(wires=[x_wires[0], used_work_wires[0]])
    cond(bit, X)(wires=[x_wires[0]])

    if compiler.active() or capture.enabled():
        x_wires = math.array(x_wires, like="jax")
        used_work_wires = math.array(used_work_wires, like="jax")

    @for_loop(1, len(x_wires))
    def _loop(i):
        bit = _get_specific_bit(value, i)
        cond(bit, X)(wires=[x_wires[i]])
        cond(bit, X)(wires=[used_work_wires[i - 1]])
        Elbow(wires=[used_work_wires[i - 1], x_wires[i], used_work_wires[i]])
        cond(bit, X)(wires=[used_work_wires[i - 1]])
        CNOT(wires=[used_work_wires[i - 1], used_work_wires[i]])
        cond(bit, X)(wires=[x_wires[i]])

    _loop()  # pylint: disable=no-value-for-parameter

    # Degenerate bound: when the effective ``value`` equals ``2 ** len(x_wires)`` its
    # low ``n`` bits are all zero, so the bitwise core above computes the constant
    # ``x < 0`` (always False) rather than the intended "always True". Bit ``n`` of
    # ``value`` is set exactly in this case, so a single conditional ``X`` fixes the
    # target wire.
    bit_n = _get_specific_bit(value, len(x_wires))
    cond(bit_n, X)(wires=[used_work_wires[len(x_wires) - 1]])

    cond(comparator.startswith(">"), _negate_output)()


add_decomps(LeftClassicalComparator, _left_classical_comparator)
